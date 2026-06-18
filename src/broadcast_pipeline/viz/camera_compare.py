"""Scene-level camera assignment comparison for debug visualization."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.manifold import TSNE

from broadcast_pipeline.camera_debug import CameraClusteringDebug, load_camera_clustering_debug
from broadcast_pipeline.config import PipelineConfig
from src.camera_assignemnt.embedding_cluster.cluster import cluster_id_to_camera_id
from src.camera_assignemnt.embedding_cluster.ensemble import normalize_member_weights

MAX_COMPARE_SCENES = 12

_CLUSTER_COLORS = (
    "#5b9dff",
    "#3ecf8e",
    "#f0b429",
    "#ff6b6b",
    "#b794f4",
    "#63b3ed",
    "#f687b3",
    "#ffd166",
    "#06d6a0",
    "#118ab2",
    "#ef476f",
    "#8338ec",
    "#fb5607",
    "#8ac926",
    "#1982c4",
    "#ffca3a",
)
_NOISE_COLOR = "#666666"


@dataclass
class SceneSelection:
    camera_id: str
    scene_id: int


def parse_vote_counts(raw: object) -> dict[str, int]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return {}
    text = str(raw)
    try:
        parsed = ast.literal_eval(text.replace("'", '"'))
    except (SyntaxError, ValueError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): int(v) for k, v in parsed.items()}


def _cosine_distance(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (norm_a * norm_b))


def _pairwise_cosine_stats(
    indices_a: list[int],
    indices_b: list[int],
    reduced_matrix: NDArray[np.float32],
) -> dict[str, float]:
    if not indices_a or not indices_b:
        return {"mean": 1.0, "min": 1.0, "max": 1.0, "std": 0.0}

    distances: list[float] = []
    for i in indices_a:
        for j in indices_b:
            if i == j:
                continue
            distances.append(_cosine_distance(reduced_matrix[i], reduced_matrix[j]))

    if not distances:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}

    arr = np.array(distances, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std()),
    }


def _pair_coassociation(
    index_a: int,
    index_b: int,
    member_labelings: NDArray[np.int32],
    member_names: list[str],
    member_weights: dict[str, float],
) -> float:
    if index_a == index_b:
        return 1.0
    weights = normalize_member_weights(member_names, member_weights)
    score = 0.0
    for row, name in zip(member_labelings, member_names, strict=True):
        label_a = int(row[index_a])
        label_b = int(row[index_b])
        if label_a >= 0 and label_a == label_b:
            score += weights[name]
    return float(score / sum(weights.values()))


def _scene_coassociation_stats(
    indices_a: list[int],
    indices_b: list[int],
    debug: CameraClusteringDebug,
) -> dict[str, float]:
    if not indices_a or not indices_b:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}

    scores: list[float] = []
    for i in indices_a:
        for j in indices_b:
            if i == j:
                continue
            scores.append(
                _pair_coassociation(
                    i,
                    j,
                    debug.member_labelings,
                    debug.member_names,
                    debug.member_weights,
                )
            )

    if not scores:
        return {"mean": 1.0, "min": 1.0, "max": 1.0, "std": 0.0}

    arr = np.array(scores, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std()),
    }


def _member_agreement(
    indices_a: list[int],
    indices_b: list[int],
    debug: CameraClusteringDebug,
) -> dict[str, float]:
    if not indices_a or not indices_b:
        return {name: 0.0 for name in debug.member_names}

    agreement: dict[str, float] = {}
    for member_idx, name in enumerate(debug.member_names):
        labels = debug.member_labelings[member_idx]
        same = 0
        total = 0
        for i in indices_a:
            for j in indices_b:
                if i == j:
                    continue
                total += 1
                if int(labels[i]) >= 0 and int(labels[i]) == int(labels[j]):
                    same += 1
        agreement[name] = float(same / total) if total else 0.0
    return agreement


def _scene_mode_cluster(indices: list[int], debug: CameraClusteringDebug) -> int:
    if not indices:
        return -1
    values = [int(debug.final_cluster_id[i]) for i in indices]
    return int(pd.Series(values).mode().iloc[0])


def _load_analysis(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / "camera_assignment_analysis.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _gt_for_scene(scene_id: int, analysis: dict[str, Any] | None) -> str | None:
    if analysis is None:
        return None
    for row in analysis.get("scene_detail", []):
        if int(row.get("scene_id", -1)) == int(scene_id):
            return str(row.get("camera_id")) if row.get("camera_id") is not None else None
    return None


def _mapped_cluster_camera(cluster_id: int, analysis: dict[str, Any] | None) -> str | None:
    if analysis is None or cluster_id < 0:
        return None
    mapping = analysis.get("cluster_to_camera_mapping") or {}
    return mapping.get(str(cluster_id)) or mapping.get(int(cluster_id))  # type: ignore[index]


def _build_scene_detail(
    selection: SceneSelection,
    scene_row: pd.Series | None,
    frame_rows: pd.DataFrame,
    debug: CameraClusteringDebug | None,
    analysis: dict[str, Any] | None,
) -> dict[str, Any]:
    scene_id = int(selection.scene_id)
    votes = parse_vote_counts(scene_row["camera_vote_counts_json"]) if scene_row is not None else {}
    total_votes = sum(votes.values())
    winner = str(scene_row["camera_id"]) if scene_row is not None else selection.camera_id
    cluster_id = int(scene_row["cluster_id"]) if scene_row is not None else -1
    winner_share = float(votes.get(winner, 0) / total_votes) if total_votes else 0.0

    frame_labels: list[dict[str, Any]] = []
    scene_frames = frame_rows[frame_rows["scene_id"] == scene_id].sort_values("frame_number")
    for row in scene_frames.itertuples(index=False):
        member_clusters: dict[str, int] = {}
        pre_merge_cluster_id = int(row.cluster_id)
        if debug is not None:
            matches = np.flatnonzero(
                (debug.scene_ids == scene_id)
                & (debug.frame_numbers == int(row.frame_number))
            )
            if len(matches):
                idx = int(matches[0])
                pre_merge_cluster_id = int(debug.final_cluster_id[idx])
                for member_idx, name in enumerate(debug.member_names):
                    member_clusters[name] = int(debug.member_labelings[member_idx, idx])
        frame_labels.append(
            {
                "frame_number": int(row.frame_number),
                "camera_id": str(row.camera_id),
                "cluster_id": int(row.cluster_id),
                "pre_merge_cluster_id": pre_merge_cluster_id,
                "member_clusters": member_clusters,
            }
        )

    indices = debug.indices_for_scene(scene_id) if debug is not None else []
    return {
        "camera_id": selection.camera_id,
        "scene_id": scene_id,
        "cluster_id": cluster_id,
        "camera_vote_counts": votes,
        "winner": winner,
        "winner_share": winner_share,
        "unanimous": len(votes) <= 1,
        "pred_noise": cluster_id < 0,
        "gt_camera": _gt_for_scene(scene_id, analysis),
        "mapped_cluster_camera": _mapped_cluster_camera(cluster_id, analysis),
        "frame_labels": frame_labels,
        "sample_indices": indices,
        "scene_mode_cluster": _scene_mode_cluster(indices, debug) if debug else cluster_id,
    }


def _verdict(
    coassoc_mean: float,
    link_threshold: float | None,
    same_cluster: bool,
    same_camera: bool,
) -> str:
    threshold = link_threshold if link_threshold is not None else 0.5
    if same_camera and same_cluster:
        return "same_cluster"
    if coassoc_mean >= threshold:
        return "would_link"
    if threshold - 0.1 <= coassoc_mean < threshold + 0.1:
        return "borderline"
    return "split"


def _build_explanations(
    per_scene: list[dict[str, Any]],
    pairwise: list[dict[str, Any]],
    debug: CameraClusteringDebug | None,
) -> list[str]:
    explanations: list[str] = []
    threshold = debug.link_threshold if debug and debug.link_threshold is not None else 0.5

    for pair in pairwise:
        a = per_scene[pair["a"]]
        b = per_scene[pair["b"]]
        label_a = f"scene {a['scene_id']} ({a['camera_id']})"
        label_b = f"scene {b['scene_id']} ({b['camera_id']})"

        if pair["verdict"] == "same_cluster":
            explanations.append(f"{label_a} and {label_b} share scene cluster and camera assignment.")
        elif pair["verdict"] == "would_link":
            explanations.append(
                f"{label_a} and {label_b} would cluster together "
                f"(mean co-association {pair['mean_coassoc']:.2f} ≥ {threshold:.2f}) "
                f"but have different scene labels."
            )
        elif pair["verdict"] == "borderline":
            explanations.append(
                f"{label_a} and {label_b} are borderline "
                f"(co-association {pair['mean_coassoc']:.2f} near threshold {threshold:.2f})."
            )
        else:
            member_bits = []
            if debug:
                for name, frac in pair.get("member_agreement", {}).items():
                    if frac < 0.5:
                        member_bits.append(f"{name} disagrees")
            member_text = f" ({', '.join(member_bits)})" if member_bits else ""
            explanations.append(
                f"{label_a} and {label_b} split "
                f"(mean co-association {pair['mean_coassoc']:.2f} < {threshold:.2f})"
                f"{member_text}."
            )

        if not a["unanimous"]:
            explanations.append(
                f"Scene {a['scene_id']} had mixed frame votes: {a['camera_vote_counts']} → winner {a['winner']}."
            )
        if not b["unanimous"]:
            explanations.append(
                f"Scene {b['scene_id']} had mixed frame votes: {b['camera_vote_counts']} → winner {b['winner']}."
            )
        if a["pred_noise"] and a["winner"] != "unknown":
            explanations.append(
                f"Scene {a['scene_id']} is noise at cluster level but majority vote picked {a['winner']}."
            )
        if b["pred_noise"] and b["winner"] != "unknown":
            explanations.append(
                f"Scene {b['scene_id']} is noise at cluster level but majority vote picked {b['winner']}."
            )

    return explanations


def _cluster_color(cluster_id: int, color_map: dict[int, str]) -> str:
    if cluster_id < 0:
        return _NOISE_COLOR
    return color_map.get(cluster_id, _NOISE_COLOR)


def _build_cluster_color_map(cluster_ids: list[int]) -> dict[int, str]:
    valid = sorted({int(c) for c in cluster_ids if int(c) >= 0})
    return {cluster_id: _CLUSTER_COLORS[idx % len(_CLUSTER_COLORS)] for idx, cluster_id in enumerate(valid)}


def _load_frame_camera_lookup(output_dir: Path) -> dict[tuple[int, int], dict[str, Any]]:
    path = output_dir / "frame_camera_results.csv"
    if not path.is_file():
        return {}
    lookup: dict[tuple[int, int], dict[str, Any]] = {}
    for row in pd.read_csv(path).itertuples(index=False):
        lookup[(int(row.scene_id), int(row.frame_number))] = {
            "cluster_id": int(row.cluster_id),
            "camera_id": str(row.camera_id),
        }
    return lookup


def _load_scene_assignment_lookup(output_dir: Path) -> dict[int, dict[str, Any]]:
    path = output_dir / "scene_assignments.csv"
    if not path.is_file():
        return {}
    lookup: dict[int, dict[str, Any]] = {}
    for row in pd.read_csv(path).itertuples(index=False):
        lookup[int(row.scene_id)] = {
            "cluster_id": int(row.cluster_id),
            "camera_id": str(row.camera_id),
        }
    return lookup


def _scene_ids_for_cameras(scene_assignments: pd.DataFrame, camera_ids: set[str]) -> list[int]:
    if not camera_ids or scene_assignments.empty:
        return []
    mask = scene_assignments["camera_id"].astype(str).isin(camera_ids)
    return scene_assignments.loc[mask, "scene_id"].astype(int).tolist()


def _projection_coords(matrix: NDArray[np.float32], *, random_state: int = 0) -> tuple[NDArray[np.float32], str]:
    n_samples = int(matrix.shape[0])
    if n_samples == 0:
        return np.zeros((0, 2), dtype=np.float32), "none"
    if n_samples == 1:
        return np.zeros((1, 2), dtype=np.float32), "single_sample"
    if n_samples == 2:
        centered = matrix - matrix.mean(axis=0)
        return centered[:, :2] if centered.shape[1] >= 2 else np.zeros((2, 2), dtype=np.float32), "centered_2d"

    if n_samples < 4:
        centered = matrix - matrix.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        return (centered @ vt[:2].T).astype(np.float32), "pca"

    perplexity = float(max(2.0, min(30.0, (n_samples - 1) / 3.0)))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(matrix).astype(np.float32), "tsne"


def build_global_projection(
    debug: CameraClusteringDebug,
    highlight_scene_ids: list[int] | None = None,
    *,
    output_dir: Path | None = None,
    random_state: int = 0,
) -> dict[str, Any]:
    """Project ensemble reduced features with t-SNE; color by post-merge pipeline cluster IDs."""
    matrix = debug.reduced_matrix.astype(np.float32)
    coords, projection_method = _projection_coords(matrix, random_state=random_state)

    frame_lookup = _load_frame_camera_lookup(output_dir) if output_dir is not None else {}
    scene_lookup = _load_scene_assignment_lookup(output_dir) if output_dir is not None else {}

    highlight = {int(s) for s in (highlight_scene_ids or [])}
    cluster_ids_for_color: list[int] = []
    points: list[dict[str, Any]] = []

    for idx in range(debug.n_samples):
        scene_id = int(debug.scene_ids[idx])
        frame_number = int(debug.frame_numbers[idx])
        pre_merge_cluster_id = int(debug.final_cluster_id[idx])
        pre_merge_camera_id = str(debug.final_camera_id[idx])

        frame_label = frame_lookup.get((scene_id, frame_number), {})
        scene_label = scene_lookup.get(scene_id, {})

        cluster_id = int(frame_label.get("cluster_id", pre_merge_cluster_id))
        camera_id = str(frame_label.get("camera_id", pre_merge_camera_id))
        scene_cluster_id = int(scene_label.get("cluster_id", cluster_id)) if scene_label else cluster_id
        scene_camera_id = str(scene_label.get("camera_id", camera_id)) if scene_label else camera_id

        cluster_ids_for_color.append(cluster_id)
        points.append(
            {
                "x": float(coords[idx, 0]),
                "y": float(coords[idx, 1]) if coords.shape[1] > 1 else 0.0,
                "scene_id": scene_id,
                "frame_number": frame_number,
                "cluster_id": cluster_id,
                "camera_id": camera_id,
                "scene_cluster_id": scene_cluster_id,
                "scene_camera_id": scene_camera_id,
                "pre_merge_cluster_id": pre_merge_cluster_id,
                "pre_merge_camera_id": pre_merge_camera_id,
                "highlighted": scene_id in highlight,
            }
        )

    color_map = _build_cluster_color_map(cluster_ids_for_color)
    legend = [
        {
            "cluster_id": cluster_id,
            "camera_id": cluster_id_to_camera_id(cluster_id) or f"cam_{cluster_id}",
            "color": color,
        }
        for cluster_id, color in sorted(color_map.items(), key=lambda item: item[0])
    ]
    if any(int(c) < 0 for c in cluster_ids_for_color):
        legend.append({"cluster_id": -1, "camera_id": "noise", "color": _NOISE_COLOR})

    for point in points:
        point["color"] = _cluster_color(int(point["cluster_id"]), color_map)

    label_source = "frame_camera_results" if frame_lookup else "camera_clustering_debug_pre_merge"

    return {
        "method": debug.method,
        "projection_method": projection_method,
        "label_source": label_source,
        "n_samples": debug.n_samples,
        "link_threshold": debug.link_threshold,
        "highlight_scene_ids": sorted(highlight),
        "points": points,
        "legend": legend,
    }


def compare_scenes(
    output_dir: Path,
    selections: list[SceneSelection],
    include_global: bool = False,
) -> dict[str, Any]:
    if len(selections) < 2:
        raise ValueError("At least two scene selections are required.")
    if len(selections) > MAX_COMPARE_SCENES:
        raise ValueError(f"At most {MAX_COMPARE_SCENES} scenes can be compared at once.")

    config = PipelineConfig(output_dir=Path(output_dir))
    scene_assignments = pd.read_csv(config.artifact("scene_assignments"))
    scene_lookup = scene_assignments.set_index("scene_id")

    frame_camera_path = config.artifact("frame_camera_results")
    if frame_camera_path.is_file():
        frame_rows = pd.read_csv(frame_camera_path)
    else:
        frame_rows = pd.DataFrame()

    debug = load_camera_clustering_debug(config.artifact("camera_clustering_debug"))
    analysis = _load_analysis(config.output_dir)

    for sel in selections:
        if int(sel.scene_id) not in scene_lookup.index:
            raise ValueError(f"Unknown scene_id: {sel.scene_id}")
        assigned_camera = str(scene_lookup.loc[int(sel.scene_id), "camera_id"])
        if assigned_camera != sel.camera_id:
            raise ValueError(
                f"Scene {sel.scene_id} is assigned to {assigned_camera}, not {sel.camera_id}."
            )

    per_scene: list[dict[str, Any]] = []
    for sel in selections:
        scene_row = scene_lookup.loc[int(sel.scene_id)]
        per_scene.append(
            _build_scene_detail(sel, scene_row, frame_rows, debug, analysis)
        )

    pairwise: list[dict[str, Any]] = []
    has_metrics = debug is not None
    link_threshold = debug.link_threshold if debug else None

    for i in range(len(selections)):
        for j in range(i + 1, len(selections)):
            indices_a = per_scene[i]["sample_indices"]
            indices_b = per_scene[j]["sample_indices"]
            cosine = (
                _pairwise_cosine_stats(indices_a, indices_b, debug.reduced_matrix)
                if debug is not None
                else {"mean": None, "min": None, "max": None, "std": None}
            )
            coassoc = (
                _scene_coassociation_stats(indices_a, indices_b, debug)
                if debug is not None
                else {"mean": None, "min": None, "max": None, "std": None}
            )
            member_agreement = (
                _member_agreement(indices_a, indices_b, debug) if debug is not None else {}
            )
            same_cluster = per_scene[i]["scene_mode_cluster"] == per_scene[j]["scene_mode_cluster"]
            same_camera = per_scene[i]["winner"] == per_scene[j]["winner"]
            coassoc_mean = coassoc["mean"] if coassoc["mean"] is not None else 0.0
            verdict = _verdict(coassoc_mean, link_threshold, same_cluster, same_camera)

            pairwise.append(
                {
                    "a": i,
                    "b": j,
                    "mean_cosine": cosine["mean"],
                    "cosine_min": cosine["min"],
                    "cosine_max": cosine["max"],
                    "cosine_std": cosine["std"],
                    "mean_coassoc": coassoc["mean"],
                    "coassoc_min": coassoc["min"],
                    "coassoc_max": coassoc["max"],
                    "coassoc_std": coassoc["std"],
                    "member_agreement": member_agreement,
                    "same_cluster_id": same_cluster,
                    "same_camera_id": same_camera,
                    "verdict": verdict,
                }
            )

    explanations = _build_explanations(per_scene, pairwise, debug)

    payload: dict[str, Any] = {
        "selections": [{"camera_id": s.camera_id, "scene_id": s.scene_id} for s in selections],
        "has_debug_artifact": debug is not None,
        "has_frame_camera_results": not frame_rows.empty,
        "pairwise": pairwise,
        "per_scene": per_scene,
        "explanations": explanations,
        "global": {
            "method": debug.method if debug else None,
            "n_samples": debug.n_samples if debug else 0,
            "link_threshold": link_threshold,
            "noise_threshold": debug.noise_threshold if debug else None,
            "member_names": debug.member_names if debug else [],
        },
    }

    if include_global and debug is not None:
        highlight_cameras = {s.camera_id for s in selections}
        highlight_scene_ids = _scene_ids_for_cameras(scene_assignments, highlight_cameras)
        payload["global_projection"] = build_global_projection(
            debug,
            highlight_scene_ids=highlight_scene_ids,
            output_dir=config.output_dir,
        )

    return payload
