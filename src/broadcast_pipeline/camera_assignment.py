from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from broadcast_pipeline.appearance_compat import (
    appearance_config_from_pipeline,
    appearance_lookup,
    load_scene_appearances,
    reconcile_scene_assignments,
)
from broadcast_pipeline.camera_debug import load_camera_clustering_debug, write_camera_clustering_debug
from broadcast_pipeline.camera_merge import apply_closeup_cluster_merge
from broadcast_pipeline.config import PipelineConfig
from src.accelerator.device import resolve_torch_device
from src.camera_assignemnt.embedding_cluster.config import (
    ClusterConfig,
    EmbeddingConfig,
    EnsembleConfig,
    PipelineConfig as CameraPipelineConfig,
)
from src.camera_assignemnt.embedding_cluster.dataset import load_scene_samples
from src.camera_assignemnt.embedding_cluster.models import PipelineOutput
from src.camera_assignemnt.embedding_cluster.pipeline import assign_cameras


def _project_root(output_dir: Path) -> Path:
    return output_dir.resolve().parent.parent


def _ensemble_config(config: PipelineConfig) -> EnsembleConfig:
    root = _project_root(config.output_dir)
    tuning_path = root / "data" / "evaluation" / "ensemble_tuning.json"
    return EnsembleConfig(tuning_path=str(tuning_path))


def _scene_midpoint_frame(scene_id: int, frame_index: pd.DataFrame) -> int:
    scene_frames = frame_index[frame_index["scene_id"] == scene_id]["frame_number"]
    if scene_frames.empty:
        return 0
    return int(scene_frames.median())


def _run_clustering(
    config: PipelineConfig,
    frame_index: pd.DataFrame,
    frame_index_path: Path,
    samples: list,
    apply_temporal: bool,
) -> PipelineOutput:
    method = "hsv" if config.fast_cameras else config.ensemble_method
    camera_pipeline = CameraPipelineConfig(
        method=method,
        samples_dir=str(config.output_dir / "frames"),
        metadata_csv=str(frame_index_path),
        cluster=ClusterConfig(apply_temporal=apply_temporal),
        embedding=EmbeddingConfig(device=resolve_torch_device(config.accelerator)),
        ensemble=_ensemble_config(config),
    )
    return assign_cameras(
        camera_pipeline,
        samples=samples,
        apply_temporal=apply_temporal,
    )


def _cluster_camera_samples(
    config: PipelineConfig,
    frame_index: pd.DataFrame,
    frame_index_path: Path,
) -> PipelineOutput:
    apply_temporal = config.camera_samples_per_scene <= 1
    all_samples = load_scene_samples(
        samples_dir=str(config.output_dir / "frames"),
        metadata_csv=str(frame_index_path),
        load_frames=False,
        sample_filter="camera",
    )
    if not all_samples:
        return PipelineOutput(method="hsv" if config.fast_cameras else config.ensemble_method)
    return _run_clustering(config, frame_index, frame_index_path, all_samples, apply_temporal)


def _scene_level_temporal_fill(frame_results: pd.DataFrame) -> pd.DataFrame:
    updated = frame_results.copy()
    for scene_id, group in updated.groupby("scene_id"):
        ordered = group.sort_values("frame_number")
        indices = ordered.index.tolist()
        camera_ids = ordered["camera_id"].astype(str).tolist()
        for pos, idx in enumerate(indices):
            if camera_ids[pos] not in ("unknown", "None", "nan", ""):
                continue
            neighbors = [
                camera_ids[j]
                for j in range(max(0, pos - 2), min(len(camera_ids), pos + 3))
                if j != pos and camera_ids[j] not in ("unknown", "None", "nan", "")
            ]
            if neighbors:
                winner = Counter(neighbors).most_common(1)[0][0]
                updated.at[idx, "camera_id"] = winner
    return updated


def majority_vote_per_scene(
    frame_results: pd.DataFrame,
    frame_index: pd.DataFrame,
    *,
    min_vote_share: float | None = None,
) -> pd.DataFrame:
    camera_frames = frame_index[frame_index["sample_role"] == "camera"][
        ["scene_id", "frame_number"]
    ]
    scene_rows: list[dict] = []
    prev_winner: str | None = None

    for scene_id, group in frame_results.groupby("scene_id", sort=True):
        group = group.copy()
        counts = Counter(group["camera_id"].fillna("unknown").astype(str).tolist())
        total_votes = sum(counts.values())
        top_count = counts.most_common(1)[0][1]
        leaders = [cam for cam, cnt in counts.items() if cnt == top_count]
        vote_share = top_count / total_votes if total_votes else 0.0
        assignment_method = "majority"

        if len(leaders) == 1 and len(counts) == 1:
            assignment_method = "unanimous"
            winner = leaders[0]
        elif len(leaders) == 1:
            winner = leaders[0]
        elif prev_winner in leaders:
            winner = prev_winner
            assignment_method = "temporal_tiebreak"
        else:
            midpoint = _scene_midpoint_frame(int(scene_id), camera_frames)
            subset = group.copy()
            subset["frame_number"] = pd.to_numeric(subset["frame_number"], errors="coerce")
            subset["dist"] = (subset["frame_number"] - midpoint).abs()
            nearest = subset.sort_values("dist").iloc[0]
            winner = str(nearest["camera_id"]) if pd.notna(nearest["camera_id"]) else "unknown"
            assignment_method = "temporal_tiebreak"

        if (
            min_vote_share is not None
            and vote_share < min_vote_share
            and assignment_method not in {"unanimous", "temporal_tiebreak"}
        ):
            midpoint = _scene_midpoint_frame(int(scene_id), camera_frames)
            subset = group.copy()
            subset["frame_number"] = pd.to_numeric(subset["frame_number"], errors="coerce")
            subset["dist"] = (subset["frame_number"] - midpoint).abs()
            mid_row = subset.sort_values("dist").iloc[0]
            winner = str(mid_row["camera_id"]) if pd.notna(mid_row["camera_id"]) else winner
            assignment_method = "mid_frame"

        winner_frames = group[group["camera_id"].astype(str) == winner]
        cluster_vals = winner_frames["cluster_id"].dropna().astype(int)
        if not cluster_vals.empty:
            non_negative = cluster_vals[cluster_vals >= 0]
            if not non_negative.empty:
                cluster_id = int(non_negative.mode().iloc[0])
            else:
                cluster_id = -1
        else:
            cluster_vals_all = group["cluster_id"].dropna().astype(int)
            cluster_id = int(cluster_vals_all.mode().iloc[0]) if not cluster_vals_all.empty else -1

        scene_rows.append(
            {
                "scene_id": int(scene_id),
                "camera_id": winner,
                "cluster_id": cluster_id,
                "camera_vote_counts_json": str(dict(counts)),
                "assignment_method": assignment_method,
                "vote_share": round(vote_share, 4),
            }
        )
        prev_winner = winner

    return pd.DataFrame(scene_rows).sort_values("scene_id")


def _results_to_frame_rows(
    output: PipelineOutput,
    frame_index: pd.DataFrame,
) -> pd.DataFrame:
    path_lookup = frame_index[frame_index["sample_role"] == "camera"].copy()
    path_lookup["frame_path"] = path_lookup["frame_path"].astype(str)

    frame_rows: list[dict] = []
    for result in output.results or []:
        result_path = str(Path(result.frame_path).resolve())
        matches = path_lookup[path_lookup["frame_path"] == result_path]
        if matches.empty:
            matches = path_lookup[
                path_lookup["frame_path"].str.endswith(Path(result.frame_path).name)
            ]
        if matches.empty:
            continue
        match = matches.iloc[0]
        frame_rows.append(
            {
                "scene_id": int(match["scene_id"]),
                "frame_number": int(match["frame_number"]),
                "frame_path": result_path,
                "sample_role": "camera",
                "cluster_id": int(result.cluster_id),
                "camera_id": result.camera_id or "unknown",
            }
        )
    return pd.DataFrame(frame_rows)


def assign_cameras_multi_frame(
    config: PipelineConfig,
    frame_index: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign cameras: global cluster → temporal fill → debug → merge → vote → reconcile."""
    frame_index_path = config.artifact("frame_index")
    frame_index_path.parent.mkdir(parents=True, exist_ok=True)
    frame_index.to_csv(frame_index_path, index=False)

    output = _cluster_camera_samples(config, frame_index, frame_index_path)
    frame_results = _results_to_frame_rows(output, frame_index)
    if frame_results.empty:
        raise RuntimeError("Camera assignment produced no frame results")

    if config.camera_scene_temporal_fill:
        frame_results = _scene_level_temporal_fill(frame_results)

    debug_path = config.artifact("camera_clustering_debug")
    clustering_debug = None
    if config.persist_camera_debug:
        write_camera_clustering_debug(debug_path, output, frame_index)
        clustering_debug = load_camera_clustering_debug(debug_path)

    scene_types = pd.read_csv(config.artifact("scene_types")) if config.artifact("scene_types").is_file() else pd.DataFrame()
    appearances = load_scene_appearances(config)
    appearance_cfg = appearance_config_from_pipeline(config) if appearances else None
    appearance_map = appearance_lookup(appearances) if appearances else {}

    if not scene_types.empty and config.camera_merge_closeup_clusters:
        frame_results, _merge_log = apply_closeup_cluster_merge(
            frame_results,
            config,
            scene_types,
            appearances=appearance_map,
            appearance_config=appearance_cfg,
        )

    frame_camera_results_path = config.artifact("frame_camera_results")
    frame_results.to_csv(frame_camera_results_path, index=False)

    scene_assignments = majority_vote_per_scene(
        frame_results,
        frame_index,
        min_vote_share=config.camera_min_vote_share,
    )
    scene_assignments = reconcile_scene_assignments(
        scene_assignments,
        appearances,
        config,
        clustering_debug=clustering_debug,
    )
    scene_lookup = scene_assignments.set_index("scene_id")

    assignment_rows: list[dict] = []
    for row in frame_index.itertuples(index=False):
        scene_id = int(getattr(row, "scene_id"))
        scene_row = scene_lookup.loc[scene_id]
        assignment_rows.append(
            {
                "scene_id": scene_id,
                "frame_number": int(getattr(row, "frame_number")),
                "seconds": float(getattr(row, "seconds")),
                "frame_path": str(getattr(row, "frame_path")),
                "sample_role": str(getattr(row, "sample_role")),
                "camera_id": scene_row["camera_id"],
                "cluster_id": int(scene_row["cluster_id"]),
            }
        )

    frame_assignments = pd.DataFrame(assignment_rows)
    return scene_assignments, frame_assignments
