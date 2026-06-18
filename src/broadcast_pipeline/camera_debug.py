"""Persist camera clustering debug artifacts for visualization and comparison."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.camera_assignemnt.embedding_cluster.models import ClusterResult, PipelineOutput

_SCENE_FRAME_PATTERN = re.compile(r"scene_(\d+)_frame_(\d+)", re.IGNORECASE)


@dataclass
class CameraClusteringDebug:
    reduced_matrix: NDArray[np.float32]
    scene_ids: NDArray[np.int32]
    frame_numbers: NDArray[np.int32]
    final_cluster_id: NDArray[np.int32]
    final_camera_id: NDArray[np.str_]
    member_labelings: NDArray[np.int32]
    member_names: list[str]
    method: str
    link_threshold: float | None
    noise_threshold: float | None
    dbscan_eps: float | None
    member_weights: dict[str, float]

    @property
    def n_samples(self) -> int:
        return int(len(self.scene_ids))

    def indices_for_scene(self, scene_id: int) -> list[int]:
        mask = self.scene_ids == int(scene_id)
        return [int(i) for i in np.flatnonzero(mask)]


def _parse_frame_number(frame_path: str) -> int:
    match = _SCENE_FRAME_PATTERN.search(Path(frame_path).stem)
    return int(match.group(2)) if match else 0


def _align_result_indices(
    results: list[ClusterResult],
    frame_index: pd.DataFrame,
) -> tuple[list[int], list[int]]:
    camera_frames = frame_index[frame_index["sample_role"] == "camera"].copy()
    camera_frames["frame_path"] = camera_frames["frame_path"].astype(str)
    path_to_row: dict[str, tuple[int, int]] = {}
    for row in camera_frames.itertuples(index=False):
        path = str(row.frame_path)
        path_to_row[path] = (int(row.scene_id), int(row.frame_number))
        path_to_row[Path(path).name] = (int(row.scene_id), int(row.frame_number))

    scene_ids: list[int] = []
    frame_numbers: list[int] = []
    for result in results:
        result_path = str(Path(result.frame_path).resolve())
        key = path_to_row.get(result_path) or path_to_row.get(Path(result_path).name)
        if key is None:
            scene_ids.append(int(result.scene_id))
            frame_numbers.append(_parse_frame_number(result.frame_path))
        else:
            scene_ids.append(key[0])
            frame_numbers.append(key[1])
    return scene_ids, frame_numbers


def write_camera_clustering_debug(
    path: Path,
    output: PipelineOutput,
    frame_index: pd.DataFrame,
) -> None:
    """Write NPZ debug bundle aligned with clustering sample order."""
    if not output.results or output.reduced_matrix is None:
        return

    scene_ids, frame_numbers = _align_result_indices(output.results, frame_index)
    final_cluster_id = np.array([int(r.cluster_id) for r in output.results], dtype=np.int32)
    final_camera_id = np.array(
        [str(r.camera_id or "unknown") for r in output.results],
        dtype="U32",
    )

    member_names = list(output.ensemble_member_names or [output.method])
    if output.ensemble_member_labelings:
        member_labelings = np.stack(
            [labels.astype(np.int32) for labels in output.ensemble_member_labelings],
            axis=0,
        )
    else:
        member_labelings = final_cluster_id.reshape(1, -1)

    meta = {
        "method": output.method,
        "link_threshold": output.ensemble_vote_threshold,
        "noise_threshold": output.ensemble_noise_threshold,
        "dbscan_eps": output.dbscan_eps,
        "member_weights": output.ensemble_member_weights or {},
        "member_names": member_names,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        reduced_matrix=output.reduced_matrix.astype(np.float32),
        scene_ids=np.array(scene_ids, dtype=np.int32),
        frame_numbers=np.array(frame_numbers, dtype=np.int32),
        final_cluster_id=final_cluster_id,
        final_camera_id=final_camera_id,
        member_labelings=member_labelings,
        meta_json=np.array([json.dumps(meta)]),
    )


def load_camera_clustering_debug(path: Path) -> CameraClusteringDebug | None:
    if not path.is_file():
        return None

    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta_json"][0]))
    member_names = list(meta.get("member_names", []))
    member_weights = {str(k): float(v) for k, v in (meta.get("member_weights") or {}).items()}

    return CameraClusteringDebug(
        reduced_matrix=data["reduced_matrix"].astype(np.float32),
        scene_ids=data["scene_ids"].astype(np.int32),
        frame_numbers=data["frame_numbers"].astype(np.int32),
        final_cluster_id=data["final_cluster_id"].astype(np.int32),
        final_camera_id=data["final_camera_id"].astype(str),
        member_labelings=data["member_labelings"].astype(np.int32),
        member_names=member_names,
        method=str(meta.get("method", "unknown")),
        link_threshold=_optional_float(meta.get("link_threshold")),
        noise_threshold=_optional_float(meta.get("noise_threshold")),
        dbscan_eps=_optional_float(meta.get("dbscan_eps")),
        member_weights=member_weights,
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
