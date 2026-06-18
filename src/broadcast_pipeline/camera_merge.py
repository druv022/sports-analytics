"""Post-cluster merge for over-segmented closeup camera clusters."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from broadcast_pipeline.appearance_compat import clusters_have_appearance_conflict
from broadcast_pipeline.camera_debug import CameraClusteringDebug, load_camera_clustering_debug
from broadcast_pipeline.config import PipelineConfig
from src.person_appearance.config import AppearanceConfig
from src.person_appearance.types import SceneAppearance
from src.camera_assignemnt.embedding_cluster.cluster import cluster_id_to_camera_id


def _cluster_centroids(
    debug: CameraClusteringDebug,
    cluster_ids: set[int],
) -> dict[int, np.ndarray]:
    centroids: dict[int, list[np.ndarray]] = {}
    for idx, cluster_id in enumerate(debug.final_cluster_id.tolist()):
        cluster_id = int(cluster_id)
        if cluster_id < 0 or cluster_id not in cluster_ids:
            continue
        centroids.setdefault(cluster_id, []).append(debug.reduced_matrix[idx])
    return {
        cluster_id: np.mean(vectors, axis=0)
        for cluster_id, vectors in centroids.items()
        if vectors
    }


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _group_size(parent: dict[int, int], root: int) -> int:
    return sum(1 for cluster_id in parent if find_parent(parent, cluster_id) == root)


def find_parent(parent: dict[int, int], cluster_id: int) -> int:
    while parent[cluster_id] != cluster_id:
        parent[cluster_id] = parent[parent[cluster_id]]
        cluster_id = parent[cluster_id]
    return cluster_id


def _build_merge_map(
    centroids: dict[int, np.ndarray],
    similarity_threshold: float,
    *,
    max_group_size: int = 3,
    can_merge: Callable[[int, int], bool] | None = None,
) -> dict[int, int]:
    cluster_ids = sorted(centroids.keys())
    parent = {cluster_id: cluster_id for cluster_id in cluster_ids}

    def union(a: int, b: int) -> None:
        if can_merge is not None and not can_merge(a, b):
            return
        root_a = find_parent(parent, a)
        root_b = find_parent(parent, b)
        if root_a == root_b:
            return
        size_a = _group_size(parent, root_a)
        size_b = _group_size(parent, root_b)
        if size_a + size_b > max_group_size:
            return
        parent[root_b] = root_a

    for i, left in enumerate(cluster_ids):
        for right in cluster_ids[i + 1 :]:
            sim = _cosine_similarity(centroids[left], centroids[right])
            if sim >= similarity_threshold:
                union(left, right)

    return {cluster_id: find_parent(parent, cluster_id) for cluster_id in cluster_ids}


def _closeup_cluster_ids(
    frame_results: pd.DataFrame,
    scene_types: pd.DataFrame,
) -> set[int]:
    closeup_scenes = set(
        scene_types.loc[scene_types["scene_type"] != "full_court", "scene_id"].astype(int)
    )
    mask = frame_results["scene_id"].isin(closeup_scenes)
    clusters = frame_results.loc[mask, "cluster_id"].astype(int)
    return {int(c) for c in clusters.tolist() if int(c) >= 0}


def apply_closeup_cluster_merge(
    frame_results: pd.DataFrame,
    config: PipelineConfig,
    scene_types: pd.DataFrame,
    debug_path: Path | None = None,
    *,
    appearances: dict[int, SceneAppearance] | None = None,
    appearance_config: AppearanceConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Merge similar closeup clusters using debug embedding centroids."""
    if not config.camera_merge_closeup_clusters:
        return frame_results, {"merged": False, "groups": []}

    path = debug_path or config.artifact("camera_clustering_debug")
    debug = load_camera_clustering_debug(path)
    if debug is None:
        return frame_results, {"merged": False, "groups": [], "reason": "missing_debug"}

    closeup_clusters = _closeup_cluster_ids(frame_results, scene_types)
    if len(closeup_clusters) < 2:
        return frame_results, {"merged": False, "groups": [], "reason": "too_few_clusters"}

    centroids = _cluster_centroids(debug, closeup_clusters)

    def can_merge(left: int, right: int) -> bool:
        if not appearances or appearance_config is None:
            return True
        return not clusters_have_appearance_conflict(
            left,
            right,
            frame_results,
            appearances,
            appearance_config,
        )

    merge_map = _build_merge_map(
        centroids,
        config.camera_merge_similarity_threshold,
        max_group_size=config.camera_merge_max_group_size,
        can_merge=can_merge if appearances and appearance_config else None,
    )

    groups: dict[int, list[int]] = {}
    for cluster_id, root in merge_map.items():
        groups.setdefault(root, []).append(cluster_id)

    merged_groups = [
        sorted(members) for members in groups.values() if len(members) > 1
    ]
    if not merged_groups:
        return frame_results, {"merged": False, "groups": []}

    updated = frame_results.copy()
    for idx, row in updated.iterrows():
        cluster_id = int(row["cluster_id"])
        if cluster_id < 0 or cluster_id not in merge_map:
            continue
        new_cluster = int(merge_map[cluster_id])
        updated.at[idx, "cluster_id"] = new_cluster
        updated.at[idx, "camera_id"] = cluster_id_to_camera_id(new_cluster) or "unknown"

    log = {
        "merged": True,
        "similarity_threshold": config.camera_merge_similarity_threshold,
        "groups": merged_groups,
        "merge_map": {str(k): v for k, v in merge_map.items()},
    }
    log_path = config.artifact("camera_merge_log")
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    return updated, log
