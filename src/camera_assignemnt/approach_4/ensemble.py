"""Ensemble cluster assignment via weighted co-association voting."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.csgraph import connected_components

from src.camera_assignemnt.approach_4.cluster import (
    apply_cluster_labels,
    temporal_fill,
)
from src.camera_assignemnt.approach_4.models import ClusterResult


def normalize_member_weights(
    member_names: list[str],
    member_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Return normalized weights keyed by member name."""
    if not member_names:
        raise ValueError("member_names must not be empty")

    if member_weights is None:
        weight = 1.0 / len(member_names)
        return {name: weight for name in member_names}

    weights = {name: float(member_weights.get(name, 0.0)) for name in member_names}
    total = sum(weights.values())
    if total <= 0:
        weight = 1.0 / len(member_names)
        return {name: weight for name in member_names}
    return {name: value / total for name, value in weights.items()}


def coassociation_matrix(
    labelings: list[NDArray[np.int64]],
    member_names: list[str],
    member_weights: dict[str, float] | None = None,
) -> NDArray[np.float32]:
    """Weighted fraction of base clusterings that place each pair in the same cluster."""
    if not labelings:
        raise ValueError("Need at least one labeling for co-association.")

    n = len(labelings[0])
    if any(len(labels) != n for labels in labelings):
        raise ValueError("All labelings must have the same length.")
    if len(labelings) != len(member_names):
        raise ValueError("labelings and member_names must have the same length.")

    weights = normalize_member_weights(member_names, member_weights)
    coassoc = np.eye(n, dtype=np.float32)
    weight_sum = float(sum(weights.values()))

    for labels, name in zip(labelings, member_names):
        weight = weights[name]
        for cluster_id in set(labels):
            if cluster_id < 0:
                continue
            members = np.flatnonzero(labels == cluster_id)
            if len(members) < 2:
                continue
            idx = np.ix_(members, members)
            coassoc[idx] += weight

    if weight_sum > 0:
        off_diag = ~np.eye(n, dtype=bool)
        coassoc[off_diag] /= weight_sum

    return coassoc


def consensus_labels(
    coassoc: NDArray[np.float32],
    link_threshold: float = 0.5,
) -> NDArray[np.int64]:
    """Merge scenes when weighted co-association exceeds the link threshold."""
    if coassoc.shape[0] == 0:
        return np.array([], dtype=np.int64)
    if coassoc.shape[0] == 1:
        return np.zeros(1, dtype=np.int64)

    adjacency = coassoc >= link_threshold
    _, labels = connected_components(adjacency, directed=False, connection="weak")
    return labels.astype(np.int64)


def member_noise_mask(
    labelings: list[NDArray[np.int64]],
    member_names: list[str],
    member_weights: dict[str, float] | None = None,
    noise_threshold: float = 0.6,
) -> NDArray[np.bool_]:
    """Mark scenes as noise when weighted noise votes exceed the threshold."""
    weights = normalize_member_weights(member_names, member_weights)
    noise_score = np.zeros(len(labelings[0]), dtype=np.float32)

    for labels, name in zip(labelings, member_names):
        noise_score += weights[name] * (labels < 0).astype(np.float32)

    return noise_score >= noise_threshold


def vote_cluster_assignments(
    results: list[ClusterResult],
    labelings: list[NDArray[np.int64]],
    member_names: list[str],
    reduced: NDArray[np.float32],
    member_weights: dict[str, float] | None = None,
    link_threshold: float = 0.5,
    noise_threshold: float = 0.6,
    temporal_window: int = 4,
) -> list[ClusterResult]:
    """Combine base cluster labelings into a single consensus assignment."""
    coassoc = coassociation_matrix(labelings, member_names, member_weights)
    consensus = consensus_labels(coassoc, link_threshold=link_threshold)
    noise_mask = member_noise_mask(
        labelings,
        member_names,
        member_weights,
        noise_threshold=noise_threshold,
    )

    final_labels = consensus.copy()
    final_labels[noise_mask] = -1

    voted = [
        ClusterResult(
            scene_idx=r.scene_idx,
            scene_id=r.scene_id,
            frame_path=r.frame_path,
            features=r.features,
        )
        for r in results
    ]
    apply_cluster_labels(voted, final_labels, reduced)
    return temporal_fill(voted, temporal_window)


def member_summary(labelings: list[NDArray[np.int64]], names: list[str]) -> dict[str, dict]:
    """Summarize each base clustering for ensemble reporting."""
    summary: dict[str, dict] = {}
    for name, labels in zip(names, labelings):
        valid = labels[labels >= 0]
        summary[name] = {
            "n_clusters": int(len(set(valid))),
            "n_noise": int(np.sum(labels < 0)),
        }
    return summary
