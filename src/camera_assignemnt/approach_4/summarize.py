"""Unsupervised cluster quality metrics."""

from __future__ import annotations

from collections import Counter

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import davies_bouldin_score, silhouette_score

from src.camera_assignemnt.approach_4.models import ClusterResult


def _cluster_sizes(labels: NDArray[np.int64]) -> dict[str, int]:
    counts = Counter(int(x) for x in labels)
    return {str(k): v for k, v in sorted(counts.items())}


def temporal_coherence(results: list[ClusterResult], window: int = 4) -> float:
    """Fraction of scenes whose cluster matches the majority among neighbours."""
    if len(results) <= 1:
        return 1.0

    labels = np.array([r.cluster_id for r in results], dtype=np.int64)
    agree = 0
    total = 0

    for i, label in enumerate(labels):
        if label < 0:
            continue
        start = max(0, i - window)
        end = min(len(labels), i + window + 1)
        neighbours = [labels[j] for j in range(start, end) if j != i and labels[j] >= 0]
        if not neighbours:
            continue
        majority = Counter(neighbours).most_common(1)[0][0]
        agree += int(label == majority)
        total += 1

    return agree / total if total else 0.0


def within_cluster_variance(reduced: NDArray[np.float32], labels: NDArray[np.int64]) -> float:
    """Mean pairwise cosine distance inside each non-noise cluster."""
    unique = sorted(set(labels) - {-1})
    if not unique:
        return float("nan")

    distances: list[float] = []
    for cluster in unique:
        members = reduced[labels == cluster]
        if len(members) < 2:
            continue
        norms = members / (np.linalg.norm(members, axis=1, keepdims=True) + 1e-8)
        sim = norms @ norms.T
        dist = 1.0 - sim
        triu = dist[np.triu_indices(len(members), k=1)]
        if triu.size:
            distances.append(float(triu.mean()))

    return float(np.mean(distances)) if distances else float("nan")


def summarize_clusters(
    results: list[ClusterResult],
    reduced: NDArray[np.float32],
    eps: float,
    method: str,
    temporal_window: int = 4,
) -> dict:
    """Build unsupervised summary dict for JSON export."""
    labels = np.array([r.cluster_id for r in results], dtype=np.int64)
    camera_ids = [r.camera_id or "unknown" for r in results]

    n_noise = int((labels == -1).sum())
    n_clusters = len(set(labels) - {-1})

    metrics: dict[str, float | None] = {
        "silhouette": None,
        "davies_bouldin": None,
        "within_cluster_variance": within_cluster_variance(reduced, labels),
        "temporal_coherence": temporal_coherence(results, window=temporal_window),
        "noise_rate": n_noise / len(results) if results else 0.0,
        "unknown_rate": camera_ids.count("unknown") / len(results) if results else 0.0,
    }

    valid = labels[labels >= 0]
    if len(set(valid)) >= 2 and len(valid) >= 3:
        mask = labels >= 0
        metrics["silhouette"] = float(silhouette_score(reduced[mask], labels[mask]))
        metrics["davies_bouldin"] = float(davies_bouldin_score(reduced[mask], labels[mask]))

    assignments = [
        {
            "scene_idx": r.scene_idx,
            "scene_id": r.scene_id,
            "frame_path": r.frame_path,
            "cluster_id": r.cluster_id,
            "camera_id": r.camera_id,
        }
        for r in results
    ]

    return {
        "method": method,
        "n_scenes": len(results),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "dbscan_eps": eps,
        "cluster_sizes": _cluster_sizes(labels),
        "metrics": metrics,
        "assignments": assignments,
    }
