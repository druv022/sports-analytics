"""Feature reduction, HDBSCAN/DBSCAN clustering, and temporal fill."""

from __future__ import annotations

import sys
import warnings
from collections import Counter

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize

from src.camera_assignemnt.embedding_cluster.config import ClusterConfig
from src.camera_assignemnt.embedding_cluster.models import ClusterResult


def hdbscan_available() -> bool:
    """Return True when hdbscan is importable in the active interpreter."""
    try:
        import hdbscan  # noqa: F401

        return True
    except ImportError:
        return False


def reduce_dimensions(
    features: NDArray[np.float32],
    n_components: int,
    random_state: int = 0,
) -> tuple[NDArray[np.float32], PCA]:
    """Fit PCA and return reduced features plus the fitted transformer."""
    n_samples, n_features = features.shape
    n_comp = min(n_components, n_samples, n_features)
    if n_comp < 1:
        raise ValueError("Need at least one sample and one feature for PCA.")

    pca = PCA(n_components=n_comp, random_state=random_state)
    reduced = pca.fit_transform(features).astype(np.float32)
    return reduced, pca


def suggest_eps(
    features: NDArray[np.float32],
    k: int = 5,
    elbow_quantile: float = 0.9,
    metric: str = "euclidean",
) -> tuple[float, NDArray[np.int64], NDArray[np.float64]]:
    """Estimate DBSCAN eps from the k-th nearest neighbour distance elbow."""
    if len(features) < 2:
        return 0.5, np.arange(len(features)), np.zeros(len(features))

    n_neighbors = min(k + 1, len(features))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(features)
    dists, _ = nbrs.kneighbors(features)
    k_dists = np.sort(dists[:, -1])

    if len(k_dists) < 3:
        return float(np.median(k_dists)), np.arange(len(k_dists)), k_dists

    cut = max(3, int(len(k_dists) * elbow_quantile))
    subset = k_dists[:cut]
    d2 = np.diff(subset, n=2)
    if len(d2) == 0:
        return float(np.percentile(k_dists, 90)), np.arange(len(k_dists)), k_dists

    elbow_idx = int(np.argmax(d2)) + 2
    elbow_idx = min(elbow_idx, len(subset) - 1)
    suggested = float(subset[elbow_idx])
    if suggested >= float(np.percentile(k_dists, 95)):
        suggested = float(np.percentile(k_dists, 90))
    return suggested, np.arange(len(k_dists)), k_dists


def prepare_features(
    raw_features: NDArray[np.float32],
    config: ClusterConfig,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Scale and optionally PCA-reduce feature matrix."""
    scaled = raw_features.astype(np.float32)
    if config.normalize_l2 and len(scaled) > 0:
        scaled = normalize(scaled, norm="l2").astype(np.float32)
    elif config.use_standard_scaler and len(scaled) > 1:
        scaled = StandardScaler().fit_transform(scaled).astype(np.float32)

    if not config.reduce_pca:
        return scaled, scaled

    n_samples, n_features = scaled.shape
    if n_samples <= 1 or n_features <= config.pca_components:
        return scaled, scaled

    reduced, _ = reduce_dimensions(
        scaled,
        n_components=config.pca_components,
        random_state=config.random_state,
    )
    return scaled, reduced


def run_dbscan(
    reduced: NDArray[np.float32],
    config: ClusterConfig,
) -> tuple[NDArray[np.int64], float]:
    """Cluster reduced features with DBSCAN; auto-calibrate eps when configured."""
    eps = config.dbscan_eps
    if eps is None or config.auto_eps:
        suggested, _, _ = suggest_eps(
            reduced,
            k=config.dbscan_min_samples,
            elbow_quantile=config.eps_elbow_quantile,
            metric=config.dbscan_metric,
        )
        eps = suggested if config.dbscan_eps is None else config.dbscan_eps

    labels = DBSCAN(
        eps=eps,
        min_samples=config.dbscan_min_samples,
        metric=config.dbscan_metric,
    ).fit_predict(reduced)
    return labels.astype(np.int64), float(eps)


def effective_hdbscan_metric(config: ClusterConfig) -> str:
    """Map unsupported HDBSCAN metrics to BallTree-compatible equivalents."""
    if config.dbscan_metric == "cosine":
        # Embeddings are L2-normalized; euclidean distance matches cosine ranking.
        return "euclidean"
    return config.dbscan_metric


def run_hdbscan(
    reduced: NDArray[np.float32],
    config: ClusterConfig,
) -> tuple[NDArray[np.int64], float]:
    """Cluster reduced features with HDBSCAN."""
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config.hdbscan_min_cluster_size,
        min_samples=config.hdbscan_min_samples,
        cluster_selection_epsilon=config.hdbscan_cluster_selection_epsilon,
        metric=effective_hdbscan_metric(config),
        core_dist_n_jobs=1,
    )
    labels = clusterer.fit_predict(reduced)
    meta = float(config.hdbscan_cluster_selection_epsilon)
    return labels.astype(np.int64), meta


def run_clusterer(
    reduced: NDArray[np.float32],
    config: ClusterConfig,
) -> tuple[NDArray[np.int64], float]:
    """Run configured clusterer (HDBSCAN by default)."""
    if config.clusterer == "dbscan":
        return run_dbscan(reduced, config)
    if hdbscan_available():
        return run_hdbscan(reduced, config)

    warnings.warn(
        "hdbscan is not installed for "
        f"{sys.executable}; falling back to DBSCAN. "
        "Install with: pip install hdbscan  (or pip install -e . in this venv)",
        stacklevel=2,
    )
    return run_dbscan(reduced, config)


def cluster_id_to_camera_id(cluster_id: int) -> str | None:
    if cluster_id < 0:
        return None
    return f"cam_{cluster_id}"


def apply_cluster_labels(
    results: list[ClusterResult],
    labels: NDArray[np.int64],
    reduced: NDArray[np.float32],
) -> list[ClusterResult]:
    for result, label, vec in zip(results, labels, reduced):
        result.cluster_id = int(label)
        result.reduced = vec
        result.camera_id = cluster_id_to_camera_id(int(label))
    return results


def temporal_fill(
    results: list[ClusterResult],
    window: int,
) -> list[ClusterResult]:
    """Fill noise-labelled scenes using majority vote over temporal neighbours."""
    cam_ids = [r.camera_id for r in results]

    for i, result in enumerate(results):
        if result.camera_id is not None:
            continue

        start = max(0, i - window)
        end = min(len(cam_ids), i + window + 1)
        neighbors = [
            cam_ids[j]
            for j in range(start, end)
            if j != i and cam_ids[j] not in (None, "unknown")
        ]
        result.camera_id = (
            Counter(neighbors).most_common(1)[0][0] if neighbors else "unknown"
        )

    return results


def cluster_features(
    results: list[ClusterResult],
    raw_features: NDArray[np.float32],
    config: ClusterConfig,
    apply_temporal: bool = True,
) -> tuple[list[ClusterResult], NDArray[np.float32], float]:
    """Full clustering pass: scale, PCA, clusterer, optional temporal fill."""
    for result, feat in zip(results, raw_features):
        result.features = feat

    _, reduced = prepare_features(raw_features, config)
    labels, cluster_param = run_clusterer(reduced, config)
    apply_cluster_labels(results, labels, reduced)

    if apply_temporal:
        temporal_fill(results, config.temporal_window)

    return results, reduced, cluster_param
