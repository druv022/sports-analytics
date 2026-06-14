"""Tests for DBSCAN clustering utilities."""

from __future__ import annotations

import numpy as np
import pytest

from src.camera_assignemnt.approach_4.cluster import (
    cluster_features,
    suggest_eps,
    temporal_fill,
)
from src.camera_assignemnt.approach_4.config import ClusterConfig
from src.camera_assignemnt.approach_4.models import ClusterResult


def _make_blob(center: np.ndarray, n: int, spread: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(0)
    return center + rng.normal(scale=spread, size=(n, center.shape[0]))


def test_suggest_eps_returns_positive():
    features = np.vstack([_make_blob(np.zeros(4), 5), _make_blob(np.ones(4) * 3, 5)])
    eps, _, dists = suggest_eps(features.astype(np.float32), k=2)
    assert eps > 0
    assert len(dists) == len(features)


def test_cluster_features_finds_two_groups():
    features = np.vstack(
        [
            _make_blob(np.zeros(8), 6),
            _make_blob(np.ones(8) * 5, 6),
        ]
    ).astype(np.float32)
    results = [
        ClusterResult(scene_idx=i, scene_id=str(i), frame_path=f"s{i}.jpg")
        for i in range(len(features))
    ]
    config = ClusterConfig(
        pca_components=4,
        clusterer="dbscan",
        dbscan_eps=0.8,
        auto_eps=False,
        dbscan_min_samples=2,
    )
    clustered, reduced, eps = cluster_features(results, features, config, apply_temporal=False)
    labels = {r.cluster_id for r in clustered}
    assert len(labels - {-1}) >= 2
    assert reduced.shape[0] == len(features)
    assert eps == 0.8


def test_cluster_features_hdbscan_finds_groups():
    hdbscan = pytest.importorskip("hdbscan")
    del hdbscan

    features = np.vstack(
        [
            _make_blob(np.zeros(8), 8),
            _make_blob(np.ones(8) * 5, 8),
        ]
    ).astype(np.float32)
    results = [
        ClusterResult(scene_idx=i, scene_id=str(i), frame_path=f"s{i}.jpg")
        for i in range(len(features))
    ]
    config = ClusterConfig(
        pca_components=4,
        clusterer="hdbscan",
        hdbscan_min_cluster_size=3,
        hdbscan_min_samples=2,
        auto_eps=False,
    )
    clustered, reduced, param = cluster_features(results, features, config, apply_temporal=False)
    labels = {r.cluster_id for r in clustered}
    assert len(labels - {-1}) >= 2
    assert reduced.shape[0] == len(features)
    assert param == 0.0


def test_effective_hdbscan_metric_maps_cosine_to_euclidean():
    from src.camera_assignemnt.approach_4.cluster import effective_hdbscan_metric

    cfg = ClusterConfig(dbscan_metric="cosine")
    assert effective_hdbscan_metric(cfg) == "euclidean"
    assert effective_hdbscan_metric(ClusterConfig(dbscan_metric="euclidean")) == "euclidean"


def test_temporal_fill_assigns_unknown_neighbors():
    results = [
        ClusterResult(scene_idx=0, scene_id="0", frame_path="a.jpg", camera_id="cam_0"),
        ClusterResult(scene_idx=1, scene_id="1", frame_path="b.jpg", camera_id=None),
        ClusterResult(scene_idx=2, scene_id="2", frame_path="c.jpg", camera_id="cam_0"),
    ]
    temporal_fill(results, window=1)
    assert results[1].camera_id == "cam_0"
