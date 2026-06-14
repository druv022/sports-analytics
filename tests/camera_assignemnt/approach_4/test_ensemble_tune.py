"""Tests for ensemble tuning helpers."""

from __future__ import annotations

import numpy as np

from src.camera_assignemnt.approach_4.ensemble_tune import (
    compute_member_weights,
    evaluate_ensemble_on_indices,
    score_member_on_indices,
    split_gt_scene_ids,
)
from src.camera_assignemnt.approach_4.models import ClusterResult


def test_compute_member_weights_prefers_better_member():
    weights = compute_member_weights(
        {
            "resnet50": {"combined_score": 0.7},
            "hsv": {"combined_score": 0.1},
            "dinov2_vits14": {"combined_score": 0.2},
        }
    )
    assert weights["resnet50"] > weights["hsv"]
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert all(weight >= 0.05 for weight in weights.values())


def test_score_member_on_indices():
    labels = np.array([0, 0, 1, -1], dtype=np.int64)
    y_true = np.array(["cam_0", "cam_0", "cam_1", "cam_1"])
    stats = score_member_on_indices(labels, y_true, np.array([0, 1, 2, 3]))
    assert 0.0 <= stats["hungarian_accuracy"] <= 1.0
    assert stats["noise_rate"] == 0.25


def test_evaluate_ensemble_on_indices():
    samples = [
        ClusterResult(scene_idx=i, scene_id=str(i), frame_path=f"{i}.jpg")
        for i in range(4)
    ]
    base_results = samples
    labelings = [
        np.array([0, 0, 1, 1], dtype=np.int64),
        np.array([0, 0, 1, 1], dtype=np.int64),
    ]
    names = ["a", "b"]
    reduced = np.eye(4, dtype=np.float32)
    y_true = np.array(["cam_0", "cam_0", "cam_1", "cam_1"])
    metrics = evaluate_ensemble_on_indices(
        samples,
        base_results,
        labelings,
        names,
        reduced,
        y_true,
        np.array([0, 1, 2, 3]),
        {"a": 0.5, "b": 0.5},
        link_threshold=0.5,
        noise_threshold=0.7,
        temporal_window=1,
    )
    assert "hungarian_accuracy" in metrics
    assert "objective" in metrics


def test_split_gt_scene_ids(tmp_path):
    gt_csv = tmp_path / "gt.csv"
    gt_csv.write_text(
        "scene_id,image_idx,camera_id\n"
        + "\n".join(f"{i},1,cam_{i % 3}" for i in range(10))
    )
    tune_ids, holdout_ids = split_gt_scene_ids(gt_csv, tune_size=4, random_state=0)
    assert len(tune_ids) == 4
    assert len(holdout_ids) == 6
    assert set(tune_ids).isdisjoint(holdout_ids)
