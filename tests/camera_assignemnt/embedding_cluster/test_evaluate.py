"""Tests for GT evaluation alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.camera_assignemnt.embedding_cluster.evaluate import (
    hungarian_mapped_accuracy,
    join_predictions_to_gt,
)


def test_hungarian_maps_clusters_to_cameras():
    y_true = np.array(["cam_a", "cam_a", "cam_b", "cam_b"])
    y_pred = np.array([0, 0, 1, 1])
    accuracy, mapping = hungarian_mapped_accuracy(y_true, y_pred)
    assert accuracy == 1.0
    assert len(mapping) == 2


def test_join_predictions_to_gt_inner_merge():
    gt = pd.DataFrame(
        {
            "scene_id": ["0", "1", "2"],
            "camera_id": ["cam_0", "cam_1", "cam_2"],
            "image_idx": [1, 1, 1],
        }
    )
    preds = pd.DataFrame(
        {
            "scene_id": ["0", "2"],
            "camera_id": ["cam_0", "cam_5"],
            "cluster_id": [0, 1],
        }
    )
    merged = join_predictions_to_gt(preds, gt)
    assert len(merged) == 2
    assert set(merged["scene_id"]) == {"0", "2"}
