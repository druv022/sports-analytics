"""Tests for ensemble co-association voting."""

from __future__ import annotations

import numpy as np

from src.camera_assignemnt.approach_4.ensemble import (
    coassociation_matrix,
    consensus_labels,
    member_noise_mask,
    normalize_member_weights,
    vote_cluster_assignments,
)
from src.camera_assignemnt.approach_4.models import ClusterResult


def test_coassociation_unanimous_labelings():
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    names = ["a", "b", "c"]
    coassoc = coassociation_matrix([labels, labels, labels], names)
    assert coassoc[0, 1] == 1.0
    assert coassoc[0, 2] == 0.0


def test_weighted_coassociation_favors_heavier_member():
    labels_a = np.array([0, 0, 1, 1], dtype=np.int64)
    labels_b = np.array([0, 1, 1, 1], dtype=np.int64)
    names = ["good", "weak"]
    coassoc = coassociation_matrix(
        [labels_a, labels_b],
        names,
        member_weights={"good": 0.9, "weak": 0.1},
    )
    assert coassoc[0, 1] > coassoc[0, 2]


def test_consensus_labels_majority_pairs():
    labels_a = np.array([0, 0, 1, 1], dtype=np.int64)
    labels_b = np.array([0, 0, 1, 1], dtype=np.int64)
    labels_c = np.array([0, 1, 1, 1], dtype=np.int64)
    names = ["a", "b", "c"]
    coassoc = coassociation_matrix([labels_a, labels_b, labels_c], names)
    consensus = consensus_labels(coassoc, link_threshold=2 / 3)
    assert consensus[0] == consensus[1]
    assert consensus[2] == consensus[3]
    assert consensus[0] != consensus[2]


def test_member_noise_mask_weighted():
    labelings = [
        np.array([-1, 0, 0, 1], dtype=np.int64),
        np.array([0, 0, 1, 1], dtype=np.int64),
    ]
    names = ["noisy", "clean"]
    noise = member_noise_mask(
        labelings,
        names,
        member_weights={"noisy": 0.2, "clean": 0.8},
        noise_threshold=0.6,
    )
    assert not noise[0]
    assert not noise[1]


def test_normalize_member_weights_defaults_to_equal():
    weights = normalize_member_weights(["a", "b", "c"])
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert set(weights) == {"a", "b", "c"}


def test_vote_cluster_assignments_applies_noise_and_temporal_fill():
    results = [
        ClusterResult(scene_idx=i, scene_id=str(i), frame_path=f"{i}.jpg")
        for i in range(4)
    ]
    labelings = [
        np.array([0, 0, 1, -1], dtype=np.int64),
        np.array([0, 0, 1, -1], dtype=np.int64),
        np.array([0, 0, 1, -1], dtype=np.int64),
    ]
    names = ["a", "b", "c"]
    reduced = np.eye(4, dtype=np.float32)

    voted = vote_cluster_assignments(
        results,
        labelings,
        names,
        reduced,
        link_threshold=2 / 3,
        noise_threshold=0.67,
        temporal_window=1,
    )

    assert voted[0].cluster_id == voted[1].cluster_id
    assert voted[2].cluster_id != voted[0].cluster_id
    assert voted[3].cluster_id == -1
    assert voted[3].camera_id == "cam_1"
    assert voted[0].camera_id == "cam_0"


def test_compute_member_weights_import():
    from src.camera_assignemnt.approach_4.ensemble_tune import compute_member_weights

    weights = compute_member_weights(
        {
            "resnet50": {"combined_score": 0.8},
            "hsv": {"combined_score": 0.2},
        }
    )
    assert weights["resnet50"] > weights["hsv"]
    assert abs(sum(weights.values()) - 1.0) < 1e-6
