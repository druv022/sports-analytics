from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

ROOT = Path(__file__).resolve().parents[3]
KEYPOINTS_JSON = ROOT / "data" / "court_reference" / "keypoints.json"
MODEL_PATH = ROOT / "models" / "tennis_court_detector.pth"


@pytest.fixture
def court_ref():
    from src.camera_assignemnt.approach_3.court_reference import load_court_reference

    return load_court_reference(KEYPOINTS_JSON)


def test_keypoints_json_exists():
    assert KEYPOINTS_JSON.is_file()


def test_court_reference_loads(court_ref):
    assert court_ref.dimension_keypoints.shape == (14, 2)
    assert court_ref.tcd_keypoints.shape == (14, 2)
    assert len(court_ref.labels) == 14


def test_bridge_roundtrip(court_ref):
    dim_pt = court_ref.dimension_keypoints[0]
    pt_h = np.array([*dim_pt, 1.0], dtype=np.float64)

    to_tcd = court_ref.H_dim_to_tcd @ pt_h
    to_tcd_xy = to_tcd[:2] / to_tcd[2]

    back_h = court_ref.H_tcd_to_dim @ np.array([*to_tcd_xy, 1.0])
    back_xy = back_h[:2] / back_h[2]

    assert np.linalg.norm(back_xy - dim_pt) < 1.0


def test_homography_estimator_synthetic(court_ref):
    from src.camera_assignemnt.approach_3.config import HomographyConfig
    from src.camera_assignemnt.approach_3.homography_estimator import estimate_homography_from_keypoints
    from src.camera_assignemnt.approach_3.types import KeypointDetection

    H_true = np.array(
        [[1.2, 0.05, 100.0], [0.02, 1.1, 50.0], [1e-5, 2e-5, 1.0]],
        dtype=np.float64,
    )

    dim = court_ref.dimension_keypoints.astype(np.float32)
    pts_h = np.hstack([dim, np.ones((14, 1))])
    scene_pts = (H_true @ pts_h.T).T
    scene_pts = scene_pts[:, :2] / scene_pts[:, 2:3]

    detection = KeypointDetection(
        points=scene_pts.astype(np.float32),
        valid=np.ones(14, dtype=bool),
        raw_points=scene_pts.astype(np.float32),
    )

    config = HomographyConfig(min_keypoints=4, ransac_thresh=3.0)
    H, H_inv, err, n_in, _, _ = estimate_homography_from_keypoints(
        detection, court_ref, config
    )

    assert H is not None
    assert n_in >= 4
    assert err < 1.0


def test_validator_synthetic():
    from src.camera_assignemnt.approach_3.homography_validator import passes_exact_gate

    assert passes_exact_gate(3.0, 2.0, 5.0, 12.0, 6)
    assert not passes_exact_gate(8.0, 2.0, 5.0, 12.0, 6)
    assert not passes_exact_gate(3.0, 20.0, 5.0, 12.0, 6)
    assert not passes_exact_gate(3.0, 2.0, 5.0, 12.0, 3)


@pytest.mark.skipif(not MODEL_PATH.is_file(), reason="TennisCourtDetector weights missing")
@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_keypoint_detector_on_sample():
    from src.camera_assignemnt.approach_3.config import HomographyConfig
    from src.camera_assignemnt.approach_3.keypoint_detector import detect_court_keypoints

    sample = ROOT / "data" / "scene_samples" / "scene_25_frame_13090.jpg"
    if not sample.is_file():
        pytest.skip("sample frame missing")

    scene = cv2.imread(str(sample))
    config = HomographyConfig(model_path=MODEL_PATH, keypoints_json=KEYPOINTS_JSON)
    detection = detect_court_keypoints(scene, config)

    assert detection.points.shape == (14, 2)
    assert detection.valid.sum() >= 4
