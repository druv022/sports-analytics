"""Tests for scene classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.camera_assignemnt.approach_1.classifier import (
    _load_cached_scene_mlp,
    classify_scene,
    compute_court_mask,
)
from src.camera_assignemnt.approach_1.config import Config
from tests.conftest import make_closeup_frame, make_hard_court_frame


def test_compute_court_mask_shape(config: Config) -> None:
    frame = make_hard_court_frame()
    ratio, mask = compute_court_mask(frame, config)
    assert mask.shape[:2] == frame.shape[:2]
    assert 0.0 <= ratio <= 1.0


def test_classify_full_court_without_model(config: Config) -> None:
    config = Config(scene_mlp_path=str(Path("models/does_not_exist.joblib")))
    _load_cached_scene_mlp.cache_clear()
    frame = make_hard_court_frame()
    scene_type, ratio, mask = classify_scene(frame, config)
    assert scene_type == "full_court"
    assert ratio > config.full_court_ratio
    assert mask.shape[:2] == frame.shape[:2]


def test_classify_closeup_without_model(config: Config) -> None:
    config = Config(scene_mlp_path=str(Path("models/does_not_exist.joblib")))
    _load_cached_scene_mlp.cache_clear()
    frame = make_closeup_frame()
    scene_type, ratio, _ = classify_scene(frame, config)
    assert scene_type == "closeup"
    assert ratio <= config.full_court_ratio


def test_classify_with_mock_mlp(tmp_path: Path, config: Config) -> None:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(max_iter=200, random_state=0),
            ),
        ]
    )
    X = np.vstack(
        [
            np.full(192, 1.0),
            np.full(192, 3.0),
        ]
    )
    y = np.array(["full_court", "closeup"])
    pipeline.fit(X, y)

    model_path = tmp_path / "scene_mlp.joblib"
    import joblib

    joblib.dump(pipeline, model_path)

    config = Config(scene_mlp_path=str(model_path))
    _load_cached_scene_mlp.cache_clear()

    frame = make_hard_court_frame(width=480, height=640)
    scene_type, _, _ = classify_scene(frame, config)
    assert scene_type in {"full_court", "closeup"}
