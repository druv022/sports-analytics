"""Tests for scene classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.camera_assignemnt.scene_classifier.classifier import (
    _load_cached_scene_mlp,
    _probe_mlp_pipeline,
    _scene_mlp_is_usable,
    _try_predict_scene_type,
    classify_scene,
    compute_court_mask,
    resolve_scene_mlp,
)
from src.camera_assignemnt.scene_classifier.config import Config
from src.camera_assignemnt.scene_classifier.models import SceneMlpError
from tests.conftest import make_closeup_frame, make_hard_court_frame


def test_compute_court_mask_shape(config: Config) -> None:
    frame = make_hard_court_frame()
    ratio, mask = compute_court_mask(frame, config)
    assert mask.shape[:2] == frame.shape[:2]
    assert 0.0 <= ratio <= 1.0


def test_classify_without_model_raises(config: Config) -> None:
    config = Config(scene_mlp_path=str(Path("models/does_not_exist.joblib")))
    _load_cached_scene_mlp.cache_clear()
    _scene_mlp_is_usable.cache_clear()
    frame = make_hard_court_frame()
    with pytest.raises(SceneMlpError, match="not found"):
        classify_scene(frame, config)
    with pytest.raises(SceneMlpError, match="not found"):
        resolve_scene_mlp(config)


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
    _scene_mlp_is_usable.cache_clear()

    frame = make_hard_court_frame(width=480, height=640)
    scene_type, _, _ = classify_scene(frame, config)
    assert scene_type in {"full_court", "closeup"}


def test_probe_mlp_pipeline_rejects_runtime_warning(config: Config) -> None:
    import warnings

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(max_iter=200, random_state=0)),
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
    mlp = pipeline.named_steps["mlp"]
    original_predict = mlp.predict

    def unstable_predict(features: np.ndarray) -> np.ndarray:
        warnings.warn("divide by zero encountered in matmul", RuntimeWarning)
        return original_predict(features)

    mlp.predict = unstable_predict  # type: ignore[method-assign]
    assert _probe_mlp_pipeline(pipeline) is False


def test_try_predict_scene_type_rejects_invalid_histogram(config: Config) -> None:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(max_iter=200, random_state=0)),
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

    assert _try_predict_scene_type(np.full(192, np.nan, dtype=np.float32), pipeline) is None
    assert _try_predict_scene_type(np.zeros(192, dtype=np.float32), pipeline) is None


def test_classify_raises_when_mlp_not_usable(
    tmp_path: Path,
    config: Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import joblib

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
    joblib.dump(pipeline, model_path)

    config = Config(scene_mlp_path=str(model_path))
    _load_cached_scene_mlp.cache_clear()
    _scene_mlp_is_usable.cache_clear()
    monkeypatch.setattr(
        "src.camera_assignemnt.scene_classifier.classifier._scene_mlp_is_usable",
        lambda _path: False,
    )

    frame = make_closeup_frame()
    with pytest.raises(SceneMlpError, match="usability checks"):
        classify_scene(frame, config)
