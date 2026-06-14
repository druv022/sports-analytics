"""Tests for MLP scene classifier training and inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.camera_assignemnt.approach_1.config import Config
from src.camera_assignemnt.approach_1.classifier import (
    ClassificationDataset,
    ClassificationSample,
    build_mlp_pipeline,
    cross_validate_scene_mlp,
    load_scene_mlp,
    predict_scene_type,
    save_scene_mlp,
    train_scene_mlp,
)


def _make_sample(
    idx: int,
    label: str,
    scene_id: str,
    value: float,
) -> ClassificationSample:
    hist = np.full(192, value, dtype=np.float32)
    return ClassificationSample(
        path=Path(f"scene_{scene_id}_frame_{idx}.jpg"),
        label=label,  # type: ignore[arg-type]
        scene_id=scene_id,
        histogram=hist,
    )


def _synthetic_dataset() -> ClassificationDataset:
    samples = [
        _make_sample(0, "full_court", "1", 1.0),
        _make_sample(1, "full_court", "2", 1.1),
        _make_sample(2, "full_court", "3", 0.9),
        _make_sample(3, "closeup", "4", 3.0),
        _make_sample(4, "closeup", "5", 3.1),
        _make_sample(5, "closeup", "6", 2.9),
        _make_sample(6, "full_court", "7", 1.05),
        _make_sample(7, "closeup", "8", 3.05),
        _make_sample(8, "full_court", "9", 0.95),
        _make_sample(9, "closeup", "10", 2.95),
    ]
    return ClassificationDataset(samples=tuple(samples))


def test_build_mlp_pipeline(config: Config) -> None:
    pipeline = build_mlp_pipeline(config)
    assert isinstance(pipeline, Pipeline)
    assert "scaler" in pipeline.named_steps
    assert "mlp" in pipeline.named_steps


def test_train_scene_mlp(config: Config) -> None:
    dataset = _synthetic_dataset()
    pipeline = train_scene_mlp(dataset.X, dataset.y, config)
    preds = pipeline.predict(dataset.X)
    assert set(preds.tolist()).issubset({"full_court", "closeup"})


def test_cross_validate_scene_mlp(config: Config) -> None:
    dataset = _synthetic_dataset()
    results = cross_validate_scene_mlp(dataset, config, n_splits=5)
    assert results["n_samples"] == len(dataset)
    assert results["n_splits"] == 5
    assert 0.0 <= results["accuracy"] <= 1.0
    assert len(results["fold_metrics"]) == 5
    assert len(results["confusion_matrix"]) == len(results["labels"])


def test_save_and_load_scene_mlp(tmp_path: Path, config: Config) -> None:
    dataset = _synthetic_dataset()
    pipeline = train_scene_mlp(dataset.X, dataset.y, config)
    model_path = tmp_path / "scene_mlp.joblib"
    save_scene_mlp(pipeline, model_path)
    loaded = load_scene_mlp(model_path)
    assert loaded.predict(dataset.X).tolist() == pipeline.predict(dataset.X).tolist()


def test_predict_scene_type_returns_mlp_label(config: Config) -> None:
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

    assert predict_scene_type(np.full(192, 1.0, dtype=np.float32), pipeline) == "full_court"
    assert predict_scene_type(np.full(192, 3.0, dtype=np.float32), pipeline) == "closeup"


def test_cross_validate_empty_dataset_raises(config: Config) -> None:
    empty = ClassificationDataset(samples=())
    with pytest.raises(ValueError, match="empty dataset"):
        cross_validate_scene_mlp(empty, config)
