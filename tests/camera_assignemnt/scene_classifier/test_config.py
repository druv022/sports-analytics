"""Tests for scene classifier config path resolution."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.camera_assignemnt.scene_classifier.classifier import resolve_scene_mlp
from src.camera_assignemnt.scene_classifier.config import Config


def _write_dummy_mlp(path: Path) -> None:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(max_iter=200, random_state=0)),
        ]
    )
    X = np.vstack([np.full(192, 1.0), np.full(192, 3.0)])
    y = np.array(["full_court", "closeup"])
    pipeline.fit(X, y)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def test_resolved_scene_mlp_path_uses_project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = tmp_path / "repo"
    model_path = project_root / "models" / "scene_mlp.joblib"
    _write_dummy_mlp(model_path)

    other_cwd = tmp_path / "other_cwd"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    config = Config(scene_mlp_path="models/scene_mlp.joblib")
    resolved = config.resolved_scene_mlp_path(project_root=project_root)
    assert resolved == model_path.resolve()
    resolve_scene_mlp(config, project_root=project_root)


def test_resolved_scene_mlp_path_falls_back_to_package_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    expected = repo_root / "models" / "scene_mlp.joblib"
    if not expected.is_file():
        pytest.skip("Bundled scene MLP not present in workspace")

    monkeypatch.chdir(tmp_path)

    config = Config(scene_mlp_path="models/scene_mlp.joblib")
    resolved = config.resolved_scene_mlp_path()
    assert resolved == expected.resolve()
