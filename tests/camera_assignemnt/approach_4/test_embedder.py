"""Tests for embedding feature extraction (mocked, no weight download)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.camera_assignemnt.approach_4.config import EmbeddingConfig
from src.camera_assignemnt.approach_4.embedder import require_torch
from src.camera_assignemnt.approach_4.models import SceneSample


def test_require_torch_raises_when_missing():
    with patch("src.camera_assignemnt.approach_4.embedder._TORCH_AVAILABLE", False):
        with pytest.raises(ImportError, match="torch and torchvision"):
            require_torch()


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("torch"),
    reason="torch not installed",
)
def test_extract_features_batch_shape():
    torch = pytest.importorskip("torch")
    from src.camera_assignemnt.approach_4.embedder import extract_features_batch

    samples = [
        SceneSample(
            scene_idx=0,
            scene_id="0",
            image_idx=1,
            frame_path="a.jpg",
            frame=np.zeros((64, 64, 3), dtype=np.uint8),
        )
    ]
    model = MagicMock()
    model.parameters.return_value = iter([torch.zeros(1)])
    model.return_value = torch.zeros(1, 128)

    config = EmbeddingConfig(batch_size=1)
    features = extract_features_batch(samples, model, config)
    assert features.shape == (1, 128)


def test_load_model_dispatches_dinov2(monkeypatch):
    torch = pytest.importorskip("torch")
    from src.camera_assignemnt.approach_4 import embedder

    mock_model = MagicMock()
    monkeypatch.setattr(embedder, "load_dinov2_vits14", lambda config, device: mock_model)
    monkeypatch.setattr(embedder, "default_device", lambda: "cpu")

    config = EmbeddingConfig(backend="dinov2_vits14", device="cpu")
    model = embedder.load_model(config)
    assert model is mock_model
