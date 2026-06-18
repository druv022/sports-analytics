"""Tests for embedding feature extraction (mocked, no weight download)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.camera_assignemnt.embedding_cluster.config import EmbeddingConfig
from src.camera_assignemnt.embedding_cluster.embedder import require_torch
from src.camera_assignemnt.embedding_cluster.models import SceneSample


def test_require_torch_raises_when_missing():
    with patch("src.camera_assignemnt.embedding_cluster.embedder._TORCH_AVAILABLE", False):
        with pytest.raises(ImportError, match="torch and torchvision"):
            require_torch()


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("torch"),
    reason="torch not installed",
)
def test_extract_features_batch_shape():
    torch = pytest.importorskip("torch")
    from src.camera_assignemnt.embedding_cluster.embedder import extract_features_batch

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
    from src.camera_assignemnt.embedding_cluster import embedder

    mock_model = MagicMock()
    monkeypatch.setattr(embedder, "load_dinov2_vits14", lambda config, device: mock_model)
    monkeypatch.setattr(embedder, "default_device", lambda: "cpu")

    config = EmbeddingConfig(backend="dinov2_vits14", device="cpu")
    model = embedder.load_model(config)
    assert model is mock_model


def test_dinov2_batch_size_cuda():
    from src.camera_assignemnt.embedding_cluster.embedder import dinov2_batch_size

    assert dinov2_batch_size("cuda") == 32
    assert dinov2_batch_size("cpu") == 2
