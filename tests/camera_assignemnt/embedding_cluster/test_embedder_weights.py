"""Tests for weight download / resolution helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.camera_assignemnt.embedding_cluster.config import EmbeddingConfig
from src.camera_assignemnt.embedding_cluster.embedder import (
    RESNET50_WEIGHTS_NAME,
    resnet50_weight_candidates,
    resolve_resnet50_weights,
)


def test_resnet50_weight_candidates_includes_weights_dir(tmp_path: Path):
    config = EmbeddingConfig(weights_dir=str(tmp_path))
    paths = resnet50_weight_candidates(config)
    assert paths[0] == tmp_path / RESNET50_WEIGHTS_NAME


def test_resolve_resnet50_weights_uses_existing_file(tmp_path: Path):
    weights = tmp_path / RESNET50_WEIGHTS_NAME
    weights.write_bytes(b"fake")
    config = EmbeddingConfig(weights_dir=str(tmp_path))
    assert resolve_resnet50_weights(config) == weights


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("torch"),
    reason="torch not installed",
)
def test_resolve_resnet50_weights_downloads_with_ssl_patch(tmp_path: Path, monkeypatch):
    cache = tmp_path / "cache" / "checkpoints"
    cache.mkdir(parents=True)
    target = cache / RESNET50_WEIGHTS_NAME

    config = EmbeddingConfig(weights_dir=str(tmp_path / "missing"))

    def fake_download(url: str, dest: Path) -> Path:
        dest.write_bytes(b"downloaded")
        return dest

    monkeypatch.setattr(
        "src.camera_assignemnt.embedding_cluster.embedder.download_file",
        fake_download,
    )
    with patch("src.camera_assignemnt.embedding_cluster.embedder.torch") as mock_torch:
        mock_torch.hub.get_dir.return_value = str(tmp_path / "cache")
        resolved = resolve_resnet50_weights(config)

    assert resolved == target
    assert target.exists()
