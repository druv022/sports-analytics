"""Tests for approach 4 config helpers."""

from __future__ import annotations

import pytest

from src.camera_assignemnt.embedding_cluster.config import (
    EmbeddingConfig,
    output_method_slug,
    resolve_method_and_backend,
)


def test_output_method_slug_hsv_and_ensemble():
    assert output_method_slug("hsv") == "hsv"
    assert output_method_slug("ensemble") == "ensemble"


def test_output_method_slug_embedding_uses_backend():
    cfg = EmbeddingConfig(backend="dinov2_vits14")
    assert output_method_slug("embedding", cfg) == "dinov2_vits14"


def test_output_method_slug_embedding_backend_shorthand():
    assert output_method_slug("resnet50") == "resnet50"
    assert output_method_slug("dinov2_vits14") == "dinov2_vits14"


def test_resolve_method_and_backend_embedding():
    method, backend, slug = resolve_method_and_backend("embedding", "dinov2_vits14")
    assert method == "embedding"
    assert backend == "dinov2_vits14"
    assert slug == "dinov2_vits14"


def test_resolve_method_and_backend_slug_shorthand():
    method, backend, slug = resolve_method_and_backend("dinov2_vits14")
    assert method == "embedding"
    assert backend == "dinov2_vits14"
    assert slug == "dinov2_vits14"


def test_output_method_slug_unknown_raises():
    with pytest.raises(ValueError, match="Unknown clustering method"):
        output_method_slug("not_a_method")
