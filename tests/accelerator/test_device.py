"""Tests for accelerator device resolution."""

from __future__ import annotations

import warnings

import pytest

from src.accelerator import device as device_mod


@pytest.fixture(autouse=True)
def _clear_warned():
    device_mod._WARNED.clear()
    yield
    device_mod._WARNED.clear()


def test_resolve_torch_device_auto_cpu(monkeypatch):
    monkeypatch.setattr(device_mod, "_torch_cuda_available", lambda: False)
    monkeypatch.setattr(device_mod, "_torch_mps_available", lambda: False)
    assert device_mod.resolve_torch_device("auto") == "cpu"


def test_resolve_torch_device_auto_cuda(monkeypatch):
    monkeypatch.setattr(device_mod, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(device_mod, "_torch_mps_available", lambda: False)
    assert device_mod.resolve_torch_device("auto") == "cuda"


def test_resolve_torch_device_cuda_fallback_warns(monkeypatch):
    monkeypatch.setattr(device_mod, "_torch_cuda_available", lambda: False)
    monkeypatch.setattr(device_mod, "_torch_mps_available", lambda: False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert device_mod.resolve_torch_device("cuda") == "cpu"
    assert any("CUDA is not available for torch" in str(w.message) for w in caught)


def test_resolve_ocr_backend_cuda_when_ep_available(monkeypatch):
    monkeypatch.setattr(device_mod, "_cuda_execution_provider_available", lambda: True)
    monkeypatch.setattr(device_mod, "_coreml_execution_provider_available", lambda: False)
    assert device_mod.resolve_ocr_backend("auto") == "cuda"


def test_resolve_ocr_backend_coreml_on_mac(monkeypatch):
    monkeypatch.setattr(device_mod, "_cuda_execution_provider_available", lambda: False)
    monkeypatch.setattr(device_mod, "_coreml_execution_provider_available", lambda: True)
    assert device_mod.resolve_ocr_backend("auto") == "coreml"


def test_resolve_ocr_use_cuda_when_ep_available(monkeypatch):
    monkeypatch.setattr(device_mod, "_cuda_execution_provider_available", lambda: True)
    monkeypatch.setattr(device_mod, "_coreml_execution_provider_available", lambda: False)
    assert device_mod.resolve_ocr_use_cuda("auto") is True


def test_resolve_ocr_use_cuda_warns_on_gpu_host_without_ort(monkeypatch):
    monkeypatch.setattr(device_mod, "_cuda_execution_provider_available", lambda: False)
    monkeypatch.setattr(device_mod, "_coreml_execution_provider_available", lambda: False)
    monkeypatch.setattr(device_mod, "_torch_cuda_available", lambda: True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert device_mod.resolve_ocr_use_cuda("auto") is False
    assert any("CUDAExecutionProvider" in str(w.message) for w in caught)


def test_warn_once_dedupes():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        device_mod.warn_once("k", "first")
        device_mod.warn_once("k", "second")
    assert len(caught) == 1
    assert "first" in str(caught[0].message)
