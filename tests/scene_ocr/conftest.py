"""Shared skips for scene OCR integration tests."""

from __future__ import annotations

import importlib.util

import pytest


def rapidocr_available() -> bool:
    if importlib.util.find_spec("rapidocr") is None:
        return False
    from rapidocr import RapidOCR

    probe = RapidOCR(params={"Global.use_cls": False})
    required = ("use_det", "text_det", "text_rec", "use_cls", "crop_text_regions")
    ok = all(hasattr(probe, name) for name in required)
    del probe
    return ok


requires_rapidocr = pytest.mark.skipif(
    not rapidocr_available(),
    reason="rapidocr >= 3.8 not installed",
)

# Backward-compatible alias for older test imports.
requires_rapidocr_144 = requires_rapidocr
