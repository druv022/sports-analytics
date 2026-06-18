"""Tests for Stage 1 text presence detection."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from src.scene_ocr.config import OcrConfig, ReadabilityConfig
from src.scene_ocr.presence import find_text_candidates


def test_find_text_candidates_empty_on_solid_frame():
    frame = np.full((200, 320, 3), 40, dtype=np.uint8)
    with patch("src.scene_ocr.presence.detect_boxes", return_value=[]):
        candidates = find_text_candidates(frame, OcrConfig(preprocess=False), ReadabilityConfig())
    assert candidates == []


def test_find_text_candidates_includes_contrast_blob():
    frame = np.full((200, 320, 3), 20, dtype=np.uint8)
    frame[30:120, 40:180] = 255
    config = ReadabilityConfig(min_blob_area=500)
    with patch("src.scene_ocr.presence.detect_boxes", return_value=[]):
        candidates = find_text_candidates(frame, OcrConfig(preprocess=False), config)
    assert any(c.source == "contrast" for c in candidates)


def test_find_text_candidates_includes_detector_boxes():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    box = np.array([10, 10, 80, 30], dtype=np.int32)
    with patch("src.scene_ocr.presence.detect_boxes", return_value=[box]):
        candidates = find_text_candidates(frame, OcrConfig(preprocess=False), ReadabilityConfig())
    assert any(c.source == "detector" for c in candidates)
