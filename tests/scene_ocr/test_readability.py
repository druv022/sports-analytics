"""Tests for readability assessment and VLM gating."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.scene_ocr.config import OcrConfig, ReadabilityConfig
from src.scene_ocr.readability import assess_readability, assess_readability_from_bgr
from src.scene_ocr.types import OcrReadabilityVerdict, TextBlobCandidate
from src.scene_ocr.vlm_client import NullVlmClient
from tests.scene_ocr.conftest import requires_rapidocr


def _mock_raw(entries: list[tuple[str, float, list[int]]]):
    return [(text, score, np.array(bbox, dtype=np.int32)) for text, score, bbox in entries]


def _mock_dt_boxes():
    return np.array(
        [[[20, 900], [300, 900], [300, 980], [20, 980]]],
        dtype=np.float32,
    )


def test_assess_readability_delegates_to_from_bgr():
    frame = np.full((120, 160, 3), 30, dtype=np.uint8)
    with patch(
        "src.scene_ocr.readability.assess_readability_from_bgr",
        return_value=MagicMock(),
    ) as mock_from_bgr:
        with patch("src.scene_ocr.readability.load_image", return_value=frame):
            assess_readability("fake.jpg", OcrConfig(preprocess=False))
    mock_from_bgr.assert_called_once()
    assert mock_from_bgr.call_args.args[0] is frame


def test_assess_readability_no_text_when_detection_empty():
    frame = np.full((120, 160, 3), 30, dtype=np.uint8)
    with patch("src.scene_ocr.readability.detect_raw", return_value=(frame, 1.0, None)):
        result = assess_readability(frame, OcrConfig(preprocess=False), ReadabilityConfig())
    assert result.verdict == OcrReadabilityVerdict.NO_TEXT
    assert result.needs_vlm is False


def test_assess_readability_uses_run_ocr_from_boxes_not_run_ocr_raw():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    raw = _mock_raw([("SINNER ITA", 0.9, [20, 900, 300, 980])])
    candidate = TextBlobCandidate(
        bbox=np.array([20, 900, 300, 980], dtype=np.int32),
        area=20000,
        source="detector",
    )
    dt_boxes = _mock_dt_boxes()
    with patch("src.scene_ocr.readability.detect_raw", return_value=(frame, 1.0, dt_boxes)):
        with patch(
            "src.scene_ocr.readability.build_text_candidates",
            return_value=[candidate],
        ):
            with patch(
                "src.scene_ocr.readability.run_ocr_from_boxes",
                return_value=raw,
            ) as from_boxes:
                with patch("src.scene_ocr.extractor.run_ocr_raw") as full_ocr:
                    result = assess_readability(
                        frame, OcrConfig(preprocess=False), ReadabilityConfig()
                    )
    from_boxes.assert_called_once()
    full_ocr.assert_not_called()
    assert result.verdict == OcrReadabilityVerdict.READABLE
    assert result.overlay_readable is True
    assert "SINNER" in result.words


def test_assess_readability_needs_vlm_for_unmatched_blob():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    blob = TextBlobCandidate(
        bbox=np.array([400, 100, 900, 500], dtype=np.int32),
        area=25000,
        source="contrast",
    )
    dt_boxes = _mock_dt_boxes()
    with patch("src.scene_ocr.readability.detect_raw", return_value=(frame, 1.0, dt_boxes)):
        with patch(
            "src.scene_ocr.readability.build_text_candidates",
            return_value=[blob],
        ):
            with patch("src.scene_ocr.readability.run_ocr_from_boxes", return_value=[]):
                result = assess_readability(frame, OcrConfig(preprocess=False), ReadabilityConfig())
    assert result.verdict == OcrReadabilityVerdict.NEEDS_VLM
    assert result.needs_vlm is True
    assert len(result.vlm_crops) >= 1


def test_assess_readability_hybrid_overlay_and_vlm():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    raw = _mock_raw([("SINNER ITA", 0.88, [20, 900, 300, 980])])
    blob = TextBlobCandidate(
        bbox=np.array([500, 80, 1100, 450], dtype=np.int32),
        area=30000,
        source="contrast",
    )
    det = TextBlobCandidate(
        bbox=np.array([20, 900, 300, 980], dtype=np.int32),
        area=20000,
        source="detector",
    )
    dt_boxes = _mock_dt_boxes()
    with patch("src.scene_ocr.readability.detect_raw", return_value=(frame, 1.0, dt_boxes)):
        with patch(
            "src.scene_ocr.readability.build_text_candidates",
            return_value=[det, blob],
        ):
            with patch("src.scene_ocr.readability.run_ocr_from_boxes", return_value=raw):
                result = assess_readability(frame, OcrConfig(preprocess=False), ReadabilityConfig())
    assert result.overlay_readable is True
    assert result.needs_vlm is True
    assert result.verdict == OcrReadabilityVerdict.NEEDS_VLM


def test_assess_readability_null_vlm_client():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    blob = TextBlobCandidate(
        bbox=np.array([400, 100, 900, 500], dtype=np.int32),
        area=25000,
        source="contrast",
    )
    dt_boxes = _mock_dt_boxes()
    with patch("src.scene_ocr.readability.detect_raw", return_value=(frame, 1.0, dt_boxes)):
        with patch(
            "src.scene_ocr.readability.build_text_candidates",
            return_value=[blob],
        ):
            with patch("src.scene_ocr.readability.run_ocr_from_boxes", return_value=[]):
                result = assess_readability(
                    frame,
                    OcrConfig(preprocess=False),
                    ReadabilityConfig(),
                    vlm_client=NullVlmClient(),
                )
    assert result.words == []


def test_assess_readability_merges_vlm_words():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    blob = TextBlobCandidate(
        bbox=np.array([400, 100, 900, 500], dtype=np.int32),
        area=25000,
        source="contrast",
    )
    vlm = MagicMock()
    vlm.read_text.return_value = ["CHASE"]
    dt_boxes = _mock_dt_boxes()
    with patch("src.scene_ocr.readability.detect_raw", return_value=(frame, 1.0, dt_boxes)):
        with patch(
            "src.scene_ocr.readability.build_text_candidates",
            return_value=[blob],
        ):
            with patch("src.scene_ocr.readability.run_ocr_from_boxes", return_value=[]):
                result = assess_readability(
                    frame,
                    OcrConfig(preprocess=False),
                    ReadabilityConfig(),
                    vlm_client=vlm,
                )
    assert "CHASE" in result.words


@requires_rapidocr
def test_integration_scene_6_assess_readability():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    image_path = root / "data" / "scene_samples" / "scene_6_frame_3356.jpg"
    if not image_path.is_file():
        pytest.skip("scene_6 frame not available")

    result = assess_readability(str(image_path), OcrConfig(), ReadabilityConfig())
    assert result.overlay_readable is True
    assert result.needs_vlm is True
    assert result.verdict == OcrReadabilityVerdict.NEEDS_VLM
    assert any(w.upper() == "SINNER" for w in result.words)
    assert len(result.vlm_crops) >= 1


@requires_rapidocr
def test_integration_solid_frame_no_text():
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    result = assess_readability(frame, OcrConfig(preprocess=False), ReadabilityConfig(min_blob_area=5000))
    assert result.verdict == OcrReadabilityVerdict.NO_TEXT
