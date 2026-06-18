"""Tests for scene OCR word extraction."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.scene_ocr.config import OcrConfig, ReadabilityConfig


def test_ocr_config_backend_batch_defaults():
    cfg = OcrConfig()
    assert cfg.resolved_rec_batch_num("cuda") == 16
    assert cfg.resolved_cls_batch_num("coreml") == 8
    assert cfg.resolved_rec_batch_num("cpu") == 6
from tests.scene_ocr.conftest import requires_rapidocr
from src.scene_ocr.extractor import (
    detect_boxes,
    lines_to_words,
    extract_detections,
    extract_words,
    load_image,
    require_ocr,
)


def _mock_ocr_output(items: list[tuple]) -> SimpleNamespace:
    boxes = [item[0] for item in items]
    txts = tuple(item[1] for item in items)
    scores = tuple(float(item[2]) for item in items)
    return SimpleNamespace(boxes=boxes, txts=txts, scores=scores)


def test_require_ocr_raises_when_missing():
    with patch("src.scene_ocr.extractor._load_rapidocr", side_effect=ImportError("missing")):
        with pytest.raises(ImportError, match="missing"):
            require_ocr()


def test_load_image_from_array():
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    loaded = load_image(frame)
    assert loaded is frame


def test_load_image_rejects_bad_shape():
    with pytest.raises(ValueError, match="HxWx3"):
        load_image(np.zeros((48, 64), dtype=np.uint8))


def test_load_image_missing_file(tmp_path: Path):
    missing = tmp_path / "missing.jpg"
    with pytest.raises(FileNotFoundError, match="Image not found"):
        load_image(missing)


def test_lines_to_words_splits_digits_and_country_codes():
    assert lines_to_words("1SINNERITA 30") == ["1", "SINNER", "ITA", "30"]
    assert lines_to_words("2 ALCARAZ ESP") == ["2", "ALCARAZ", "ESP"]


def test_extract_words_splits_scoreboard_tokens(monkeypatch):
    fake_result = [
        ([[0, 0], [1, 0], [1, 1], [0, 1]], "1SINNERITA", 0.9),
        ([[0, 0], [1, 0], [1, 1], [0, 1]], "2 ALCARAZ ESP", 0.88),
    ]
    mock_engine = MagicMock(return_value=_mock_ocr_output(fake_result))
    monkeypatch.setattr(
        "src.scene_ocr.extractor._get_engine",
        lambda _key: mock_engine,
    )

    config = OcrConfig(preprocess=False)
    words = extract_words(np.zeros((100, 100, 3), dtype=np.uint8), config)
    assert words == ["1", "SINNER", "ITA", "2", "ALCARAZ", "ESP"]


def test_extract_words_splits_and_filters(monkeypatch):
    fake_result = [
        ([[0, 0], [1, 0], [1, 1], [0, 1]], "SET 2", 0.9),
        ([[0, 0], [1, 0], [1, 1], [0, 1]], "DEUCE!", 0.4),
        ([[0, 0], [1, 0], [1, 1], [0, 1]], '"15-30"', 0.85),
    ]
    mock_engine = MagicMock(return_value=_mock_ocr_output(fake_result))
    monkeypatch.setattr(
        "src.scene_ocr.extractor._get_engine",
        lambda _key: mock_engine,
    )

    words = extract_words(np.zeros((100, 100, 3), dtype=np.uint8), OcrConfig(preprocess=False))
    assert words == ["SET", "2", "15-30"]


def test_extract_words_dedupes_case_insensitive(monkeypatch):
    fake_result = [
        ([[0, 0], [1, 0], [1, 1], [0, 1]], "Set", 0.9),
        ([[0, 0], [1, 0], [1, 1], [0, 1]], "set", 0.88),
    ]
    mock_engine = MagicMock(return_value=_mock_ocr_output(fake_result))
    monkeypatch.setattr(
        "src.scene_ocr.extractor._get_engine",
        lambda _key: mock_engine,
    )

    words = extract_words(np.zeros((100, 100, 3), dtype=np.uint8), OcrConfig(preprocess=False))
    assert words == ["Set"]


def test_extract_words_no_dedupe(monkeypatch):
    fake_result = [
        ([[0, 0], [1, 0], [1, 1], [0, 1]], "Set", 0.9),
        ([[0, 0], [1, 0], [1, 1], [0, 1]], "set", 0.88),
    ]
    mock_engine = MagicMock(return_value=_mock_ocr_output(fake_result))
    monkeypatch.setattr(
        "src.scene_ocr.extractor._get_engine",
        lambda _key: mock_engine,
    )

    config = OcrConfig(dedupe=False, preprocess=False)
    words = extract_words(np.zeros((100, 100, 3), dtype=np.uint8), config)
    assert words == ["Set", "set"]


def test_extract_detections_returns_line_objects(monkeypatch):
    bbox = np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=np.float32)
    fake_result = [(bbox.tolist(), "GAME SET", 0.92)]
    mock_engine = MagicMock(return_value=_mock_ocr_output(fake_result))
    monkeypatch.setattr(
        "src.scene_ocr.extractor._get_engine",
        lambda _key: mock_engine,
    )

    detections = extract_detections(np.zeros((100, 100, 3), dtype=np.uint8), OcrConfig(preprocess=False))
    assert len(detections) == 1
    assert detections[0].text == "GAME SET"
    assert detections[0].confidence == pytest.approx(0.92)
    assert detections[0].bbox.shape == (4,)


def test_detect_boxes_uses_rapidocr_text_det_api(monkeypatch):
    polygon = np.array([[0, 0], [40, 0], [40, 20], [0, 20]], dtype=np.float32)
    mock_text_det = MagicMock(
        return_value=SimpleNamespace(boxes=np.array([polygon]), scores=[0.9], elapse=0.01)
    )
    mock_text_det.postprocess_op.box_thresh = 0.3
    mock_text_det.postprocess_op.unclip_ratio = 1.6
    mock_engine = MagicMock()
    mock_engine.use_det = True
    mock_engine.text_det = mock_text_det
    monkeypatch.setattr(
        "src.scene_ocr.extractor._get_engine",
        lambda _key: mock_engine,
    )

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = detect_boxes(frame, OcrConfig(preprocess=False), ReadabilityConfig(min_box_height_px=5))
    assert len(boxes) == 1
    assert boxes[0].tolist() == [0, 0, 40, 20]


def test_detect_boxes_skips_when_use_det_disabled(monkeypatch):
    mock_engine = MagicMock()
    mock_engine.use_det = False
    monkeypatch.setattr(
        "src.scene_ocr.extractor._get_engine",
        lambda _key: mock_engine,
    )

    boxes = detect_boxes(
        np.zeros((100, 100, 3), dtype=np.uint8),
        OcrConfig(preprocess=False),
        ReadabilityConfig(),
    )
    assert boxes == []


def test_get_engine_passes_cuda_params(monkeypatch):
    created: dict = {}

    class FakeRapidOCR:
        use_det = True
        use_cls = True
        text_det = MagicMock(return_value=SimpleNamespace(boxes=None))

        def __init__(self, params=None, **kwargs):
            created["params"] = params or {}

    monkeypatch.setattr("src.scene_ocr.extractor.RapidOCR", FakeRapidOCR)
    monkeypatch.setattr("src.scene_ocr.extractor._load_rapidocr", lambda: FakeRapidOCR)
    from src.scene_ocr.extractor import _EngineKey, _get_engine, clear_ocr_engine_cache

    clear_ocr_engine_cache()
    _get_engine(_EngineKey(True, "cuda", 0.3, 1.8, 16, 16))
    params = created["params"]
    assert params["Global.use_cls"] is True
    assert params["EngineConfig.onnxruntime.use_cuda"] is True
    assert params["Rec.rec_batch_num"] == 16
    assert params["Cls.cls_batch_num"] == 16
    assert params["Det.box_thresh"] == 0.3
    assert params["Det.unclip_ratio"] == 1.8
    assert params["EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search"] == "HEURISTIC"


def test_get_engine_passes_coreml_params(monkeypatch):
    created: dict = {}

    class FakeRapidOCR:
        def __init__(self, params=None, **kwargs):
            created["params"] = params or {}

    monkeypatch.setattr("src.scene_ocr.extractor.RapidOCR", FakeRapidOCR)
    monkeypatch.setattr("src.scene_ocr.extractor._load_rapidocr", lambda: FakeRapidOCR)
    from src.scene_ocr.extractor import _EngineKey, _get_engine, clear_ocr_engine_cache

    clear_ocr_engine_cache()
    _get_engine(_EngineKey(False, "coreml", 0.3, 1.8, 8, 8))
    assert created["params"]["EngineConfig.onnxruntime.use_coreml"] is True


def test_run_ocr_from_boxes_uses_text_rec(monkeypatch):
    polygon = np.array([[0, 0], [40, 0], [40, 20], [0, 20]], dtype=np.float32)
    mock_engine = MagicMock()
    mock_engine.use_cls = False
    mock_engine.crop_text_regions.return_value = [np.zeros((20, 40, 3), dtype=np.uint8)]
    mock_engine.recognize_txt.return_value = SimpleNamespace(
        txts=("HELLO",),
        scores=[0.95],
    )
    monkeypatch.setattr(
        "src.scene_ocr.extractor._get_engine",
        lambda _key: mock_engine,
    )

    from src.scene_ocr.extractor import run_ocr_from_boxes

    variant = np.zeros((100, 100, 3), dtype=np.uint8)
    raw = run_ocr_from_boxes(variant, 1.0, np.array([polygon]), OcrConfig(preprocess=False))
    assert raw[0][0] == "HELLO"
    mock_engine.recognize_txt.assert_called_once()


def test_extract_words_empty_when_no_text(monkeypatch):
    mock_engine = MagicMock(return_value=SimpleNamespace(boxes=None, txts=None, scores=None))
    monkeypatch.setattr(
        "src.scene_ocr.extractor._get_engine",
        lambda _key: mock_engine,
    )

    words = extract_words(np.zeros((100, 100, 3), dtype=np.uint8), OcrConfig(preprocess=False))
    assert words == []


@requires_rapidocr
def test_integration_extract_words_from_synthetic_image():
    image = np.full((120, 400, 3), 255, dtype=np.uint8)
    cv2.putText(
        image,
        "DEUCE",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )

    words = extract_words(image, OcrConfig(min_confidence=0.3))
    assert any("DEUCE" in word.upper() for word in words)
