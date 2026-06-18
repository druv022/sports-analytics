"""Lightweight scene OCR via RapidOCR (PP-OCR ONNX models).

Requires the unified ``rapidocr`` package (>=3.8) with onnxruntime.
"""

from __future__ import annotations

import importlib
import re
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import cv2
import numpy as np
from numpy.typing import NDArray

from src.accelerator.device import OcrBackend, resolve_ocr_backend
from src.scene_ocr.config import OcrConfig, ReadabilityConfig
from src.scene_ocr.geometry import bbox_height, polygon_to_xyxy, scale_xyxy
from src.scene_ocr.types import OcrDetection

if TYPE_CHECKING:
    from rapidocr import RapidOCR as RapidOCRType

_OCR_IMPORT_ERROR: ImportError | None = None
RapidOCR: Any = None
_ENGINE_LOCK = threading.Lock()

_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+(?:-\d+)+|\d+")
_EDGE_PUNCT_RE = re.compile(r"^[^\w\-]+|[^\w\-]+$", re.UNICODE)
_COUNTRY_SUFFIXES = frozenset(
    {
        "ITA",
        "ESP",
        "USA",
        "GBR",
        "FRA",
        "GER",
        "AUS",
        "CAN",
        "SUI",
        "SRB",
        "GRE",
        "JPN",
        "CHN",
        "KOR",
        "RUS",
        "ARG",
        "BRA",
        "MEX",
        "NED",
        "BEL",
        "CRO",
        "CZE",
        "POL",
        "AUT",
        "DEN",
        "SWE",
        "NOR",
        "FIN",
        "RSA",
        "TPE",
        "UKR",
        "KAZ",
        "GEO",
        "BUL",
        "ROU",
        "HUN",
        "SVK",
        "TUR",
        "ISR",
        "IND",
        "NZL",
    }
)


class _EngineKey(NamedTuple):
    use_cls: bool
    backend: OcrBackend
    box_thresh: float
    unclip_ratio: float
    rec_batch_num: int
    cls_batch_num: int


def _load_rapidocr() -> Any:
    global RapidOCR, _OCR_IMPORT_ERROR
    if RapidOCR is not None:
        return RapidOCR
    if _OCR_IMPORT_ERROR is not None:
        raise _OCR_IMPORT_ERROR
    try:
        module = importlib.import_module("rapidocr")
        RapidOCR = module.RapidOCR
        _verify_rapidocr_api(RapidOCR)
        return RapidOCR
    except ImportError as exc:
        if "rapidocr >=" in str(exc) or "missing RapidOCR attributes" in str(exc):
            _OCR_IMPORT_ERROR = exc
            raise
        _OCR_IMPORT_ERROR = ImportError(
            "rapidocr is required for scene OCR. "
            f"Install into the active interpreter ({sys.executable}) with: "
            "pip install -e '.[ocr]'"
        )
        raise _OCR_IMPORT_ERROR from exc


def _verify_rapidocr_api(rapid_ocr_cls: Any) -> None:
    probe = rapid_ocr_cls(params={"Global.use_cls": False})
    required = ("use_det", "text_det", "text_rec", "use_cls", "crop_text_regions")
    missing = [name for name in required if not hasattr(probe, name)]
    del probe
    if missing:
        raise ImportError(
            "rapidocr >= 3.8 is required (missing RapidOCR attributes: "
            f"{', '.join(missing)}). Upgrade with: pip install -U 'rapidocr>=3.8'"
        )


def require_ocr() -> None:
    _load_rapidocr()


def _config_ocr_backend(config: OcrConfig) -> OcrBackend:
    if config.use_cuda is False:
        return "cpu"
    if config.use_cuda is True:
        return resolve_ocr_backend("auto")
    return resolve_ocr_backend("auto")


def _engine_key(config: OcrConfig, backend: OcrBackend) -> _EngineKey:
    return _EngineKey(
        use_cls=config.use_angle_cls,
        backend=backend,
        box_thresh=config.box_thresh,
        unclip_ratio=config.unclip_ratio,
        rec_batch_num=config.resolved_rec_batch_num(backend),
        cls_batch_num=config.resolved_cls_batch_num(backend),
    )


def _engine_params(key: _EngineKey) -> dict[str, Any]:
    params: dict[str, Any] = {
        "Global.use_cls": key.use_cls,
        "Det.box_thresh": key.box_thresh,
        "Det.unclip_ratio": key.unclip_ratio,
        "Rec.rec_batch_num": key.rec_batch_num,
        "Cls.cls_batch_num": key.cls_batch_num,
    }
    if key.backend == "cuda":
        params["EngineConfig.onnxruntime.use_cuda"] = True
        params["EngineConfig.onnxruntime.intra_op_num_threads"] = 1
        params["EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search"] = "HEURISTIC"
    elif key.backend == "coreml":
        params["EngineConfig.onnxruntime.use_coreml"] = True
        params["EngineConfig.onnxruntime.intra_op_num_threads"] = 1
    return params


def _warmup_engine(engine: Any) -> None:
    if not getattr(engine, "use_det", False):
        return
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    try:
        with _ENGINE_LOCK:
            engine.text_det(dummy)
    except Exception:
        pass


@lru_cache(maxsize=8)
def _get_engine(key: _EngineKey) -> RapidOCRType:
    rapid_ocr_cls = _load_rapidocr()
    engine = rapid_ocr_cls(params=_engine_params(key))
    _warmup_engine(engine)
    return engine


def clear_ocr_engine_cache() -> None:
    """Clear cached RapidOCR engines after pip install or device changes."""
    _get_engine.cache_clear()


def load_image(image: str | Path | NDArray) -> NDArray:
    """Load or validate a BGR uint8 image array."""
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        loaded = cv2.imread(str(path))
        if loaded is None:
            raise ValueError(f"Failed to decode image: {path}")
        return loaded

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected path or ndarray, got {type(image).__name__}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 BGR image, got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {image.dtype}")
    return image


def preprocess_image(bgr: NDArray, config: OcrConfig) -> NDArray:
    """Apply CLAHE and optional upscaling for OCR."""
    out = bgr
    if config.clahe:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        out = cv2.cvtColor(
            cv2.merge([l_channel, a_channel, b_channel]),
            cv2.COLOR_LAB2BGR,
        )
    if config.scale != 1.0:
        out = cv2.resize(
            out,
            None,
            fx=config.scale,
            fy=config.scale,
            interpolation=cv2.INTER_CUBIC,
        )
    return out


def _ocr_input(bgr: NDArray, config: OcrConfig) -> tuple[NDArray, float]:
    if config.preprocess:
        return preprocess_image(bgr, config), config.scale
    return bgr, 1.0


def normalize_word(token: str) -> str:
    return _EDGE_PUNCT_RE.sub("", token)


def split_merged_token(token: str) -> list[str]:
    if not token.isalpha() or len(token) <= 3:
        return [token]
    suffix = token[-3:].upper()
    prefix = token[:-3]
    if suffix in _COUNTRY_SUFFIXES and prefix.isalpha() and len(prefix) >= 2:
        return [prefix, suffix]
    return [token]


def lines_to_words(text: str) -> list[str]:
    words: list[str] = []
    for raw_token in _TOKEN_RE.findall(text):
        for token in split_merged_token(raw_token):
            word = normalize_word(token)
            if word:
                words.append(word)
    return words


def is_plausible_word(word: str, min_len: int = 2) -> bool:
    if len(word) < min_len:
        return False
    alnum = sum(ch.isalnum() for ch in word)
    return alnum / len(word) >= 0.6


def detect_raw(
    bgr: NDArray,
    ocr_config: OcrConfig,
) -> tuple[NDArray, float, NDArray | None]:
    """Run text detection once; return preprocessed image, scale, and raw polygons."""
    backend = _config_ocr_backend(ocr_config)
    engine = _get_engine(_engine_key(ocr_config, backend))
    if not engine.use_det:
        variant, scale = _ocr_input(bgr, ocr_config)
        return variant, scale, None

    variant, scale = _ocr_input(bgr, ocr_config)
    with _ENGINE_LOCK:
        det_out = engine.text_det(variant)
    if det_out.boxes is None or len(det_out.boxes) < 1:
        return variant, scale, None
    return variant, scale, det_out.boxes


def detect_boxes(
    bgr: NDArray,
    ocr_config: OcrConfig,
    readability_config: ReadabilityConfig,
) -> list[NDArray]:
    """Run text detection only; return xyxy boxes in original image coordinates."""
    _variant, scale, dt_boxes = detect_raw(bgr, ocr_config)
    if dt_boxes is None:
        return []

    boxes: list[NDArray] = []
    for box in dt_boxes:
        xyxy = scale_xyxy(polygon_to_xyxy(np.asarray(box)), scale)
        if bbox_height(xyxy) >= readability_config.min_box_height_px:
            boxes.append(xyxy)
    return boxes


def _merge_raw_results(
    filter_boxes: list[NDArray],
    filter_rec_res: list[tuple[str, float]],
    scale: float,
) -> list[tuple[str, float, NDArray]]:
    merged: dict[str, tuple[str, float, NDArray]] = {}
    for box, rec_result in zip(filter_boxes, filter_rec_res):
        text, score = rec_result[0], float(rec_result[1])
        if not text or not str(text).strip():
            continue
        cleaned = str(text).strip()
        xyxy = scale_xyxy(polygon_to_xyxy(np.asarray(box)), scale)
        existing = merged.get(cleaned)
        if existing is None or score > existing[1]:
            merged[cleaned] = (cleaned, score, xyxy)
    return list(merged.values())


def _parse_ocr_result(
    result: Any,
    scale: float,
) -> list[tuple[str, float, NDArray]]:
    if result is None:
        return []

    if hasattr(result, "txts"):
        if result.txts is None:
            return []
        if result.boxes is None:
            return []
        merged: dict[str, tuple[str, float, NDArray]] = {}
        scores = result.scores or [1.0] * len(result.txts)
        for bbox, text, score in zip(result.boxes, result.txts, scores):
            if not text or not str(text).strip():
                continue
            cleaned = str(text).strip()
            xyxy = scale_xyxy(polygon_to_xyxy(np.asarray(bbox)), scale)
            existing = merged.get(cleaned)
            score_f = float(score)
            if existing is None or score_f > existing[1]:
                merged[cleaned] = (cleaned, score_f, xyxy)
        return list(merged.values())

    if not result:
        return []

    merged = {}
    for item in result:
        if len(item) < 3:
            continue
        bbox, text, score = item[0], item[1], float(item[2])
        if not text or not str(text).strip():
            continue
        cleaned = str(text).strip()
        xyxy = scale_xyxy(polygon_to_xyxy(np.asarray(bbox)), scale)
        existing = merged.get(cleaned)
        if existing is None or score > existing[1]:
            merged[cleaned] = (cleaned, score, xyxy)
    return list(merged.values())


def run_ocr_from_boxes(
    variant: NDArray,
    scale: float,
    dt_boxes: NDArray,
    config: OcrConfig,
) -> list[tuple[str, float, NDArray]]:
    """Run cls+rec on precomputed detection boxes (skips second detection pass)."""
    backend = _config_ocr_backend(config)
    engine = _get_engine(_engine_key(config, backend))
    img_crop_list = engine.crop_text_regions(variant, dt_boxes)

    with _ENGINE_LOCK:
        if engine.use_cls:
            img_crop_list, _ = engine.cls_and_rotate(img_crop_list)
        rec_res = engine.recognize_txt(img_crop_list)
    if rec_res.txts is None or not rec_res.txts:
        return []

    filter_boxes: list[NDArray] = []
    filter_rec_res: list[tuple[str, float]] = []
    scores = rec_res.scores or [1.0] * len(rec_res.txts)
    for box, txt, score in zip(dt_boxes, rec_res.txts, scores):
        cleaned = str(txt).strip()
        if cleaned:
            filter_boxes.append(np.asarray(box))
            filter_rec_res.append((cleaned, float(score)))
    if not filter_boxes:
        return []
    return _merge_raw_results(filter_boxes, filter_rec_res, scale)


def run_ocr_raw(bgr: NDArray, config: OcrConfig) -> list[tuple[str, float, NDArray]]:
    """Run full OCR; bboxes are in original image coordinates."""
    backend = _config_ocr_backend(config)
    engine = _get_engine(_engine_key(config, backend))
    variant, scale = _ocr_input(bgr, config)
    with _ENGINE_LOCK:
        result = engine(
            variant,
            use_det=True,
            use_cls=config.use_angle_cls,
            use_rec=True,
            box_thresh=config.box_thresh,
            unclip_ratio=config.unclip_ratio,
            text_score=config.engine_text_score,
        )
    return _parse_ocr_result(result, scale)


def raw_to_detections(
    raw: list[tuple[str, float, NDArray]],
    min_confidence: float,
) -> list[OcrDetection]:
    return [
        OcrDetection(text=text, confidence=score, bbox=bbox)
        for text, score, bbox in raw
        if score >= min_confidence
    ]


def extract_detections(
    image: str | Path | NDArray,
    config: OcrConfig | None = None,
) -> list[OcrDetection]:
    """Return line-level OCR detections before word splitting."""
    cfg = config or OcrConfig()
    bgr = load_image(image)
    raw = run_ocr_raw(bgr, cfg)
    return raw_to_detections(raw, cfg.min_confidence)


def extract_words(
    image: str | Path | NDArray,
    config: OcrConfig | None = None,
) -> list[str]:
    """Return words found in the image, filtered by confidence."""
    cfg = config or OcrConfig()
    words: list[str] = []
    seen: set[str] = set()

    for detection in extract_detections(image, cfg):
        for word in lines_to_words(detection.text):
            if cfg.dedupe:
                key = word.casefold()
                if key in seen:
                    continue
                seen.add(key)
            words.append(word)

    return words
