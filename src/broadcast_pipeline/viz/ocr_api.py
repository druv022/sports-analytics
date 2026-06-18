from __future__ import annotations

import cv2
import numpy as np

from src.scene_ocr.config import OcrConfig
from src.scene_ocr.extractor import extract_detections


def default_viz_ocr_config() -> OcrConfig:
    return OcrConfig(min_confidence=0.5, preprocess=True, scale=1.5, dedupe=True)


def _decode_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode uploaded image")
    return bgr


def run_ocr_on_bytes(image_bytes: bytes, config: OcrConfig | None = None) -> dict:
    """Run line-level OCR on raw image bytes and return JSON-serializable output."""
    config = config or default_viz_ocr_config()
    bgr = _decode_image(image_bytes)
    height, width = bgr.shape[:2]
    detections = extract_detections(bgr, config)

    payload_detections = [
        {
            "text": det.text,
            "confidence": float(det.confidence),
            "bbox": [int(x) for x in det.bbox.tolist()],
        }
        for det in detections
    ]

    return {
        "verdict": "readable" if payload_detections else "no_text",
        "detections": payload_detections,
        "image_width": int(width),
        "image_height": int(height),
    }
