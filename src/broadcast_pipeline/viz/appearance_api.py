from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from broadcast_pipeline.appearance_compat import appearance_config_from_pipeline
from broadcast_pipeline.config import PipelineConfig
from src.person_appearance.extractor import analyze_frame, default_segmenter
from src.person_appearance.segmenter import PersonSegmenter


_segmenter_cache: dict[str, PersonSegmenter] = {}


def _decode_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode uploaded image")
    return bgr


def _mask_contours(mask: np.ndarray | None) -> list[list[list[int]]]:
    if mask is None:
        return []
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result: list[list[list[int]]] = []
    for contour in contours:
        if contour.size < 6:
            continue
        points = contour.reshape(-1, 2).tolist()
        result.append([[int(x), int(y)] for x, y in points])
    return result


def get_segmenter(output_dir: Path | None = None, config: PipelineConfig | None = None) -> PersonSegmenter:
    """Lazy singleton segmenter keyed by resolved model path."""
    pipeline_config = config or PipelineConfig(output_dir=Path(output_dir or "data/pipeline"))
    appearance_cfg = appearance_config_from_pipeline(pipeline_config)
    model_key = str(appearance_cfg.resolved_model_path().resolve())
    if model_key not in _segmenter_cache:
        _segmenter_cache[model_key] = default_segmenter(appearance_cfg)
    return _segmenter_cache[model_key]


def clear_segmenter_cache() -> None:
    _segmenter_cache.clear()


def run_appearance_on_bytes(
    image_bytes: bytes,
    *,
    output_dir: Path | None = None,
    config: PipelineConfig | None = None,
    segmenter: PersonSegmenter | None = None,
    scene_id: int = -1,
    frame_number: int = -1,
) -> dict:
    """Run person segmentation on raw image bytes and return JSON-serializable output."""
    pipeline_config = config or PipelineConfig(output_dir=Path(output_dir or "data/pipeline"))
    appearance_cfg = appearance_config_from_pipeline(pipeline_config)
    bgr = _decode_image(image_bytes)
    height, width = bgr.shape[:2]

    if segmenter is None:
        segmenter = get_segmenter(config=pipeline_config)

    result = analyze_frame(
        bgr,
        scene_id=scene_id,
        frame_number=frame_number,
        frame_path="",
        config=appearance_cfg,
        segmenter=segmenter,
    )

    detections = []
    for det in result.detections:
        detections.append(
            {
                "bbox": [int(v) for v in det.bbox_xyxy],
                "confidence": float(det.confidence),
                "clothing_color": det.clothing_color or "neutral",
                "mask_contours": _mask_contours(det.mask),
            }
        )

    return {
        "person_count": int(result.person_count),
        "status": result.status,
        "person_colors": list(result.person_colors),
        "primary_bgr": list(result.primary_bgr) if result.primary_bgr else [],
        "confidence": float(result.confidence),
        "detections": detections,
        "image_width": int(width),
        "image_height": int(height),
    }
