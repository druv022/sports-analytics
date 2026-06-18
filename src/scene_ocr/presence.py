"""Stage 1 text-presence detection for scene OCR."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from src.scene_ocr.config import OcrConfig, OverlayRegion, ReadabilityConfig
from src.scene_ocr.extractor import detect_boxes
from src.scene_ocr.geometry import bbox_area, crop_region, detection_in_region, offset_xyxy, polygon_to_xyxy
from src.scene_ocr.types import TextBlobCandidate


def _contrast_blobs(
    bgr: NDArray,
    region: OverlayRegion,
    config: ReadabilityConfig,
) -> list[TextBlobCandidate]:
    crop, offset = crop_region(bgr, region)
    if crop.size == 0:
        return []

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, config.blob_threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[TextBlobCandidate] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < config.min_blob_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        local_xyxy = np.array([x, y, x + w, y + h], dtype=np.int32)
        full_xyxy = offset_xyxy(local_xyxy, offset)
        candidates.append(
            TextBlobCandidate(
                bbox=full_xyxy,
                area=float(area),
                source="contrast",
            )
        )
    return candidates


def _overlay_region_candidates_from_boxes(
    boxes: list[NDArray],
    regions: tuple[OverlayRegion, ...],
    width: int,
    height: int,
) -> list[TextBlobCandidate]:
    candidates: list[TextBlobCandidate] = []
    for region in regions:
        for box in boxes:
            xyxy = polygon_to_xyxy(box)
            if detection_in_region(xyxy, region, width, height):
                candidates.append(
                    TextBlobCandidate(
                        bbox=xyxy,
                        area=bbox_area(xyxy),
                        source="overlay_region",
                    )
                )
    return candidates


def _dedupe_candidates(candidates: list[TextBlobCandidate], iou_threshold: float) -> list[TextBlobCandidate]:
    from src.scene_ocr.geometry import bbox_iou

    kept: list[TextBlobCandidate] = []
    for candidate in sorted(candidates, key=lambda c: c.area, reverse=True):
        if any(bbox_iou(candidate.bbox, other.bbox) > iou_threshold for other in kept):
            continue
        kept.append(candidate)
    return kept


def build_text_candidates(
    bgr: NDArray,
    detected_boxes: list[NDArray],
    readability_config: ReadabilityConfig,
) -> list[TextBlobCandidate]:
    """Build candidate regions from one detector pass plus cheap contrast blobs."""
    height, width = bgr.shape[:2]
    candidates: list[TextBlobCandidate] = [
        TextBlobCandidate(
            bbox=polygon_to_xyxy(box),
            area=bbox_area(box),
            source="detector",
        )
        for box in detected_boxes
    ]

    sponsor = next(
        (r for r in readability_config.overlay_regions if r.name == "sponsor_band"),
        readability_config.overlay_regions[-1],
    )
    candidates.extend(_contrast_blobs(bgr, sponsor, readability_config))
    candidates.extend(
        _overlay_region_candidates_from_boxes(
            detected_boxes,
            readability_config.overlay_regions,
            width,
            height,
        )
    )
    return _dedupe_candidates(candidates, readability_config.crop_iou_dedupe)


def find_text_candidates(
    bgr: NDArray,
    ocr_config: OcrConfig,
    readability_config: ReadabilityConfig,
) -> list[TextBlobCandidate]:
    """Return regions that likely contain text."""
    detected_boxes = detect_boxes(bgr, ocr_config, readability_config)
    return build_text_candidates(bgr, detected_boxes, readability_config)
