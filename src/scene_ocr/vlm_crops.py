"""Build VLM-ready crops from unread text candidates."""

from __future__ import annotations

from numpy.typing import NDArray

from src.scene_ocr.config import ReadabilityConfig
from src.scene_ocr.geometry import (
    bbox_iou,
    crop_xyxy,
    pad_xyxy,
    polygon_to_xyxy,
    resize_max_dim,
)
from src.scene_ocr.types import OcrDetection, TextBlobCandidate


def _matched_detection(
    candidate: TextBlobCandidate,
    detections: list[OcrDetection],
    iou_threshold: float,
) -> OcrDetection | None:
    best: OcrDetection | None = None
    best_iou = 0.0
    for detection in detections:
        iou = bbox_iou(candidate.bbox, detection.bbox)
        if iou > best_iou:
            best_iou = iou
            best = detection
    if best_iou >= iou_threshold:
        return best
    return None


def build_vlm_crops(
    bgr: NDArray,
    candidates: list[TextBlobCandidate],
    detections: list[OcrDetection],
    all_detections: list[OcrDetection],
    config: ReadabilityConfig,
    confidence_readable: float,
) -> list[NDArray]:
    """Build padded, deduped crops for VLM escalation."""
    height, width = bgr.shape[:2]
    priority: list[NDArray] = []

    contrast = [c for c in candidates if c.source == "contrast"]
    for candidate in sorted(contrast, key=lambda c: c.area, reverse=True):
        match = _matched_detection(candidate, all_detections, config.crop_iou_dedupe)
        if match is not None and match.confidence >= confidence_readable:
            continue
        priority.append(candidate.bbox)

    for detection in all_detections:
        if detection.confidence >= confidence_readable:
            continue
        if any(bbox_iou(detection.bbox, polygon_to_xyxy(box)) > config.crop_iou_dedupe for box in priority):
            continue
        priority.append(polygon_to_xyxy(detection.bbox))

    for candidate in candidates:
        if candidate.source != "overlay_region":
            continue
        match = _matched_detection(candidate, detections, config.crop_iou_dedupe)
        if match is not None and match.confidence >= confidence_readable:
            continue
        if any(bbox_iou(candidate.bbox, polygon_to_xyxy(box)) > config.crop_iou_dedupe for box in priority):
            continue
        priority.append(candidate.bbox)

    crops: list[NDArray] = []
    crop_boxes: list[NDArray] = []
    for xyxy in priority:
        padded = pad_xyxy(xyxy, config.crop_pad_ratio, width, height)
        if any(bbox_iou(padded, existing) > config.crop_iou_dedupe for existing in crop_boxes):
            continue
        crop = resize_max_dim(crop_xyxy(bgr, padded), config.max_crop_dim_px)
        if crop.size == 0:
            continue
        crops.append(crop)
        crop_boxes.append(padded)
        if len(crops) >= config.max_vlm_crops:
            break
    return crops
