"""Readability assessment and VLM escalation gating."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.scene_ocr.config import OcrConfig, OverlayRegion, ReadabilityConfig
from src.scene_ocr.extractor import (
    detect_raw,
    is_plausible_word,
    lines_to_words,
    load_image,
    run_ocr_from_boxes,
)
from src.scene_ocr.geometry import bbox_height, bbox_iou, detection_in_region, polygon_to_xyxy, scale_xyxy
from src.scene_ocr.presence import build_text_candidates
from src.scene_ocr.types import (
    OcrDetection,
    OcrReadability,
    OcrReadabilityVerdict,
    TextBlobCandidate,
)
from src.scene_ocr.vlm_client import DEFAULT_VLM_PROMPT, NullVlmClient, VlmOcrClient
from src.scene_ocr.vlm_crops import build_vlm_crops


def _overlay_regions_for_readability(config: ReadabilityConfig) -> tuple[OverlayRegion, ...]:
    return tuple(r for r in config.overlay_regions if r.name in {"score_bug", "logo"})


def _detections_in_overlay(
    detections: list[OcrDetection],
    regions: tuple[OverlayRegion, ...],
    width: int,
    height: int,
) -> list[OcrDetection]:
    matched: list[OcrDetection] = []
    for detection in detections:
        if any(detection_in_region(detection.bbox, region, width, height) for region in regions):
            matched.append(detection)
    return matched


def _compute_overlay_readable(
    detections: list[OcrDetection],
    regions: tuple[OverlayRegion, ...],
    width: int,
    height: int,
    config: ReadabilityConfig,
) -> bool:
    overlay = _detections_in_overlay(detections, regions, width, height)
    for detection in overlay:
        if detection.confidence < config.confidence_readable:
            continue
        if any(is_plausible_word(word) for word in lines_to_words(detection.text)):
            return True
    return False


def _candidate_needs_vlm(
    candidate: TextBlobCandidate,
    all_detections: list[OcrDetection],
    config: ReadabilityConfig,
) -> bool:
    best_conf = 0.0
    for detection in all_detections:
        if bbox_iou(candidate.bbox, detection.bbox) >= config.crop_iou_dedupe:
            best_conf = max(best_conf, detection.confidence)

    if candidate.source == "contrast":
        return best_conf < config.confidence_readable

    if candidate.source == "detector":
        return best_conf < config.confidence_readable

    return best_conf < config.confidence_weak


def _compute_needs_vlm(
    candidates: list[TextBlobCandidate],
    all_detections: list[OcrDetection],
    config: ReadabilityConfig,
    reasons: list[str],
) -> bool:
    for candidate in candidates:
        if candidate.source not in {"contrast", "detector", "overlay_region"}:
            continue
        if _candidate_needs_vlm(candidate, all_detections, config):
            if candidate.source == "contrast":
                reasons.append(f"unmatched_blob_area={int(candidate.area)}")
            else:
                reasons.append(f"unmatched_{candidate.source}_area={int(candidate.area)}")
            return True
    return False


def _collect_words(
    detections: list[OcrDetection],
    ocr_config: OcrConfig,
) -> list[str]:
    words: list[str] = []
    seen: set[str] = set()
    for detection in detections:
        for word in lines_to_words(detection.text):
            if ocr_config.dedupe:
                key = word.casefold()
                if key in seen:
                    continue
                seen.add(key)
            words.append(word)
    return words


def _empty_result(reasons: list[str]) -> OcrReadability:
    return OcrReadability(
        verdict=OcrReadabilityVerdict.NO_TEXT,
        words=[],
        detections=[],
        text_candidates=[],
        vlm_crops=[],
        overlay_readable=False,
        needs_vlm=False,
        reasons=reasons,
    )


def assess_readability_from_bgr(
    bgr: NDArray,
    ocr_config: OcrConfig | None = None,
    readability_config: ReadabilityConfig | None = None,
    vlm_client: VlmOcrClient | None = None,
) -> OcrReadability:
    """Assess whether lightweight OCR suffices or VLM escalation is needed."""
    ocr_cfg = ocr_config or OcrConfig()
    read_cfg = readability_config or ReadabilityConfig()
    reasons: list[str] = []

    height, width = bgr.shape[:2]

    variant, scale, dt_boxes = detect_raw(bgr, ocr_cfg)
    if dt_boxes is None:
        reasons.append("no_text_candidates")
        return _empty_result(reasons)

    detected_boxes = []
    for box in dt_boxes:
        xyxy = scale_xyxy(polygon_to_xyxy(np.asarray(box)), scale)
        if bbox_height(xyxy) >= read_cfg.min_box_height_px:
            detected_boxes.append(xyxy)

    candidates = build_text_candidates(bgr, detected_boxes, read_cfg)
    if not candidates:
        reasons.append("no_text_candidates")
        return _empty_result(reasons)

    reasons.append(f"text_candidates={len(candidates)}")

    raw = run_ocr_from_boxes(variant, scale, dt_boxes, ocr_cfg)
    all_detections = [
        OcrDetection(text=text, confidence=score, bbox=bbox) for text, score, bbox in raw
    ]
    filtered_detections = [
        d for d in all_detections if d.confidence >= ocr_cfg.min_confidence
    ]

    if all_detections:
        max_conf = max(d.confidence for d in all_detections)
        reasons.append(f"max_confidence={max_conf:.2f}")

    overlay_regions = _overlay_regions_for_readability(read_cfg)
    overlay_readable = _compute_overlay_readable(
        all_detections,
        overlay_regions,
        width,
        height,
        read_cfg,
    )
    if overlay_readable:
        reasons.append("overlay_readable=true")

    needs_vlm = _compute_needs_vlm(candidates, all_detections, read_cfg, reasons)
    words = _collect_words(filtered_detections, ocr_cfg)

    vlm_crops: list[NDArray] = []
    if needs_vlm:
        vlm_crops = build_vlm_crops(
            bgr,
            candidates,
            filtered_detections,
            all_detections,
            read_cfg,
            read_cfg.confidence_readable,
        )
        reasons.append(f"vlm_crop_count={len(vlm_crops)}")

    if needs_vlm:
        verdict = OcrReadabilityVerdict.NEEDS_VLM
    elif overlay_readable or words:
        verdict = OcrReadabilityVerdict.READABLE
    else:
        verdict = OcrReadabilityVerdict.NO_TEXT

    result = OcrReadability(
        verdict=verdict,
        words=words,
        detections=filtered_detections,
        text_candidates=candidates,
        vlm_crops=vlm_crops,
        overlay_readable=overlay_readable,
        needs_vlm=needs_vlm,
        reasons=reasons,
    )

    if needs_vlm and vlm_client is not None and vlm_crops and not isinstance(vlm_client, NullVlmClient):
        vlm_words = vlm_client.read_text(vlm_crops, DEFAULT_VLM_PROMPT)
        if vlm_words:
            merged = list(words)
            seen = {w.casefold() for w in merged}
            for word in vlm_words:
                key = word.casefold()
                if key not in seen:
                    seen.add(key)
                    merged.append(word)
            result = OcrReadability(
                verdict=result.verdict,
                words=merged,
                detections=result.detections,
                text_candidates=result.text_candidates,
                vlm_crops=result.vlm_crops,
                overlay_readable=result.overlay_readable,
                needs_vlm=result.needs_vlm,
                reasons=[*result.reasons, f"vlm_words={len(vlm_words)}"],
            )

    return result


def assess_readability(
    image: str | Path | NDArray,
    ocr_config: OcrConfig | None = None,
    readability_config: ReadabilityConfig | None = None,
    vlm_client: VlmOcrClient | None = None,
) -> OcrReadability:
    """Assess readability from a file path or BGR image array."""
    bgr = load_image(image)
    return assess_readability_from_bgr(
        bgr,
        ocr_config=ocr_config,
        readability_config=readability_config,
        vlm_client=vlm_client,
    )
