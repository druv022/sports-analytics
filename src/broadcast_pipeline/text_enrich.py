from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from broadcast_pipeline.config import PipelineConfig
from broadcast_pipeline.progress import log_info
from src.scene_ocr.geometry import bbox_area, bbox_iou

ENRICHED_COLUMNS = [
    "scene_id",
    "frame_number",
    "seconds",
    "camera_id",
    "words_json",
    "detections_json",
    "verdict",
    "used_unk",
]


def load_processed_frame_ocr(config: PipelineConfig) -> pd.DataFrame:
    enriched_path = config.artifact("frame_ocr_enriched")
    if enriched_path.is_file():
        return pd.read_csv(enriched_path)
    return pd.read_csv(config.artifact("frame_ocr"))


def _parse_detections(raw: object) -> list[dict]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    if not isinstance(raw, str) or not raw.strip():
        return []
    data = json.loads(raw)
    return data if isinstance(data, list) else []


def _serialize_detections(detections: list[dict]) -> str:
    return json.dumps(detections)


def _label_readability(detections: list[dict], multiplier: float) -> list[dict]:
    if not detections:
        return detections
    areas = [bbox_area(np.array(det["bbox"], dtype=np.float64)) for det in detections]
    avg_area = sum(areas) / len(areas)
    threshold = multiplier * avg_area if avg_area > 0 else 0.0
    labeled: list[dict] = []
    for det, area in zip(detections, areas):
        entry = dict(det)
        entry["readability_label"] = "good" if area >= threshold else "partial"
        labeled.append(entry)
    return labeled


def _best_iou_match(
    bbox: np.ndarray,
    candidates: list[dict],
    used: set[int],
    iou_threshold: float,
) -> int | None:
    best_idx: int | None = None
    best_iou = iou_threshold
    for idx, candidate in enumerate(candidates):
        if idx in used:
            continue
        iou = bbox_iou(bbox, np.array(candidate["bbox"], dtype=np.float64))
        if iou > best_iou:
            best_iou = iou
            best_idx = idx
    return best_idx


def _build_camera_runs(frames: list) -> list[list]:
    if not frames:
        return []
    runs: list[list] = [[frames[0]]]
    for frame in frames[1:]:
        prev = runs[-1][-1]
        same_camera = str(getattr(frame, "camera_id")) == str(getattr(prev, "camera_id"))
        if same_camera:
            runs[-1].append(frame)
        else:
            runs.append([frame])
    return runs


def _stabilize_run(
    run_frames: list,
    iou_threshold: float,
    readability_multiplier: float,
) -> list[dict]:
    if not run_frames:
        return []

    frame_states: list[dict] = []
    next_region_id = 0
    region_texts: dict[int, list[str]] = {}

    for frame_idx, frame in enumerate(run_frames):
        detections = _parse_detections(getattr(frame, "detections_json", "[]"))
        for det in detections:
            det.setdefault("source", "ocr")

        if frame_idx == 0:
            stabilized: list[dict] = []
            for det in detections:
                entry = dict(det)
                entry["region_id"] = next_region_id
                region_texts.setdefault(next_region_id, []).append(str(entry.get("text", "")))
                next_region_id += 1
                stabilized.append(entry)
            frame_states.append(
                {
                    "frame": frame,
                    "detections": _label_readability(stabilized, readability_multiplier),
                }
            )
            continue

        prev_detections = frame_states[-1]["detections"]
        used_curr: set[int] = set()
        stabilized = []
        matched_prev: set[int] = set()

        for prev_det in prev_detections:
            prev_bbox = np.array(prev_det["bbox"], dtype=np.float64)
            match_idx = _best_iou_match(prev_bbox, detections, used_curr, iou_threshold)
            if match_idx is not None:
                used_curr.add(match_idx)
                matched_prev.add(id(prev_det))
                curr_det = dict(detections[match_idx])
                curr_det["region_id"] = prev_det["region_id"]
                curr_det.setdefault("source", "ocr")
                region_texts.setdefault(prev_det["region_id"], []).append(
                    str(curr_det.get("text", ""))
                )
                stabilized.append(curr_det)
            else:
                carried = dict(prev_det)
                carried["source"] = "carried"
                region_texts.setdefault(prev_det["region_id"], []).append(
                    str(carried.get("text", ""))
                )
                stabilized.append(carried)

        for idx, det in enumerate(detections):
            if idx in used_curr:
                continue
            entry = dict(det)
            entry["region_id"] = next_region_id
            entry.setdefault("source", "ocr")
            region_texts.setdefault(next_region_id, []).append(str(entry.get("text", "")))
            next_region_id += 1
            stabilized.append(entry)

        frame_states.append(
            {
                "frame": frame,
                "detections": _label_readability(stabilized, readability_multiplier),
            }
        )

    region_mode = {
        region_id: Counter(texts).most_common(1)[0][0]
        for region_id, texts in region_texts.items()
        if texts
    }

    enriched_rows: list[dict] = []
    for state in frame_states:
        frame = state["frame"]
        detections = []
        words: list[str] = []
        seen_words: set[str] = set()
        for det in state["detections"]:
            entry = dict(det)
            region_id = entry.get("region_id")
            if region_id in region_mode:
                entry["text"] = region_mode[region_id]
            detections.append(entry)
            word = str(entry.get("text", "")).strip()
            if word:
                key = word.casefold()
                if key not in seen_words:
                    seen_words.add(key)
                    words.append(word)

        enriched_rows.append(
            {
                "scene_id": int(getattr(frame, "scene_id")),
                "frame_number": int(getattr(frame, "frame_number")),
                "seconds": float(getattr(frame, "seconds")),
                "camera_id": str(getattr(frame, "camera_id")),
                "words_json": json.dumps(words),
                "detections_json": _serialize_detections(detections),
                "verdict": str(getattr(frame, "verdict")),
                "used_unk": bool(getattr(frame, "used_unk")),
            }
        )

    return enriched_rows


def enrich_ocr_observations(
    config: PipelineConfig,
    frame_ocr: pd.DataFrame,
    frame_assignments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    del frame_assignments  # reserved for future cross-checks against assignments
    if not config.enrich_enabled:
        return frame_ocr.copy()

    if "detections_json" not in frame_ocr.columns:
        log_info("  enrich: detections_json missing — passing frame_ocr through unchanged")
        return frame_ocr.copy()

    enriched_rows: list[dict] = []
    for _, group in frame_ocr.groupby("scene_id"):
        ordered = group.sort_values("frame_number")
        frames = list(ordered.itertuples(index=False))
        for run in _build_camera_runs(frames):
            enriched_rows.extend(
                _stabilize_run(
                    run,
                    config.enrich_region_iou,
                    config.readability_size_multiplier,
                )
            )

    if not enriched_rows:
        return frame_ocr.copy()

    return pd.DataFrame(enriched_rows, columns=ENRICHED_COLUMNS)


def enrich_is_complete(output_path: Path, frame_ocr: pd.DataFrame) -> bool:
    if not output_path.is_file() or output_path.stat().st_size == 0:
        return False
    if frame_ocr.empty:
        return True
    enriched = pd.read_csv(output_path, usecols=["scene_id", "frame_number"])
    expected = {
        (int(row.scene_id), int(row.frame_number))
        for row in frame_ocr.itertuples(index=False)
    }
    done = {
        (int(row.scene_id), int(row.frame_number))
        for row in enriched.itertuples(index=False)
    }
    return expected <= done
