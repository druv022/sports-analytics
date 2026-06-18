"""Load stored pipeline OCR detections and join association mappings for the viz."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from broadcast_pipeline.config import PipelineConfig


class PipelineOcrLoadError(Exception):
    """Raised when stored OCR artifacts are missing or invalid."""


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _parse_bbox(raw: object) -> list[int] | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    if isinstance(raw, str):
        token = raw.strip()
        if not token or token == "[]":
            return None
        try:
            bbox = json.loads(token)
        except json.JSONDecodeError:
            return None
    else:
        bbox = raw
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    return [int(v) for v in bbox]


def _parse_detections(raw: object) -> list[dict]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _display_text(raw: str, mapped: str | None) -> str:
    if not mapped or raw.casefold() == mapped.casefold():
        return mapped or raw
    return f"{raw} → {mapped}"


def _assoc_key(record: dict) -> tuple:
    return (
        record.get("raw_text"),
        record.get("mapped_complete_text"),
        record.get("bbox_json"),
        bool(record.get("enrich_applied")),
        record.get("ocr_raw_text"),
    )


def _association_record(row: object) -> dict:
    enrich_applied = bool(getattr(row, "enrich_applied", False))
    ocr_raw = getattr(row, "ocr_raw_text", None)
    if pd.isna(ocr_raw):
        ocr_raw = None
    else:
        ocr_raw = str(ocr_raw) if ocr_raw else None
    return {
        "raw_text": str(getattr(row, "raw_text", "")),
        "mapped_complete_text": str(getattr(row, "mapped_complete_text", "")),
        "text_kind": str(getattr(row, "text_kind", "")),
        "mapping_confidence": float(getattr(row, "mapping_confidence", 0.0)),
        "readability_label": str(getattr(row, "readability_label", "partial")),
        "bbox_json": getattr(row, "bbox_json", "[]"),
        "enrich_applied": enrich_applied,
        "ocr_raw_text": ocr_raw,
    }


def _pick_association(candidates: list[dict]) -> dict | None:
    if not candidates:
        return None

    def rank(record: dict) -> tuple:
        kind_score = 0 if record["text_kind"] == "complete" else 1
        enrich_score = 0 if record["enrich_applied"] else 1
        return (kind_score, enrich_score, -record["mapping_confidence"])

    return min(candidates, key=rank)


def _detection_entry(det: dict, mapping: dict | None, *, display_from: str) -> dict:
    text = str(det.get("text", "")).strip()
    entry: dict = {
        "text": text,
        "confidence": float(det.get("confidence", 0.0)),
        "bbox": [int(v) for v in det["bbox"]],
        "source": str(det.get("source", "ocr")),
    }
    if mapping is None:
        entry["display_text"] = text
        return entry

    mapped = mapping["mapped_complete_text"]
    entry.update(
        {
            "mapped_complete_text": mapped,
            "text_kind": mapping["text_kind"],
            "mapping_confidence": mapping["mapping_confidence"],
            "readability_label": mapping["readability_label"],
            "enrich_applied": mapping["enrich_applied"],
            "ocr_raw_text": mapping["ocr_raw_text"],
            "display_text": _display_text(display_from, mapped),
        }
    )
    return entry


def _enriched_entry(record: dict) -> dict:
    bbox = _parse_bbox(record["bbox_json"])
    if bbox is None:
        raise ValueError("enriched entry requires bbox")
    raw_text = record["raw_text"]
    ocr_raw = record["ocr_raw_text"] or raw_text
    mapped = record["mapped_complete_text"]
    return {
        "text": raw_text,
        "confidence": float(record["mapping_confidence"]),
        "bbox": bbox,
        "source": "enriched",
        "mapped_complete_text": mapped,
        "text_kind": record["text_kind"],
        "mapping_confidence": record["mapping_confidence"],
        "readability_label": record["readability_label"],
        "enrich_applied": True,
        "ocr_raw_text": record["ocr_raw_text"],
        "display_text": _display_text(ocr_raw, mapped),
    }


def build_frame_ocr_payload(
    frame_number: int,
    frame_ocr: pd.DataFrame,
    associated: pd.DataFrame,
) -> dict:
    ocr_rows = frame_ocr[frame_ocr["frame_number"] == frame_number]
    if ocr_rows.empty:
        raise PipelineOcrLoadError(f"No stored OCR for frame {frame_number}")

    ocr_row = ocr_rows.iloc[0]
    detections = _parse_detections(getattr(ocr_row, "detections_json", "[]"))
    assoc_rows = associated[associated["frame_number"] == frame_number]

    by_raw: dict[str, list[dict]] = defaultdict(list)
    by_ocr_raw: dict[str, list[dict]] = defaultdict(list)
    enriched_with_bbox: list[dict] = []

    for row in assoc_rows.itertuples(index=False):
        record = _association_record(row)
        by_raw[record["raw_text"].casefold()].append(record)
        if record["enrich_applied"] and record["ocr_raw_text"]:
            by_ocr_raw[record["ocr_raw_text"].casefold()].append(record)
        if record["enrich_applied"] and _parse_bbox(record["bbox_json"]) is not None:
            enriched_with_bbox.append(record)

    results: list[dict] = []
    matched_assoc: set[tuple] = set()

    for det in detections:
        text = str(det.get("text", "")).strip()
        if not text or "bbox" not in det:
            continue
        key = text.casefold()
        mapping = _pick_association(by_raw.get(key, []))
        display_from = text
        if mapping is None:
            mapping = _pick_association(by_ocr_raw.get(key, []))
            if mapping is not None:
                display_from = mapping["ocr_raw_text"] or text
        results.append(_detection_entry(det, mapping, display_from=display_from))
        if mapping is not None:
            matched_assoc.add(_assoc_key(mapping))

    for record in enriched_with_bbox:
        if _assoc_key(record) in matched_assoc:
            continue
        raw_key = record["raw_text"].casefold()
        if any(item["text"].casefold() == raw_key for item in results):
            continue
        try:
            results.append(_enriched_entry(record))
        except ValueError:
            continue

    return {
        "frame_number": frame_number,
        "scene_id": int(ocr_row["scene_id"]),
        "camera_id": str(ocr_row["camera_id"]),
        "source": "frame_ocr.csv",
        "detections": results,
    }


@dataclass
class PipelineOcrIndex:
    output_dir: Path
    frame_ocr: pd.DataFrame
    associated: pd.DataFrame

    def frame_payload(self, frame_number: int) -> dict:
        return build_frame_ocr_payload(frame_number, self.frame_ocr, self.associated)


def load_pipeline_ocr_index(output_dir: Path) -> PipelineOcrIndex:
    config = PipelineConfig(output_dir=Path(output_dir))
    frame_ocr_path = config.artifact("frame_ocr")
    if not frame_ocr_path.is_file():
        raise PipelineOcrLoadError(
            f"Missing {frame_ocr_path.name} in {config.output_dir}. "
            "Run the pipeline through the ocr stage first."
        )

    frame_ocr = _read_csv(frame_ocr_path)
    if frame_ocr.empty:
        raise PipelineOcrLoadError(f"{frame_ocr_path.name} is empty.")

    associated = _read_csv(config.artifact("frame_text_associated"))
    return PipelineOcrIndex(
        output_dir=config.output_dir,
        frame_ocr=frame_ocr,
        associated=associated,
    )
