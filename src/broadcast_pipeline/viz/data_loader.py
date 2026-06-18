from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from broadcast_pipeline.config import PipelineConfig
from broadcast_pipeline.text_reference import _normalize_approved
from broadcast_pipeline.viz.frame_paths import resolve_frame_under_output
from broadcast_pipeline.viz.frame_ranges import parse_frame_ranges


class TimelineLoadError(Exception):
    """Raised when required pipeline artifacts are missing or invalid."""


@dataclass
class FrameInfo:
    frame_number: int
    scene_id: int
    seconds: float
    frame_path: str
    camera_id: str | None = None


@dataclass
class TimelineBundle:
    output_dir: Path
    rows: list[dict]
    suggestions: list[str]
    frame_lookup: dict[int, FrameInfo] = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    associated_provenance: dict[tuple[str, str, str, str], dict[int, dict]] = field(
        default_factory=dict
    )

    def _aggregate_associated_key(self, row: dict) -> tuple[str, str, str, str]:
        camera_id = str(row.get("camera_id", ""))
        text_kind = str(row.get("text_kind", ""))
        mapped = str(row.get("mapped_complete_text", ""))
        text = str(row.get("text", ""))
        return (camera_id, text_kind, text, mapped)

    def row_detail(self, row: dict) -> dict:
        frames = parse_frame_ranges(str(row.get("frame_ranges", "")))
        prov_key = self._aggregate_associated_key(row)
        frame_provenance = self.associated_provenance.get(prov_key, {})
        frame_details = []
        for frame_number in frames:
            detail: dict = {"frame_number": frame_number}
            prov = frame_provenance.get(frame_number)
            if prov:
                detail.update(prov)
            info = self.frame_lookup.get(frame_number)
            if info is None:
                frame_details.append(detail)
            else:
                frame_details.append(
                    {
                        **detail,
                        "scene_id": info.scene_id,
                        "seconds": info.seconds,
                        "camera_id": info.camera_id,
                        "has_image": bool(info.frame_path),
                    }
                )
        return {**row, "frames": frames, "frame_details": frame_details}

    def search_rows(self, query: str) -> list[dict]:
        if not query.strip():
            return list(self.rows)
        needle = query.strip().casefold()
        return [
            row
            for row in self.rows
            if needle in str(row.get("mapped_complete_text", "")).casefold()
        ]

    def search_suggestions(self, query: str, limit: int = 20) -> list[str]:
        if not query.strip():
            return self.suggestions[:limit]
        needle = query.strip().casefold()

        def rank(text: str) -> tuple[int, str]:
            folded = text.casefold()
            if folded.startswith(needle):
                return (0, text)
            if needle in folded:
                return (1, text)
            return (2, text)

        matches = [s for s in self.suggestions if needle in s.casefold()]
        matches.sort(key=rank)
        return matches[:limit]

    def find_row(self, camera_id: str, mapped: str, text: str | None = None) -> dict | None:
        for row in self.rows:
            if row.get("camera_id") != camera_id:
                continue
            if str(row.get("mapped_complete_text", "")) != mapped:
                continue
            if text is not None and str(row.get("text", "")) != text:
                continue
            return row
        return None


def _load_associated_provenance(
    associated_df: pd.DataFrame,
) -> dict[tuple[str, str, str, str], dict[int, dict]]:
    if associated_df.empty or "enrich_applied" not in associated_df.columns:
        return {}

    provenance: dict[tuple[str, str, str, str], dict[int, dict]] = {}
    for row in associated_df.itertuples(index=False):
        key = (
            str(getattr(row, "camera_id", "")),
            str(getattr(row, "text_kind", "")),
            str(getattr(row, "raw_text", "")),
            str(getattr(row, "mapped_complete_text", "")),
        )
        frame_number = int(getattr(row, "frame_number"))
        enrich_applied = bool(getattr(row, "enrich_applied", False))
        ocr_raw = getattr(row, "ocr_raw_text", None)
        if pd.isna(ocr_raw):
            ocr_raw = None
        else:
            ocr_raw = str(ocr_raw) if ocr_raw else None
        provenance.setdefault(key, {})[frame_number] = {
            "enrich_applied": enrich_applied,
            "ocr_raw_text": ocr_raw,
            "associated_text": str(getattr(row, "raw_text", "")),
        }
    return provenance


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _build_suggestions(
    complete_df: pd.DataFrame,
    partial_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []

    def add(text: str) -> None:
        token = str(text).strip()
        if not token:
            return
        key = token.casefold()
        if key in seen:
            return
        seen.add(key)
        ordered.append(token)

    for df in (complete_df, partial_df):
        if df.empty or "mapped_complete_text" not in df.columns:
            continue
        for value in df["mapped_complete_text"].dropna().astype(str):
            add(value)

    if not reference_df.empty and "complete_text" in reference_df.columns:
        approved = reference_df.get("approved", pd.Series(dtype=bool))
        for idx, value in reference_df["complete_text"].dropna().astype(str).items():
            if idx in approved.index and _normalize_approved(approved.loc[idx]):
                add(value)

    return sorted(ordered, key=str.casefold)


def _dataframe_to_rows(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    rows = df.to_dict(orient="records")
    for row in rows:
        for key, value in row.items():
            if pd.isna(value):
                row[key] = None
    return rows


def _load_frame_lookup(
    output_dir: Path,
    frame_index: pd.DataFrame,
    frame_assignments: pd.DataFrame,
) -> dict[int, FrameInfo]:
    if frame_index.empty:
        return {}

    ocr_frames = frame_index[frame_index["sample_role"] == "ocr"].copy()
    if ocr_frames.empty:
        return {}

    assign_lookup: dict[int, dict] = {}
    if not frame_assignments.empty:
        for row in frame_assignments.itertuples(index=False):
            assign_lookup[int(row.frame_number)] = {
                "camera_id": str(getattr(row, "camera_id", "")) or None,
                "seconds": float(getattr(row, "seconds", 0.0)),
            }

    lookup: dict[int, FrameInfo] = {}
    for row in ocr_frames.itertuples(index=False):
        frame_number = int(row.frame_number)
        path = resolve_frame_under_output(Path(str(row.frame_path)), output_dir)
        assign = assign_lookup.get(frame_number, {})
        lookup[frame_number] = FrameInfo(
            frame_number=frame_number,
            scene_id=int(row.scene_id),
            seconds=float(assign.get("seconds", getattr(row, "seconds", 0.0))),
            frame_path=str(path),
            camera_id=assign.get("camera_id"),
        )
    return lookup


def _load_summary(config: PipelineConfig) -> dict:
    path = config.artifact("pipeline_summary")
    if not path.is_file():
        return {"output_dir": str(config.output_dir)}
    return json.loads(path.read_text(encoding="utf-8"))


def load_timeline_bundle(output_dir: Path) -> TimelineBundle:
    config = PipelineConfig(output_dir=Path(output_dir))
    complete_path = config.artifact("aggregated_complete")
    partial_path = config.artifact("aggregated_partial")

    if not complete_path.is_file() and not partial_path.is_file():
        raise TimelineLoadError(
            f"Missing aggregate CSVs in {config.output_dir}. "
            "Run the pipeline through the aggregate stage first."
        )

    complete_df = _read_csv(complete_path)
    partial_df = _read_csv(partial_path)
    reference_df = _read_csv(config.reference_csv)
    frame_index = _read_csv(config.artifact("frame_index"))
    frame_assignments = _read_csv(config.artifact("frame_assignments"))
    associated_df = _read_csv(config.artifact("frame_text_associated"))

    rows = _dataframe_to_rows(complete_df) + _dataframe_to_rows(partial_df)
    suggestions = _build_suggestions(complete_df, partial_df, reference_df)
    frame_lookup = _load_frame_lookup(config.output_dir, frame_index, frame_assignments)
    summary = _load_summary(config)
    associated_provenance = _load_associated_provenance(associated_df)

    return TimelineBundle(
        output_dir=config.output_dir,
        rows=rows,
        suggestions=suggestions,
        frame_lookup=frame_lookup,
        summary=summary,
        associated_provenance=associated_provenance,
    )
