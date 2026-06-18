from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from broadcast_pipeline.config import PipelineConfig
from src.scene_ocr.extractor import is_plausible_word

BASE_COLUMNS = [
    "complete_text",
    "approved",
    "first_seen_scene_id",
    "first_seen_frame",
    "discovery_count",
]


def _normalize_approved(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def _load_reference(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=BASE_COLUMNS)
    df = pd.read_csv(path)
    for col in BASE_COLUMNS:
        if col not in df.columns:
            if col == "approved":
                df[col] = False
            elif col == "discovery_count":
                df[col] = 0
            else:
                df[col] = pd.NA
    if "approved" in df.columns:
        df["approved"] = df["approved"].map(_normalize_approved)
    return df


def _discover_complete_tokens(frame_ocr: pd.DataFrame, unk_token: str) -> pd.DataFrame:
    discoveries: dict[str, dict] = {}

    for row in frame_ocr.itertuples(index=False):
        raw_words = json.loads(getattr(row, "words_json"))
        for word in raw_words:
            token = str(word).strip()
            if not token or token == unk_token:
                continue
            if not is_plausible_word(token):
                continue
            key = token.casefold()
            if key not in discoveries:
                discoveries[key] = {
                    "complete_text": token,
                    "first_seen_scene_id": int(getattr(row, "scene_id")),
                    "first_seen_frame": int(getattr(row, "frame_number")),
                    "count": 1,
                }
            else:
                discoveries[key]["count"] += 1

    if not discoveries:
        return pd.DataFrame(columns=["complete_text", "first_seen_scene_id", "first_seen_frame", "count"])
    return pd.DataFrame(list(discoveries.values()))


def update_text_reference(
    config: PipelineConfig,
    frame_ocr: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    path = config.reference_csv
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_reference(path)
    discovered = _discover_complete_tokens(frame_ocr, config.unk_token)

    if existing.empty:
        merged = pd.DataFrame(columns=BASE_COLUMNS)
    else:
        merged = existing.copy()

    known = {
        str(row.complete_text).casefold()
        for row in merged.itertuples(index=False)
        if hasattr(row, "complete_text") and pd.notna(row.complete_text)
    }

    n_new = 0
    for item in discovered.itertuples(index=False):
        key = str(item.complete_text).casefold()
        if key in known:
            mask = merged["complete_text"].astype(str).str.casefold() == key
            merged.loc[mask, "discovery_count"] = (
                pd.to_numeric(merged.loc[mask, "discovery_count"], errors="coerce").fillna(0)
                + int(item.count)
            )
            continue

        n_new += 1
        new_row = {col: pd.NA for col in merged.columns} if not merged.empty else {}
        new_row.update(
            {
                "complete_text": item.complete_text,
                "approved": config.default_new_text_approved,
                "first_seen_scene_id": int(item.first_seen_scene_id),
                "first_seen_frame": int(item.first_seen_frame),
                "discovery_count": int(item.count),
            }
        )
        merged = pd.concat([merged, pd.DataFrame([new_row])], ignore_index=True)
        known.add(key)

    for col in BASE_COLUMNS:
        if col not in merged.columns:
            merged[col] = pd.NA

    merged.to_csv(path, index=False)
    return merged, n_new
