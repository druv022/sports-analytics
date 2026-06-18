from __future__ import annotations

import json

import pandas as pd

from broadcast_pipeline.config import PipelineConfig
from broadcast_pipeline.text_reference import _load_reference, _normalize_approved
from src.scene_ocr.extractor import is_plausible_word


def _lcs_length(a: str, b: str) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def _longest_common_substring(partial: str, reference: str) -> tuple[int, int]:
    """Return length and 0-based start index in reference for the longest contiguous match."""
    if not partial or not reference:
        return 0, -1

    best_len = 0
    best_ref_start = -1
    prev = [0] * (len(reference) + 1)
    for i in range(1, len(partial) + 1):
        curr = [0] * (len(reference) + 1)
        for j in range(1, len(reference) + 1):
            if partial[i - 1] == reference[j - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] > best_len:
                    best_len = curr[j]
                    best_ref_start = j - curr[j]
            else:
                curr[j] = 0
        prev = curr
    return best_len, best_ref_start


def _approved_complete_texts(reference: pd.DataFrame) -> list[str]:
    texts: list[str] = []
    for row in reference.itertuples(index=False):
        if not _normalize_approved(getattr(row, "approved", False)):
            continue
        text = str(getattr(row, "complete_text", "")).strip()
        if text:
            texts.append(text)
    return texts


def _reference_has_text(reference: pd.DataFrame, token: str) -> bool:
    token_fold = token.casefold()
    for row in reference.itertuples(index=False):
        text = str(getattr(row, "complete_text", "")).strip()
        if text and text.casefold() == token_fold:
            return True
    return False


def _best_complete_match(
    partial: str,
    candidates: list[str],
    *,
    min_match_chars: int,
    min_reference_coverage: float,
    min_prefix_coverage: float,
    min_token_coverage: float,
) -> tuple[str | None, float]:
    best_text: str | None = None
    best_score = 0.0
    partial_fold = partial.casefold()
    partial_len = len(partial_fold)

    for candidate in candidates:
        cand_fold = candidate.casefold()
        block_len, ref_start = _longest_common_substring(partial_fold, cand_fold)
        if block_len < min_match_chars:
            continue

        if block_len / partial_len < min_token_coverage:
            continue

        ref_coverage = block_len / len(cand_fold)
        threshold = (
            min_prefix_coverage if ref_start == 0 else min_reference_coverage
        )
        if ref_coverage < threshold:
            continue

        if ref_coverage > best_score:
            best_score = ref_coverage
            best_text = candidate

    if best_text is None:
        return None, best_score
    return best_text, best_score


def _is_complete_token(token: str, approved: list[str], min_len: int = 3) -> bool:
    if len(token) < min_len:
        return False
    if not is_plausible_word(token):
        return False
    token_fold = token.casefold()
    return any(token_fold == cand.casefold() for cand in approved)


def _detection_lookup(detections: list[dict]) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for det in detections:
        text = str(det.get("text", "")).strip()
        if not text:
            continue
        key = text.casefold()
        if key not in lookup:
            lookup[key] = det
    return lookup


def _detection_metadata(token: str, detections: list[dict]) -> tuple[str, str]:
    lookup = _detection_lookup(detections)
    det = lookup.get(token.casefold())
    if det is None:
        return "partial", "[]"
    readability = str(det.get("readability_label", "partial"))
    bbox = det.get("bbox", [])
    return readability, json.dumps(bbox)


def _build_raw_words_lookup(
    raw_frame_ocr: pd.DataFrame | None,
) -> dict[tuple[int, int], list[str]]:
    if raw_frame_ocr is None or raw_frame_ocr.empty:
        return {}
    lookup: dict[tuple[int, int], list[str]] = {}
    for row in raw_frame_ocr.itertuples(index=False):
        key = (int(getattr(row, "scene_id")), int(getattr(row, "frame_number")))
        words = json.loads(getattr(row, "words_json"))
        lookup[key] = [str(word).strip() for word in words if str(word).strip()]
    return lookup


def _best_raw_word_match(token: str, raw_words: list[str]) -> str | None:
    if not raw_words:
        return None
    token_fold = token.casefold()
    best_word: str | None = None
    best_lcs = 0
    for word in raw_words:
        lcs = _lcs_length(token_fold, word.casefold())
        if lcs > best_lcs:
            best_lcs = lcs
            best_word = word
    return best_word if best_lcs > 0 else None


def _enrich_provenance(
    token: str,
    scene_id: int,
    frame_number: int,
    raw_lookup: dict[tuple[int, int], list[str]],
) -> tuple[bool, str | None]:
    raw_words = raw_lookup.get((scene_id, frame_number), [])
    if not raw_lookup:
        return False, None
    raw_folded = {word.casefold() for word in raw_words}
    if token.casefold() in raw_folded:
        return False, token
    return True, _best_raw_word_match(token, raw_words)


def _with_provenance(
    row: dict,
    token: str,
    scene_id: int,
    frame_number: int,
    raw_lookup: dict[tuple[int, int], list[str]],
) -> dict:
    enrich_applied, ocr_raw_text = _enrich_provenance(
        token, scene_id, frame_number, raw_lookup
    )
    row["enrich_applied"] = enrich_applied
    row["ocr_raw_text"] = ocr_raw_text if enrich_applied else None
    return row


def associate_text(
    config: PipelineConfig,
    frame_ocr: pd.DataFrame,
    reference: pd.DataFrame | None = None,
    raw_frame_ocr: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if reference is None:
        reference = _load_reference(config.reference_csv)

    approved = _approved_complete_texts(reference)
    approved_fold = {text.casefold(): text for text in approved}
    raw_lookup = _build_raw_words_lookup(raw_frame_ocr)

    associated_rows: list[dict] = []
    dropped_rows: list[dict] = []

    for row in frame_ocr.itertuples(index=False):
        scene_id = int(getattr(row, "scene_id"))
        frame_number = int(getattr(row, "frame_number"))
        camera_id = str(getattr(row, "camera_id"))
        words = json.loads(getattr(row, "words_json"))
        detections = []
        if hasattr(row, "detections_json"):
            raw_detections = getattr(row, "detections_json")
            if isinstance(raw_detections, str) and raw_detections.strip():
                detections = json.loads(raw_detections)

        for raw in words:
            token = str(raw).strip()
            readability_label, bbox_json = _detection_metadata(token, detections)
            if not token or token == config.unk_token:
                dropped_rows.append(
                    {
                        "scene_id": scene_id,
                        "frame_number": frame_number,
                        "camera_id": camera_id,
                        "raw_text": token,
                        "reason": "unk_or_empty",
                    }
                )
                continue

            if _is_complete_token(token, approved):
                canonical = approved_fold[token.casefold()]
                associated_rows.append(
                    _with_provenance(
                        {
                            "scene_id": scene_id,
                            "frame_number": frame_number,
                            "camera_id": camera_id,
                            "raw_text": token,
                            "text_kind": "complete",
                            "mapped_complete_text": canonical,
                            "mapping_confidence": 1.0,
                            "readability_label": readability_label,
                            "bbox_json": bbox_json,
                        },
                        token,
                        scene_id,
                        frame_number,
                        raw_lookup,
                    )
                )
                continue

            mapped, confidence = _best_complete_match(
                token,
                approved,
                min_match_chars=config.association_min_match_chars,
                min_reference_coverage=config.association_min_reference_coverage,
                min_prefix_coverage=config.association_min_prefix_coverage,
                min_token_coverage=config.association_min_token_coverage,
            )
            if mapped is not None:
                associated_rows.append(
                    _with_provenance(
                        {
                            "scene_id": scene_id,
                            "frame_number": frame_number,
                            "camera_id": camera_id,
                            "raw_text": token,
                            "text_kind": "partial",
                            "mapped_complete_text": mapped,
                            "mapping_confidence": confidence,
                            "readability_label": readability_label,
                            "bbox_json": bbox_json,
                        },
                        token,
                        scene_id,
                        frame_number,
                        raw_lookup,
                    )
                )
                continue

            if _reference_has_text(reference, token):
                reason = "unapproved"
            else:
                reason = "no_match"
            dropped_rows.append(
                {
                    "scene_id": scene_id,
                    "frame_number": frame_number,
                    "camera_id": camera_id,
                    "raw_text": token,
                    "reason": reason,
                }
            )

    associated = pd.DataFrame(associated_rows)
    dropped = pd.DataFrame(
        dropped_rows,
        columns=["scene_id", "frame_number", "camera_id", "raw_text", "reason"],
    )
    return associated, dropped
