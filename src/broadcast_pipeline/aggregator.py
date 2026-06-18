from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from broadcast_pipeline.config import PipelineConfig
from broadcast_pipeline.scene_extractor import load_scenes
from broadcast_pipeline.text_reference import _normalize_approved
from broadcast_pipeline.types import PipelineSummary, VideoMeta
from broadcast_pipeline.video_meta import load_video_meta


AGGREGATE_CSV_COLUMNS = [
    "camera_id",
    "text",
    "text_kind",
    "mapped_complete_text",
    "total_duration_sec",
    "frame_ranges",
    "n_frames_present",
    "n_frames_good",
    "n_frames_partial",
    "n_frames_enriched",
    "dominant_readability",
]


def _scene_end_lookup(scenes: list) -> dict[int, int]:
    return {scene.scene_id: scene.end_frame for scene in scenes}


def _slot_durations(
    frame_ocr: pd.DataFrame,
    scenes: list,
    fps: float,
) -> dict[tuple[int, int], float]:
    ends = _scene_end_lookup(scenes)
    durations: dict[tuple[int, int], float] = {}
    if fps <= 0:
        fps = 1.0

    for scene_id, group in frame_ocr.groupby("scene_id"):
        ordered = group.sort_values("frame_number")
        frames = ordered["frame_number"].astype(int).tolist()
        end_frame = ends.get(int(scene_id), frames[-1] + 1 if frames else 0)
        for idx, frame_number in enumerate(frames):
            if idx + 1 < len(frames):
                delta = frames[idx + 1] - frame_number
            else:
                delta = max(1, end_frame - frame_number)
            durations[(int(scene_id), int(frame_number))] = delta / fps

    return durations


def _merge_frame_ranges(frames: list[int]) -> str:
    if not frames:
        return ""
    frames = sorted(set(frames))
    ranges: list[str] = []
    start = frames[0]
    prev = frames[0]
    for frame in frames[1:]:
        if frame == prev + 1:
            prev = frame
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = frame
        prev = frame
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ";".join(ranges)


def _bucket_total_duration(
    duration_frames: set[tuple[int, int]],
    durations: dict[tuple[int, int], float],
) -> float:
    return sum(durations.get(frame_key, 0.0) for frame_key in duration_frames)


def _new_aggregate_bucket() -> dict:
    return {
        "duration_frames": set(),
        "frames": [],
        "frames_present": set(),
        "frames_good": set(),
        "frames_partial": set(),
        "frames_enriched": set(),
    }


def _accumulate_association_row(
    bucket: dict,
    *,
    scene_id: int,
    frame_number: int,
    readability: str,
    enrich_applied: bool,
    claim_key: tuple[str, str],
    mapped_claimed: dict[tuple[str, str], set[tuple[int, int]]],
) -> None:
    frame_key = (scene_id, frame_number)
    bucket["frames"].append(frame_number)
    bucket["frames_present"].add(frame_number)
    if readability == "good":
        bucket["frames_good"].add(frame_number)
    else:
        bucket["frames_partial"].add(frame_number)
    if enrich_applied:
        bucket["frames_enriched"].add(frame_number)

    if frame_key not in mapped_claimed[claim_key]:
        bucket["duration_frames"].add(frame_key)
        mapped_claimed[claim_key].add(frame_key)


def aggregate_text_timeline(
    config: PipelineConfig,
    associated: pd.DataFrame,
    frame_ocr: pd.DataFrame,
    meta: VideoMeta,
    scenes: list,
) -> tuple[pd.DataFrame, pd.DataFrame, PipelineSummary]:
    durations = _slot_durations(frame_ocr, scenes, meta.fps)

    complete_groups: dict[tuple[str, str], dict] = {}
    partial_groups: dict[tuple[str, str, str], dict] = {}
    mapped_claimed: dict[tuple[str, str], set[tuple[int, int]]] = defaultdict(set)

    complete_rows = associated[associated["text_kind"] == "complete"]
    partial_rows_df = associated[associated["text_kind"] != "complete"]

    for row in complete_rows.itertuples(index=False):
        scene_id = int(getattr(row, "scene_id"))
        frame_number = int(getattr(row, "frame_number"))
        camera_id = str(getattr(row, "camera_id"))
        mapped = str(getattr(row, "mapped_complete_text"))
        readability = str(getattr(row, "readability_label", "partial"))
        key = (camera_id, mapped)
        bucket = complete_groups.setdefault(key, _new_aggregate_bucket())
        _accumulate_association_row(
            bucket,
            scene_id=scene_id,
            frame_number=frame_number,
            readability=readability,
            enrich_applied=bool(getattr(row, "enrich_applied", False)),
            claim_key=key,
            mapped_claimed=mapped_claimed,
        )

    for row in partial_rows_df.itertuples(index=False):
        scene_id = int(getattr(row, "scene_id"))
        frame_number = int(getattr(row, "frame_number"))
        camera_id = str(getattr(row, "camera_id"))
        raw_text = str(getattr(row, "raw_text"))
        mapped = str(getattr(row, "mapped_complete_text"))
        readability = str(getattr(row, "readability_label", "partial"))
        bucket_key = (camera_id, raw_text, mapped)
        claim_key = (camera_id, mapped)
        bucket = partial_groups.setdefault(bucket_key, _new_aggregate_bucket())
        _accumulate_association_row(
            bucket,
            scene_id=scene_id,
            frame_number=frame_number,
            readability=readability,
            enrich_applied=bool(getattr(row, "enrich_applied", False)),
            claim_key=claim_key,
            mapped_claimed=mapped_claimed,
        )

    def _dominant_readability(value: dict) -> str:
        n_good = len(value["frames_good"])
        n_partial = len(value["frames_partial"])
        return "good" if n_good >= n_partial else "partial"

    complete_rows = [
        {
            "camera_id": key[0],
            "text": key[1],
            "text_kind": "complete",
            "mapped_complete_text": key[1],
            "total_duration_sec": _bucket_total_duration(value["duration_frames"], durations),
            "frame_ranges": _merge_frame_ranges(value["frames"]),
            "n_frames_present": len(value["frames_present"]),
            "n_frames_good": len(value["frames_good"]),
            "n_frames_partial": len(value["frames_partial"]),
            "n_frames_enriched": len(value["frames_enriched"]),
            "dominant_readability": _dominant_readability(value),
        }
        for key, value in sorted(complete_groups.items())
    ]

    partial_rows = [
        {
            "camera_id": key[0],
            "text": key[1],
            "text_kind": "partial",
            "mapped_complete_text": key[2],
            "total_duration_sec": _bucket_total_duration(value["duration_frames"], durations),
            "frame_ranges": _merge_frame_ranges(value["frames"]),
            "n_frames_present": len(value["frames_present"]),
            "n_frames_good": len(value["frames_good"]),
            "n_frames_partial": len(value["frames_partial"]),
            "n_frames_enriched": len(value["frames_enriched"]),
            "dominant_readability": _dominant_readability(value),
        }
        for key, value in sorted(partial_groups.items())
    ]

    complete_df = pd.DataFrame(complete_rows, columns=AGGREGATE_CSV_COLUMNS)
    partial_df = pd.DataFrame(partial_rows, columns=AGGREGATE_CSV_COLUMNS)

    summary = PipelineSummary(
        video_meta=meta,
        n_scenes=len(scenes),
        n_frames=len(frame_ocr),
        n_cameras=associated["camera_id"].nunique() if not associated.empty else 0,
        output_dir=config.output_dir,
    )
    return complete_df, partial_df, summary


def load_aggregate_inputs(config: PipelineConfig) -> tuple[VideoMeta, list]:
    meta = load_video_meta(config.artifact("video_meta"))
    scenes = load_scenes(config.artifact("scenes"))
    return meta, scenes


def write_pipeline_summary(
    config: PipelineConfig,
    summary: PipelineSummary,
    reference: pd.DataFrame,
    n_new_reference: int = 0,
    *,
    complete_df: pd.DataFrame | None = None,
    partial_df: pd.DataFrame | None = None,
) -> None:
    n_approved = (
        int(reference["approved"].map(_normalize_approved).sum())
        if not reference.empty and "approved" in reference.columns
        else 0
    )
    n_frames_good = 0
    n_frames_partial = 0
    n_text_presence_events = 0
    for df in (complete_df, partial_df):
        if df is None or df.empty:
            continue
        if "n_frames_good" in df.columns:
            n_frames_good += int(df["n_frames_good"].sum())
        if "n_frames_partial" in df.columns:
            n_frames_partial += int(df["n_frames_partial"].sum())
        if "n_frames_present" in df.columns:
            n_text_presence_events += int(df["n_frames_present"].sum())

    payload = {
        "video_path": str(summary.video_meta.path) if summary.video_meta else "",
        "duration_sec": summary.video_meta.duration_sec if summary.video_meta else 0.0,
        "n_scenes": summary.n_scenes,
        "n_ocr_frames": summary.n_frames,
        "n_cameras": summary.n_cameras,
        "n_approved_text": n_approved,
        "n_new_reference_text": n_new_reference,
        "readability_size_multiplier": config.readability_size_multiplier,
        "n_frames_good": n_frames_good,
        "n_frames_partial": n_frames_partial,
        "n_text_presence_events": n_text_presence_events,
        "output_dir": str(config.output_dir),
    }
    path = config.artifact("pipeline_summary")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
