from __future__ import annotations

import csv
import gc
import json
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from broadcast_pipeline.config import PipelineConfig
from broadcast_pipeline.progress import ProgressTracker, log_info
from src.accelerator.device import (
    log_accelerator_summary,
    resolve_ocr_backend,
    resolve_ocr_use_cuda,
    resolve_torch_device,
)
from src.accelerator.install import check_gpu_stack, warn_gpu_stack_gaps
from src.scene_ocr.config import OcrConfig, OpenAIVlmConfig, ReadabilityConfig
from src.scene_ocr.extractor import lines_to_words, load_image
from src.scene_ocr.geometry import polygon_to_xyxy
from src.scene_ocr.readability import assess_readability_from_bgr
from src.scene_ocr.types import OcrDetection
from src.scene_ocr.vlm_client import NullVlmClient, OpenAIVlmClient

OCR_COLUMNS = [
    "scene_id",
    "frame_number",
    "seconds",
    "camera_id",
    "words_json",
    "detections_json",
    "verdict",
    "used_unk",
]


def _pipeline_ocr_config(config: PipelineConfig) -> OcrConfig:
    return OcrConfig(
        scale=config.ocr_scale,
        preprocess=config.ocr_preprocess,
        use_cuda=resolve_ocr_use_cuda(config.accelerator),
        rec_batch_num=config.ocr_rec_batch,
        cls_batch_num=config.ocr_cls_batch,
    )


def _apply_unk(result, config: PipelineConfig) -> tuple[list[str], bool]:
    words = list(result.words)
    if not result.needs_vlm:
        return words, False
    if config.enable_vlm:
        return words, False
    n_unk = len(result.vlm_crops) if result.vlm_crops else 1
    for _ in range(n_unk):
        words.append(config.unk_token)
    return words, True


def _word_level_detections(detections: list[OcrDetection]) -> list[dict]:
    """Expand line-level detections to word-level records (full line bbox per word)."""
    entries: list[dict] = []
    for detection in detections:
        words = lines_to_words(detection.text)
        if not words:
            continue
        x0, y0, x1, y1 = (int(v) for v in polygon_to_xyxy(detection.bbox))
        for word in words:
            entries.append(
                {
                    "text": word,
                    "confidence": float(detection.confidence),
                    "bbox": [x0, y0, x1, y1],
                    "source": "ocr",
                }
            )
    return entries


def _detections_to_json(detections: list[OcrDetection]) -> str:
    return json.dumps(_word_level_detections(detections))


def ocr_frame_keys(frame_index: pd.DataFrame) -> set[tuple[int, int]]:
    ocr_frames = frame_index[frame_index["sample_role"] == "ocr"]
    return {
        (int(row.scene_id), int(row.frame_number))
        for row in ocr_frames.itertuples(index=False)
    }


def ocr_done_keys(output_path: Path) -> set[tuple[int, int]]:
    if not output_path.is_file() or output_path.stat().st_size == 0:
        return set()
    df = pd.read_csv(output_path, usecols=["scene_id", "frame_number"])
    return {
        (int(row.scene_id), int(row.frame_number))
        for row in df.itertuples(index=False)
    }


def ocr_is_complete(output_path: Path, frame_index: pd.DataFrame) -> bool:
    expected = ocr_frame_keys(frame_index)
    if not expected:
        return output_path.is_file()
    return expected <= ocr_done_keys(output_path)


def _resolve_camera_id(
    row: Any,
    assign_lookup: pd.DataFrame,
    frame_assignments: pd.DataFrame,
) -> str:
    scene_id = int(getattr(row, "scene_id"))
    frame_number = int(getattr(row, "frame_number"))
    lookup_key = (scene_id, frame_number, "ocr")
    if lookup_key in assign_lookup.index:
        return str(assign_lookup.loc[lookup_key]["camera_id"])
    fallback = frame_assignments[frame_assignments["scene_id"] == scene_id]
    return str(fallback.iloc[0]["camera_id"]) if not fallback.empty else "unknown"


def _process_ocr_row(
    row: Any,
    *,
    ocr_config: OcrConfig,
    readability_config: ReadabilityConfig,
    vlm_client,
    config: PipelineConfig,
    assign_lookup: pd.DataFrame,
    frame_assignments: pd.DataFrame,
    bgr,
) -> dict:
    scene_id = int(getattr(row, "scene_id"))
    frame_number = int(getattr(row, "frame_number"))
    camera_id = _resolve_camera_id(row, assign_lookup, frame_assignments)

    result = assess_readability_from_bgr(
        bgr,
        ocr_config,
        readability_config,
        vlm_client=vlm_client,
    )
    words, used_unk = _apply_unk(result, config)
    row_dict = {
        "scene_id": scene_id,
        "frame_number": frame_number,
        "seconds": float(getattr(row, "seconds")),
        "camera_id": camera_id,
        "words_json": json.dumps(words),
        "detections_json": _detections_to_json(result.detections),
        "verdict": result.verdict.value,
        "used_unk": used_unk,
    }
    del result, words, bgr
    return row_dict


def run_segment_ocr(
    config: PipelineConfig,
    frame_index: pd.DataFrame,
    frame_assignments: pd.DataFrame,
    *,
    output_path: Path | None = None,
) -> pd.DataFrame:
    ocr_frames = frame_index[frame_index["sample_role"] == "ocr"].copy()
    assign_lookup = frame_assignments.set_index(["scene_id", "frame_number", "sample_role"])

    ocr_config = _pipeline_ocr_config(config)
    readability_config = ReadabilityConfig()
    ocr_backend = resolve_ocr_backend(config.accelerator)
    log_accelerator_summary(resolve_torch_device(config.accelerator), ocr_backend)
    if config.accelerator in ("auto", "cuda", "mps"):
        warn_gpu_stack_gaps(check_gpu_stack(config.accelerator), config.accelerator)
    if ocr_backend == "cpu" and config.accelerator in ("auto", "cuda"):
        log_info("  OCR is running on CPU (install onnxruntime-gpu for CUDA acceleration)")

    gc_every_n = config.ocr_gc_interval if ocr_config.use_cuda else 1
    flush_every_n = max(1, config.ocr_csv_flush_interval)
    prefetch_workers = max(0, config.ocr_prefetch_workers)
    if config.enable_vlm:
        vlm_client = OpenAIVlmClient(OpenAIVlmConfig())
    else:
        vlm_client = NullVlmClient()

    processed = ocr_done_keys(output_path) if output_path else set()
    if processed:
        log_info(f"  OCR resume: {len(processed)} frame(s) already stored")

    writer: csv.DictWriter | None = None
    csv_file = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not output_path.is_file() or output_path.stat().st_size == 0
        csv_file = output_path.open(
            "a" if output_path.is_file() and not write_header else "w",
            newline="",
            encoding="utf-8",
        )
        writer = csv.DictWriter(csv_file, fieldnames=OCR_COLUMNS)
        if write_header:
            writer.writeheader()
            csv_file.flush()

    progress = ProgressTracker(len(ocr_frames), "OCR")
    rows_written_since_flush = 0
    processed_count = 0
    load_future: Future | None = None
    next_load_path: Path | None = None

    executor: ThreadPoolExecutor | None = None
    if prefetch_workers > 0:
        executor = ThreadPoolExecutor(max_workers=prefetch_workers)

    try:
        for idx, row in enumerate(ocr_frames.itertuples(index=False)):
            scene_id = int(getattr(row, "scene_id"))
            frame_number = int(getattr(row, "frame_number"))
            frame_key = (scene_id, frame_number)
            if frame_key in processed:
                progress.advance()
                continue

            frame_path = Path(getattr(row, "frame_path"))
            if not frame_path.is_file():
                progress.advance()
                continue

            if executor is not None:
                if load_future is not None and next_load_path == frame_path:
                    bgr = load_future.result()
                    load_future = None
                    next_load_path = None
                else:
                    bgr = load_image(frame_path)

                for ahead_idx in range(idx + 1, len(ocr_frames)):
                    ahead = ocr_frames.iloc[ahead_idx]
                    ahead_key = (int(ahead["scene_id"]), int(ahead["frame_number"]))
                    if ahead_key in processed:
                        continue
                    ahead_path = Path(ahead["frame_path"])
                    if not ahead_path.is_file():
                        continue
                    load_future = executor.submit(load_image, ahead_path)
                    next_load_path = ahead_path
                    break
            else:
                bgr = load_image(frame_path)

            row_dict = _process_ocr_row(
                row,
                ocr_config=ocr_config,
                readability_config=readability_config,
                vlm_client=vlm_client,
                config=config,
                assign_lookup=assign_lookup,
                frame_assignments=frame_assignments,
                bgr=bgr,
            )

            if writer is not None:
                writer.writerow(row_dict)
                rows_written_since_flush += 1
                if rows_written_since_flush >= flush_every_n:
                    csv_file.flush()
                    rows_written_since_flush = 0
            processed.add(frame_key)
            progress.advance()
            processed_count += 1
            if processed_count % gc_every_n == 0:
                gc.collect()
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    if csv_file is not None:
        csv_file.flush()
        csv_file.close()

    if output_path is not None:
        return pd.read_csv(output_path)

    raise RuntimeError("run_segment_ocr requires output_path for incremental storage")
