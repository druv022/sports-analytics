"""OCR stack verification, benchmarking, and auto-tuning for the broadcast pipeline."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from broadcast_pipeline.config import PipelineConfig
from broadcast_pipeline.ocr_runner import _pipeline_ocr_config
from src.accelerator.device import resolve_ocr_backend, resolve_torch_device
from src.accelerator.install import check_gpu_stack, gpu_install_hints
from src.scene_ocr.config import OcrConfig, ReadabilityConfig
from src.scene_ocr.extractor import clear_ocr_engine_cache, load_image
from src.scene_ocr.readability import assess_readability, assess_readability_from_bgr


@dataclass(frozen=True)
class OcrAutotuneOptions:
    benchmark_limit: int = 20
    run_extract_if_missing: bool = True
    run_cameras_if_missing: bool = True
    prefetch_speedup_threshold: float = 1.03
    cuda_batch_candidates: tuple[int, ...] = (16, 24, 32)
    min_frames_for_batch_search: int = 5


@dataclass(frozen=True)
class OcrBenchmarkResult:
    label: str
    fps: float
    p50_s: float
    p95_s: float
    elapsed_s: float
    frame_count: int


@dataclass
class OcrAutotuneReport:
    torch_device: str
    ocr_backend: str
    sync: OcrBenchmarkResult
    prefetch: OcrBenchmarkResult | None
    batch_trials: list[tuple[int, OcrBenchmarkResult]] = field(default_factory=list)
    chosen_prefetch_workers: int = 2
    chosen_rec_batch: int | None = None
    chosen_cls_batch: int | None = None
    warnings: list[str] = field(default_factory=list)
    ocr_elapsed_s: float | None = None
    ocr_frame_count: int | None = None


RunStagesFn = Callable[[PipelineConfig, str, str], Any]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[idx]


def _benchmark_result(label: str, latencies: list[float], elapsed: float) -> OcrBenchmarkResult:
    count = len(latencies)
    fps = count / elapsed if elapsed > 0 else 0.0
    return OcrBenchmarkResult(
        label=label,
        fps=fps,
        p50_s=statistics.median(latencies) if latencies else 0.0,
        p95_s=_percentile(latencies, 95),
        elapsed_s=elapsed,
        frame_count=count,
    )


def collect_ocr_frame_paths(
    config: PipelineConfig,
    *,
    limit: int | None = None,
) -> list[Path]:
    frame_index_path = config.artifact("frame_index")
    if not frame_index_path.is_file():
        return []

    frame_index = pd.read_csv(frame_index_path)
    ocr_rows = frame_index[frame_index["sample_role"] == "ocr"]
    paths: list[Path] = []
    for row in ocr_rows.itertuples(index=False):
        path = Path(str(getattr(row, "frame_path")))
        if path.is_file():
            paths.append(path)
        if limit is not None and len(paths) >= limit:
            break
    return paths


def verify_ocr_accelerator(config: PipelineConfig) -> tuple[str, str, list[str]]:
    """Return torch device, OCR backend, and warning messages."""
    warnings: list[str] = []
    torch_device = resolve_torch_device(config.accelerator)
    ocr_backend = resolve_ocr_backend(config.accelerator)
    gpu_report = check_gpu_stack(config.accelerator)

    if gpu_report.ocr_cuda_missing and config.accelerator in ("auto", "cuda"):
        warnings.append(
            "CUDAExecutionProvider missing; OCR is on CPU. "
            + (gpu_install_hints(gpu_report)[0] if gpu_install_hints(gpu_report) else "")
        )
    if ocr_backend == "cpu" and config.accelerator in ("auto", "cuda"):
        warnings.append(
            "Expected GPU OCR but backend resolved to CPU — re-run the install cell "
            "(onnxruntime-gpu on Colab Pro)."
        )
    if torch_device == "cuda" and ocr_backend == "cuda":
        warnings.append(f"OCR stack OK: torch={torch_device}, OCR={ocr_backend}")
    else:
        warnings.append(f"OCR stack: torch={torch_device}, OCR={ocr_backend}")

    return torch_device, ocr_backend, warnings


def benchmark_ocr_paths(
    paths: list[Path],
    config: PipelineConfig,
    *,
    prefetch_workers: int,
    label: str | None = None,
) -> OcrBenchmarkResult:
    if not paths:
        return _benchmark_result(label or "empty", [], 0.0)

    ocr_config = _pipeline_ocr_config(config)
    readability_config = ReadabilityConfig()
    latencies: list[float] = []
    started = time.perf_counter()

    if prefetch_workers <= 0:
        for path in paths:
            t0 = time.perf_counter()
            assess_readability(path, ocr_config, readability_config)
            latencies.append(time.perf_counter() - t0)
    else:
        with ThreadPoolExecutor(max_workers=prefetch_workers) as executor:
            load_future: Future | None = None
            next_path: Path | None = None
            for idx, path in enumerate(paths):
                if load_future is not None and next_path == path:
                    bgr = load_future.result()
                else:
                    bgr = load_image(path)

                if idx + 1 < len(paths):
                    next_path = paths[idx + 1]
                    load_future = executor.submit(load_image, next_path)
                else:
                    load_future = None
                    next_path = None

                t0 = time.perf_counter()
                assess_readability_from_bgr(bgr, ocr_config, readability_config)
                latencies.append(time.perf_counter() - t0)

    elapsed = time.perf_counter() - started
    result_label = label or (f"prefetch={prefetch_workers}" if prefetch_workers else "sync")
    return _benchmark_result(result_label, latencies, elapsed)


def _has_ocr_samples(config: PipelineConfig) -> bool:
    frame_index_path = config.artifact("frame_index")
    if not frame_index_path.is_file():
        return False
    frame_index = pd.read_csv(frame_index_path)
    return not frame_index[frame_index["sample_role"] == "ocr"].empty


def ensure_ocr_prerequisites(
    config: PipelineConfig,
    run_stages: RunStagesFn,
    options: OcrAutotuneOptions,
) -> list[str]:
    """Run upstream stages when artifacts required for OCR are missing."""
    actions: list[str] = []
    frame_index_path = config.artifact("frame_index")
    assignments_path = config.artifact("frame_assignments")

    if options.run_extract_if_missing and (
        not frame_index_path.is_file() or not _has_ocr_samples(config)
    ):
        run_stages(config, "meta", "extract")
        actions.append("meta → extract")

    if options.run_cameras_if_missing and not assignments_path.is_file():
        run_stages(config, "filter", "cameras")
        actions.append("filter → cameras")

    return actions


def autotune_ocr_settings(
    config: PipelineConfig,
    paths: list[Path],
    options: OcrAutotuneOptions,
    *,
    locked: dict[str, Any] | None = None,
) -> OcrAutotuneReport:
    """Benchmark sync/prefetch and optional CUDA batch sizes; apply winners to *config*."""
    locked = locked or {}
    torch_device, ocr_backend, warnings = verify_ocr_accelerator(config)

    clear_ocr_engine_cache()
    sync = benchmark_ocr_paths(paths, config, prefetch_workers=0, label="sync")
    prefetch = benchmark_ocr_paths(paths, config, prefetch_workers=2, label="prefetch")

    if locked.get("ocr_prefetch_workers") is not None:
        chosen_prefetch = int(config.ocr_prefetch_workers)
    elif prefetch.fps >= sync.fps * options.prefetch_speedup_threshold:
        chosen_prefetch = 2
    else:
        chosen_prefetch = 0

    batch_trials: list[tuple[int, OcrBenchmarkResult]] = []
    chosen_rec = config.ocr_rec_batch
    chosen_cls = config.ocr_cls_batch

    search_batches = (
        ocr_backend == "cuda"
        and locked.get("ocr_rec_batch") is None
        and len(paths) >= options.min_frames_for_batch_search
    )
    if search_batches:
        best_fps = -1.0
        best_batch = OcrConfig().resolved_rec_batch_num("cuda")
        for batch_size in options.cuda_batch_candidates:
            config.ocr_rec_batch = batch_size
            config.ocr_cls_batch = (
                batch_size if locked.get("ocr_cls_batch") is None else config.ocr_cls_batch
            )
            clear_ocr_engine_cache()
            trial = benchmark_ocr_paths(
                paths,
                config,
                prefetch_workers=chosen_prefetch,
                label=f"batch={batch_size}",
            )
            batch_trials.append((batch_size, trial))
            if trial.fps > best_fps:
                best_fps = trial.fps
                best_batch = batch_size
        chosen_rec = best_batch
        if locked.get("ocr_cls_batch") is None:
            chosen_cls = best_batch

    config.ocr_prefetch_workers = chosen_prefetch
    if locked.get("ocr_rec_batch") is None:
        config.ocr_rec_batch = chosen_rec
    if locked.get("ocr_cls_batch") is None:
        config.ocr_cls_batch = chosen_cls

    return OcrAutotuneReport(
        torch_device=torch_device,
        ocr_backend=ocr_backend,
        sync=sync,
        prefetch=prefetch,
        batch_trials=batch_trials,
        chosen_prefetch_workers=chosen_prefetch,
        chosen_rec_batch=config.ocr_rec_batch,
        chosen_cls_batch=config.ocr_cls_batch,
        warnings=warnings,
    )


def print_autotune_report(report: OcrAutotuneReport) -> None:
    for message in report.warnings:
        print(message)
    for result in (report.sync, report.prefetch):
        if result is None:
            continue
        print(
            f"{result.label}: {result.fps:.2f} fps "
            f"(p50={result.p50_s:.3f}s, p95={result.p95_s:.3f}s, n={result.frame_count})"
        )
    for batch_size, trial in report.batch_trials:
        print(f"  batch {batch_size}: {trial.fps:.2f} fps")
    print(
        "Chosen settings: "
        f"prefetch_workers={report.chosen_prefetch_workers}, "
        f"rec_batch={report.chosen_rec_batch}, cls_batch={report.chosen_cls_batch}"
    )
    if report.ocr_frame_count is not None and report.ocr_elapsed_s is not None:
        fps = (
            report.ocr_frame_count / report.ocr_elapsed_s
            if report.ocr_elapsed_s > 0
            else 0.0
        )
        print(
            f"OCR stage: {report.ocr_frame_count} frames in {report.ocr_elapsed_s:.1f}s "
            f"({fps:.2f} fps)"
        )


def run_ocr_autotune_pipeline(
    config: PipelineConfig,
    run_stages: RunStagesFn | None = None,
    *,
    options: OcrAutotuneOptions | None = None,
    locked: dict[str, Any] | None = None,
    run_ocr: bool = True,
) -> OcrAutotuneReport:
    """Verify stack, ensure prerequisites, benchmark, tune, and optionally run the OCR stage."""
    if run_stages is None:
        from broadcast_pipeline.orchestrator import run_stages as _run_stages

        run_stages = _run_stages
    opts = options or OcrAutotuneOptions()
    locked = locked or {}

    upstream = ensure_ocr_prerequisites(config, run_stages, opts)
    if upstream:
        print("Ran upstream stages:", ", ".join(upstream))

    paths = collect_ocr_frame_paths(config, limit=opts.benchmark_limit)
    if not paths:
        raise FileNotFoundError(
            "No OCR sample frames found. Check frame_index.csv and extracted JPEG paths."
        )

    report = autotune_ocr_settings(config, paths, opts, locked=locked)
    print_autotune_report(report)

    if run_ocr:
        clear_ocr_engine_cache()
        t0 = time.perf_counter()
        summary = run_stages(config, "ocr", "ocr")
        report.ocr_elapsed_s = time.perf_counter() - t0
        report.ocr_frame_count = summary.n_frames
        print_autotune_report(report)

    return report
