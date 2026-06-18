#!/usr/bin/env python3
"""Benchmark OCR throughput on sample frames."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from broadcast_pipeline.config import PipelineConfig
from broadcast_pipeline.ocr_autotune import (
    OcrAutotuneOptions,
    autotune_ocr_settings,
    collect_ocr_frame_paths,
    print_autotune_report,
    verify_ocr_accelerator,
)
from src.accelerator.device import resolve_ocr_backend, resolve_torch_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames-dir",
        type=Path,
        required=True,
        help="Directory containing OCR sample images",
    )
    parser.add_argument("--limit", type=int, default=20, help="Max frames to benchmark")
    parser.add_argument("--ocr-scale", type=float, default=1.5)
    parser.add_argument("--rec-batch", type=int, default=None)
    parser.add_argument("--cls-batch", type=int, default=None)
    parser.add_argument(
        "--prefetch",
        action="store_true",
        help="Benchmark and auto-select prefetch vs sync",
    )
    parser.add_argument(
        "--accelerator",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(
        p
        for p in args.frames_dir.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        print(f"No images found under {args.frames_dir}", file=sys.stderr)
        return 1

    config = PipelineConfig(
        output_dir=args.frames_dir,
        ocr_scale=args.ocr_scale,
        accelerator=args.accelerator,
        ocr_rec_batch=args.rec_batch,
        ocr_cls_batch=args.cls_batch,
    )
    backend = resolve_ocr_backend(args.accelerator)
    torch_device = resolve_torch_device(args.accelerator)
    _, _, warnings = verify_ocr_accelerator(config)
    for message in warnings:
        print(message)

    if args.prefetch:
        report = autotune_ocr_settings(
            config,
            paths,
            OcrAutotuneOptions(benchmark_limit=len(paths)),
            locked={
                "ocr_rec_batch": args.rec_batch,
                "ocr_cls_batch": args.cls_batch,
            },
        )
        print_autotune_report(report)
    else:
        from broadcast_pipeline.ocr_autotune import benchmark_ocr_paths

        sync = benchmark_ocr_paths(paths, config, prefetch_workers=0, label="sync")
        print(
            f"sync: {sync.fps:.2f} fps (p50={sync.p50_s:.3f}s, p95={sync.p95_s:.3f}s) "
            f"torch={torch_device}, OCR backend={backend}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
