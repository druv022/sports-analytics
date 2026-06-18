
"""End-to-end broadcast pipeline: extract frames -> camera ID -> OCR -> aggregation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from broadcast_pipeline.config import PipelineConfig, PipelineStep
from broadcast_pipeline.orchestrator import run_pipeline
from broadcast_pipeline.progress import log_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT / "data" / "Untitled.mp4")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "pipeline")
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=None,
        help="User-editable approved complete text CSV",
    )
    parser.add_argument("--camera-samples", type=int, default=5)
    parser.add_argument("--ocr-samples-per-sec", type=float, default=2.0)
    parser.add_argument(
        "--ocr-scale",
        type=float,
        default=1.5,
        help="OCR upscale factor (lower uses less memory; default 1.5)",
    )
    parser.add_argument(
        "--ocr-rec-batch",
        type=int,
        default=None,
        help="RapidOCR recognition batch size (default: backend-aware)",
    )
    parser.add_argument(
        "--ocr-cls-batch",
        type=int,
        default=None,
        help="RapidOCR angle-classifier batch size (default: backend-aware)",
    )
    parser.add_argument(
        "--ocr-prefetch-workers",
        type=int,
        default=2,
        help="Thread pool size for prefetching next frame load (0 disables)",
    )
    parser.add_argument("--enable-vlm", action="store_true")
    parser.add_argument("--fast-cameras", action="store_true", help="HSV-only camera clustering")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--from-step",
        choices=(
            "all",
            "meta",
            "extract",
            "filter",
            "appearance",
            "cameras",
            "ocr",
            "reference",
            "enrich",
            "associate",
            "aggregate",
        ),
        default="all",
        help="First pipeline stage to run (inclusive)",
    )
    parser.add_argument(
        "--to-step",
        choices=(
            "all",
            "meta",
            "extract",
            "filter",
            "appearance",
            "cameras",
            "ocr",
            "reference",
            "enrich",
            "associate",
            "aggregate",
        ),
        default=None,
        help="Last pipeline stage to run (inclusive). Defaults to aggregate.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PipelineConfig(
        video_path=args.video,
        output_dir=args.output_dir,
        reference_csv=args.reference_csv,
        camera_samples_per_scene=args.camera_samples,
        ocr_samples_per_sec=args.ocr_samples_per_sec,
        ocr_scale=args.ocr_scale,
        ocr_rec_batch=args.ocr_rec_batch,
        ocr_cls_batch=args.ocr_cls_batch,
        ocr_prefetch_workers=args.ocr_prefetch_workers,
        enable_vlm=args.enable_vlm,
        fast_cameras=args.fast_cameras,
        resume=args.resume,
        from_step=args.from_step,  # type: ignore[arg-type]
        to_step=args.to_step,  # type: ignore[arg-type]
    )
    step_range = config.from_step if config.to_step is None else f"{config.from_step}→{config.to_step}"
    log_info(
        f"Pipeline start — video={config.video_path.name}, "
        f"output={config.output_dir}, steps={step_range}"
    )
    if config.resume:
        log_info("Resume enabled — reusing existing stage artifacts where present")
    if config.fast_cameras:
        log_info("Fast camera mode — HSV-only clustering")
    log_info(
        f"OCR settings — scale={config.ocr_scale}, samples/sec={config.ocr_samples_per_sec}, "
        f"prefetch_workers={config.ocr_prefetch_workers}"
    )

    summary = run_pipeline(config)

    log_info(f"Pipeline finished — output={config.output_dir}")
    if summary.artifacts:
        for name, path in summary.artifacts.items():
            print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
