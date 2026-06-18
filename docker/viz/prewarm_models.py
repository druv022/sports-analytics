#!/usr/bin/env python3
"""Warm the appearance segmenter in the runtime image."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def prewarm_appearance(model_path: Path) -> None:
    from broadcast_pipeline.config import PipelineConfig
    from broadcast_pipeline.viz.appearance_api import get_segmenter

    if not model_path.is_file():
        raise FileNotFoundError(f"Person segmentation model missing: {model_path}")

    config = PipelineConfig(output_dir=Path("/app/data/pipeline"))
    get_segmenter(config=config)
    print(f"Appearance segmenter ready ({model_path})")


def main() -> int:
    model_path = Path("/app/models/person_seg/yolo11n-seg.onnx")
    prewarm_appearance(model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
