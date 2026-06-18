#!/usr/bin/env python3
"""Download or export YOLO11n-seg ONNX weights for person appearance analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "models" / "person_seg" / "yolo11n-seg.onnx"


def export_with_ultralytics(output_path: Path, imgsz: int) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is required to export YOLO11 weights. "
            "Install with: pip install -e '.[appearance-export]'"
        ) from exc

    model = YOLO("yolo11n-seg.pt")
    exported = model.export(
        format="onnx",
        task="segment",
        opset=17,
        dynamic=False,
        imgsz=imgsz,
    )
    exported_path = Path(str(exported))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if exported_path.resolve() != output_path.resolve():
        output_path.write_bytes(exported_path.read_bytes())
    print(f"Exported ONNX model to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help="Destination ONNX path",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    if args.output.is_file():
        print(f"Model already exists: {args.output}")
        return 0

    export_with_ultralytics(args.output, args.imgsz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
