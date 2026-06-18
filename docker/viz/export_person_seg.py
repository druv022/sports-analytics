#!/usr/bin/env python3
"""Export YOLO11n-seg ONNX weights for the timeline viz container."""

from __future__ import annotations

import argparse
from pathlib import Path


def export_person_seg(output_path: Path, imgsz: int = 640) -> None:
    from ultralytics import YOLO

    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported = YOLO("yolo11n-seg.pt").export(
        format="onnx",
        task="segment",
        opset=17,
        dynamic=False,
        imgsz=imgsz,
    )
    exported_path = Path(str(exported))
    if exported_path.resolve() != output_path.resolve():
        output_path.write_bytes(exported_path.read_bytes())
    print(f"Exported ONNX model to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/models/person_seg/yolo11n-seg.onnx"),
    )
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    export_person_seg(args.output, args.imgsz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
