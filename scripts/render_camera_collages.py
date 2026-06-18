#!/usr/bin/env python3
"""Render one static verification collage per assigned camera."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from broadcast_pipeline.viz.camera_collage import CameraCollageLoadError  # noqa: E402
from broadcast_pipeline.viz.camera_collage_render import (  # noqa: E402
    CollageRenderConfig,
    render_camera_collages,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "pipeline",
        help="Pipeline output directory with scene_assignments.csv and frames/",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=None,
        help="Where to write collages (default: <output-dir>/camera_collages)",
    )
    parser.add_argument("--thumb-width", type=int, default=320)
    parser.add_argument("--thumb-height", type=int, default=180)
    parser.add_argument(
        "--layout",
        choices=("timeline", "grid"),
        default="timeline",
        help="timeline = one row per scene; grid = compact mid-frame grid",
    )
    parser.add_argument(
        "--slots",
        default="mid",
        help="Comma-separated frame slots: begin,mid,end (timeline layout only)",
    )
    parser.add_argument("--grid-columns", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    slot_names = tuple(part.strip() for part in args.slots.split(",") if part.strip())  # type: ignore[assignment]
    if not slot_names:
        print("Provide at least one slot name.", file=sys.stderr)
        return 1

    config = CollageRenderConfig(
        thumb_width=args.thumb_width,
        thumb_height=args.thumb_height,
        slots=slot_names,  # type: ignore[arg-type]
        layout=args.layout,
        grid_columns=args.grid_columns,
    )

    try:
        paths = render_camera_collages(args.output_dir, dest_dir=args.dest_dir, config=config)
    except CameraCollageLoadError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    dest = args.dest_dir or (args.output_dir / "camera_collages")
    print(f"Wrote {len(paths)} collage(s) to {dest.resolve()}")
    for path in paths:
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
