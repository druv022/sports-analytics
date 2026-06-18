#!/usr/bin/env python3
"""Serve the timeline visualization web app for broadcast pipeline outputs."""

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

from broadcast_pipeline.viz.server import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "pipeline",
        help="Pipeline output directory with aggregate CSVs",
    )
    parser.add_argument(
        "--static-dir",
        type=Path,
        default=ROOT / "static" / "timeline_viz",
        help="Directory containing index.html and assets",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import uvicorn
    except ImportError:
        print(
            "Missing viz dependencies. Install with: pip install -e '.[viz,appearance,ocr]'",
            file=sys.stderr,
        )
        return 1

    app = create_app(args.output_dir.resolve(), args.static_dir.resolve())
    print(f"Timeline viz: http://{args.host}:{args.port}")
    print(f"Pipeline output: {args.output_dir.resolve()}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
