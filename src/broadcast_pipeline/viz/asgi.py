"""ASGI entrypoint for container deployment (Cloud Run, Docker, etc.)."""

from __future__ import annotations

import os
from pathlib import Path

from broadcast_pipeline.viz.server import create_app

_output_dir = Path(os.environ.get("OUTPUT_DIR", "/app/data/pipeline")).resolve()
_static_dir = Path(os.environ.get("STATIC_DIR", "/app/static/timeline_viz")).resolve()

app = create_app(_output_dir, _static_dir)
