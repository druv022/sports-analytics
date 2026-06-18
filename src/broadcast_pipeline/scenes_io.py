"""Load scenes.json without scene-detection runtime dependencies."""

from __future__ import annotations
import json
from pathlib import Path

from broadcast_pipeline.types import Scene


def load_scenes(path: Path) -> list[Scene]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        Scene(
            scene_id=int(item["scene_id"]),
            start_frame=int(item["start_frame"]),
            end_frame=int(item["end_frame"]),
            start_sec=float(item["start_sec"]),
            end_sec=float(item["end_sec"]),
        )
        for item in data
    ]
