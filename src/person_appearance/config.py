"""Configuration for person appearance analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

PRIMARY_PALETTE: tuple[str, ...] = (
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "white",
    "black",
    "neutral",
)

# Approximate LAB centroids for palette matching (L, a, b).
PALETTE_LAB_CENTROIDS: dict[str, tuple[float, float, float]] = {
    "red": (53.0, 80.0, 67.0),
    "orange": (74.0, 48.0, 74.0),
    "yellow": (97.0, -21.0, 94.0),
    "green": (87.0, -86.0, 83.0),
    "blue": (32.0, 79.0, -108.0),
    "purple": (60.0, 98.0, -60.0),
    "white": (95.0, 0.0, 0.0),
    "black": (15.0, 0.0, 0.0),
    "neutral": (70.0, 0.0, 0.0),
}

COCO_PERSON_CLASS_ID = 0

ColorMethod = Literal["median_lab", "histogram_hsv"]
ColorMaskRegion = Literal["torso", "full"]
DominantTrackPolicy = Literal["biggest", "consistent"]


@dataclass
class AppearanceConfig:
    model_path: Path | None = None
    imgsz: int = 640
    min_confidence: float = 0.5
    color_tolerance: float = 18.0
    count_slack: int = 1
    min_sequence_match: int = 2
    torso_y_start: float = 0.25
    torso_y_end: float = 0.70
    min_saturation: float = 0.12
    nms_iou: float = 0.45
    person_class_id: int = COCO_PERSON_CLASS_ID
    palette: tuple[str, ...] = field(default_factory=lambda: PRIMARY_PALETTE)
    color_method: ColorMethod = "histogram_hsv"
    color_mask_region: ColorMaskRegion = "torso"
    dominant_track_policy: DominantTrackPolicy = "consistent"
    track_match_iou: float = 0.3
    mask_erode_px: int = 1
    histogram_h_bins: int = 32

    def resolved_model_path(self, project_root: Path | None = None) -> Path:
        if self.model_path is not None:
            return Path(self.model_path)
        root = project_root or Path.cwd()
        return root / "models" / "person_seg" / "yolo11n-seg.onnx"
