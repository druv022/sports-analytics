"""Data types for person appearance analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

AppearanceStatus = Literal["ok", "low_conf", "no_person", "error"]


@dataclass(frozen=True)
class PersonDetection:
    """A single person instance in a frame."""

    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    mask: NDArray[np.uint8] | None = None
    clothing_color: str | None = None


@dataclass(frozen=True)
class FrameAppearance:
    """Appearance analysis for one camera-sample frame."""

    scene_id: int
    frame_number: int
    frame_path: str
    person_count: int
    person_colors: tuple[str, ...]
    confidence: float
    status: AppearanceStatus
    primary_bgr: tuple[int, int, int] | None = None
    detections: tuple[PersonDetection, ...] = ()


@dataclass(frozen=True)
class SceneAppearance:
    """Primary-color summary for a scene."""

    scene_id: int
    scene_type: str
    person_count: int
    person_colors: tuple[str, ...]
    appearance_signature: str
    confidence: float
    status: AppearanceStatus
    primary_bgr: tuple[int, int, int] | None = None
    dominant_track_frames: int = 0
    dominant_track_median_area: int = 0
