from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class VideoMeta:
    path: Path
    fps: float
    frame_count: int
    duration_sec: float
    width: int
    height: int
    fps_source: str = "scenedetect"


@dataclass
class Scene:
    scene_id: int
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float


@dataclass
class FrameRecord:
    scene_id: int
    frame_number: int
    seconds: float
    frame_path: Path
    sample_role: Literal["camera", "ocr"]


@dataclass
class SceneResult:
    scene_id: int
    scene_type: str
    camera_id: str
    cluster_id: int
    camera_vote_counts: dict[str, int]


@dataclass
class FrameOcrObservation:
    scene_id: int
    frame_number: int
    seconds: float
    camera_id: str
    words: list[str]
    verdict: str
    used_unk: bool
    detections: list[dict] | None = None


@dataclass
class AggregatedTextRow:
    camera_id: str
    text: str
    text_kind: Literal["complete", "partial"]
    mapped_complete_text: str | None
    total_duration_sec: float
    frame_ranges: str
    n_frames_present: int = 0
    n_frames_good: int = 0
    n_frames_partial: int = 0
    n_frames_enriched: int = 0
    dominant_readability: str = "partial"


@dataclass
class PipelineSummary:
    video_meta: VideoMeta | None = None
    n_scenes: int = 0
    n_frames: int = 0
    n_cameras: int = 0
    n_approved_text: int = 0
    n_new_reference_text: int = 0
    output_dir: Path = field(default_factory=lambda: Path("data/pipeline"))
    artifacts: dict[str, Path] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
