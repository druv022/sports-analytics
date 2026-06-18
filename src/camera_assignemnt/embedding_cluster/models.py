"""Data models for approach 4 camera clustering."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

Frame = NDArray[np.uint8]


@dataclass
class SceneSample:
    """One representative frame per detected scene."""

    scene_idx: int
    scene_id: str
    image_idx: int
    frame_path: str
    frame: Frame | None = None


@dataclass
class ClusterResult:
    """Per-scene clustering output."""

    scene_idx: int
    scene_id: str
    frame_path: str
    features: NDArray[np.float32] | None = None
    reduced: NDArray[np.float32] | None = None
    cluster_id: int = -1
    camera_id: str | None = None


@dataclass
class PipelineOutput:
    """Full pipeline result."""

    results: list[ClusterResult] = field(default_factory=list)
    reduced_matrix: NDArray[np.float32] | None = None
    dbscan_eps: float | None = None
    method: str = "hsv"
    ensemble_vote_threshold: float | None = None
    ensemble_noise_threshold: float | None = None
    ensemble_member_weights: dict[str, float] | None = None
    ensemble_member_stats: dict[str, dict] | None = None
    ensemble_member_labelings: list[NDArray[np.int64]] | None = None
    ensemble_member_names: list[str] | None = None
