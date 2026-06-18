"""Configuration and tournament-specific HSV constants."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Surface = Literal["clay", "hard", "grass"]

COURT_HSV_RANGES: dict[Surface, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "hard": ((90, 60, 60), (130, 255, 255)),
    "clay": ((5, 80, 80), (20, 255, 255)),
    "grass": ((35, 40, 40), (85, 255, 255)),
}

DEFAULT_HSV_REGION_WEIGHTS: dict[str, float] = {
    "full": 1.0,
    "top_left": 1.0,
    "top_right": 1.0,
    "bottom_left": 1.0,
    "bottom_right": 1.0,
    "center": 1.5,
}


@dataclass
class Config:
    """Pipeline hyper-parameters for approach 1."""

    surface: Surface = "hard"

    full_court_ratio: float = 0.35

    classification_dir: str = "data/classification"
    scene_mlp_path: str = "models/scene_mlp.joblib"

    hough_threshold: int = 50
    hough_min_length: int = 40
    hough_max_gap: int = 8

    horizontal_angle_max_deg: float = 12.0
    max_lines_for_vp: int = 80
    line_dedup_midpoint_px: float = 10.0
    line_dedup_angle_deg: float = 5.0

    min_lines_for_vp: int = 5

    ransac_iterations: int = 200
    ransac_inlier_px: float = 5.0
    ransac_seed: int = 0

    dbscan_eps: float = 0.06
    dbscan_min_samples: int = 2

    hsv_dbscan_eps: float = 0.35
    histogram_bins: int = 16
    hsv_region_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_HSV_REGION_WEIGHTS)
    )

    ensemble_vp_weight: float = 0.7
    ensemble_hsv_weight: float = 0.3
    vp_confidence_min: float = 0.20
    vp_confidence_strong: float = 0.35

    temporal_window: int = 4

    def resolved_scene_mlp_path(self, project_root: Path | None = None) -> Path:
        """Resolve the scene MLP path against project root, not only CWD."""
        raw = Path(self.scene_mlp_path)
        if raw.is_file():
            return raw.resolve()

        candidates: list[Path] = []
        seen: set[Path] = set()

        def _add(path: Path) -> None:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                candidates.append(resolved)

        if raw.is_absolute():
            _add(raw)
        else:
            if project_root is not None:
                _add(Path(project_root) / raw)
            _add(Path(__file__).resolve().parents[3] / raw)
            _add(Path.cwd() / raw)

        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return candidates[0] if candidates else raw.resolve()

    def scene_mlp_search_paths(self, project_root: Path | None = None) -> tuple[Path, ...]:
        """Candidate paths checked by :meth:`resolved_scene_mlp_path`."""
        raw = Path(self.scene_mlp_path)
        paths: list[Path] = []
        seen: set[Path] = set()

        def _add(path: Path) -> None:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                paths.append(resolved)

        if raw.is_file():
            return (raw.resolve(),)
        if raw.is_absolute():
            _add(raw)
        else:
            if project_root is not None:
                _add(Path(project_root) / raw)
            _add(Path(__file__).resolve().parents[3] / raw)
            _add(Path.cwd() / raw)
        return tuple(paths)
