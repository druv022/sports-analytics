from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class HomographyConfig:
    """Parameters for keypoint-based homography estimation."""

    model_path: Path | None = None
    keypoints_json: Path | None = None
    device: str | None = None

    input_size: tuple[int, int] = (640, 360)
    heatmap_thresh: int = 170
    heatmap_max_radius: int = 25

    use_refine_kps: bool = True
    use_tcd_homography_repair: bool = False

    min_keypoints: int = 4
    ransac_thresh: float = 5.0
    max_line_error_px: float = 5.0
    max_scene_line_error_px: float = 12.0

    overlay_alpha: float = 0.45
    composite_layer_alpha: float = 0.15

    def resolved_model_path(self) -> Path:
        if self.model_path is not None:
            return Path(self.model_path)
        root = Path(__file__).resolve().parents[3]
        return root / "models" / "tennis_court_detector.pth"

    def resolved_keypoints_json(self) -> Path:
        if self.keypoints_json is not None:
            return Path(self.keypoints_json)
        root = Path(__file__).resolve().parents[3]
        return root / "data" / "court_reference" / "keypoints.json"
