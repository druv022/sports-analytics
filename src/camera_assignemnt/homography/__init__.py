from .config import HomographyConfig
from .court_reference import CourtReferenceData, load_court_reference
from .homography_validator import line_alignment_error, passes_exact_gate, reference_space_line_error

__all__ = [
    "HomographyConfig",
    "HomographyResult",
    "CourtReferenceData",
    "KeypointDetection",
    "ManualCalibrator",
    "detect_court_keypoints",
    "draw_backprojected_lines",
    "draw_keypoints",
    "estimate_homography",
    "estimate_homography_from_keypoints",
    "line_alignment_error",
    "load_court_reference",
    "map_scene_to_reference",
    "overlay_on_reference",
    "passes_exact_gate",
    "reference_space_line_error",
    "stack_overlays_on_reference",
    "warp_scene_to_reference",
]


def __getattr__(name: str):
    if name == "HomographyResult":
        from .homography_projector import HomographyResult
        return HomographyResult
    if name in {
        "ManualCalibrator",
        "draw_backprojected_lines",
        "draw_keypoints",
        "estimate_homography",
        "map_scene_to_reference",
        "overlay_on_reference",
        "stack_overlays_on_reference",
        "warp_scene_to_reference",
    }:
        from . import homography_projector as hp
        return getattr(hp, name)
    if name in {"detect_court_keypoints", "KeypointDetection"}:
        from .keypoint_detector import KeypointDetection, detect_court_keypoints
        return KeypointDetection if name == "KeypointDetection" else detect_court_keypoints
    if name == "estimate_homography_from_keypoints":
        from .homography_estimator import estimate_homography_from_keypoints
        return estimate_homography_from_keypoints
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
