"""
Homography projection via TennisCourtDetector keypoints.

Maps broadcast frames onto the Court_dimension.png reference using:
1. 14-keypoint CNN detection (TennisCourtDetector)
2. TCD homography repair for occluded points
3. Bridge transform to Court_dimension.png
4. Back-projection validation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import HomographyConfig
from .court_reference import CourtReferenceData, load_court_reference
from .homography_estimator import estimate_homography_from_keypoints
from .homography_validator import line_alignment_error, passes_exact_gate, reference_space_line_error
from .types import KeypointDetection

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class HomographyResult:
    """Outcome of homography estimation for one scene frame."""

    success: bool
    exact: bool = False
    H: np.ndarray | None = None
    H_inv: np.ndarray | None = None
    reproj_error: float = float("inf")
    line_alignment_error_px: float = float("inf")
    reference_line_error_px: float = float("inf")
    inlier_count: int = 0
    n_keypoints_detected: int = 0
    matched_ref: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float32))
    matched_scene: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float32))
    scene_keypoints: np.ndarray = field(default_factory=lambda: np.full((14, 2), np.nan, np.float32))
    keypoint_valid: np.ndarray = field(default_factory=lambda: np.zeros(14, dtype=bool))
    message: str = ""


# ---------------------------------------------------------------------------
# Warp / overlay helpers
# ---------------------------------------------------------------------------

_PALETTE = [
    (0, 255, 0),
    (0, 165, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 255),
]


def warp_scene_to_reference(
    scene: np.ndarray,
    H_inv: np.ndarray,
    ref_shape: Tuple[int, int],
) -> np.ndarray:
    """Warp a scene frame onto reference-image pixel coordinates."""
    ref_h, ref_w = ref_shape
    return cv2.warpPerspective(
        scene,
        H_inv,
        (ref_w, ref_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def overlay_on_reference(
    reference: np.ndarray,
    warped_scene: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend warped scene over reference where the warp is non-black."""
    mask = (warped_scene.sum(axis=2) > 0).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = np.clip(mask * alpha, 0.0, 1.0)[:, :, np.newaxis]

    ref_f = reference.astype(np.float32)
    scene_f = warped_scene.astype(np.float32)
    blended = ref_f * (1.0 - mask) + scene_f * mask
    return blended.astype(np.uint8)


def stack_overlays_on_reference(
    reference: np.ndarray,
    warped_scenes: list[np.ndarray],
    alpha: float = 0.15,
) -> np.ndarray:
    """Composite multiple warped scenes onto one reference canvas."""
    canvas = reference.copy().astype(np.float32)

    for warped in warped_scenes:
        mask = (warped.sum(axis=2) > 0).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = np.clip(mask * alpha, 0.0, 1.0)[:, :, np.newaxis]
        canvas = canvas * (1.0 - mask) + warped.astype(np.float32) * mask

    return canvas.astype(np.uint8)


# ---------------------------------------------------------------------------
# Estimation pipeline
# ---------------------------------------------------------------------------


def estimate_homography(
    scene: np.ndarray,
    config: HomographyConfig | None = None,
    court_ref: CourtReferenceData | None = None,
) -> HomographyResult:
    """Estimate dimension -> scene homography from CNN keypoints."""
    from .keypoint_detector import detect_court_keypoints

    config = config or HomographyConfig()
    court_ref = court_ref or load_court_reference(config.resolved_keypoints_json())

    try:
        detection = detect_court_keypoints(scene, config)
    except (FileNotFoundError, ImportError) as exc:
        return HomographyResult(success=False, message=str(exc))

    n_detected = int(detection.valid.sum())
    if n_detected < config.min_keypoints:
        return HomographyResult(
            success=False,
            n_keypoints_detected=n_detected,
            scene_keypoints=detection.points,
            keypoint_valid=detection.valid,
            message=f"Too few keypoints detected ({n_detected}/{config.min_keypoints})",
        )

    H, H_inv, reproj_err, n_in, matched_ref, matched_scene = estimate_homography_from_keypoints(
        detection, court_ref, config
    )

    if H is None or H_inv is None:
        return HomographyResult(
            success=False,
            n_keypoints_detected=n_detected,
            scene_keypoints=detection.points,
            keypoint_valid=detection.valid,
            message="Homography computation failed",
        )

    line_err = line_alignment_error(scene, H, court_ref)

    return HomographyResult(
        success=True,
        exact=False,
        H=H,
        H_inv=H_inv,
        reproj_error=reproj_err,
        line_alignment_error_px=line_err,
        inlier_count=n_in,
        n_keypoints_detected=n_detected,
        matched_ref=matched_ref,
        matched_scene=matched_scene,
        scene_keypoints=detection.points,
        keypoint_valid=detection.valid,
        message="ok",
    )


def map_scene_to_reference(
    scene: np.ndarray,
    reference: np.ndarray,
    config: HomographyConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, HomographyResult]:
    """
    Warp scene onto reference coordinates and return overlay.

    Returns (warped_scene, overlay, result).
    """
    config = config or HomographyConfig()
    ref_shape = reference.shape[:2]
    court_ref = load_court_reference(config.resolved_keypoints_json())

    result = estimate_homography(scene, config, court_ref)
    if not result.success or result.H_inv is None:
        return np.zeros_like(reference), reference.copy(), result

    warped = warp_scene_to_reference(scene, result.H_inv, ref_shape)
    overlay = overlay_on_reference(reference, warped, alpha=config.overlay_alpha)

    ref_line_err = reference_space_line_error(scene, result.H_inv, reference, court_ref)
    result.reference_line_error_px = ref_line_err
    result.exact = passes_exact_gate(
        ref_line_err,
        result.line_alignment_error_px,
        config.max_line_error_px,
        config.max_scene_line_error_px,
        result.inlier_count,
    )
    if result.success and not result.exact:
        result.message = (
            f"line alignment ref={ref_line_err:.1f}px scene={result.line_alignment_error_px:.1f}px "
            f"(limits {config.max_line_error_px}/{config.max_scene_line_error_px}px)"
        )

    return warped, overlay, result


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def draw_keypoints(
    scene: np.ndarray,
    detection: KeypointDetection,
    labels: list[str] | None = None,
) -> np.ndarray:
    from .keypoint_detector import draw_detected_keypoints

    return draw_detected_keypoints(scene, detection, labels)


def draw_backprojected_lines(
    scene: np.ndarray,
    H_dim_to_scene: np.ndarray,
    court_ref: CourtReferenceData | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw reference court lines back-projected onto the scene."""
    from .court_reference import dimension_line_segments

    court_ref = court_ref or load_court_reference()
    vis = scene.copy()
    for p1, p2 in dimension_line_segments(court_ref):
        pts = np.array([p1, p2], dtype=np.float32).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(pts, H_dim_to_scene).reshape(-1, 2).astype(int)
        cv2.line(vis, tuple(proj[0]), tuple(proj[1]), color, 2, cv2.LINE_AA)
    return vis


class ManualCalibrator:
    """Interactive 4-point calibration fallback."""

    def pick_points(
        self,
        image: np.ndarray,
        n: int = 4,
        title: str = "Click points — press any key when done",
    ) -> np.ndarray:
        pts: list[list[float]] = []
        vis = image.copy()

        def _cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(pts) < n:
                pts.append([float(x), float(y)])
                cv2.circle(vis, (x, y), 6, (0, 255, 0), -1)
                cv2.putText(
                    vis, str(len(pts)), (x + 8, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )
                cv2.imshow(title, vis)

        cv2.imshow(title, vis)
        cv2.setMouseCallback(title, _cb)
        cv2.waitKey(0)
        cv2.destroyWindow(title)

        if len(pts) < n:
            raise RuntimeError(f"Only {len(pts)}/{n} points were picked.")

        return np.array(pts, dtype=np.float32)
