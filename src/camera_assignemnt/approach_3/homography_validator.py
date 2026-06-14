from __future__ import annotations

import cv2
import numpy as np

from .court_reference import CourtReferenceData, build_dimension_line_mask, load_court_reference


def line_alignment_error(
    scene: np.ndarray,
    H_dim_to_scene: np.ndarray,
    court_ref: CourtReferenceData | None = None,
    white_thresh: int = 200,
) -> float:
    """
    Back-project dimension court lines onto the scene and measure alignment
    with white court-line pixels. Lower is better (px).
    """
    court_ref = court_ref or load_court_reference()
    line_mask = build_dimension_line_mask(court_ref, thickness=3)

    h, w = scene.shape[:2]
    warped_lines = cv2.warpPerspective(
        line_mask,
        H_dim_to_scene,
        (w, h),
        flags=cv2.INTER_NEAREST,
    )

    gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    _, white = cv2.threshold(gray, white_thresh, 255, cv2.THRESH_BINARY)
    dist = cv2.distanceTransform(255 - white, cv2.DIST_L2, 3)

    ys, xs = np.where(warped_lines > 0)
    if len(xs) == 0:
        return float("inf")

    return float(np.median(dist[ys, xs]))


def reference_space_line_error(
    scene: np.ndarray,
    H_scene_to_dim: np.ndarray,
    reference: np.ndarray,
    court_ref: CourtReferenceData | None = None,
    white_thresh: int = 200,
) -> float:
    """Measure warped scene white lines vs reference diagram lines (dimension space)."""
    court_ref = court_ref or load_court_reference()
    h, w = court_ref.reference_size

    warped = cv2.warpPerspective(scene, H_scene_to_dim, (w, h))
    wgray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, wlines = cv2.threshold(wgray, white_thresh, 255, cv2.THRESH_BINARY)

    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    _, ref_lines = cv2.threshold(ref_gray, 128, 255, cv2.THRESH_BINARY_INV)
    dist = cv2.distanceTransform(255 - ref_lines, cv2.DIST_L2, 3)

    court_mask = build_dimension_line_mask(court_ref, thickness=8)
    ys, xs = np.where((wlines > 0) & (court_mask > 0))
    if len(xs) == 0:
        return float("inf")

    return float(np.median(dist[ys, xs]))


def passes_exact_gate(
    reference_line_error_px: float,
    scene_line_error_px: float,
    max_reference_line_error_px: float,
    max_scene_line_error_px: float,
    inlier_count: int,
    min_inliers: int = 4,
) -> bool:
    return (
        inlier_count >= min_inliers
        and reference_line_error_px <= max_reference_line_error_px
        and scene_line_error_px <= max_scene_line_error_px
    )
