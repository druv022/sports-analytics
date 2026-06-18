from __future__ import annotations

import cv2
import numpy as np

from .config import HomographyConfig
from .court_reference import CourtReferenceData
from .types import KeypointDetection


def estimate_homography_from_keypoints(
    detection: KeypointDetection,
    court_ref: CourtReferenceData,
    config: HomographyConfig,
) -> tuple[np.ndarray | None, np.ndarray | None, float, int, np.ndarray, np.ndarray]:
    """
    Estimate H mapping dimension reference -> scene from detected keypoints.

    Returns (H_dim_to_scene, H_scene_to_dim, reproj_error, inlier_count,
             matched_dim_pts, matched_scene_pts).
    """
    valid_idx = np.where(detection.valid)[0]
    if len(valid_idx) < config.min_keypoints:
        return None, None, float("inf"), 0, np.empty((0, 2)), np.empty((0, 2))

    src = court_ref.dimension_keypoints[valid_idx].astype(np.float32)
    dst = detection.points[valid_idx].astype(np.float32)

    H_dim_to_scene, mask = cv2.findHomography(
        src,
        dst,
        cv2.RANSAC,
        config.ransac_thresh,
    )
    if H_dim_to_scene is None:
        return None, None, float("inf"), 0, np.empty((0, 2)), np.empty((0, 2))

    inlier_mask = mask.ravel().astype(bool)
    n_in = int(inlier_mask.sum())

    pts_h = np.hstack([src, np.ones((len(src), 1))]).astype(np.float64)
    proj = (H_dim_to_scene @ pts_h.T).T
    proj_xy = proj[:, :2] / proj[:, 2:3]
    err = float(np.linalg.norm(proj_xy - dst, axis=1)[inlier_mask].mean()) if n_in else float("inf")

    H_scene_to_dim = np.linalg.inv(H_dim_to_scene)

    matched_dim = court_ref.dimension_keypoints[valid_idx[inlier_mask]]
    matched_scene = dst[inlier_mask]

    return (
        H_dim_to_scene,
        H_scene_to_dim,
        err,
        n_in,
        matched_dim,
        matched_scene,
    )
