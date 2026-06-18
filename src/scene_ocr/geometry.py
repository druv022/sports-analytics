"""Bounding-box helpers for scene OCR."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from src.scene_ocr.config import OverlayRegion


def polygon_to_xyxy(bbox: NDArray) -> NDArray:
    """Convert a polygon (4, 2) or xyxy (4,) bbox to integer xyxy."""
    arr = np.asarray(bbox, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == 2:
        xs = arr[:, 0]
        ys = arr[:, 1]
        return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.int32)
    if arr.shape == (4,):
        x0, y0, x1, y1 = arr
        return np.array([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], dtype=np.int32)
    raise ValueError(f"Unsupported bbox shape: {arr.shape}")


def bbox_area(xyxy: NDArray) -> float:
    x0, y0, x1, y1 = polygon_to_xyxy(xyxy)
    return float(max(0, x1 - x0) * max(0, y1 - y0))


def bbox_height(xyxy: NDArray) -> float:
    _, y0, _, y1 = polygon_to_xyxy(xyxy)
    return float(max(0, y1 - y0))


def scale_xyxy(xyxy: NDArray, factor: float) -> NDArray:
    if factor == 1.0:
        return polygon_to_xyxy(xyxy)
    scaled = polygon_to_xyxy(xyxy).astype(np.float64) / factor
    return np.round(scaled).astype(np.int32)


def bbox_iou(a: NDArray, b: NDArray) -> float:
    ax0, ay0, ax1, ay1 = polygon_to_xyxy(a)
    bx0, by0, bx1, by1 = polygon_to_xyxy(b)
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter == 0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def bbox_center(xyxy: NDArray) -> tuple[float, float]:
    x0, y0, x1, y1 = polygon_to_xyxy(xyxy)
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def point_in_region(x: float, y: float, region: OverlayRegion, width: int, height: int) -> bool:
    rx0 = region.x0 * width
    ry0 = region.y0 * height
    rx1 = region.x1 * width
    ry1 = region.y1 * height
    return rx0 <= x <= rx1 and ry0 <= y <= ry1


def detection_in_region(bbox: NDArray, region: OverlayRegion, width: int, height: int) -> bool:
    cx, cy = bbox_center(bbox)
    return point_in_region(cx, cy, region, width, height)


def crop_region(bgr: NDArray, region: OverlayRegion) -> tuple[NDArray, NDArray]:
    """Return (crop, xyxy offset bbox in full image)."""
    h, w = bgr.shape[:2]
    x0 = int(region.x0 * w)
    y0 = int(region.y0 * h)
    x1 = int(region.x1 * w)
    y1 = int(region.y1 * h)
    return bgr[y0:y1, x0:x1].copy(), np.array([x0, y0, x1, y1], dtype=np.int32)


def offset_xyxy(xyxy: NDArray, offset: NDArray) -> NDArray:
    ox0, oy0, _, _ = polygon_to_xyxy(offset)
    x0, y0, x1, y1 = polygon_to_xyxy(xyxy)
    return np.array([x0 + ox0, y0 + oy0, x1 + ox0, y1 + oy0], dtype=np.int32)


def pad_xyxy(xyxy: NDArray, pad_ratio: float, width: int, height: int) -> NDArray:
    x0, y0, x1, y1 = polygon_to_xyxy(xyxy)
    bw = x1 - x0
    bh = y1 - y0
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)
    return np.array(
        [
            max(0, x0 - pad_x),
            max(0, y0 - pad_y),
            min(width, x1 + pad_x),
            min(height, y1 + pad_y),
        ],
        dtype=np.int32,
    )


def crop_xyxy(bgr: NDArray, xyxy: NDArray) -> NDArray:
    x0, y0, x1, y1 = polygon_to_xyxy(xyxy)
    return bgr[y0:y1, x0:x1].copy()


def resize_max_dim(bgr: NDArray, max_dim: int) -> NDArray:
    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return bgr
    scale = max_dim / longest
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
