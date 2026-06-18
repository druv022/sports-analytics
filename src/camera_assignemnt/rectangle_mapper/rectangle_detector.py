from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import numpy as np

from .config import RectangleConfig

ImageInput = Union[str, Path, np.ndarray]


@dataclass
class DetectedRectangle:
    """One detected rectangle."""
    corners: np.ndarray          # (4, 2) float32, ordered TL, TR, BR, BL
    area: float
    center: tuple[float, float]
    angle: float                 # degrees; meaningful for min_area_rect mode
    contour: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Image I/O and basic geometry
# ---------------------------------------------------------------------------

def load_image(image: ImageInput) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    img = cv2.imread(str(image))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image}")
    return img


def _to_gray(image: np.ndarray, blur_ksize: int) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if blur_ksize > 1:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    return gray


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _box_corners_from_min_area_rect(rect: tuple) -> np.ndarray:
    box = cv2.boxPoints(rect)
    return _order_corners(box)


def _passes_aspect_ratio(corners: np.ndarray, config: RectangleConfig) -> bool:
    w = float(np.linalg.norm(corners[1] - corners[0]))
    h = float(np.linalg.norm(corners[3] - corners[0]))
    if w == 0 or h == 0:
        return False
    ratio = max(w, h) / min(w, h)
    return config.min_aspect_ratio <= ratio <= config.max_aspect_ratio


def _fold_angle_rad(theta: float) -> float:
    """Map angle to [0, pi/2] so parallel lines match regardless of direction."""
    theta = abs(theta) % np.pi
    if theta > np.pi / 2:
        theta = np.pi - theta
    return theta


def _segment_angle_rad(x1: float, y1: float, x2: float, y2: float) -> float:
    return _fold_angle_rad(float(np.arctan2(y2 - y1, x2 - x1)))


def _point_segment_distance(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    vx, vy = x2 - x1, y2 - y1
    seg_len_sq = vx * vx + vy * vy
    if seg_len_sq == 0.0:
        return float(np.hypot(px - x1, py - y1))

    t = ((px - x1) * vx + (py - y1) * vy) / seg_len_sq
    t = float(np.clip(t, 0.0, 1.0))
    proj_x = x1 + t * vx
    proj_y = y1 + t * vy
    return float(np.hypot(px - proj_x, py - proj_y))


# ---------------------------------------------------------------------------
# Masks and edge images
# ---------------------------------------------------------------------------

def _court_mask(bgr: np.ndarray, config: RectangleConfig) -> np.ndarray | None:
    if not config.use_court_mask_for_lines:
        return None
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array(config.court_hsv_low, dtype=np.uint8),
        np.array(config.court_hsv_high, dtype=np.uint8),
    )
    k = max(1, int(config.court_mask_dilate))
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(mask, kernel)


def _white_line_mask(bgr: np.ndarray, config: RectangleConfig) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array(config.white_hsv_low, dtype=np.uint8),
        np.array(config.white_hsv_high, dtype=np.uint8),
    )
    court_mask = _court_mask(bgr, config)
    if court_mask is not None:
        mask = cv2.bitwise_and(mask, court_mask)
    return mask


def _contour_edge_image(
    bgr: np.ndarray,
    gray: np.ndarray,
    config: RectangleConfig,
) -> np.ndarray:
    """Build contour edge map from white lines, gray, or schematic canny."""
    if config.contour_edge_source == "white_mask":
        base = _white_line_mask(bgr, config)
    elif config.contour_edge_source == "canny_hough":
        base = cv2.Canny(gray, config.canny_low, config.canny_high)
    else:
        base = gray

    edges = cv2.Canny(base, config.canny_low, config.canny_high)
    return cv2.dilate(edges, None, iterations=1)


# ---------------------------------------------------------------------------
# Hough line detection and alignment filters
# ---------------------------------------------------------------------------

def detect_line_segments(
    bgr: np.ndarray,
    gray: np.ndarray,
    config: RectangleConfig,
) -> np.ndarray:
    """Return line segments as (N, 4) array: x1, y1, x2, y2."""
    if config.line_source == "white_hough":
        line_mask = _white_line_mask(bgr, config)
    else:
        line_mask = cv2.Canny(gray, config.line_canny_low, config.line_canny_high)

    edges = cv2.Canny(line_mask, config.line_canny_low, config.line_canny_high)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=config.line_hough_threshold,
        minLineLength=config.line_hough_min_length,
        maxLineGap=config.line_hough_max_gap,
    )
    if lines is None:
        return np.empty((0, 4), dtype=np.float32)
    return lines[:, 0, :].astype(np.float32)


def _edge_line_support_fraction(
    p0: np.ndarray,
    p1: np.ndarray,
    lines: np.ndarray,
    config: RectangleConfig,
) -> float:
    """Return fraction of sampled edge points lying on a Hough segment (0..1)."""
    if lines.shape[0] == 0:
        return 0.0

    edge_angle = _segment_angle_rad(float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1]))
    angle_thresh = np.radians(config.line_angle_thresh_deg)
    n_samples = max(2, int(config.line_edge_samples))

    hits = 0
    for t in np.linspace(0.0, 1.0, n_samples):
        px = float(p0[0] + t * (p1[0] - p0[0]))
        py = float(p0[1] + t * (p1[1] - p0[1]))

        for x1, y1, x2, y2 in lines:
            dist = _point_segment_distance(px, py, x1, y1, x2, y2)
            line_angle = _segment_angle_rad(x1, y1, x2, y2)
            angle_diff = abs(edge_angle - line_angle)
            if (
                dist <= config.line_distance_thresh_px
                and angle_diff <= angle_thresh
            ):
                hits += 1
                break

    return hits / n_samples


def _rectangle_perimeter_line_support(
    corners: np.ndarray,
    lines: np.ndarray,
    config: RectangleConfig,
) -> float:
    """Length-weighted fraction of rectangle perimeter supported by Hough lines."""
    total_len = 0.0
    supported_len = 0.0

    for i in range(4):
        p0 = corners[i]
        p1 = corners[(i + 1) % 4]
        edge_len = float(np.linalg.norm(p1 - p0))
        if edge_len <= 0.0:
            continue
        support = _edge_line_support_fraction(p0, p1, lines, config)
        total_len += edge_len
        supported_len += support * edge_len

    if total_len <= 0.0:
        return 0.0
    return supported_len / total_len


def _passes_size_filter(
    rect: DetectedRectangle,
    image_area: float,
    config: RectangleConfig,
) -> bool:
    if not config.require_size_filter:
        return True
    norm_area = float(rect.area) / float(image_area)
    return config.min_area_ratio <= norm_area <= config.max_area_ratio


def _passes_line_alignment(
    corners: np.ndarray,
    lines: np.ndarray,
    config: RectangleConfig,
) -> bool:
    if not config.require_line_alignment:
        return True

    if lines.shape[0] == 0:
        return config.line_filter_if_no_lines == "pass"

    if config.line_use_perimeter_support:
        support = _rectangle_perimeter_line_support(corners, lines, config)
        return support >= config.line_min_perimeter_support

    aligned_edges = 0
    for i in range(4):
        p0 = corners[i]
        p1 = corners[(i + 1) % 4]
        if _edge_line_support_fraction(p0, p1, lines, config) >= config.line_min_edge_support:
            aligned_edges += 1
    return aligned_edges >= config.line_min_aligned_edges


def _passes_rectangle_filters(
    rect: DetectedRectangle,
    lines: np.ndarray | None,
    image_area: float,
    config: RectangleConfig,
) -> bool:
    if not _passes_size_filter(rect, image_area, config):
        return False
    if lines is None:
        lines = np.empty((0, 4), dtype=np.float32)
    return _passes_line_alignment(rect.corners, lines, config)


def rectangle_perimeter_line_support(
    rect: DetectedRectangle,
    lines: np.ndarray,
    config: RectangleConfig,
) -> float:
    """Public helper for debug/visualization."""
    return _rectangle_perimeter_line_support(rect.corners, lines, config)


# ---------------------------------------------------------------------------
# Rectangle builders
# ---------------------------------------------------------------------------

def _detected_rectangle_from_quad_approx(
    contour: np.ndarray,
    approx: np.ndarray,
) -> DetectedRectangle | None:
    """Build a DetectedRectangle from a 4-point polygon approximation."""
    corners = _order_corners(approx.reshape(4, 2))
    area = float(cv2.contourArea(approx))
    cx = float(np.mean(corners[:, 0]))
    cy = float(np.mean(corners[:, 1]))

    return DetectedRectangle(
        corners=corners,
        area=area,
        center=(cx, cy),
        angle=0.0,
        contour=contour,
    )


def _detected_rectangle_from_min_area_rect(
    contour: np.ndarray,
    config: RectangleConfig,
    image_area: float,
) -> DetectedRectangle | None:
    """Build a DetectedRectangle from cv2.minAreaRect, with filtering."""
    if len(contour) < 5:
        return None

    rect = cv2.minAreaRect(contour)
    (_, _), (rw, rh), angle = rect
    box_area = float(rw * rh)
    box_area_ratio = box_area / image_area

    if box_area_ratio < config.min_area_ratio or box_area_ratio > config.max_area_ratio:
        return None

    corners = _box_corners_from_min_area_rect(rect)
    if not _passes_aspect_ratio(corners, config):
        return None

    return DetectedRectangle(
        corners=corners,
        area=box_area,
        center=(rect[0][0], rect[0][1]),
        angle=float(angle),
        contour=contour,
    )


def _fit_quad_from_polygon(
    contour: np.ndarray,
    approx: np.ndarray,
) -> DetectedRectangle | None:
    """Turn 4..6 vertex polygon into ordered quad + area."""
    if len(approx) < 4:
        return None

    if len(approx) == 4:
        return _detected_rectangle_from_quad_approx(contour, approx)

    rect = cv2.minAreaRect(approx.reshape(-1, 2).astype(np.float32))
    (_, _), (rw, rh), angle = rect
    if rw <= 1.0 or rh <= 1.0:
        return None

    corners = _box_corners_from_min_area_rect(rect)
    area = float(rw * rh)
    cx, cy = rect[0]
    return DetectedRectangle(
        corners=corners,
        area=area,
        center=(float(cx), float(cy)),
        angle=float(angle),
        contour=contour,
    )


def _quad_from_corners(corners: np.ndarray) -> DetectedRectangle | None:
    corners = _order_corners(corners.astype(np.float32))
    if not cv2.isContourConvex(corners.reshape(-1, 1, 2)):
        return None
    area = float(cv2.contourArea(corners))
    if area <= 0:
        return None
    cx = float(np.mean(corners[:, 0]))
    cy = float(np.mean(corners[:, 1]))
    return DetectedRectangle(
        corners=corners,
        area=area,
        center=(cx, cy),
        angle=0.0,
        contour=None,
    )


def _is_valid_quad(
    corners: np.ndarray,
    image_shape: tuple[int, int],
    config: RectangleConfig,
) -> bool:
    h, w = image_shape
    image_area = float(h * w)
    if not _passes_aspect_ratio(corners, config):
        return False

    area = float(cv2.contourArea(corners.astype(np.float32)))
    if area / image_area < config.min_area_ratio or area / image_area > config.max_area_ratio:
        return False

    for i in range(4):
        side = float(np.linalg.norm(corners[i] - corners[(i + 1) % 4]))
        if side < config.line_quad_min_side_px:
            return False

    margin = 10
    for x, y in corners:
        if x < -margin or y < -margin or x > w + margin or y > h + margin:
            return False
    return True


def _dedupe_rectangles(
    rectangles: list[DetectedRectangle],
    min_center_distance: float = 25.0,
    max_count: int | None = None,
) -> list[DetectedRectangle]:
    deduped: list[DetectedRectangle] = []
    for rect in sorted(rectangles, key=lambda r: r.area, reverse=True):
        if all(
            np.linalg.norm(np.array(rect.center) - np.array(existing.center)) > min_center_distance
            for existing in deduped
        ):
            deduped.append(rect)
        if max_count is not None and len(deduped) >= max_count:
            break
    return deduped


# ---------------------------------------------------------------------------
# Line-intersection quad detection
# ---------------------------------------------------------------------------

def _infinite_line_from_segment(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> tuple[float, float, float]:
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    norm = float(np.hypot(a, b))
    if norm < 1e-6:
        return 0.0, 0.0, 0.0
    return a / norm, b / norm, c / norm


def _intersect_infinite_lines(
    l1: tuple[float, float, float],
    l2: tuple[float, float, float],
) -> tuple[float, float] | None:
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return float(x), float(y)


def _merge_line_segments(
    lines: np.ndarray,
    config: RectangleConfig,
) -> list[tuple[float, float, float]]:
    """Merge Hough segments into unique infinite lines (a, b, c)."""
    if lines.shape[0] == 0:
        return []

    merged: list[tuple[float, float, float, float]] = []
    angle_tol = np.radians(config.line_quad_merge_angle_deg)
    rho_tol = config.line_quad_merge_rho_px

    for x1, y1, x2, y2 in lines:
        inf = _infinite_line_from_segment(float(x1), float(y1), float(x2), float(y2))
        if inf == (0.0, 0.0, 0.0):
            continue
        a, b, c = inf
        theta = _fold_angle_rad(float(np.arctan2(b, -a if abs(a) > 1e-6 else 1e-6)))
        rho = -c

        placed = False
        for i, (ma, mb, mc, mtheta) in enumerate(merged):
            if abs(theta - mtheta) <= angle_tol and abs(rho - (-mc)) <= rho_tol:
                merged[i] = ((ma + a) / 2, (mb + b) / 2, (mc + c) / 2, (mtheta + theta) / 2)
                placed = True
                break
        if not placed:
            merged.append((a, b, c, theta))

    return [(a, b, c) for a, b, c, _ in merged]


def _split_line_families(
    infinite_lines: list[tuple[float, float, float]],
    split_deg: float,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """Split merged lines into two angle families."""
    if len(infinite_lines) < 2:
        return infinite_lines, []

    angles = [
        _fold_angle_rad(float(np.arctan2(b, -a if abs(a) > 1e-6 else 1e-6)))
        for a, b, _ in infinite_lines
    ]

    median = float(np.median(angles))
    family_a: list[tuple[float, float, float]] = []
    family_b: list[tuple[float, float, float]] = []
    split = np.radians(split_deg)

    for line, ang in zip(infinite_lines, angles):
        d = abs(ang - median)
        d = min(d, np.pi / 2 - d)
        if d <= split:
            family_a.append(line)
        else:
            family_b.append(line)

    if len(family_b) < 2:
        family_b = [line for line in infinite_lines if line not in family_a]
    return family_a, family_b


def detect_rectangles_from_line_quads(
    bgr: np.ndarray,
    gray: np.ndarray,
    config: RectangleConfig,
    line_segments: np.ndarray | None = None,
) -> list[DetectedRectangle]:
    """Build quads from intersections of two Hough line families."""
    h, w = gray.shape[:2]
    image_area = float(h * w)
    lines = line_segments if line_segments is not None else detect_line_segments(bgr, gray, config)
    merged = _merge_line_segments(lines, config)
    fam_a, fam_b = _split_line_families(merged, config.line_quad_angle_split_deg)

    if (
        len(fam_a) < config.line_quad_min_family_lines
        or len(fam_b) < config.line_quad_min_family_lines
    ):
        return []

    fam_a = fam_a[: min(len(fam_a), 6)]
    fam_b = fam_b[: min(len(fam_b), 6)]
    candidates: list[DetectedRectangle] = []

    for i in range(len(fam_a)):
        for j in range(i + 1, len(fam_a)):
            for k in range(len(fam_b)):
                for m in range(k + 1, len(fam_b)):
                    corners: list[tuple[float, float]] = []
                    for la in (fam_a[i], fam_a[j]):
                        for lb in (fam_b[k], fam_b[m]):
                            pt = _intersect_infinite_lines(la, lb)
                            if pt is not None:
                                corners.append(pt)
                    if len(corners) != 4:
                        continue

                    quad = _quad_from_corners(np.array(corners, dtype=np.float32))
                    if quad is None:
                        continue
                    if not _is_valid_quad(quad.corners, (h, w), config):
                        continue
                    if not _passes_rectangle_filters(quad, lines, image_area, config):
                        continue
                    candidates.append(quad)

    return _dedupe_rectangles(
        candidates,
        min_center_distance=25.0,
        max_count=config.line_quad_max_candidates,
    )


# ---------------------------------------------------------------------------
# Contour-based detection
# ---------------------------------------------------------------------------

def detect_rectangles_contour(
    bgr: np.ndarray,
    gray: np.ndarray,
    config: RectangleConfig,
    line_segments: np.ndarray | None = None,
) -> list[DetectedRectangle]:
    h, w = gray.shape[:2]
    image_area = float(h * w)

    edges = _contour_edge_image(bgr, gray, config)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rectangles: list[DetectedRectangle] = []
    lines = line_segments if line_segments is not None else np.empty((0, 4), dtype=np.float32)

    for contour in contours:
        area = cv2.contourArea(contour)
        area_ratio = area / image_area
        if area_ratio < config.min_area_ratio or area_ratio > config.max_area_ratio:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(
            contour,
            config.approx_epsilon_ratio * peri,
            True,
        )
        n_vertices = len(approx)

        is_valid_poly = (
            config.approx_min_vertices <= n_vertices <= config.approx_max_vertices
            and (not config.require_convex or cv2.isContourConvex(approx))
        )

        if is_valid_poly and config.fit_quad_from_polygon:
            if n_vertices == 4:
                rect_obj = _detected_rectangle_from_quad_approx(contour, approx)
            else:
                rect_obj = _fit_quad_from_polygon(contour, approx)

            if rect_obj is None:
                continue
            if not _passes_aspect_ratio(rect_obj.corners, config):
                continue
            if not _passes_rectangle_filters(rect_obj, lines, image_area, config):
                continue
            rectangles.append(rect_obj)
            continue

        if config.use_min_area_fallback and area_ratio >= config.min_area_fallback_ratio:
            rect_obj = _detected_rectangle_from_min_area_rect(contour, config, image_area)
            if rect_obj is None:
                continue
            if not _passes_rectangle_filters(rect_obj, lines, image_area, config):
                continue
            rectangles.append(rect_obj)

    rectangles.sort(key=lambda r: r.area, reverse=True)
    return rectangles


def detect_rectangles_min_area(
    gray: np.ndarray,
    config: RectangleConfig,
) -> list[DetectedRectangle]:
    h, w = gray.shape[:2]
    image_area = float(h * w)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rectangles: list[DetectedRectangle] = []

    for contour in contours:
        rect_obj = _detected_rectangle_from_min_area_rect(contour, config, image_area)
        if rect_obj is None:
            continue
        if rect_obj.area / image_area < config.min_box_area_ratio:
            continue
        rectangles.append(rect_obj)

    rectangles.sort(key=lambda r: r.area, reverse=True)
    return rectangles


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect_rectangles(
    image: ImageInput,
    config: RectangleConfig | None = None,
) -> list[DetectedRectangle]:
    """
    Detect rectangles in an image.

    Modes:
      - contour: white-line / gray edge contours + optional quad fitting
      - line_quad: Hough line families -> intersection quads
      - min_area_rect: Otsu threshold + minAreaRect

    Filters (when enabled):
      - require_size_filter: min_area_ratio .. max_area_ratio
      - require_line_alignment + line_use_perimeter_support:
        keep only if >= line_min_perimeter_support (default 70%)

    Returns:
        List of DetectedRectangle, sorted largest-first.
    """
    config = config or RectangleConfig()
    bgr = load_image(image)
    gray = _to_gray(bgr, config.blur_ksize)
    image_area = float(bgr.shape[0] * bgr.shape[1])

    need_lines = config.require_line_alignment or config.mode == "line_quad"
    line_segments = detect_line_segments(bgr, gray, config) if need_lines else None

    if config.mode == "contour":
        return detect_rectangles_contour(bgr, gray, config, line_segments)

    if config.mode == "line_quad":
        rects = detect_rectangles_from_line_quads(bgr, gray, config, line_segments)
        rects = [
            rect
            for rect in rects
            if _passes_rectangle_filters(rect, line_segments, image_area, config)
        ]
        rects.sort(key=lambda r: r.area, reverse=True)
        return rects

    if config.mode == "min_area_rect":
        rects = detect_rectangles_min_area(gray, config)
        if config.require_line_alignment or config.require_size_filter:
            rects = [
                rect
                for rect in rects
                if _passes_rectangle_filters(rect, line_segments, image_area, config)
            ]
        return rects

    raise ValueError(f"Unknown mode: {config.mode}")