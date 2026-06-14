#!/usr/bin/env python3
"""Detect 14 court keypoints on Court_dimension.png from line intersections."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.annotate_court_keypoints import DEFAULT_IMAGE, DEFAULT_OUTPUT, LABELS, build_json


def _cluster(vals: list[float], tol: float) -> list[float]:
    vals = sorted(vals)
    clusters: list[list[float]] = []
    for v in vals:
        if not clusters or abs(v - clusters[-1][0]) > tol:
            clusters.append([float(v), 1.0])
        else:
            c = clusters[-1]
            c[0] = (c[0] * c[1] + v) / (c[1] + 1.0)
            c[1] += 1.0
    return [c[0] for c in clusters]


def detect_dimension_keypoints(image_path: Path) -> list[list[float]]:
    """Return 14 (x, y) points aligned with TCD label order."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    lines = cv2.HoughLinesP(bw, 1, np.pi / 180, 100, minLineLength=200, maxLineGap=20)
    if lines is None:
        raise RuntimeError("Could not detect court lines on reference diagram")

    horiz: list[tuple[int, int, int, int]] = []
    vert: list[tuple[int, int, int, int]] = []
    for line in lines:
        x1, y1, x2, y2 = [int(v) for v in line[0]]
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if ang < 10 or ang > 170:
            horiz.append((x1, y1, x2, y2))
        elif 80 < ang < 100:
            vert.append((x1, y1, x2, y2))

    if not horiz or not vert:
        raise RuntimeError("Insufficient horizontal/vertical court lines detected")

    ys = _cluster([y1 for _, y1, _, _ in horiz] + [y2 for _, _, _, y2 in horiz], tol=20)
    xs = _cluster([x1 for x1, _, _, _ in vert] + [x2 for _, _, x2, _ in vert], tol=20)

    if len(ys) < 4 or len(xs) < 2:
        raise RuntimeError(f"Expected 4+ horizontal and 2+ vertical lines, got {len(ys)} / {len(xs)}")

    # Keep the four main court horizontals; ignore outer dimension annotations.
    if len(ys) > 4:
        ys = ys[:4]
    y_top, y_service_far, y_service_near, y_bottom = ys
    x_left, x_right = xs[0], xs[-2 if len(xs) >= 2 else -1]

    if len(xs) >= 7:
        x_left_singles = xs[2]
        x_right_singles = xs[-3]
        x_center = xs[len(xs) // 2]
    else:
        width = x_right - x_left
        singles_inset = width * (4.5 / 36.0)
        x_left_singles = x_left + singles_inset
        x_right_singles = x_right - singles_inset
        x_center = (x_left + x_right) / 2.0

    points = [
        [x_left, y_top],  # 0 baseline_top_left
        [x_right, y_top],  # 1 baseline_top_right
        [x_left, y_bottom],  # 2 baseline_bottom_left
        [x_right, y_bottom],  # 3 baseline_bottom_right
        [x_left_singles, y_top],  # 4 left_inner_top
        [x_left_singles, y_bottom],  # 5 left_inner_bottom
        [x_right_singles, y_top],  # 6 right_inner_top
        [x_right_singles, y_bottom],  # 7 right_inner_bottom
        [x_left_singles, y_service_far],  # 8 top_inner_left
        [x_right_singles, y_service_far],  # 9 top_inner_right
        [x_left_singles, y_service_near],  # 10 bottom_inner_left
        [x_right_singles, y_service_near],  # 11 bottom_inner_right
        [x_center, y_service_far],  # 12 middle_line_top
        [x_center, y_service_near],  # 13 middle_line_bottom
    ]
    return [[float(x), float(y)] for x, y in points]


def main() -> None:
    points = detect_dimension_keypoints(DEFAULT_IMAGE)
    data = build_json(points, DEFAULT_IMAGE)
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote {DEFAULT_OUTPUT}")
    for i, (label, pt) in enumerate(zip(LABELS, points)):
        print(f"  {i:2d} {label:22s} {pt}")


if __name__ == "__main__":
    main()
