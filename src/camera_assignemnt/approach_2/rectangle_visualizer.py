from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np

from .config import RectangleConfig
from .rectangle_detector import DetectedRectangle, detect_rectangles, load_image

PathLike = Union[str, Path]


def draw_rectangles(
    image: np.ndarray,
    rectangles: list[DetectedRectangle],
    *,
    box_color: tuple[int, int, int] = (0, 255, 0),
    corner_color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
    show_labels: bool = True,
) -> np.ndarray:
    vis = image.copy()

    for i, rect in enumerate(rectangles):
        pts = rect.corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], isClosed=True, color=box_color, thickness=thickness)

        for j, (x, y) in enumerate(rect.corners.astype(int)):
            cv2.circle(vis, (x, y), 5, corner_color, -1, lineType=cv2.LINE_AA)
            if show_labels:
                cv2.putText(
                    vis,
                    str(j),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    corner_color,
                    1,
                    cv2.LINE_AA,
                )

        if show_labels:
            cx, cy = map(int, rect.center)
            cv2.putText(
                vis,
                f"#{i} area={rect.area:.0f}",
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2,
                cv2.LINE_AA,
            )

    return vis


def visualize_rectangles(
    image_path: PathLike,
    config: RectangleConfig | None = None,
    *,
    save_path: PathLike | None = None,
    show: bool = True,
    window_name: str = "Rectangles",
) -> tuple[np.ndarray, np.ndarray, list[DetectedRectangle]]:
    config = config or RectangleConfig()
    image = load_image(image_path)
    rectangles = detect_rectangles(image, config)
    annotated = draw_rectangles(image, rectangles)
    if config.require_line_alignment:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        from .rectangle_detector import detect_line_segments
        lines = detect_line_segments(image, gray, config)
        for x1, y1, x2, y2 in lines.astype(int):
            cv2.line(annotated, (x1, y1), (x2, y2), (255, 0, 0), 1, cv2.LINE_AA)
    from .rectangle_detector import (
    detect_line_segments, _rectangle_perimeter_line_support, _to_gray
    )
    gray = _to_gray(image, config.blur_ksize)
    lines = detect_line_segments(image, gray, config)
    for i, r in enumerate(rectangles):
        s = _rectangle_perimeter_line_support(r.corners, lines, config)
        print(f"#{i} area={r.area:.0f} perimeter_support={s:.2f}")

    cv2.putText(
        annotated,
        f"{config.mode}: {len(rectangles)} rectangles",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    if save_path is not None:
        cv2.imwrite(str(save_path), annotated)

    if show:
        cv2.imshow(window_name, annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image, annotated, rectangles