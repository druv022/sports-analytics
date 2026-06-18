from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np

from .rectangle_detector import DetectedRectangle, load_image
from .rectangle_mapper import RectangleMappingResult

PathLike = Union[str, Path]

# Distinct BGR colors for matched pairs
MATCH_COLORS: list[tuple[int, int, int]] = [
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 165, 255),
    (203, 192, 255),
    (147, 20, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 255, 255),
    (180, 105, 255),
]


def _color_for_match(match_id: int) -> tuple[int, int, int]:
    return MATCH_COLORS[(match_id - 1) % len(MATCH_COLORS)]


def _resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == target_height:
        return image
    scale = target_height / h
    new_w = max(1, int(w * scale))
    return cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_AREA)


def _scale_rect(
    rect: DetectedRectangle,
    sx: float,
    sy: float,
) -> DetectedRectangle:
    corners = rect.corners.copy()
    corners[:, 0] *= sx
    corners[:, 1] *= sy
    center = (rect.center[0] * sx, rect.center[1] * sy)
    area = rect.area * sx * sy
    return DetectedRectangle(
        corners=corners,
        area=area,
        center=center,
        angle=rect.angle,
        contour=rect.contour,
    )


def _draw_single_rectangle(
    image: np.ndarray,
    rect: DetectedRectangle,
    label: str,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    pts = rect.corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)

    cx, cy = map(int, rect.center)
    cv2.circle(image, (cx, cy), 6, color, -1, lineType=cv2.LINE_AA)

    cv2.putText(
        image,
        label,
        (cx + 8, cy - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_mapping_side_by_side(
    ref_image: np.ndarray,
    scene_image: np.ndarray,
    mapping: RectangleMappingResult,
    ref_rects: list[DetectedRectangle],
    scene_rects: list[DetectedRectangle],
    *,
    gap: int = 40,
    target_height: int = 900,
    draw_unmatched: bool = True,
) -> np.ndarray:
    """
    Create side-by-side visualization:
      left  = reference image
      right = scene image
    Matched rectangles share the same number and color.
    """
    ref_h, ref_w = ref_image.shape[:2]
    scene_h, scene_w = scene_image.shape[:2]

    ref_vis = _resize_to_height(ref_image, target_height)
    scene_vis = _resize_to_height(scene_image, target_height)

    ref_sy = target_height / ref_h
    ref_sx = ref_vis.shape[1] / ref_w
    scene_sy = target_height / scene_h
    scene_sx = scene_vis.shape[1] / scene_w

    canvas_h = target_height
    canvas_w = ref_vis.shape[1] + gap + scene_vis.shape[1]
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    canvas[:, : ref_vis.shape[1]] = ref_vis
    canvas[:, ref_vis.shape[1] + gap :] = scene_vis

    x_offset_scene = ref_vis.shape[1] + gap

    cv2.putText(
        canvas,
        "Reference",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Scene",
        (x_offset_scene + 20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    matched_ref_idx = {m.ref_index for m in mapping.matches}
    matched_scene_idx = {m.scene_index for m in mapping.matches}

    for match in mapping.matches:
        color = _color_for_match(match.match_id)
        label = str(match.match_id)

        ref_rect = _scale_rect(ref_rects[match.ref_index], ref_sx, ref_sy)
        scene_rect = _scale_rect(scene_rects[match.scene_index], scene_sx, scene_sy)

        _draw_single_rectangle(canvas, ref_rect, label, color, thickness=3)

        scene_rect_shifted = DetectedRectangle(
            corners=scene_rect.corners.copy(),
            area=scene_rect.area,
            center=(scene_rect.center[0] + x_offset_scene, scene_rect.center[1]),
            angle=scene_rect.angle,
            contour=scene_rect.contour,
        )
        scene_rect_shifted.corners[:, 0] += x_offset_scene
        _draw_single_rectangle(canvas, scene_rect_shifted, label, color, thickness=3)

        ref_center = (int(ref_rect.center[0]), int(ref_rect.center[1]))
        scene_center = (int(scene_rect_shifted.center[0]), int(scene_rect_shifted.center[1]))
        cv2.line(canvas, ref_center, scene_center, color, 2, cv2.LINE_AA)

        mid_x = (ref_center[0] + scene_center[0]) // 2
        mid_y = (ref_center[1] + scene_center[1]) // 2
        cv2.putText(
            canvas,
            f"c={match.cost:.2f}",
            (mid_x - 30, mid_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    if draw_unmatched:
        gray = (128, 128, 128)
        for i, rect in enumerate(ref_rects):
            if i in matched_ref_idx:
                continue
            scaled = _scale_rect(rect, ref_sx, ref_sy)
            _draw_single_rectangle(canvas, scaled, f"U{i}", gray, thickness=1)

        for j, rect in enumerate(scene_rects):
            if j in matched_scene_idx:
                continue
            scaled = _scale_rect(rect, scene_sx, scene_sy)
            scaled.corners[:, 0] += x_offset_scene
            shifted = DetectedRectangle(
                corners=scaled.corners,
                area=scaled.area,
                center=(scaled.center[0] + x_offset_scene, scaled.center[1]),
                angle=scaled.angle,
                contour=scaled.contour,
            )
            _draw_single_rectangle(canvas, shifted, f"U{j}", gray, thickness=1)

    return canvas


def visualize_rectangle_mapping(
    ref_path: PathLike,
    scene_path: PathLike,
    mapping: RectangleMappingResult,
    ref_rects: list[DetectedRectangle],
    scene_rects: list[DetectedRectangle],
    *,
    save_path: PathLike | None = None,
    show: bool = True,
    window_name: str = "Rectangle Mapping",
) -> np.ndarray:
    ref_image = load_image(ref_path)
    scene_image = load_image(scene_path)

    canvas = draw_mapping_side_by_side(
        ref_image,
        scene_image,
        mapping,
        ref_rects,
        scene_rects,
    )

    if save_path is not None:
        cv2.imwrite(str(save_path), canvas)

    if show:
        cv2.imshow(window_name, canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return canvas