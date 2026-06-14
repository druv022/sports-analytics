from .config import RectangleConfig
from .rectangle_detector import DetectedRectangle, detect_rectangles
from .rectangle_mapper import (
    RectangleMappingConfig,
    RectangleMappingResult,
    RectangleMatch,
    match_rectangles,
    match_rectangles_by_rank,
    match_rectangles_hungarian,
)
from .mapping_visualizer import visualize_rectangle_mapping, draw_mapping_side_by_side

try:
    from .rectangle_visualizer import draw_rectangles, visualize_rectangles
except ImportError:
    from .rectangle_visualizer import draw_rectangles, visualize_rectangles

__all__ = [
    "RectangleConfig",
    "RectangleMappingConfig",
    "RectangleMappingResult",
    "RectangleMatch",
    "DetectedRectangle",
    "detect_rectangles",
    "match_rectangles",
    "match_rectangles_by_rank",
    "match_rectangles_hungarian",
    "draw_rectangles",
    "visualize_rectangles",
    "draw_mapping_side_by_side",
    "visualize_rectangle_mapping",
]