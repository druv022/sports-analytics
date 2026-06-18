from dataclasses import dataclass
from typing import Literal

DetectionMode = Literal["contour", "min_area_rect", "line_quad"]
ContourEdgeSource = Literal["gray", "white_mask", "canny_hough"]

@dataclass
class RectangleConfig:
    mode: DetectionMode = "contour"

    # Preprocessing
    blur_ksize: int = 5
    canny_low: int = 50
    canny_high: int = 150

    # Contour filtering
    min_area_ratio: float = 0.001
    max_area_ratio: float = 0.95
    approx_epsilon_ratio: float = 0.025
    min_aspect_ratio: float = 0.2
    max_aspect_ratio: float = 8.0
    require_convex: bool = True

    # #4: contour edges from white court lines instead of full gray frame
    contour_edge_source: ContourEdgeSource = "white_mask"

    # #5: accept 4..N-gon approximations, then fit a quad
    approx_min_vertices: int = 4
    approx_max_vertices: int = 6
    fit_quad_from_polygon: bool = True

    use_min_area_fallback: bool = False   # disable junk minAreaRect blobs on scenes
    min_area_fallback_ratio: float = 0.05

    min_box_area_ratio: float = 0.001

    # Line-alignment filter
    require_line_alignment: bool = False
    line_source: Literal["white_hough", "canny_hough"] = "white_hough"
    white_hsv_low: tuple[int, int, int] = (0, 0, 185)
    white_hsv_high: tuple[int, int, int] = (180, 40, 255)
    use_court_mask_for_lines: bool = True
    court_hsv_low: tuple[int, int, int] = (90, 60, 60)
    court_hsv_high: tuple[int, int, int] = (130, 255, 255)
    court_mask_dilate: int = 20
    line_hough_threshold: int = 40
    line_hough_min_length: int = 30
    line_hough_max_gap: int = 12
    line_canny_low: int = 50
    line_canny_high: int = 150
    line_distance_thresh_px: float = 8.0
    line_angle_thresh_deg: float = 20.0
    line_min_edge_support: float = 0.50
    line_min_aligned_edges: int = 3
    line_edge_samples: int = 11
    line_filter_if_no_lines: Literal["pass", "reject"] = "pass"

    # #3: quad construction from line intersections
    line_quad_angle_split_deg: float = 25.0
    line_quad_min_family_lines: int = 2
    line_quad_merge_angle_deg: float = 8.0
    line_quad_merge_rho_px: float = 15.0
    line_quad_max_candidates: int = 30
    line_quad_min_side_px: float = 20.0

    # Size filter (explicit, applied after detection)
    require_size_filter: bool = True

    # Perimeter line visibility (fraction of border on detected lines)
    line_min_perimeter_support: float = 0.70   # >70% of edge length
    line_use_perimeter_support: bool = True      # use perimeter metric instead of edge count