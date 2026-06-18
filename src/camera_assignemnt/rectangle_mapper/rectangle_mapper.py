from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .rectangle_detector import DetectedRectangle

MatchingMode = Literal["rank", "hungarian"]


@dataclass
class RectangleMappingConfig:
    """Weights and thresholds for rectangle matching."""

    matching_mode: MatchingMode = "rank"

    # Cost weights (should sum to ~1.0)
    weight_area: float = 0.80
    weight_aspect: float = 0.10
    weight_center: float = 0.05
    weight_angle: float = 0.05

    # Structural penalties
    containment_penalty: float = 3.0
    adjacency_penalty: float = 2.0

    # Adjacency detection
    adjacency_edge_gap: float = 8.0
    adjacency_center_factor: float = 1.25

    # Rank matching: max allowed |norm_area_ref - norm_area_scene|
    rank_area_tolerance: float = 0.35

    # Reject matches above this cost (None = no filter)
    max_match_cost: float | None = None

    # Limit how many rectangles to match (largest N from each side)
    max_rectangles: int | None = None


@dataclass
class RectangleMatch:
    """One reference -> scene rectangle correspondence."""
    match_id: int
    ref_index: int
    scene_index: int
    cost: float
    ref: DetectedRectangle
    scene: DetectedRectangle


@dataclass
class RectangleMappingResult:
    matches: list[RectangleMatch]
    unmatched_ref: list[int]
    unmatched_scene: list[int]
    cost_matrix: np.ndarray
    ref_indices: list[int]
    scene_indices: list[int]


def _image_shape(
    rects: list[DetectedRectangle],
    image_shape: tuple[int, int] | None,
) -> tuple[int, int]:
    if image_shape is not None:
        h, w = image_shape
        return int(h), int(w)

    if not rects:
        raise ValueError("image_shape is required when rectangle list is empty")

    max_x = max(float(r.corners[:, 0].max()) for r in rects)
    max_y = max(float(r.corners[:, 1].max()) for r in rects)
    return int(max_y + 1), int(max_x + 1)


def _aspect_ratio(rect: DetectedRectangle) -> float:
    w = float(np.linalg.norm(rect.corners[1] - rect.corners[0]))
    h = float(np.linalg.norm(rect.corners[3] - rect.corners[0]))
    if min(w, h) == 0:
        return 0.0
    return max(w, h) / min(w, h)


def _normalized_area(rect: DetectedRectangle, image_area: float) -> float:
    return float(rect.area) / float(image_area)


def _normalized_center(rect: DetectedRectangle, width: int, height: int) -> np.ndarray:
    cx, cy = rect.center
    return np.array([cx / width, cy / height], dtype=np.float32)


def _rect_width(rect: DetectedRectangle) -> float:
    return float(np.linalg.norm(rect.corners[1] - rect.corners[0]))


def _containment_matrix(rects: list[DetectedRectangle]) -> np.ndarray:
    """containment[i, j] == True if rects[i] fully contains rects[j]."""
    n = len(rects)
    matrix = np.zeros((n, n), dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            inside = cv2.pointPolygonTest(
                rects[i].corners.astype(np.float32),
                rects[j].center,
                measureDist=False,
            )
            if inside >= 0:
                matrix[i, j] = True

    return matrix


def _min_corner_to_rect_distance(
    corners: np.ndarray,
    rect: DetectedRectangle,
) -> float:
    poly = rect.corners.astype(np.float32)
    distances = [
        float(cv2.pointPolygonTest(poly, (float(x), float(y)), measureDist=True))
        for x, y in corners
    ]
    return min(distances)


def _are_adjacent(
    rect_a: DetectedRectangle,
    rect_b: DetectedRectangle,
    edge_gap: float,
    center_factor: float,
) -> bool:
    dist_ab = _min_corner_to_rect_distance(rect_a.corners, rect_b)
    dist_ba = _min_corner_to_rect_distance(rect_b.corners, rect_a)

    boundary_gap = min(abs(dist_ab), abs(dist_ba))
    if 0.0 <= boundary_gap <= edge_gap:
        return True

    center_dist = float(
        np.linalg.norm(np.array(rect_a.center) - np.array(rect_b.center))
    )
    avg_width = (_rect_width(rect_a) + _rect_width(rect_b)) / 2.0
    return center_dist <= center_factor * avg_width


def _adjacency_matrix(
    rects: list[DetectedRectangle],
    edge_gap: float,
    center_factor: float,
) -> np.ndarray:
    n = len(rects)
    matrix = np.zeros((n, n), dtype=bool)

    for i in range(n):
        for j in range(i + 1, n):
            if _are_adjacent(rects[i], rects[j], edge_gap, center_factor):
                matrix[i, j] = True
                matrix[j, i] = True

    return matrix


def _containment_violation_count(
    ref_contains: np.ndarray,
    scene_contains: np.ndarray,
    ref_to_scene: dict[int, int],
) -> int:
    violations = 0
    for i, scene_i in ref_to_scene.items():
        for j, scene_j in ref_to_scene.items():
            if i == j:
                continue
            if ref_contains[i, j] != scene_contains[scene_i, scene_j]:
                violations += 1
    return violations


def _adjacency_violation_count(
    ref_adj: np.ndarray,
    scene_adj: np.ndarray,
    ref_to_scene: dict[int, int],
) -> int:
    violations = 0
    for i, scene_i in ref_to_scene.items():
        for j, scene_j in ref_to_scene.items():
            if i >= j:
                continue
            if ref_adj[i, j] != scene_adj[scene_i, scene_j]:
                violations += 1
    return violations


def _structural_violation_count(
    ref_contains: np.ndarray,
    scene_contains: np.ndarray,
    ref_adj: np.ndarray,
    scene_adj: np.ndarray,
    ref_to_scene: dict[int, int],
    config: RectangleMappingConfig,
) -> tuple[int, int, float]:
    containment_v = _containment_violation_count(
        ref_contains, scene_contains, ref_to_scene
    )
    adjacency_v = _adjacency_violation_count(ref_adj, scene_adj, ref_to_scene)
    penalty = (
        config.containment_penalty * containment_v
        + config.adjacency_penalty * adjacency_v
    )
    return containment_v, adjacency_v, penalty


def _pair_cost(
    ref: DetectedRectangle,
    scene: DetectedRectangle,
    ref_area_norm: float,
    scene_area_norm: float,
    ref_center_norm: np.ndarray,
    scene_center_norm: np.ndarray,
    config: RectangleMappingConfig,
) -> float:
    area_cost = abs(ref_area_norm - scene_area_norm)

    ref_aspect = _aspect_ratio(ref)
    scene_aspect = _aspect_ratio(scene)
    aspect_cost = abs(ref_aspect - scene_aspect) / max(ref_aspect, scene_aspect, 1e-6)

    center_cost = float(np.linalg.norm(ref_center_norm - scene_center_norm))

    angle_diff = abs(ref.angle - scene.angle) % 180.0
    angle_diff = min(angle_diff, 180.0 - angle_diff)
    angle_cost = angle_diff / 90.0

    return (
        config.weight_area * area_cost
        + config.weight_aspect * aspect_cost
        + config.weight_center * center_cost
        + config.weight_angle * angle_cost
    )


def _select_rectangles(
    rects: list[DetectedRectangle],
    max_rectangles: int | None,
) -> tuple[list[DetectedRectangle], list[int]]:
    """Select rectangles sorted largest-first."""
    indexed = list(enumerate(rects))
    indexed.sort(key=lambda item: item[1].area, reverse=True)

    if max_rectangles is not None:
        indexed = indexed[:max_rectangles]

    selected = [rect for _, rect in indexed]
    original_indices = [idx for idx, _ in indexed]
    return selected, original_indices


def build_cost_matrix(
    ref_rects: list[DetectedRectangle],
    scene_rects: list[DetectedRectangle],
    ref_shape: tuple[int, int],
    scene_shape: tuple[int, int],
    config: RectangleMappingConfig,
) -> np.ndarray:
    ref_h, ref_w = ref_shape
    scene_h, scene_w = scene_shape
    ref_image_area = float(ref_h * ref_w)
    scene_image_area = float(scene_h * scene_w)

    n_ref = len(ref_rects)
    n_scene = len(scene_rects)
    cost = np.zeros((n_ref, n_scene), dtype=np.float64)

    ref_area_norm = [_normalized_area(r, ref_image_area) for r in ref_rects]
    scene_area_norm = [_normalized_area(r, scene_image_area) for r in scene_rects]

    ref_centers = [_normalized_center(r, ref_w, ref_h) for r in ref_rects]
    scene_centers = [_normalized_center(r, scene_w, scene_h) for r in scene_rects]

    for i, ref in enumerate(ref_rects):
        for j, scene in enumerate(scene_rects):
            cost[i, j] = _pair_cost(
                ref,
                scene,
                ref_area_norm[i],
                scene_area_norm[j],
                ref_centers[i],
                scene_centers[j],
                config,
            )

    return cost


def _pad_cost_matrix(cost: np.ndarray, n: int) -> np.ndarray:
    n_ref, n_scene = cost.shape
    padded = np.full((n, n), fill_value=1e6, dtype=np.float64)
    padded[:n_ref, :n_scene] = cost
    return padded


def _finalize_matches(
    matches: list[RectangleMatch],
    ref_orig_idx: list[int],
    scene_orig_idx: list[int],
    n_ref: int,
    n_scene: int,
    matched_ref_ranks: set[int],
    matched_scene_ranks: set[int],
    cost_matrix: np.ndarray,
) -> RectangleMappingResult:
    matches.sort(key=lambda m: m.ref.area, reverse=True)
    for new_id, match in enumerate(matches, start=1):
        match.match_id = new_id

    unmatched_ref = [
        ref_orig_idx[i] for i in range(n_ref) if i not in matched_ref_ranks
    ]
    unmatched_scene = [
        scene_orig_idx[j] for j in range(n_scene) if j not in matched_scene_ranks
    ]

    return RectangleMappingResult(
        matches=matches,
        unmatched_ref=unmatched_ref,
        unmatched_scene=unmatched_scene,
        cost_matrix=cost_matrix,
        ref_indices=ref_orig_idx,
        scene_indices=scene_orig_idx,
    )


def match_rectangles_by_rank(
    ref_rects: list[DetectedRectangle],
    scene_rects: list[DetectedRectangle],
    ref_shape: tuple[int, int] | None = None,
    scene_shape: tuple[int, int] | None = None,
    config: RectangleMappingConfig | None = None,
) -> RectangleMappingResult:
    """
    Match by size rank: i-th largest reference -> i-th largest scene.

    Outer/large rectangles are paired first, then progressively smaller inner ones.
    """
    config = config or RectangleMappingConfig()

    if not ref_rects or not scene_rects:
        return RectangleMappingResult(
            matches=[],
            unmatched_ref=list(range(len(ref_rects))),
            unmatched_scene=list(range(len(scene_rects))),
            cost_matrix=np.empty((0, 0)),
            ref_indices=list(range(len(ref_rects))),
            scene_indices=list(range(len(scene_rects))),
        )

    ref_selected, ref_orig_idx = _select_rectangles(ref_rects, config.max_rectangles)
    scene_selected, scene_orig_idx = _select_rectangles(scene_rects, config.max_rectangles)

    ref_shape = _image_shape(ref_selected, ref_shape)
    scene_shape = _image_shape(scene_selected, scene_shape)

    ref_h, ref_w = ref_shape
    scene_h, scene_w = scene_shape
    ref_image_area = float(ref_h * ref_w)
    scene_image_area = float(scene_h * scene_w)

    n_ref = len(ref_selected)
    n_scene = len(scene_selected)
    n_pairs = min(n_ref, n_scene)

    ref_contains = _containment_matrix(ref_selected)
    scene_contains = _containment_matrix(scene_selected)
    ref_adj = _adjacency_matrix(
        ref_selected,
        config.adjacency_edge_gap,
        config.adjacency_center_factor,
    )
    scene_adj = _adjacency_matrix(
        scene_selected,
        config.adjacency_edge_gap,
        config.adjacency_center_factor,
    )

    cost_matrix = build_cost_matrix(
        ref_selected,
        scene_selected,
        ref_shape,
        scene_shape,
        config,
    )

    matches: list[RectangleMatch] = []
    ref_to_scene: dict[int, int] = {}
    matched_ref_ranks: set[int] = set()
    matched_scene_ranks: set[int] = set()

    for rank in range(n_pairs):
        ref = ref_selected[rank]
        scene = scene_selected[rank]

        ref_norm = _normalized_area(ref, ref_image_area)
        scene_norm = _normalized_area(scene, scene_image_area)
        area_diff = abs(ref_norm - scene_norm)

        if area_diff > config.rank_area_tolerance:
            continue

        ref_center_norm = _normalized_center(ref, ref_w, ref_h)
        scene_center_norm = _normalized_center(scene, scene_w, scene_h)

        base = _pair_cost(
            ref,
            scene,
            ref_norm,
            scene_norm,
            ref_center_norm,
            scene_center_norm,
            config,
        )

        trial_map = dict(ref_to_scene)
        trial_map[rank] = rank
        _, _, penalty = _structural_violation_count(
            ref_contains,
            scene_contains,
            ref_adj,
            scene_adj,
            trial_map,
            config,
        )
        total_cost = base + penalty

        if config.max_match_cost is not None and total_cost > config.max_match_cost:
            continue

        ref_to_scene[rank] = rank
        matched_ref_ranks.add(rank)
        matched_scene_ranks.add(rank)

        matches.append(
            RectangleMatch(
                match_id=len(matches) + 1,
                ref_index=ref_orig_idx[rank],
                scene_index=scene_orig_idx[rank],
                cost=total_cost,
                ref=ref,
                scene=scene,
            )
        )

    return _finalize_matches(
        matches,
        ref_orig_idx,
        scene_orig_idx,
        n_ref,
        n_scene,
        matched_ref_ranks,
        matched_scene_ranks,
        cost_matrix,
    )


def match_rectangles_hungarian(
    ref_rects: list[DetectedRectangle],
    scene_rects: list[DetectedRectangle],
    ref_shape: tuple[int, int] | None = None,
    scene_shape: tuple[int, int] | None = None,
    config: RectangleMappingConfig | None = None,
) -> RectangleMappingResult:
    """
    Match rectangles using Hungarian algorithm (global cost minimization).

    Note: with unequal counts this may NOT preserve size rank ordering.
    Prefer match_rectangles_by_rank for court template mapping.
    """
    config = config or RectangleMappingConfig()

    if not ref_rects or not scene_rects:
        return RectangleMappingResult(
            matches=[],
            unmatched_ref=list(range(len(ref_rects))),
            unmatched_scene=list(range(len(scene_rects))),
            cost_matrix=np.empty((0, 0)),
            ref_indices=list(range(len(ref_rects))),
            scene_indices=list(range(len(scene_rects))),
        )

    ref_selected, ref_orig_idx = _select_rectangles(ref_rects, config.max_rectangles)
    scene_selected, scene_orig_idx = _select_rectangles(scene_rects, config.max_rectangles)

    ref_shape = _image_shape(ref_selected, ref_shape)
    scene_shape = _image_shape(scene_selected, scene_shape)

    base_cost = build_cost_matrix(
        ref_selected,
        scene_selected,
        ref_shape,
        scene_shape,
        config,
    )

    n_ref = len(ref_selected)
    n_scene = len(scene_selected)
    n = max(n_ref, n_scene)

    ref_contains = _containment_matrix(ref_selected)
    scene_contains = _containment_matrix(scene_selected)
    ref_adj = _adjacency_matrix(
        ref_selected,
        config.adjacency_edge_gap,
        config.adjacency_center_factor,
    )
    scene_adj = _adjacency_matrix(
        scene_selected,
        config.adjacency_edge_gap,
        config.adjacency_center_factor,
    )

    row_ind, col_ind = linear_sum_assignment(_pad_cost_matrix(base_cost, n))

    preliminary: dict[int, int] = {}
    for r, c in zip(row_ind, col_ind):
        if r < n_ref and c < n_scene:
            preliminary[r] = c

    final_cost = base_cost.copy()
    for i in range(n_ref):
        for j in range(n_scene):
            trial_map = dict(preliminary)
            trial_map[i] = j
            _, _, penalty = _structural_violation_count(
                ref_contains,
                scene_contains,
                ref_adj,
                scene_adj,
                trial_map,
                config,
            )
            final_cost[i, j] = base_cost[i, j] + penalty

    row_ind, col_ind = linear_sum_assignment(_pad_cost_matrix(final_cost, n))

    matches: list[RectangleMatch] = []
    matched_ref_ranks: set[int] = set()
    matched_scene_ranks: set[int] = set()

    for r, c in zip(row_ind, col_ind):
        if r >= n_ref or c >= n_scene:
            continue

        cost_value = float(final_cost[r, c])
        if config.max_match_cost is not None and cost_value > config.max_match_cost:
            continue

        matches.append(
            RectangleMatch(
                match_id=len(matches) + 1,
                ref_index=ref_orig_idx[r],
                scene_index=scene_orig_idx[c],
                cost=cost_value,
                ref=ref_selected[r],
                scene=scene_selected[c],
            )
        )
        matched_ref_ranks.add(r)
        matched_scene_ranks.add(c)

    return _finalize_matches(
        matches,
        ref_orig_idx,
        scene_orig_idx,
        n_ref,
        n_scene,
        matched_ref_ranks,
        matched_scene_ranks,
        final_cost,
    )


def match_rectangles(
    ref_rects: list[DetectedRectangle],
    scene_rects: list[DetectedRectangle],
    ref_shape: tuple[int, int] | None = None,
    scene_shape: tuple[int, int] | None = None,
    config: RectangleMappingConfig | None = None,
) -> RectangleMappingResult:
    """Dispatch to rank or Hungarian matching based on config.matching_mode."""
    config = config or RectangleMappingConfig()

    if config.matching_mode == "rank":
        return match_rectangles_by_rank(
            ref_rects, scene_rects, ref_shape, scene_shape, config
        )
    if config.matching_mode == "hungarian":
        return match_rectangles_hungarian(
            ref_rects, scene_rects, ref_shape, scene_shape, config
        )

    raise ValueError(f"Unknown matching_mode: {config.matching_mode}")