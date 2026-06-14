
from src.camera_split_segment import VIDEO_PATH, extract_frames_from_video

import cv2


# def main():
#     print("Step 1: Scene detection and frame extraction")
#     # extract_frames_from_video(VIDEO_PATH)

#     print("\nStep 2: Camera assignment")
# def main():
#     image_path = "data/scene_samples/scene_0_frame_374.jpg"
#     reference_map_path = "data/Court_dimension.png"

from src.camera_assignemnt.approach_2.config import RectangleConfig
from src.camera_assignemnt.approach_2.rectangle_detector import detect_rectangles, load_image
from src.camera_assignemnt.approach_2.rectangle_mapper import (
    RectangleMappingConfig,
    match_rectangles,
)
from src.camera_assignemnt.approach_2.mapping_visualizer import visualize_rectangle_mapping
from src.camera_assignemnt.approach_2.rectangle_visualizer import visualize_rectangles


def main():
    reference_path = "data/Court_dimension.png"
    scene_path = "data/scene_samples/scene_25_frame_13090.jpg"

    detect_config = RectangleConfig(
        mode="line_quad",
        require_size_filter=True,
        min_area_ratio=0.008,
        max_area_ratio=0.08,
        max_aspect_ratio=6.0,
        require_line_alignment=True,
        line_use_perimeter_support=True,
        line_min_perimeter_support=0.70,
        line_edge_samples=21,
        line_distance_thresh_px=8.0,
        line_angle_thresh_deg=20.0,
        line_filter_if_no_lines="reject",
        line_source="white_hough",
        use_court_mask_for_lines=True,
        use_min_area_fallback=False,
        line_quad_max_candidates=15,
    )

    ref_detect_config = RectangleConfig(
        mode="contour",
        contour_edge_source="canny_hough",
        fit_quad_from_polygon=True,
        approx_min_vertices=4,
        approx_max_vertices=6,
        canny_low=30,
        canny_high=120,
        min_area_ratio=0.01,
        use_min_area_fallback=True,
        max_aspect_ratio=5.0,
        require_line_alignment=False,
        line_source="canny_hough",
        use_court_mask_for_lines=False,
    )


    ref_image = load_image(reference_path)
    scene_image = load_image(scene_path)

    ref_rects = detect_rectangles(ref_image, ref_detect_config)
    scene_rects = detect_rectangles(scene_image, detect_config)

    # Outer court (ref[0]) has no scene equivalent under scene size filter — skip for matching.
    ref_rects_for_match = ref_rects[1:]

    ref_area = ref_image.shape[0] * ref_image.shape[1]
    scene_area = scene_image.shape[0] * scene_image.shape[1]

    print(f"Reference rectangles: {len(ref_rects)} (matching inner {len(ref_rects_for_match)})")
    for i, r in enumerate(ref_rects[:6]):
        print(f"  ref[{i}] norm_area={r.area / ref_area:.4f}")

    print(f"\nScene rectangles: {len(scene_rects)}")
    for i, r in enumerate(scene_rects[:6]):
        print(f"  scene[{i}] norm_area={r.area / scene_area:.4f}")

    mapping_config = RectangleMappingConfig(
        matching_mode="hungarian",
        weight_area=0.50,
        weight_aspect=0.10,
        weight_center=0.35,
        weight_angle=0.05,
        containment_penalty=0.0,
        adjacency_penalty=0.0,
        max_match_cost=0.15,
        max_rectangles=10,
    )

    mapping = match_rectangles(
        ref_rects_for_match,
        scene_rects,
        ref_shape=ref_image.shape[:2],
        scene_shape=scene_image.shape[:2],
        config=mapping_config,
    )

    # Map match indices back onto full ref_rects (index 0 = outer court, skipped above).
    for m in mapping.matches:
        m.ref_index += 1
    mapping.unmatched_ref = [0] + [i + 1 for i in mapping.unmatched_ref]

    print(f"\nMatched pairs (Hungarian, cost <= {mapping_config.max_match_cost}): {len(mapping.matches)}")
    for m in mapping.matches:
        ref_norm = m.ref.area / ref_area
        scene_norm = m.scene.area / scene_area
        print(
            f"  #{m.match_id}: ref[{m.ref_index}] norm={ref_norm:.4f} "
            f"-> scene[{m.scene_index}] norm={scene_norm:.4f} "
            f"cost={m.cost:.3f}"
        )

    print(f"\nUnmatched ref indices: {mapping.unmatched_ref}")
    print(f"Unmatched scene indices: {mapping.unmatched_scene}")

    visualize_rectangle_mapping(
        reference_path,
        scene_path,
        mapping,
        ref_rects,
        scene_rects,
        save_path="data/verification/rectangle_mapping.jpg",
        show=True,
    )
    # visualize_rectangles(ref_image, ref_detect_config, save_path="data/verification/rectangle_mapping.jpg", show=True)


if __name__ == "__main__":
    main()