from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from broadcast_pipeline.config import PipelineConfig
from broadcast_pipeline.progress import ProgressTracker
from src.person_appearance.config import AppearanceConfig
from src.person_appearance.extractor import (
    analyze_frame,
    build_scene_appearances,
    default_segmenter,
    frame_appearances_to_dataframe,
    load_frame_image,
    scene_appearances_to_dataframe,
)
from src.person_appearance.segmenter import PersonSegmenter


def _appearance_config(config: PipelineConfig, project_root: Path | None = None) -> AppearanceConfig:
    root = project_root or config.output_dir.resolve().parent.parent
    model_path = config.appearance_model_path
    if model_path is None:
        resolved = root / "models" / "person_seg" / "yolo11n-seg.onnx"
    else:
        resolved = Path(model_path)
    return AppearanceConfig(
        model_path=resolved,
        imgsz=config.appearance_imgsz,
        min_confidence=config.appearance_min_confidence,
        color_tolerance=config.appearance_color_tolerance,
        count_slack=config.appearance_count_slack,
        min_sequence_match=config.appearance_min_sequence_match,
        color_method=config.appearance_color_method,
        color_mask_region=config.appearance_color_mask_region,
        dominant_track_policy=config.appearance_dominant_track_policy,
        track_match_iou=config.appearance_track_match_iou,
        mask_erode_px=config.appearance_mask_erode_px,
        histogram_h_bins=config.appearance_histogram_h_bins,
    )


def _load_scene_type_lookup(config: PipelineConfig) -> dict[int, str]:
    path = config.artifact("scene_types")
    if not path.is_file():
        return {}
    scene_types = pd.read_csv(path)
    return {
        int(row.scene_id): str(row.scene_type)
        for row in scene_types.itertuples(index=False)
    }


def _jsonify_list_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = out[column].apply(json.dumps)
    return out


def analyze_scenes(
    config: PipelineConfig,
    frame_index: pd.DataFrame,
    *,
    segmenter: PersonSegmenter | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    appearance_cfg = _appearance_config(config)
    if segmenter is None:
        segmenter = default_segmenter(appearance_cfg)

    scene_type_lookup = _load_scene_type_lookup(config)
    camera_df = frame_index[frame_index["sample_role"] == "camera"].copy()
    frame_results: list = []
    progress = ProgressTracker(len(camera_df), "Appearance analysis")

    for row in camera_df.itertuples(index=False):
        frame_path = str(getattr(row, "frame_path"))
        image = load_frame_image(frame_path)
        if image is None:
            progress.advance()
            continue
        frame_results.append(
            analyze_frame(
                image,
                scene_id=int(getattr(row, "scene_id")),
                frame_number=int(getattr(row, "frame_number")),
                frame_path=frame_path,
                config=appearance_cfg,
                segmenter=segmenter,
            )
        )
        progress.advance()

    scene_results = build_scene_appearances(frame_results, scene_type_lookup, appearance_cfg)
    frame_df = _jsonify_list_columns(
        frame_appearances_to_dataframe(frame_results),
        ("person_colors_json", "primary_bgr_json"),
    )
    scene_df = _jsonify_list_columns(
        scene_appearances_to_dataframe(scene_results),
        ("person_colors_json", "primary_bgr_json"),
    )
    return frame_df, scene_df
