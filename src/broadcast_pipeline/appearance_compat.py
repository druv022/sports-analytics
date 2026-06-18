"""Load and apply scene appearance constraints for camera assignment."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from broadcast_pipeline.camera_debug import CameraClusteringDebug
from broadcast_pipeline.config import PipelineConfig
from src.camera_assignemnt.embedding_cluster.cluster import cluster_id_to_camera_id
from src.person_appearance.config import AppearanceConfig
from src.person_appearance.signature import normalize_signature, signatures_compatible
from src.person_appearance.types import SceneAppearance

FULL_COURT = "full_court"


def appearance_config_from_pipeline(config: PipelineConfig) -> AppearanceConfig:
    root = config.output_dir.resolve().parent.parent
    model_path = config.appearance_model_path or (root / "models" / "person_seg" / "yolo11n-seg.onnx")
    return AppearanceConfig(
        model_path=Path(model_path),
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


def _parse_bgr(raw: object) -> tuple[int, int, int] | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    if isinstance(raw, str):
        parsed = json.loads(raw) if raw else []
    else:
        parsed = raw
    if not parsed or len(parsed) < 3:
        return None
    return (int(parsed[0]), int(parsed[1]), int(parsed[2]))


def _parse_colors(raw: object) -> tuple[str, ...]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ()
    if isinstance(raw, str):
        parsed = json.loads(raw)
    else:
        parsed = raw
    return tuple(str(c) for c in parsed)


def load_scene_appearances(config: PipelineConfig) -> list[SceneAppearance]:
    path = config.artifact("scene_appearance")
    if not path.is_file():
        return []
    df = pd.read_csv(path)
    appearances: list[SceneAppearance] = []
    for row in df.itertuples(index=False):
        raw_signature = str(getattr(row, "appearance_signature", ""))
        appearances.append(
            SceneAppearance(
                scene_id=int(row.scene_id),
                scene_type=str(getattr(row, "scene_type", "closeup")),
                person_count=int(row.person_count),
                person_colors=_parse_colors(getattr(row, "person_colors_json", "[]")),
                appearance_signature=normalize_signature(raw_signature),
                confidence=float(getattr(row, "confidence", 0.0)),
                status=str(getattr(row, "status", "error")),  # type: ignore[arg-type]
                primary_bgr=_parse_bgr(getattr(row, "primary_bgr_json", None)),
                dominant_track_frames=int(getattr(row, "dominant_track_frames", 0)),
                dominant_track_median_area=int(getattr(row, "dominant_track_median_area", 0)),
            )
        )
    return appearances


def appearance_lookup(appearances: list[SceneAppearance]) -> dict[int, SceneAppearance]:
    return {app.scene_id: app for app in appearances}


def is_appearance_eligible(app: SceneAppearance | None) -> bool:
    return (
        app is not None
        and app.scene_type != FULL_COURT
        and app.status == "ok"
        and bool(app.appearance_signature)
    )


def _modal_cluster_for_scene(debug: CameraClusteringDebug, scene_id: int) -> int | None:
    indices = debug.indices_for_scene(scene_id)
    if not indices:
        return None
    clusters = [
        int(debug.final_cluster_id[i])
        for i in indices
        if int(debug.final_cluster_id[i]) >= 0
    ]
    if not clusters:
        return None
    return int(Counter(clusters).most_common(1)[0][0])


def _assign_split_group(
    updated: pd.DataFrame,
    split_group: list[int],
    config: PipelineConfig,
    clustering_debug: CameraClusteringDebug | None,
    next_cluster: int,
) -> int:
    if len(split_group) < config.camera_reconcile_min_split_size:
        return next_cluster

    if config.camera_reconcile_reuse_labels and clustering_debug is not None:
        modal_by_scene = {
            scene_id: _modal_cluster_for_scene(clustering_debug, scene_id)
            for scene_id in split_group
        }
        valid = {sid: cid for sid, cid in modal_by_scene.items() if cid is not None}
        if valid:
            for scene_id, cluster_id in valid.items():
                camera_id = cluster_id_to_camera_id(cluster_id) or f"cam_{cluster_id}"
                mask = updated["scene_id"] == scene_id
                updated.loc[mask, "camera_id"] = camera_id
                updated.loc[mask, "cluster_id"] = cluster_id
            return next_cluster

    new_cluster = next_cluster
    new_camera = f"cam_{new_cluster}"
    next_cluster += 1
    for scene_id in split_group:
        mask = updated["scene_id"] == scene_id
        updated.loc[mask, "camera_id"] = new_camera
        updated.loc[mask, "cluster_id"] = new_cluster
    return next_cluster


def clusters_have_appearance_conflict(
    left_cluster: int,
    right_cluster: int,
    frame_results: pd.DataFrame,
    appearances: dict[int, SceneAppearance],
    config: AppearanceConfig,
) -> bool:
    left_scenes = {
        int(s)
        for s in frame_results.loc[frame_results["cluster_id"] == left_cluster, "scene_id"].unique()
    }
    right_scenes = {
        int(s)
        for s in frame_results.loc[frame_results["cluster_id"] == right_cluster, "scene_id"].unique()
    }
    for left_id in left_scenes:
        left_app = appearances.get(left_id)
        if not is_appearance_eligible(left_app):
            continue
        for right_id in right_scenes:
            right_app = appearances.get(right_id)
            if not is_appearance_eligible(right_app):
                continue
            if left_app is not None and right_app is not None:
                if not signatures_compatible(left_app, right_app, config):
                    return True
    return False


def reconcile_scene_assignments(
    scene_assignments: pd.DataFrame,
    appearances: list[SceneAppearance],
    config: PipelineConfig,
    *,
    clustering_debug: CameraClusteringDebug | None = None,
) -> pd.DataFrame:
    if not config.camera_appearance_reconcile or not appearances:
        return scene_assignments

    app_cfg = appearance_config_from_pipeline(config)
    lookup = appearance_lookup(appearances)
    updated = scene_assignments.copy()
    next_cluster = int(updated["cluster_id"].max()) + 1 if not updated.empty else 0

    for camera_id, group in updated.groupby("camera_id"):
        eligible = [
            int(sid)
            for sid in group["scene_id"].tolist()
            if is_appearance_eligible(lookup.get(int(sid)))
        ]
        if len(eligible) < 2:
            continue

        groups: list[list[int]] = []
        for scene_id in eligible:
            app = lookup[scene_id]
            placed = False
            for compat_group in groups:
                if all(
                    signatures_compatible(app, lookup[other], app_cfg)
                    for other in compat_group
                ):
                    compat_group.append(scene_id)
                    placed = True
                    break
            if not placed:
                groups.append([scene_id])

        if len(groups) <= 1:
            continue

        groups.sort(key=len, reverse=True)
        for split_group in groups[1:]:
            next_cluster = _assign_split_group(
                updated,
                split_group,
                config,
                clustering_debug,
                next_cluster,
            )

    return updated.sort_values("scene_id")
