from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from broadcast_pipeline.appearance_compat import (
    clusters_have_appearance_conflict,
    reconcile_scene_assignments,
)
from broadcast_pipeline.appearance_filter import analyze_scenes
from broadcast_pipeline.camera_debug import CameraClusteringDebug
from broadcast_pipeline.config import PipelineConfig
from src.person_appearance.config import AppearanceConfig
from src.person_appearance.extractor import analyze_frame
from src.person_appearance.segmenter import MockPersonSegmenter
from src.person_appearance.types import SceneAppearance


def _scene(scene_id: int, count: int, colors: tuple[str, ...], scene_type: str = "closeup") -> SceneAppearance:
    primary = colors[0] if colors else ""
    return SceneAppearance(
        scene_id=scene_id,
        scene_type=scene_type,
        person_count=count,
        person_colors=(primary,) if primary else (),
        appearance_signature=primary,
        confidence=0.9,
        status="ok",
    )


def _debug_for_scenes(cluster_by_scene: dict[int, int]) -> CameraClusteringDebug:
    scene_ids: list[int] = []
    cluster_ids: list[int] = []
    for scene_id, cluster_id in sorted(cluster_by_scene.items()):
        scene_ids.extend([scene_id, scene_id])
        cluster_ids.extend([cluster_id, cluster_id])
    n = len(scene_ids)
    return CameraClusteringDebug(
        reduced_matrix=np.zeros((n, 2), dtype=np.float32),
        scene_ids=np.array(scene_ids, dtype=np.int32),
        frame_numbers=np.array([0, 10] * len(cluster_by_scene), dtype=np.int32),
        final_cluster_id=np.array(cluster_ids, dtype=np.int32),
        final_camera_id=np.array([f"cam_{c}" for c in cluster_ids], dtype="U32"),
        member_labelings=np.zeros((1, n), dtype=np.int32),
        member_names=["mock"],
        method="ensemble",
        link_threshold=None,
        noise_threshold=None,
        dbscan_eps=None,
        member_weights={},
    )


def test_analyze_scenes_with_mock_segmenter(tmp_path: Path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    image_path = frames_dir / "scene_0_frame_0.jpg"
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    image[:, :] = (30, 30, 220)
    cv2.imwrite(str(image_path), image)

    frame_index = pd.DataFrame(
        [
            {
                "scene_id": 0,
                "frame_number": 0,
                "frame_path": str(image_path),
                "sample_role": "camera",
                "seconds": 0.0,
            }
        ]
    )
    scene_types = pd.DataFrame([{"scene_id": 0, "scene_type": "closeup", "vote_counts_json": "{}"}])
    scene_types.to_csv(tmp_path / "scene_types.csv", index=False)

    config = PipelineConfig(output_dir=tmp_path, appearance_enabled=True)
    segmenter = MockPersonSegmenter(detections=[((20, 20, 80, 100), 0.95)])
    frame_df, scene_df = analyze_scenes(config, frame_index, segmenter=segmenter)

    assert len(frame_df) == 1
    assert len(scene_df) == 1
    assert scene_df.iloc[0]["person_count"] == 1
    assert ":" not in str(scene_df.iloc[0]["appearance_signature"])
    assert "primary_bgr_json" in frame_df.columns


def test_reconcile_splits_incompatible_closeups():
    appearances = [
        _scene(1, 2, ("blue", "white")),
        _scene(2, 2, ("red", "yellow")),
    ]
    scene_assignments = pd.DataFrame(
        [
            {"scene_id": 1, "camera_id": "cam_0", "cluster_id": 0},
            {"scene_id": 2, "camera_id": "cam_0", "cluster_id": 0},
        ]
    )
    config = PipelineConfig(camera_reconcile_min_split_size=1)
    debug = _debug_for_scenes({1: 0, 2: 1})
    updated = reconcile_scene_assignments(
        scene_assignments,
        appearances,
        config,
        clustering_debug=debug,
    )
    camera_ids = set(updated["camera_id"].tolist())
    assert len(camera_ids) == 2
    assert updated.loc[updated.scene_id == 2, "camera_id"].iloc[0] == "cam_1"


def test_reconcile_reuses_premerge_cluster_label():
    appearances = [
        _scene(1, 1, ("blue",)),
        _scene(4, 1, ("blue",)),
        _scene(2, 1, ("red",)),
        _scene(3, 1, ("red",)),
    ]
    scene_assignments = pd.DataFrame(
        [
            {"scene_id": 1, "camera_id": "cam_0", "cluster_id": 0},
            {"scene_id": 2, "camera_id": "cam_0", "cluster_id": 0},
            {"scene_id": 3, "camera_id": "cam_0", "cluster_id": 0},
            {"scene_id": 4, "camera_id": "cam_0", "cluster_id": 0},
        ]
    )
    config = PipelineConfig(camera_reconcile_min_split_size=2)
    debug = _debug_for_scenes({1: 0, 4: 0, 2: 5, 3: 5})
    updated = reconcile_scene_assignments(
        scene_assignments,
        appearances,
        config,
        clustering_debug=debug,
    )
    assert updated.loc[updated.scene_id == 2, "camera_id"].iloc[0] == "cam_5"
    assert updated.loc[updated.scene_id == 3, "camera_id"].iloc[0] == "cam_5"
    assert updated.loc[updated.scene_id == 1, "camera_id"].iloc[0] == "cam_0"
    assert "cam_6" not in set(updated["camera_id"].tolist())


def test_reconcile_skips_singleton_split():
    appearances = [
        _scene(1, 2, ("blue", "white")),
        _scene(2, 2, ("red", "yellow")),
        _scene(3, 2, ("blue", "white")),
    ]
    scene_assignments = pd.DataFrame(
        [
            {"scene_id": 1, "camera_id": "cam_0", "cluster_id": 0},
            {"scene_id": 2, "camera_id": "cam_0", "cluster_id": 0},
            {"scene_id": 3, "camera_id": "cam_0", "cluster_id": 0},
        ]
    )
    config = PipelineConfig(camera_reconcile_min_split_size=2)
    debug = _debug_for_scenes({1: 0, 2: 1, 3: 0})
    updated = reconcile_scene_assignments(
        scene_assignments,
        appearances,
        config,
        clustering_debug=debug,
    )
    assert updated.loc[updated.scene_id == 2, "camera_id"].iloc[0] == "cam_0"


def test_clusters_have_appearance_conflict():
    appearances = {
        1: _scene(1, 2, ("blue", "white")),
        2: _scene(2, 2, ("red", "yellow")),
    }
    frame_results = pd.DataFrame(
        [
            {"scene_id": 1, "cluster_id": 0},
            {"scene_id": 2, "cluster_id": 1},
        ]
    )
    config = AppearanceConfig()
    assert clusters_have_appearance_conflict(0, 1, frame_results, appearances, config)
