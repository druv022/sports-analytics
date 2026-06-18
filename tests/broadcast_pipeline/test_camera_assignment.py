from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import pytest

from broadcast_pipeline.camera_assignment import (
    _cluster_camera_samples,
    _run_clustering,
    assign_cameras_multi_frame,
    majority_vote_per_scene,
)
from broadcast_pipeline.camera_merge import _build_merge_map, _cosine_similarity
from broadcast_pipeline.config import PipelineConfig
from src.camera_assignemnt.embedding_cluster.models import ClusterResult, PipelineOutput


def test_majority_vote_unanimous_and_mid_frame_fallback():
    frame_results = pd.DataFrame(
        [
            {"scene_id": 0, "frame_number": 10, "camera_id": "cam_1", "cluster_id": 1},
            {"scene_id": 0, "frame_number": 50, "camera_id": "cam_1", "cluster_id": 1},
            {"scene_id": 1, "frame_number": 10, "camera_id": "cam_2", "cluster_id": 2},
            {"scene_id": 1, "frame_number": 50, "camera_id": "cam_3", "cluster_id": 3},
            {"scene_id": 1, "frame_number": 90, "camera_id": "cam_2", "cluster_id": 2},
        ]
    )
    frame_index = pd.DataFrame(
        [
            {"scene_id": 0, "frame_number": 10, "sample_role": "camera"},
            {"scene_id": 0, "frame_number": 50, "sample_role": "camera"},
            {"scene_id": 1, "frame_number": 10, "sample_role": "camera"},
            {"scene_id": 1, "frame_number": 50, "sample_role": "camera"},
            {"scene_id": 1, "frame_number": 90, "sample_role": "camera"},
        ]
    )
    out = majority_vote_per_scene(frame_results, frame_index, min_vote_share=0.6)
    assert out.loc[out.scene_id == 0, "assignment_method"].iloc[0] == "unanimous"
    scene1 = out[out.scene_id == 1].iloc[0]
    assert scene1["camera_id"] in {"cam_2", "cam_3"}
    assert "assignment_method" in scene1


def test_majority_vote_winner_aligned_cluster_id():
    frame_results = pd.DataFrame(
        [
            {"scene_id": 2, "frame_number": 10, "camera_id": "cam_1", "cluster_id": -1},
            {"scene_id": 2, "frame_number": 50, "camera_id": "cam_1", "cluster_id": 5},
            {"scene_id": 2, "frame_number": 90, "camera_id": "cam_2", "cluster_id": 2},
        ]
    )
    frame_index = pd.DataFrame(
        [
            {"scene_id": 2, "frame_number": 10, "sample_role": "camera"},
            {"scene_id": 2, "frame_number": 50, "sample_role": "camera"},
            {"scene_id": 2, "frame_number": 90, "sample_role": "camera"},
        ]
    )
    out = majority_vote_per_scene(frame_results, frame_index)
    row = out.iloc[0]
    assert row["camera_id"] == "cam_1"
    assert int(row["cluster_id"]) == 5


def test_build_merge_map_merges_similar_clusters():
    centroids = {
        1: np.array([1.0, 0.0], dtype=np.float32),
        2: np.array([0.99, 0.1], dtype=np.float32),
        3: np.array([0.0, 1.0], dtype=np.float32),
    }
    merge_map = _build_merge_map(centroids, similarity_threshold=0.85)
    assert merge_map[1] == merge_map[2]
    assert merge_map[3] != merge_map[1]
    assert _cosine_similarity(centroids[1], centroids[2]) >= 0.85


def test_build_merge_map_blocks_appearance_conflict():
    centroids = {
        1: np.array([1.0, 0.0], dtype=np.float32),
        2: np.array([0.99, 0.1], dtype=np.float32),
    }
    merge_map = _build_merge_map(
        centroids,
        similarity_threshold=0.85,
        can_merge=lambda _a, _b: False,
    )
    assert merge_map[1] != merge_map[2]


def test_build_merge_map_respects_max_group_size():
    centroids = {
        1: np.array([1.0, 0.0, 0.0], dtype=np.float32),
        2: np.array([0.99, 0.05, 0.0], dtype=np.float32),
        3: np.array([0.98, 0.08, 0.0], dtype=np.float32),
        4: np.array([0.97, 0.10, 0.0], dtype=np.float32),
    }
    merge_map = _build_merge_map(centroids, similarity_threshold=0.85, max_group_size=3)
    roots = {merge_map[cid] for cid in centroids}
    largest_group = max(
        sum(1 for cid, root in merge_map.items() if root == r) for r in roots
    )
    assert largest_group <= 3


def test_cluster_camera_samples_runs_single_global_pass(tmp_path: Path, monkeypatch):
    config = PipelineConfig(output_dir=tmp_path)
    frame_index = pd.DataFrame(
        [
            {
                "scene_id": 0,
                "frame_number": 0,
                "frame_path": str(tmp_path / "frames" / "camera" / "scene_0_frame_0.jpg"),
                "sample_role": "camera",
                "seconds": 0.0,
            }
        ]
    )
    frame_index_path = config.artifact("frame_index")
    frame_index_path.parent.mkdir(parents=True, exist_ok=True)
    frame_index.to_csv(frame_index_path, index=False)

    sample = object()
    calls: list[list] = []

    def fake_load(*_args, **_kwargs):
        return [sample]

    def fake_run(_config, _frame_index, _path, samples, _apply_temporal):
        calls.append(samples)
        return PipelineOutput(method="ensemble", results=[])

    monkeypatch.setattr(
        "broadcast_pipeline.camera_assignment.load_scene_samples",
        fake_load,
    )
    monkeypatch.setattr(
        "broadcast_pipeline.camera_assignment._run_clustering",
        fake_run,
    )

    _cluster_camera_samples(config, frame_index, frame_index_path)
    assert len(calls) == 1
    assert calls[0] == [sample]


def test_run_clustering_does_not_pass_appearance_features(tmp_path: Path, monkeypatch):
    config = PipelineConfig(output_dir=tmp_path)
    frame_index_path = config.artifact("frame_index")
    frame_index_path.parent.mkdir(parents=True, exist_ok=True)
    frame_index_path.write_text("scene_id,frame_number,frame_path,sample_role,seconds\n")

    captured: dict = {}

    def fake_assign_cameras(pipeline_cfg, **_kwargs):
        captured["appearance_features_by_scene"] = pipeline_cfg.appearance_features_by_scene
        captured["appearance_feature_weight"] = pipeline_cfg.appearance_feature_weight
        return PipelineOutput(method="ensemble")

    monkeypatch.setattr(
        "broadcast_pipeline.camera_assignment.assign_cameras",
        fake_assign_cameras,
    )

    _run_clustering(config, pd.DataFrame(), frame_index_path, [], apply_temporal=False)
    assert captured["appearance_features_by_scene"] is None
    assert captured["appearance_feature_weight"] == 0.0


def test_pipeline_config_removed_stratification_fields():
    config = PipelineConfig()
    assert not hasattr(config, "camera_stratify_by_appearance")
    assert not hasattr(config, "camera_stratify_by_scene_type")
    assert not hasattr(config, "appearance_feature_weight")
    assert config.camera_reconcile_min_split_size == 2
    assert config.camera_reconcile_reuse_labels is True


def _write_scene_appearance_csv(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "scene_id": 0,
                "scene_type": "closeup",
                "person_count": 1,
                "person_colors_json": '["blue"]',
                "appearance_signature": "blue",
                "confidence": 0.9,
                "status": "ok",
            },
            {
                "scene_id": 1,
                "scene_type": "closeup",
                "person_count": 1,
                "person_colors_json": '["blue"]',
                "appearance_signature": "blue",
                "confidence": 0.9,
                "status": "ok",
            },
            {
                "scene_id": 2,
                "scene_type": "closeup",
                "person_count": 1,
                "person_colors_json": '["red"]',
                "appearance_signature": "red",
                "confidence": 0.9,
                "status": "ok",
            },
        ]
    ).to_csv(path, index=False)


def _build_e2e_fixture(tmp_path: Path) -> tuple[PipelineConfig, pd.DataFrame, PipelineOutput]:
    frames_dir = tmp_path / "frames" / "camera"
    frames_dir.mkdir(parents=True)
    rows: list[dict] = []
    results: list[ClusterResult] = []
    vectors: list[np.ndarray] = []

    for scene_id in (0, 1, 2):
        cluster_id = 0
        camera_id = "cam_0"
        vector = np.array([1.0, 0.0], dtype=np.float32)
        for frame_number in (0, 10):
            frame_path = frames_dir / f"scene_{scene_id}_frame_{frame_number}.jpg"
            image = np.zeros((40, 40, 3), dtype=np.uint8)
            cv2.imwrite(str(frame_path), image)
            rows.append(
                {
                    "scene_id": scene_id,
                    "frame_number": frame_number,
                    "frame_path": str(frame_path),
                    "sample_role": "camera",
                    "seconds": float(frame_number),
                }
            )
            results.append(
                ClusterResult(
                    scene_idx=len(results),
                    scene_id=str(scene_id),
                    frame_path=str(frame_path),
                    cluster_id=cluster_id,
                    camera_id=camera_id,
                )
            )
            vectors.append(vector)

    frame_index = pd.DataFrame(rows)
    output = PipelineOutput(
        results=results,
        reduced_matrix=np.vstack(vectors).astype(np.float32),
        method="ensemble",
        ensemble_member_names=["mock"],
        ensemble_member_labelings=[np.array([r.cluster_id for r in results], dtype=np.int32)],
    )

    config = PipelineConfig(output_dir=tmp_path)
    pd.DataFrame(
        [
            {"scene_id": 0, "scene_type": "closeup", "vote_counts_json": "{}"},
            {"scene_id": 1, "scene_type": "closeup", "vote_counts_json": "{}"},
            {"scene_id": 2, "scene_type": "closeup", "vote_counts_json": "{}"},
        ]
    ).to_csv(config.artifact("scene_types"), index=False)
    _write_scene_appearance_csv(config.artifact("scene_appearance"))
    return config, frame_index, output


def test_assign_cameras_multi_frame_merge_vote_reconcile_e2e(tmp_path: Path):
    config, frame_index, mock_output = _build_e2e_fixture(tmp_path)
    config.camera_reconcile_min_split_size = 1
    for idx, result in enumerate(mock_output.results or []):
        if result.scene_id == "2":
            result.cluster_id = 1
            result.camera_id = "cam_1"
            mock_output.reduced_matrix[idx] = np.array([0.99, 0.01], dtype=np.float32)
    mock_output.ensemble_member_labelings = [
        np.array([r.cluster_id for r in mock_output.results], dtype=np.int32)
    ]

    def _force_merge(frame_results, _config, _scene_types, **_kwargs):
        updated = frame_results.copy()
        updated["cluster_id"] = 0
        updated["camera_id"] = "cam_0"
        return updated, {"merged": True, "groups": [[0, 1]]}

    with patch(
        "broadcast_pipeline.camera_assignment._cluster_camera_samples",
        return_value=mock_output,
    ), patch(
        "broadcast_pipeline.camera_assignment.apply_closeup_cluster_merge",
        side_effect=_force_merge,
    ):
        scene_assignments, _frame_assignments = assign_cameras_multi_frame(config, frame_index)

    by_scene = scene_assignments.set_index("scene_id")
    assert by_scene.loc[1, "camera_id"] == "cam_0"
    assert by_scene.loc[2, "camera_id"] == "cam_1"


def test_assign_cameras_multi_frame_reconcile_with_premerge_reuse(tmp_path: Path):
    config, frame_index, mock_output = _build_e2e_fixture(tmp_path)
    config.camera_reconcile_min_split_size = 1
    for idx, result in enumerate(mock_output.results or []):
        if result.scene_id == "2":
            result.cluster_id = 1
            result.camera_id = "cam_1"
            mock_output.reduced_matrix[idx] = np.array([0.99, 0.01], dtype=np.float32)
    mock_output.ensemble_member_labelings = [
        np.array([r.cluster_id for r in mock_output.results], dtype=np.int32)
    ]

    def _force_merge(frame_results, _config, _scene_types, **_kwargs):
        updated = frame_results.copy()
        updated["cluster_id"] = 0
        updated["camera_id"] = "cam_0"
        return updated, {"merged": True, "groups": [[0, 1]]}

    with patch(
        "broadcast_pipeline.camera_assignment._cluster_camera_samples",
        return_value=mock_output,
    ), patch(
        "broadcast_pipeline.camera_assignment.apply_closeup_cluster_merge",
        side_effect=_force_merge,
    ):
        scene_assignments, _ = assign_cameras_multi_frame(config, frame_index)

    by_scene = scene_assignments.set_index("scene_id")
    assert by_scene.loc[1, "camera_id"] == "cam_0"
    assert by_scene.loc[2, "camera_id"] == "cam_1"


def test_assign_cameras_multi_frame_reconcile_disabled_keeps_shared_camera(tmp_path: Path):
    config, frame_index, mock_output = _build_e2e_fixture(tmp_path)
    config.camera_appearance_reconcile = False

    with patch(
        "broadcast_pipeline.camera_assignment._cluster_camera_samples",
        return_value=mock_output,
    ):
        scene_assignments, _ = assign_cameras_multi_frame(config, frame_index)

    by_scene = scene_assignments.set_index("scene_id")
    assert by_scene.loc[1, "camera_id"] == by_scene.loc[2, "camera_id"]


def test_assign_cameras_multi_frame_singleton_split_skipped(tmp_path: Path):
    config, frame_index, mock_output = _build_e2e_fixture(tmp_path)
    config.camera_reconcile_min_split_size = 2

    only_scene2 = [r for r in mock_output.results or [] if r.scene_id == "2"]
    mock_output.results = [r for r in mock_output.results or [] if r.scene_id != "1"]
    mock_output.reduced_matrix = mock_output.reduced_matrix[[0, 1, 4, 5]]
    mock_output.ensemble_member_labelings = [
        np.array([r.cluster_id for r in mock_output.results], dtype=np.int32)
    ]

    pd.DataFrame(
        [
            {"scene_id": 0, "scene_type": "closeup", "vote_counts_json": "{}"},
            {"scene_id": 2, "scene_type": "closeup", "vote_counts_json": "{}"},
        ]
    ).to_csv(config.artifact("scene_types"), index=False)
    pd.DataFrame(
        [
                {
                    "scene_id": 0,
                    "scene_type": "closeup",
                    "person_count": 1,
                    "person_colors_json": '["blue"]',
                    "appearance_signature": "blue",
                    "confidence": 0.9,
                    "status": "ok",
                },
                {
                    "scene_id": 2,
                    "scene_type": "closeup",
                    "person_count": 1,
                    "person_colors_json": '["red"]',
                    "appearance_signature": "red",
                    "confidence": 0.9,
                    "status": "ok",
                },
        ]
    ).to_csv(config.artifact("scene_appearance"), index=False)

    frame_index = frame_index[frame_index["scene_id"] != 1].reset_index(drop=True)

    with patch(
        "broadcast_pipeline.camera_assignment._cluster_camera_samples",
        return_value=mock_output,
    ):
        scene_assignments, _ = assign_cameras_multi_frame(config, frame_index)

    by_scene = scene_assignments.set_index("scene_id")
    assert by_scene.loc[0, "camera_id"] == by_scene.loc[2, "camera_id"]
