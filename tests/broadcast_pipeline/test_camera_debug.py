from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from broadcast_pipeline.camera_debug import (
    CameraClusteringDebug,
    load_camera_clustering_debug,
    write_camera_clustering_debug,
)
from src.camera_assignemnt.embedding_cluster.models import ClusterResult, PipelineOutput


def _make_output(n: int = 4) -> PipelineOutput:
    results = [
        ClusterResult(
            scene_idx=i,
            scene_id=str(i // 2),
            frame_path=f"/tmp/scene_{i // 2}_frame_{10 + i}.jpg",
            cluster_id=i % 2,
            camera_id=f"cam_{i % 2}",
        )
        for i in range(n)
    ]
    reduced = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )
    labelings = [
        np.array([0, 0, 1, 1], dtype=np.int64),
        np.array([0, 0, 1, 1], dtype=np.int64),
    ]
    return PipelineOutput(
        results=results,
        reduced_matrix=reduced,
        dbscan_eps=0.5,
        method="ensemble",
        ensemble_vote_threshold=0.5,
        ensemble_noise_threshold=0.6,
        ensemble_member_weights={"hsv": 0.5, "resnet50": 0.5},
        ensemble_member_labelings=labelings,
        ensemble_member_names=["hsv", "resnet50"],
    )


def test_write_and_load_camera_clustering_debug(tmp_path: Path) -> None:
    frame_index = pd.DataFrame(
        {
            "scene_id": [0, 0, 1, 1],
            "frame_number": [10, 11, 20, 21],
            "frame_path": [
                str(tmp_path / "scene_0_frame_10.jpg"),
                str(tmp_path / "scene_0_frame_11.jpg"),
                str(tmp_path / "scene_1_frame_20.jpg"),
                str(tmp_path / "scene_1_frame_21.jpg"),
            ],
            "sample_role": ["camera"] * 4,
        }
    )
    output = _make_output()
    debug_path = tmp_path / "camera_clustering_debug.npz"
    write_camera_clustering_debug(debug_path, output, frame_index)

    loaded = load_camera_clustering_debug(debug_path)
    assert loaded is not None
    assert loaded.n_samples == 4
    assert loaded.method == "ensemble"
    assert loaded.member_names == ["hsv", "resnet50"]
    assert loaded.indices_for_scene(0) == [0, 1]
    assert loaded.indices_for_scene(1) == [2, 3]


def test_pairwise_cosine_from_compare_module(tmp_path: Path) -> None:
    from broadcast_pipeline.viz.camera_compare import _pairwise_cosine_stats

    reduced = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    stats = _pairwise_cosine_stats([0], [1], reduced)
    assert stats["mean"] == pytest.approx(1.0, abs=1e-5)


def test_build_global_projection_tsne_and_pipeline_labels(tmp_path: Path) -> None:
    from broadcast_pipeline.viz.camera_compare import build_global_projection

    output_dir = tmp_path / "pipeline"
    output_dir.mkdir()
    frame_index = pd.DataFrame(
        {
            "scene_id": [0, 0, 1, 1],
            "frame_number": [10, 20, 30, 40],
            "frame_path": [f"s{i}.jpg" for i in range(4)],
            "sample_role": ["camera"] * 4,
        }
    )
    results = [
        ClusterResult(scene_idx=i, scene_id=str(i // 2), frame_path=f"s{i}.jpg", cluster_id=i % 2)
        for i in range(4)
    ]
    output = PipelineOutput(
        results=results,
        reduced_matrix=np.array(
            [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0], [0.05, 0.95]],
            dtype=np.float32,
        ),
        method="ensemble",
        ensemble_member_labelings=[np.array([0, 0, 1, 1], dtype=np.int32)],
        ensemble_member_names=["hsv"],
    )
    write_camera_clustering_debug(output_dir / "camera_clustering_debug.npz", output, frame_index)
    pd.DataFrame(
        [
            {"scene_id": 0, "frame_number": 10, "camera_id": "cam_0", "cluster_id": 5},
            {"scene_id": 0, "frame_number": 20, "camera_id": "cam_0", "cluster_id": 5},
            {"scene_id": 1, "frame_number": 30, "camera_id": "cam_9", "cluster_id": 9},
            {"scene_id": 1, "frame_number": 40, "camera_id": "cam_9", "cluster_id": 9},
        ]
    ).to_csv(output_dir / "frame_camera_results.csv", index=False)
    pd.DataFrame(
        [
            {"scene_id": 0, "camera_id": "cam_0", "cluster_id": 5},
            {"scene_id": 1, "camera_id": "cam_9", "cluster_id": 9},
        ]
    ).to_csv(output_dir / "scene_assignments.csv", index=False)

    debug = load_camera_clustering_debug(output_dir / "camera_clustering_debug.npz")
    assert debug is not None
    projection = build_global_projection(debug, output_dir=output_dir, random_state=0)
    assert projection["projection_method"] == "tsne"
    assert projection["label_source"] == "frame_camera_results"
    assert projection["points"][0]["cluster_id"] == 5
    assert projection["points"][0]["pre_merge_cluster_id"] == 0
    assert projection["points"][2]["cluster_id"] == 9
    assert any(item["cluster_id"] == 5 for item in projection["legend"])


def test_scene_ids_for_cameras_highlights_all_scenes_on_camera() -> None:
    from broadcast_pipeline.viz.camera_compare import _scene_ids_for_cameras

    scene_assignments = pd.DataFrame(
        [
            {"scene_id": 0, "camera_id": "cam_0", "cluster_id": 0},
            {"scene_id": 1, "camera_id": "cam_0", "cluster_id": 0},
            {"scene_id": 2, "camera_id": "cam_1", "cluster_id": 1},
        ]
    )
    assert _scene_ids_for_cameras(scene_assignments, {"cam_0"}) == [0, 1]
    assert sorted(_scene_ids_for_cameras(scene_assignments, {"cam_0", "cam_1"})) == [0, 1, 2]


def test_compare_scenes_with_fixture(tmp_path: Path) -> None:
    from broadcast_pipeline.viz.camera_compare import SceneSelection, compare_scenes

    output_dir = tmp_path / "pipeline"
    output_dir.mkdir()
    frame_index = pd.DataFrame(
        {
            "scene_id": [0, 1],
            "frame_number": [10, 20],
            "frame_path": [
                str(output_dir / "scene_0_frame_10.jpg"),
                str(output_dir / "scene_1_frame_20.jpg"),
            ],
            "sample_role": ["camera", "camera"],
        }
    )
    output = PipelineOutput(
        results=[
            ClusterResult(
                scene_idx=0,
                scene_id="0",
                frame_path=str(output_dir / "scene_0_frame_10.jpg"),
                cluster_id=0,
                camera_id="cam_0",
            ),
            ClusterResult(
                scene_idx=1,
                scene_id="1",
                frame_path=str(output_dir / "scene_1_frame_20.jpg"),
                cluster_id=1,
                camera_id="cam_1",
            ),
        ],
        reduced_matrix=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        dbscan_eps=0.5,
        method="hsv",
        ensemble_member_labelings=[np.array([0, 1], dtype=np.int64)],
        ensemble_member_names=["hsv"],
    )
    write_camera_clustering_debug(output_dir / "camera_clustering_debug.npz", output, frame_index)

    pd.DataFrame(
        [
            {"scene_id": 0, "camera_id": "cam_0", "cluster_id": 0, "camera_vote_counts_json": "{'cam_0': 1}"},
            {"scene_id": 1, "camera_id": "cam_1", "cluster_id": 1, "camera_vote_counts_json": "{'cam_1': 1}"},
        ]
    ).to_csv(output_dir / "scene_assignments.csv", index=False)
    pd.DataFrame(
        [
            {
                "scene_id": 0,
                "frame_number": 10,
                "frame_path": str(output_dir / "scene_0_frame_10.jpg"),
                "camera_id": "cam_0",
                "cluster_id": 0,
            },
            {
                "scene_id": 1,
                "frame_number": 20,
                "frame_path": str(output_dir / "scene_1_frame_20.jpg"),
                "camera_id": "cam_1",
                "cluster_id": 1,
            },
        ]
    ).to_csv(output_dir / "frame_camera_results.csv", index=False)

    payload = compare_scenes(
        output_dir,
        [
            SceneSelection(camera_id="cam_0", scene_id=0),
            SceneSelection(camera_id="cam_1", scene_id=1),
        ],
        include_global=True,
    )
    assert payload["has_debug_artifact"] is True
    assert len(payload["pairwise"]) == 1
    assert payload["pairwise"][0]["mean_cosine"] == pytest.approx(1.0, abs=1e-5)
    assert "global_projection" in payload
    projection = payload["global_projection"]
    assert projection["projection_method"] in {"tsne", "pca", "centered_2d"}
    assert projection["label_source"] == "frame_camera_results"
    assert len(projection["points"]) == 2
    assert projection["points"][0]["cluster_id"] == 0
    assert projection["points"][0]["camera_id"] == "cam_0"
    assert projection["points"][0]["highlighted"] is True
    assert projection["highlight_scene_ids"] == [0, 1]
    assert "legend" in projection
