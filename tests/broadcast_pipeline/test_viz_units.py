from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from broadcast_pipeline.camera_debug import write_camera_clustering_debug
from broadcast_pipeline.viz.camera_collage import (
    CameraCollageLoadError,
    load_camera_collage_bundle,
    pick_scene_slots,
)
from broadcast_pipeline.viz.camera_collage_render import CollageRenderConfig, render_camera_collages
from broadcast_pipeline.viz.data_loader import TimelineLoadError, load_timeline_bundle
from broadcast_pipeline.viz.frame_paths import resolve_frame_under_output
from broadcast_pipeline.viz.frame_ranges import parse_frame_ranges
from broadcast_pipeline.viz.appearance_api import clear_segmenter_cache, run_appearance_on_bytes
from broadcast_pipeline.viz.appearance_loader import load_appearance_bundle
from broadcast_pipeline.viz.pipeline_ocr import (
    PipelineOcrLoadError,
    load_pipeline_ocr_index,
)
from broadcast_pipeline.viz.server import create_app
from src.camera_assignemnt.embedding_cluster.models import ClusterResult, PipelineOutput
from src.person_appearance.segmenter import MockPersonSegmenter


def test_parse_frame_ranges():
    assert parse_frame_ranges("10-12;20") == [10, 11, 12, 20]
    assert parse_frame_ranges("5") == [5]
    assert parse_frame_ranges("") == []


def _write_fixture_bundle(tmp_path: Path) -> Path:
    output_dir = tmp_path / "pipeline"
    output_dir.mkdir()

    (output_dir / "aggregated_complete.csv").write_text(
        "camera_id,text,text_kind,mapped_complete_text,total_duration_sec,frame_ranges,"
        "n_frames_present,n_frames_good,n_frames_partial,n_frames_enriched,dominant_readability\n"
        "cam_0,PLAYER,complete,PLAYER,1.5,10-12,3,2,1,1,good\n"
        "cam_1,SET,complete,SET,0.5,20,1,1,0,0,good\n",
        encoding="utf-8",
    )
    (output_dir / "aggregated_partial.csv").write_text(
        "camera_id,text,text_kind,mapped_complete_text,total_duration_sec,frame_ranges,"
        "n_frames_present,n_frames_good,n_frames_partial,n_frames_enriched,dominant_readability\n"
        "cam_0,PLA,partial,PLAYER,0.3,11,1,0,1,0,partial\n",
        encoding="utf-8",
    )
    (output_dir / "frame_text_associated.csv").write_text(
        "scene_id,frame_number,camera_id,raw_text,text_kind,mapped_complete_text,"
        "mapping_confidence,readability_label,bbox_json,enrich_applied,ocr_raw_text\n"
        "0,10,cam_0,PLAYER,complete,PLAYER,1.0,good,\"[10,10,100,40]\",false,\n"
        "0,11,cam_0,PLA,partial,PLAYER,0.8,partial,\"[12,12,90,38]\",false,\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "scene_id": 0,
                "frame_number": 10,
                "seconds": 0.33,
                "camera_id": "cam_0",
                "words_json": json.dumps(["PLAYER"]),
                "detections_json": json.dumps(
                    [
                        {
                            "text": "PLAYER",
                            "confidence": 0.95,
                            "bbox": [10, 10, 100, 40],
                            "source": "ocr",
                        }
                    ]
                ),
                "verdict": "readable",
                "used_unk": False,
            },
            {
                "scene_id": 0,
                "frame_number": 11,
                "seconds": 0.66,
                "camera_id": "cam_0",
                "words_json": json.dumps(["PLA"]),
                "detections_json": json.dumps(
                    [
                        {
                            "text": "PLA",
                            "confidence": 0.72,
                            "bbox": [12, 12, 90, 38],
                            "source": "ocr",
                        }
                    ]
                ),
                "verdict": "readable",
                "used_unk": False,
            },
        ]
    ).to_csv(output_dir / "frame_ocr.csv", index=False)
    (output_dir / "approved_text_reference.csv").write_text(
        "complete_text,approved,first_seen_scene_id,first_seen_frame,discovery_count\n"
        "PLAYER,true,0,10,2\n"
        "GAME,false,0,15,1\n",
        encoding="utf-8",
    )
    (output_dir / "frame_index.csv").write_text(
        "scene_id,frame_number,seconds,frame_path,sample_role\n"
        "0,10,0.33,frames/scene_0_frame_10.jpg,ocr\n"
        "0,11,0.66,frames/scene_0_frame_11.jpg,ocr\n"
        "0,10,0.33,frames/scene_0_frame_10.jpg,camera\n"
        "0,50,1.66,frames/scene_0_frame_50.jpg,camera\n"
        "0,90,3.00,frames/scene_0_frame_90.jpg,camera\n"
        "1,20,0.66,frames/scene_1_frame_20.jpg,camera\n"
        "1,40,1.33,frames/scene_1_frame_40.jpg,camera\n",
        encoding="utf-8",
    )
    (output_dir / "scene_assignments.csv").write_text(
        "scene_id,camera_id,cluster_id,camera_vote_counts_json\n"
        '0,cam_0,0,"{""cam_0"": 3}"\n'
        '1,cam_1,1,"{""cam_1"": 2}"\n',
        encoding="utf-8",
    )
    (output_dir / "scenes.json").write_text(
        '[{"scene_id":0,"start_frame":0,"end_frame":100,"start_sec":0.0,"end_sec":3.33},'
        '{"scene_id":1,"start_frame":100,"end_frame":200,"start_sec":3.33,"end_sec":6.66}]',
        encoding="utf-8",
    )
    (output_dir / "frame_assignments.csv").write_text(
        "scene_id,frame_number,seconds,frame_path,sample_role,camera_id,cluster_id\n"
        "0,10,0.33,frames/ocr/scene_0_frame_10.jpg,ocr,cam_0,0\n"
        "0,11,0.66,frames/ocr/scene_0_frame_11.jpg,ocr,cam_0,0\n",
        encoding="utf-8",
    )
    (output_dir / "pipeline_summary.json").write_text(
        '{"video_path":"test.mp4","duration_sec":30.0,"n_cameras":2,"n_ocr_frames":2,"output_dir":"'
        + str(output_dir)
        + '"}',
        encoding="utf-8",
    )

    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True)
    blank = np.zeros((32, 48, 3), dtype=np.uint8)
    for name in (
        "scene_0_frame_10.jpg",
        "scene_0_frame_11.jpg",
        "scene_0_frame_50.jpg",
        "scene_0_frame_90.jpg",
        "scene_1_frame_20.jpg",
        "scene_1_frame_40.jpg",
    ):
        cv2.imwrite(str(frame_dir / name), blank)

    return output_dir


def _write_appearance_artifacts(output_dir: Path) -> None:
    (output_dir / "frame_appearance.csv").write_text(
        "scene_id,frame_number,frame_path,person_count,person_colors_json,primary_bgr_json,confidence,status\n"
        '0,10,frames/scene_0_frame_10.jpg,2,"[""red""]","[20, 20, 220]",0.90,ok\n'
        '0,50,frames/scene_0_frame_50.jpg,2,"[""red""]","[20, 20, 220]",0.88,ok\n'
        '0,90,frames/scene_0_frame_90.jpg,3,"[""red""]","[20, 20, 220]",0.75,ok\n'
        '1,20,frames/scene_1_frame_20.jpg,1,"[""blue""]","[220, 20, 20]",0.92,ok\n'
        '1,40,frames/scene_1_frame_40.jpg,1,"[""blue""]","[220, 20, 20]",0.89,ok\n',
        encoding="utf-8",
    )
    (output_dir / "scene_appearance.csv").write_text(
        "scene_id,scene_type,person_count,person_colors_json,appearance_signature,primary_bgr_json,dominant_track_frames,dominant_track_median_area,confidence,status\n"
        '0,closeup,2,"[""red""]","red","[20, 20, 220]",3,1200,0.88,ok\n'
        '1,closeup,1,"[""blue""]","blue","[220, 20, 20]",2,800,0.91,ok\n',
        encoding="utf-8",
    )
    (output_dir / "scene_types.csv").write_text(
        "scene_id,scene_type\n0,closeup\n1,closeup\n",
        encoding="utf-8",
    )


def _write_debug_artifacts(output_dir: Path) -> None:
    frame_index = pd.read_csv(output_dir / "frame_index.csv")
    camera_rows = frame_index[frame_index["sample_role"] == "camera"]
    results: list[ClusterResult] = []
    vectors: list[list[float]] = []
    for idx, row in enumerate(camera_rows.itertuples(index=False)):
        frame_path = str((output_dir / row.frame_path).resolve())
        cluster_id = 0 if int(row.scene_id) == 0 else 1
        results.append(
            ClusterResult(
                scene_idx=idx,
                scene_id=str(row.scene_id),
                frame_path=frame_path,
                cluster_id=cluster_id,
                camera_id=f"cam_{cluster_id}",
            )
        )
        vectors.append([1.0, 0.0] if cluster_id == 0 else [0.0, 1.0])

    output = PipelineOutput(
        results=results,
        reduced_matrix=np.array(vectors, dtype=np.float32),
        dbscan_eps=0.5,
        method="hsv",
        ensemble_member_labelings=[np.array([r.cluster_id for r in results], dtype=np.int64)],
        ensemble_member_names=["hsv"],
    )
    write_camera_clustering_debug(output_dir / "camera_clustering_debug.npz", output, frame_index)

    frame_camera = camera_rows.copy()
    frame_camera["camera_id"] = frame_camera["scene_id"].map({0: "cam_0", 1: "cam_1"})
    frame_camera["cluster_id"] = frame_camera["scene_id"]
    frame_camera[["scene_id", "frame_number", "frame_path", "camera_id", "cluster_id"]].to_csv(
        output_dir / "frame_camera_results.csv",
        index=False,
    )


def test_load_timeline_bundle(tmp_path):
    output_dir = _write_fixture_bundle(tmp_path)
    bundle = load_timeline_bundle(output_dir)

    assert len(bundle.rows) == 3
    assert "PLAYER" in bundle.suggestions
    assert "SET" in bundle.suggestions
    assert "GAME" not in bundle.suggestions
    assert 10 in bundle.frame_lookup
    assert bundle.frame_lookup[10].camera_id == "cam_0"

    matches = bundle.search_rows("player")
    assert len(matches) == 2
    assert not bundle.search_rows("cam_0")

    suggestions = bundle.search_suggestions("pl")
    assert suggestions[0].casefold().startswith("pl") or "PL" in suggestions[0]

    row = bundle.find_row("cam_0", "PLAYER", "PLAYER")
    assert row is not None
    detail = bundle.row_detail(row)
    assert detail["frames"] == [10, 11, 12]
    frame10 = next(item for item in detail["frame_details"] if item["frame_number"] == 10)
    assert frame10["enrich_applied"] is False
    assert frame10["associated_text"] == "PLAYER"


def test_row_detail_includes_enrich_provenance(tmp_path):
    output_dir = _write_fixture_bundle(tmp_path)
    associated_path = output_dir / "frame_text_associated.csv"
    associated_path.write_text(
        "scene_id,frame_number,camera_id,raw_text,text_kind,mapped_complete_text,"
        "mapping_confidence,readability_label,bbox_json,enrich_applied,ocr_raw_text\n"
        "0,10,cam_0,CHASE,complete,CHASE,1.0,partial,[],true,CHAREO\n",
        encoding="utf-8",
    )
    complete_path = output_dir / "aggregated_complete.csv"
    complete_path.write_text(
        "camera_id,text,text_kind,mapped_complete_text,total_duration_sec,frame_ranges,"
        "n_frames_present,n_frames_good,n_frames_partial,n_frames_enriched,dominant_readability\n"
        "cam_0,CHASE,complete,CHASE,1.0,10,1,0,1,1,partial\n",
        encoding="utf-8",
    )
    (output_dir / "aggregated_partial.csv").write_text(
        "camera_id,text,text_kind,mapped_complete_text,total_duration_sec,frame_ranges,"
        "n_frames_present,n_frames_good,n_frames_partial,n_frames_enriched,dominant_readability\n",
        encoding="utf-8",
    )
    bundle = load_timeline_bundle(output_dir)
    row = bundle.find_row("cam_0", "CHASE", "CHASE")
    assert row is not None
    detail = bundle.row_detail(row)
    frame10 = detail["frame_details"][0]
    assert frame10["enrich_applied"] is True
    assert frame10["ocr_raw_text"] == "CHAREO"
    assert frame10["associated_text"] == "CHASE"


def test_api_row_includes_enrich_provenance(tmp_path):
    output_dir = _write_fixture_bundle(tmp_path)
    associated_path = output_dir / "frame_text_associated.csv"
    associated_path.write_text(
        "scene_id,frame_number,camera_id,raw_text,text_kind,mapped_complete_text,"
        "mapping_confidence,readability_label,bbox_json,enrich_applied,ocr_raw_text\n"
        "0,10,cam_0,CHASE,complete,CHASE,1.0,partial,[],true,CHAREO\n",
        encoding="utf-8",
    )
    (output_dir / "aggregated_complete.csv").write_text(
        "camera_id,text,text_kind,mapped_complete_text,total_duration_sec,frame_ranges,"
        "n_frames_present,n_frames_good,n_frames_partial,n_frames_enriched,dominant_readability\n"
        "cam_0,CHASE,complete,CHASE,1.0,10,1,0,1,1,partial\n",
        encoding="utf-8",
    )
    (output_dir / "aggregated_partial.csv").write_text(
        "camera_id,text,text_kind,mapped_complete_text,total_duration_sec,frame_ranges,"
        "n_frames_present,n_frames_good,n_frames_partial,n_frames_enriched,dominant_readability\n",
        encoding="utf-8",
    )
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    app = create_app(output_dir, static_dir)
    client = TestClient(app)

    res = client.get(
        "/api/row",
        params={"camera_id": "cam_0", "mapped": "CHASE", "text": "CHASE"},
    )
    assert res.status_code == 200
    detail = res.json()
    assert detail["frame_details"][0]["enrich_applied"] is True
    assert detail["frame_details"][0]["ocr_raw_text"] == "CHAREO"


def test_load_timeline_bundle_missing_raises(tmp_path):
    with pytest.raises(TimelineLoadError):
        load_timeline_bundle(tmp_path / "missing")


def test_build_frame_ocr_payload_joins_associations(tmp_path):
    output_dir = _write_fixture_bundle(tmp_path)
    index = load_pipeline_ocr_index(output_dir)
    payload = index.frame_payload(10)

    assert payload["frame_number"] == 10
    assert payload["source"] == "frame_ocr.csv"
    assert len(payload["detections"]) == 1
    det = payload["detections"][0]
    assert det["text"] == "PLAYER"
    assert det["mapped_complete_text"] == "PLAYER"
    assert det["display_text"] == "PLAYER"
    assert det["bbox"] == [10, 10, 100, 40]


def test_build_frame_ocr_payload_enriched_entry(tmp_path):
    output_dir = _write_fixture_bundle(tmp_path)
    assoc_path = output_dir / "frame_text_associated.csv"
    assoc_path.write_text(
        "scene_id,frame_number,camera_id,raw_text,text_kind,mapped_complete_text,"
        "mapping_confidence,readability_label,bbox_json,enrich_applied,ocr_raw_text\n"
        "0,10,cam_0,CHASE,complete,CHASE,1.0,partial,\"[50,50,80,60]\",true,CHASEO\n",
        encoding="utf-8",
    )
    ocr_path = output_dir / "frame_ocr.csv"
    pd.DataFrame(
        [
            {
                "scene_id": 0,
                "frame_number": 10,
                "seconds": 0.33,
                "camera_id": "cam_0",
                "words_json": json.dumps(["CHASEO"]),
                "detections_json": json.dumps(
                    [
                        {
                            "text": "CHASEO",
                            "confidence": 0.64,
                            "bbox": [48, 48, 82, 62],
                            "source": "ocr",
                        }
                    ]
                ),
                "verdict": "readable",
                "used_unk": False,
            }
        ]
    ).to_csv(ocr_path, index=False)

    payload = load_pipeline_ocr_index(output_dir).frame_payload(10)
    assert len(payload["detections"]) == 1
    det = payload["detections"][0]
    assert det["text"] == "CHASEO"
    assert det["mapped_complete_text"] == "CHASE"
    assert det["display_text"] == "CHASEO → CHASE"
    assert det["enrich_applied"] is True


def test_load_pipeline_ocr_index_missing_raises(tmp_path):
    with pytest.raises(PipelineOcrLoadError):
        load_pipeline_ocr_index(tmp_path / "missing")


def test_api_search_and_row(tmp_path):
    output_dir = _write_fixture_bundle(tmp_path)
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    app = create_app(output_dir, static_dir)
    client = TestClient(app)

    res = client.get("/api/search", params={"q": "player"})
    assert res.status_code == 200
    data = res.json()
    assert len(data["rows"]) == 2
    assert "PLAYER" in data["suggestions"]

    res = client.get(
        "/api/row",
        params={"camera_id": "cam_0", "mapped": "PLAYER", "text": "PLAYER"},
    )
    assert res.status_code == 200
    detail = res.json()
    assert detail["frames"] == [10, 11, 12]

    res = client.get("/api/frames/10")
    assert res.status_code == 200

    res = client.get("/api/pipeline/ocr/10")
    assert res.status_code == 200
    data = res.json()
    assert data["detections"][0]["text"] == "PLAYER"
    assert data["detections"][0]["mapped_complete_text"] == "PLAYER"


def test_api_pipeline_ocr_missing_frame(tmp_path):
    output_dir = _write_fixture_bundle(tmp_path)
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    app = create_app(output_dir, static_dir)
    client = TestClient(app)

    res = client.get("/api/pipeline/ocr/999")
    assert res.status_code == 404


def test_pick_scene_slots_full():
    rows = [
        {"frame_number": 10, "frame_path": "frames/a.jpg"},
        {"frame_number": 50, "frame_path": "frames/b.jpg"},
        {"frame_number": 90, "frame_path": "frames/c.jpg"},
    ]
    slots = pick_scene_slots(rows, Path("/tmp"))
    assert [s.slot for s in slots] == ["begin", "mid", "end"]
    assert [s.frame_number for s in slots] == [10, 50, 90]


def test_pick_scene_slots_deduped():
    rows = [
        {"frame_number": 10, "frame_path": "frames/a.jpg"},
        {"frame_number": 20, "frame_path": "frames/b.jpg"},
    ]
    slots = pick_scene_slots(rows, Path("/tmp"))
    assert len(slots) == 2
    assert slots[0].slot == "begin"
    assert slots[0].frame_number == 10
    assert slots[1].slot == "mid"
    assert slots[1].frame_number == 20


def test_load_camera_collage_bundle(tmp_path):
    output_dir = _write_fixture_bundle(tmp_path)
    bundle = load_camera_collage_bundle(output_dir)

    assert bundle.camera_ids == ["cam_0", "cam_1"]
    assert bundle.scene_count("cam_0") == 1
    assert bundle.scene_count("cam_1") == 1

    scene0 = bundle.scenes_for_camera("cam_0")[0]
    assert scene0.scene_id == 0
    assert scene0.cluster_id == 0
    assert scene0.camera_vote_counts == {"cam_0": 3}
    assert scene0.unanimous is True
    assert len(scene0.frames) == 3
    assert scene0.frames[0].frame_number == 10
    assert scene0.frames[1].frame_number == 50
    assert scene0.frames[2].frame_number == 90


def test_load_camera_collage_bundle_missing_raises(tmp_path):
    with pytest.raises(CameraCollageLoadError):
        load_camera_collage_bundle(tmp_path / "missing")


def test_api_camera_collage(tmp_path):
    output_dir = _write_fixture_bundle(tmp_path)
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    app = create_app(output_dir, static_dir)
    client = TestClient(app)

    res = client.get("/api/cameras")
    assert res.status_code == 200
    data = res.json()
    assert data["camera_ids"] == ["cam_0", "cam_1"]
    assert data["counts"]["cam_0"] == 1
    assert data["has_debug_artifact"] is False

    res = client.get("/api/cameras/cam_0/scenes")
    assert res.status_code == 200
    scenes = res.json()["scenes"]
    assert len(scenes) == 1
    assert scenes[0]["cluster_id"] == 0
    assert scenes[0]["camera_vote_counts"]["cam_0"] == 3
    assert len(scenes[0]["frames"]) == 3
    assert scenes[0]["frames"][0]["image_url"] == "/api/scene-images/0/begin"

    res = client.get("/api/scene-images/0/begin")
    assert res.status_code == 200

    res = client.get("/api/cameras/missing/scenes")
    assert res.status_code == 404


def test_resolve_frame_under_output_host_absolute_path(tmp_path):
    output_dir = tmp_path / "pipeline"
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True)
    image = frames_dir / "scene_0_frame_10.jpg"
    image.write_bytes(b"jpg")

    host_path = Path("/Users/example/project/data/pipeline/frames/scene_0_frame_10.jpg")
    resolved = resolve_frame_under_output(host_path, output_dir)
    assert resolved == image.resolve()


def test_api_scene_images_remaps_host_absolute_paths(tmp_path):
    output_dir = _write_fixture_bundle(tmp_path)
    frame_index = pd.read_csv(output_dir / "frame_index.csv")
    host_root = tmp_path / "host_machine"
    frame_index["frame_path"] = frame_index["frame_path"].apply(
        lambda rel: str(host_root / "data/pipeline" / rel)
    )
    frame_index.to_csv(output_dir / "frame_index.csv", index=False)

    static_dir = tmp_path / "static"
    static_dir.mkdir()
    app = create_app(output_dir, static_dir)
    client = TestClient(app)

    res = client.get("/api/scene-images/0/begin")
    assert res.status_code == 200


def test_api_camera_compare_and_global(tmp_path):
    output_dir = _write_fixture_bundle(tmp_path)
    _write_debug_artifacts(output_dir)
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    app = create_app(output_dir, static_dir)
    client = TestClient(app)

    res = client.get("/api/cameras")
    assert res.json()["has_debug_artifact"] is True

    res = client.post(
        "/api/cameras/compare",
        json={
            "selections": [
                {"camera_id": "cam_0", "scene_id": 0},
                {"camera_id": "cam_1", "scene_id": 1},
            ],
            "include_global": True,
        },
    )
    assert res.status_code == 200
    payload = res.json()
    assert payload["has_debug_artifact"] is True
    assert len(payload["pairwise"]) == 1
    assert "global_projection" in payload

    res = client.get("/api/camera-debug/global", params=[("scene_id", 0), ("scene_id", 1)])
    assert res.status_code == 200
    data = res.json()
    assert data["projection_method"] in {"tsne", "pca", "centered_2d"}
    assert len(data["points"]) > 0
    assert "legend" in data


def test_render_camera_collages(tmp_path):
    output_dir = _write_fixture_bundle(tmp_path)
    dest = tmp_path / "collages"
    paths = render_camera_collages(
        output_dir,
        dest_dir=dest,
        config=CollageRenderConfig(layout="timeline", slots=("begin", "mid", "end")),
    )
    assert len(paths) == 2
    assert (dest / "cam_0.jpg").is_file()
    assert (dest / "cam_1.jpg").is_file()
    assert all(path.stat().st_size > 0 for path in paths)


def test_load_appearance_bundle(tmp_path):
    output_dir = _write_fixture_bundle(tmp_path)
    _write_appearance_artifacts(output_dir)
    bundle = load_appearance_bundle(output_dir)

    assert bundle.has_appearance_artifacts is True
    assert len(bundle.scene_by_id) == 2
    assert bundle.scene_by_id[0].person_count == 2
    assert bundle.scene_by_id[0].has_count_variance is True
    assert bundle.scene_by_id[1].has_count_variance is False

    summary = bundle.build_summary()
    assert summary["n_scenes"] == 2
    assert summary["n_frames"] == 5
    assert summary["person_count_histogram"][2] == 1
    assert summary["n_scenes_with_count_variance"] == 1

    groups = bundle.compatibility_groups()
    assert 0 in groups
    assert 1 in groups
    assert groups[0] != groups[1]


def test_load_appearance_bundle_empty(tmp_path):
    output_dir = _write_fixture_bundle(tmp_path)
    bundle = load_appearance_bundle(output_dir)
    assert bundle.has_appearance_artifacts is False
    assert bundle.build_summary()["n_scenes"] == 0


def test_run_appearance_on_bytes_mocked():
    clear_segmenter_cache()
    segmenter = MockPersonSegmenter(detections=[((5, 5, 25, 35), 0.93)])
    img = np.zeros((40, 50, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", img)
    assert ok

    payload = run_appearance_on_bytes(
        encoded.tobytes(),
        segmenter=segmenter,
        scene_id=0,
        frame_number=10,
    )
    assert payload["person_count"] == 1
    assert payload["status"] == "ok"
    assert payload["image_width"] == 50
    assert payload["image_height"] == 40
    assert len(payload["detections"]) == 1
    assert payload["detections"][0]["bbox"] == [5, 5, 25, 35]
    assert payload["detections"][0]["mask_contours"]


def test_api_appearance(tmp_path):
    output_dir = _write_fixture_bundle(tmp_path)
    _write_appearance_artifacts(output_dir)
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    app = create_app(output_dir, static_dir)
    client = TestClient(app)

    res = client.get("/api/appearance")
    assert res.status_code == 200
    data = res.json()
    assert data["has_artifacts"] is True
    assert data["summary"]["n_scenes"] == 2
    assert data["scenes_with_issues"] == [0]

    res = client.get("/api/appearance/scenes")
    assert res.status_code == 200
    scenes = res.json()["scenes"]
    assert len(scenes) == 2
    assert scenes[0]["has_count_variance"] is True

    res = client.get("/api/appearance/scene/0")
    assert res.status_code == 200
    detail = res.json()
    assert detail["scene"]["scene_id"] == 0
    assert len(detail["frames"]) == 3
    assert len(detail["slots"]) == 3

    with patch("broadcast_pipeline.viz.server.run_appearance_on_bytes") as mock_seg:
        mock_seg.return_value = {
            "person_count": 2,
            "status": "ok",
            "person_colors": ["red", "black"],
            "confidence": 0.9,
            "detections": [
                {
                    "bbox": [1, 2, 10, 12],
                    "confidence": 0.9,
                    "clothing_color": "red",
                    "mask_contours": [[[1, 2], [10, 2], [10, 12]]],
                }
            ],
            "image_width": 48,
            "image_height": 32,
        }
        res = client.get("/api/appearance/segment/frame/50")
    assert res.status_code == 200
    assert res.json()["person_count"] == 2

    with patch("broadcast_pipeline.viz.server.run_appearance_on_bytes") as mock_seg:
        mock_seg.return_value = {
            "person_count": 2,
            "status": "ok",
            "person_colors": ["red", "black"],
            "confidence": 0.9,
            "detections": [],
            "image_width": 48,
            "image_height": 32,
        }
        res = client.get("/api/appearance/segment/scene/0/begin")
    assert res.status_code == 200

    res = client.get("/api/appearance/scene/99")
    assert res.status_code == 404


def test_api_appearance_without_artifacts(tmp_path):
    output_dir = _write_fixture_bundle(tmp_path)
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    app = create_app(output_dir, static_dir)
    client = TestClient(app)

    res = client.get("/api/appearance")
    assert res.status_code == 200
    assert res.json()["has_artifacts"] is False

    res = client.get("/api/appearance/scenes")
    assert res.status_code == 200
    assert res.json()["scenes"] == []
