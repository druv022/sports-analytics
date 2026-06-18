from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from broadcast_pipeline.camera_assignment import majority_vote_per_scene
from broadcast_pipeline.config import PipelineConfig
from broadcast_pipeline.ocr_runner import _pipeline_ocr_config
from broadcast_pipeline.progress import ProgressTracker
from broadcast_pipeline.scene_extractor import camera_sample_frames, ocr_sample_frames
from broadcast_pipeline.text_associate import (
    associate_text,
    _best_complete_match,
    _enrich_provenance,
)
from broadcast_pipeline.text_reference import update_text_reference
from broadcast_pipeline.aggregator import (
    AGGREGATE_CSV_COLUMNS,
    _merge_frame_ranges,
    _slot_durations,
    aggregate_text_timeline,
)
from broadcast_pipeline.artifacts import resolve_stage_range
from broadcast_pipeline.orchestrator import run_stages
from broadcast_pipeline.types import PipelineSummary
from broadcast_pipeline.types import Scene, VideoMeta


def test_resolve_stage_range_full_pipeline():
    assert resolve_stage_range("all", None) == (
        "meta",
        "extract",
        "filter",
        "appearance",
        "cameras",
        "ocr",
        "reference",
        "enrich",
        "associate",
        "aggregate",
    )
    assert resolve_stage_range("cameras", None) == (
        "cameras",
        "ocr",
        "reference",
        "enrich",
        "associate",
        "aggregate",
    )


def test_resolve_stage_range_single_stage():
    assert resolve_stage_range("cameras", "cameras") == ("cameras",)


def test_resolve_stage_range_bounded_slice():
    assert resolve_stage_range("cameras", "enrich") == (
        "cameras",
        "ocr",
        "reference",
        "enrich",
    )


def test_resolve_stage_range_rejects_inverted_range():
    import pytest

    with pytest.raises(ValueError, match="after to_step"):
        resolve_stage_range("aggregate", "meta")


def test_run_stages_sets_range_and_restores_config(monkeypatch):
    config = PipelineConfig(
        video_path=Path("video.mp4"),
        output_dir=Path("out"),
        from_step="all",
        to_step=None,
    )
    seen: list[tuple[str, str | None]] = []

    def fake_run(cfg: PipelineConfig) -> PipelineSummary:
        seen.append((cfg.from_step, cfg.to_step))
        return PipelineSummary(output_dir=cfg.output_dir)

    monkeypatch.setattr("broadcast_pipeline.orchestrator.run_pipeline", fake_run)
    summary = run_stages(config, "meta", "extract", log_range=False)
    assert seen == [("meta", "extract")]
    assert config.from_step == "all"
    assert config.to_step is None
    assert summary.output_dir == Path("out")


def test_progress_tracker_emits_milestones(capsys):
    tracker = ProgressTracker(10, "test", step_pct=50)
    tracker.advance(5)
    tracker.advance(5)
    output = capsys.readouterr().out
    assert "test: 5/10 (50%)" in output
    assert "test: 10/10 (100%)" in output


def test_pipeline_ocr_config_respects_accelerator(monkeypatch):
    monkeypatch.setattr(
        "broadcast_pipeline.ocr_runner.resolve_ocr_use_cuda",
        lambda accelerator: accelerator == "cuda",
    )
    cfg = PipelineConfig(accelerator="cuda")
    ocr_cfg = _pipeline_ocr_config(cfg)
    assert ocr_cfg.use_cuda is True

    cfg_cpu = PipelineConfig(accelerator="cpu")
    ocr_cfg_cpu = _pipeline_ocr_config(cfg_cpu)
    assert ocr_cfg_cpu.use_cuda is False


def test_pipeline_ocr_config_passes_batch_overrides():
    cfg = PipelineConfig(ocr_rec_batch=32, ocr_cls_batch=24)
    ocr_cfg = _pipeline_ocr_config(cfg)
    assert ocr_cfg.rec_batch_num == 32
    assert ocr_cfg.cls_batch_num == 24


def test_camera_sample_frames_dedup():
    frames = camera_sample_frames(0, 10, 5)
    assert frames[0] == 0
    assert frames[-1] == 9
    assert len(frames) == len(set(frames))


def test_ocr_sample_frames_step():
    frames = ocr_sample_frames(0, 120, fps=60.0, ocr_hz=2.0)
    assert frames[0] == 0
    assert all(frames[i] < frames[i + 1] for i in range(len(frames) - 1))


def test_ocr_sample_frames_disabled():
    assert ocr_sample_frames(0, 100, fps=30.0, ocr_hz=0.0) == []


def test_majority_vote_per_scene_tie_midpoint():
    frame_results = pd.DataFrame(
        [
            {"scene_id": 1, "frame_number": 10, "camera_id": "cam_0", "cluster_id": 0},
            {"scene_id": 1, "frame_number": 100, "camera_id": "cam_1", "cluster_id": 1},
        ]
    )
    frame_index = pd.DataFrame(
        [
            {"scene_id": 1, "frame_number": 10, "sample_role": "camera"},
            {"scene_id": 1, "frame_number": 100, "sample_role": "camera"},
        ]
    )
    out = majority_vote_per_scene(frame_results, frame_index)
    assert out.iloc[0]["camera_id"] in {"cam_0", "cam_1"}


def test_best_complete_match():
    mapped, score = _best_complete_match("SIN", ["SINNER", "GAME"], min_score=0.5)
    assert mapped == "SINNER"
    assert score >= 0.5


def test_best_complete_match_requires_three_chars():
    mapped, _score = _best_complete_match("SI", ["SINNER", "GAME"], min_score=0.1)
    assert mapped is None


def test_best_complete_match_three_char_subsequence():
    mapped, score = _best_complete_match("GAM", ["GAME", "PLAYER"], min_score=0.5)
    assert mapped == "GAME"
    assert score >= 0.5


def test_update_text_reference_preserves_extra_columns(tmp_path):
    ref_path = tmp_path / "approved_text_reference.csv"
    ref_path.write_text(
        "complete_text,approved,first_seen_scene_id,first_seen_frame,discovery_count,notes\n"
        "PLAYER,true,0,10,1,keep\n",
        encoding="utf-8",
    )
    config = PipelineConfig(output_dir=tmp_path, reference_csv=ref_path)
    frame_ocr = pd.DataFrame(
        [
            {
                "scene_id": 0,
                "frame_number": 10,
                "camera_id": "cam_0",
                "words_json": '["PLAYER", "SET"]',
                "verdict": "readable",
                "used_unk": False,
            }
        ]
    )
    merged, n_new = update_text_reference(config, frame_ocr)
    assert n_new == 1
    assert "notes" in merged.columns
    assert merged.loc[merged["complete_text"] == "PLAYER", "notes"].iloc[0] == "keep"


def test_associate_text_drops_unapproved(tmp_path):
    ref_path = tmp_path / "approved_text_reference.csv"
    ref_path.write_text(
        "complete_text,approved,first_seen_scene_id,first_seen_frame,discovery_count\n"
        "PLAYER,false,0,10,1\n",
        encoding="utf-8",
    )
    config = PipelineConfig(output_dir=tmp_path, reference_csv=ref_path)
    frame_ocr = pd.DataFrame(
        [
            {
                "scene_id": 0,
                "frame_number": 10,
                "seconds": 1.0,
                "camera_id": "cam_0",
                "words_json": '["PLAYER"]',
                "verdict": "readable",
                "used_unk": False,
            }
        ]
    )
    associated, dropped = associate_text(config, frame_ocr)
    assert associated.empty
    assert not dropped.empty


def test_merge_frame_ranges():
    assert _merge_frame_ranges([10, 11, 12, 20]) == "10-12;20"


def test_associate_text_propagates_readability_metadata(tmp_path):
    ref_path = tmp_path / "approved_text_reference.csv"
    ref_path.write_text(
        "complete_text,approved,first_seen_scene_id,first_seen_frame,discovery_count\n"
        "PLAYER,true,0,10,1\n",
        encoding="utf-8",
    )
    config = PipelineConfig(output_dir=tmp_path, reference_csv=ref_path)
    frame_ocr = pd.DataFrame(
        [
            {
                "scene_id": 0,
                "frame_number": 10,
                "seconds": 1.0,
                "camera_id": "cam_0",
                "words_json": '["PLAYER"]',
                "detections_json": json.dumps(
                    [
                        {
                            "text": "PLAYER",
                            "confidence": 0.9,
                            "bbox": [10, 10, 100, 40],
                            "source": "ocr",
                            "readability_label": "good",
                        }
                    ]
                ),
                "verdict": "readable",
                "used_unk": False,
            }
        ]
    )
    associated, dropped = associate_text(config, frame_ocr)
    assert dropped.empty
    assert associated.iloc[0]["readability_label"] == "good"
    assert json.loads(associated.iloc[0]["bbox_json"]) == [10, 10, 100, 40]


def test_enrich_provenance_marks_token_not_in_raw(tmp_path):
    ref_path = tmp_path / "approved_text_reference.csv"
    ref_path.write_text(
        "complete_text,approved,first_seen_scene_id,first_seen_frame,discovery_count\n"
        "CHASE,true,0,10,1\n",
        encoding="utf-8",
    )
    config = PipelineConfig(output_dir=tmp_path, reference_csv=ref_path)
    enriched = pd.DataFrame(
        [
            {
                "scene_id": 9,
                "frame_number": 5109,
                "seconds": 85.0,
                "camera_id": "cam_0",
                "words_json": '["CHASE"]',
                "verdict": "readable",
                "used_unk": False,
            }
        ]
    )
    raw = pd.DataFrame(
        [
            {
                "scene_id": 9,
                "frame_number": 5109,
                "seconds": 85.0,
                "camera_id": "cam_0",
                "words_json": '["CHAREO"]',
                "verdict": "needs_vlm",
                "used_unk": True,
            }
        ]
    )
    associated, dropped = associate_text(config, enriched, raw_frame_ocr=raw)
    assert dropped.empty
    row = associated.iloc[0]
    assert row["raw_text"] == "CHASE"
    assert row["text_kind"] == "complete"
    assert bool(row["enrich_applied"]) is True
    assert row["ocr_raw_text"] == "CHAREO"


def test_enrich_provenance_not_applied_when_raw_contains_token(tmp_path):
    ref_path = tmp_path / "approved_text_reference.csv"
    ref_path.write_text(
        "complete_text,approved,first_seen_scene_id,first_seen_frame,discovery_count\n"
        "PLAYER,true,0,10,1\n",
        encoding="utf-8",
    )
    config = PipelineConfig(output_dir=tmp_path, reference_csv=ref_path)
    frame = pd.DataFrame(
        [
            {
                "scene_id": 0,
                "frame_number": 10,
                "seconds": 1.0,
                "camera_id": "cam_0",
                "words_json": '["PLAYER"]',
                "verdict": "readable",
                "used_unk": False,
            }
        ]
    )
    associated, _dropped = associate_text(config, frame, raw_frame_ocr=frame)
    row = associated.iloc[0]
    assert bool(row["enrich_applied"]) is False
    assert row["ocr_raw_text"] is None or pd.isna(row["ocr_raw_text"])


def test_enrich_provenance_unit():
    raw_lookup = {(0, 10): ["CHAREO"]}
    applied, raw_text = _enrich_provenance("CHASE", 0, 10, raw_lookup)
    assert applied is True
    assert raw_text == "CHAREO"


def test_aggregate_includes_presence_and_readability(tmp_path):
    config = PipelineConfig(output_dir=tmp_path)
    associated = pd.DataFrame(
        [
            {
                "scene_id": 0,
                "frame_number": 0,
                "camera_id": "cam_0",
                "raw_text": "PLAYER",
                "text_kind": "complete",
                "mapped_complete_text": "PLAYER",
                "mapping_confidence": 1.0,
                "readability_label": "good",
                "bbox_json": "[10,10,100,40]",
            },
            {
                "scene_id": 0,
                "frame_number": 30,
                "camera_id": "cam_0",
                "raw_text": "PLAYER",
                "text_kind": "complete",
                "mapped_complete_text": "PLAYER",
                "mapping_confidence": 1.0,
                "readability_label": "partial",
                "bbox_json": "[10,10,50,20]",
            },
        ]
    )
    frame_ocr = pd.DataFrame(
        [
            {"scene_id": 0, "frame_number": 0, "seconds": 0.0, "camera_id": "cam_0"},
            {"scene_id": 0, "frame_number": 30, "seconds": 1.0, "camera_id": "cam_0"},
        ]
    )
    meta = VideoMeta(
        path=tmp_path / "video.mp4",
        fps=30.0,
        frame_count=60,
        duration_sec=2.0,
        width=1920,
        height=1080,
    )
    scenes = [Scene(scene_id=0, start_frame=0, end_frame=60, start_sec=0.0, end_sec=2.0)]
    complete_df, partial_df, _summary = aggregate_text_timeline(
        config, associated, frame_ocr, meta, scenes
    )
    assert partial_df.empty
    assert list(partial_df.columns) == AGGREGATE_CSV_COLUMNS
    row = complete_df.iloc[0]
    assert row["n_frames_present"] == 2
    assert row["n_frames_good"] == 1
    assert row["n_frames_partial"] == 1
    assert row["dominant_readability"] == "good"
    assert row["n_frames_enriched"] == 0


def test_aggregate_empty_partial_csv_roundtrip(tmp_path):
    config = PipelineConfig(output_dir=tmp_path)
    associated = pd.DataFrame(
        [
            {
                "scene_id": 0,
                "frame_number": 0,
                "camera_id": "cam_0",
                "raw_text": "PLAYER",
                "text_kind": "complete",
                "mapped_complete_text": "PLAYER",
                "mapping_confidence": 1.0,
                "readability_label": "good",
                "bbox_json": "[]",
            },
        ]
    )
    frame_ocr = pd.DataFrame(
        [{"scene_id": 0, "frame_number": 0, "seconds": 0.0, "camera_id": "cam_0"}]
    )
    meta = VideoMeta(
        path=tmp_path / "video.mp4",
        fps=30.0,
        frame_count=60,
        duration_sec=2.0,
        width=1920,
        height=1080,
    )
    scenes = [Scene(scene_id=0, start_frame=0, end_frame=60, start_sec=0.0, end_sec=2.0)]
    _complete_df, partial_df, _summary = aggregate_text_timeline(
        config, associated, frame_ocr, meta, scenes
    )
    partial_path = tmp_path / "aggregated_partial.csv"
    partial_df.to_csv(partial_path, index=False)
    loaded = pd.read_csv(partial_path)
    assert list(loaded.columns) == AGGREGATE_CSV_COLUMNS
    assert loaded.empty


def test_aggregate_counts_enriched_frames(tmp_path):
    config = PipelineConfig(output_dir=tmp_path)
    associated = pd.DataFrame(
        [
            {
                "scene_id": 0,
                "frame_number": 0,
                "camera_id": "cam_0",
                "raw_text": "CHASE",
                "text_kind": "complete",
                "mapped_complete_text": "CHASE",
                "mapping_confidence": 1.0,
                "readability_label": "partial",
                "bbox_json": "[]",
                "enrich_applied": True,
                "ocr_raw_text": "CHAREO",
            },
            {
                "scene_id": 0,
                "frame_number": 30,
                "camera_id": "cam_0",
                "raw_text": "CHASE",
                "text_kind": "complete",
                "mapped_complete_text": "CHASE",
                "mapping_confidence": 1.0,
                "readability_label": "partial",
                "bbox_json": "[]",
                "enrich_applied": False,
                "ocr_raw_text": None,
            },
        ]
    )
    frame_ocr = pd.DataFrame(
        [
            {"scene_id": 0, "frame_number": 0, "seconds": 0.0, "camera_id": "cam_0"},
            {"scene_id": 0, "frame_number": 30, "seconds": 1.0, "camera_id": "cam_0"},
        ]
    )
    meta = VideoMeta(
        path=tmp_path / "video.mp4",
        fps=30.0,
        frame_count=60,
        duration_sec=2.0,
        width=1920,
        height=1080,
    )
    scenes = [Scene(scene_id=0, start_frame=0, end_frame=60, start_sec=0.0, end_sec=2.0)]
    complete_df, partial_df, _summary = aggregate_text_timeline(
        config, associated, frame_ocr, meta, scenes
    )
    assert partial_df.empty
    row = complete_df.iloc[0]
    assert row["n_frames_present"] == 2
    assert row["n_frames_enriched"] == 1


def test_slot_durations():
    frame_ocr = pd.DataFrame(
        [
            {"scene_id": 0, "frame_number": 0},
            {"scene_id": 0, "frame_number": 30},
        ]
    )
    scenes = [Scene(scene_id=0, start_frame=0, end_frame=60, start_sec=0.0, end_sec=1.0)]
    durations = _slot_durations(frame_ocr, scenes, fps=30.0)
    assert durations[(0, 0)] == 1.0
    assert durations[(0, 30)] == 1.0
