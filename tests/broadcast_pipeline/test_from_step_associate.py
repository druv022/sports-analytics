from __future__ import annotations

import json

import pandas as pd
import pytest

from broadcast_pipeline.aggregator import aggregate_text_timeline, write_pipeline_summary
from broadcast_pipeline.artifacts import validate_stage_inputs
from broadcast_pipeline.config import PipelineConfig
from broadcast_pipeline.orchestrator import run_pipeline
from broadcast_pipeline.types import Scene, VideoMeta


def test_from_step_associate_without_ocr(tmp_path):
    config = PipelineConfig(output_dir=tmp_path, from_step="associate")
    with pytest.raises(FileNotFoundError):
        validate_stage_inputs(config, "associate")


def test_associate_and_aggregate_from_artifacts(tmp_path):
    config = PipelineConfig(output_dir=tmp_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    meta = VideoMeta(
        path=tmp_path / "video.mp4",
        fps=30.0,
        frame_count=300,
        duration_sec=10.0,
        width=1920,
        height=1080,
    )
    config.artifact("video_meta").write_text(
        json.dumps(
            {
                "path": str(meta.path),
                "fps": meta.fps,
                "frame_count": meta.frame_count,
                "duration_sec": meta.duration_sec,
                "width": meta.width,
                "height": meta.height,
                "fps_source": "test",
            }
        ),
        encoding="utf-8",
    )
    config.artifact("scenes").write_text(
        json.dumps(
            [
                {
                    "scene_id": 0,
                    "start_frame": 0,
                    "end_frame": 60,
                    "start_sec": 0.0,
                    "end_sec": 2.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "scene_id": 0,
                "frame_number": 0,
                "seconds": 0.0,
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
    ).to_csv(config.artifact("frame_ocr"), index=False)
    pd.DataFrame(
        [
            {
                "scene_id": 0,
                "frame_number": 0,
                "seconds": 0.0,
                "frame_path": "frame.jpg",
                "sample_role": "ocr",
                "camera_id": "cam_0",
                "cluster_id": 0,
            }
        ]
    ).to_csv(config.artifact("frame_assignments"), index=False)

    config.reference_csv.write_text(
        "complete_text,approved,first_seen_scene_id,first_seen_frame,discovery_count\n"
        "PLAYER,true,0,0,1\n",
        encoding="utf-8",
    )

    summary = run_pipeline(
        PipelineConfig(
            output_dir=tmp_path,
            video_path=tmp_path / "video.mp4",
            from_step="associate",
        )
    )

    assert config.artifact("aggregated_complete").is_file()
    assert config.artifact("pipeline_summary").is_file()
    assert summary.artifacts["aggregated_complete"].exists()

    complete_df = pd.read_csv(config.artifact("aggregated_complete"))
    assert "n_frames_present" in complete_df.columns
    assert complete_df.iloc[0]["n_frames_present"] == 1

    summary_json = json.loads(config.artifact("pipeline_summary").read_text(encoding="utf-8"))
    assert summary_json["readability_size_multiplier"] == 1.25
    assert summary_json["n_text_presence_events"] == 1
