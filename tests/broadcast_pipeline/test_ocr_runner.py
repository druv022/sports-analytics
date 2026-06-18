from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from broadcast_pipeline.config import PipelineConfig
from broadcast_pipeline.ocr_runner import (
    _detections_to_json,
    ocr_done_keys,
    ocr_frame_keys,
    ocr_is_complete,
    run_segment_ocr,
)
from src.scene_ocr.types import OcrDetection, OcrReadability, OcrReadabilityVerdict


def test_ocr_is_complete_when_all_frames_present(tmp_path):
    frame_index = pd.DataFrame(
        [
            {"scene_id": 0, "frame_number": 10, "sample_role": "ocr"},
            {"scene_id": 1, "frame_number": 20, "sample_role": "ocr"},
        ]
    )
    out = tmp_path / "frame_ocr.csv"
    out.write_text(
        "scene_id,frame_number,seconds,camera_id,words_json,verdict,used_unk\n"
        "0,10,1.0,cam_0,[],no_text,False\n"
        "1,20,2.0,cam_0,[],no_text,False\n",
        encoding="utf-8",
    )
    assert ocr_frame_keys(frame_index) == {(0, 10), (1, 20)}
    assert ocr_done_keys(out) == {(0, 10), (1, 20)}
    assert ocr_is_complete(out, frame_index)


def test_ocr_is_complete_false_for_partial(tmp_path):
    frame_index = pd.DataFrame(
        [
            {"scene_id": 0, "frame_number": 10, "sample_role": "ocr"},
            {"scene_id": 1, "frame_number": 20, "sample_role": "ocr"},
        ]
    )
    out = tmp_path / "frame_ocr.csv"
    out.write_text(
        "scene_id,frame_number,seconds,camera_id,words_json,verdict,used_unk\n"
        "0,10,1.0,cam_0,[],no_text,False\n",
        encoding="utf-8",
    )
    assert not ocr_is_complete(out, frame_index)


def test_detections_to_json_expands_line_to_words():
    detection = OcrDetection(
        text="HELLO WORLD",
        confidence=0.95,
        bbox=np.array([[0, 0], [200, 0], [200, 40], [0, 40]]),
    )
    payload = json.loads(_detections_to_json([detection]))
    assert [entry["text"] for entry in payload] == ["HELLO", "WORLD"]
    assert payload[0]["bbox"] == [0, 0, 200, 40]


def test_run_segment_ocr_writes_detections_json(tmp_path):
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    frame_path = frame_dir / "frame.jpg"
    frame_path.write_bytes(b"not-a-real-jpeg")

    frame_index = pd.DataFrame(
        [
            {
                "scene_id": 0,
                "frame_number": 10,
                "seconds": 0.5,
                "frame_path": str(frame_path),
                "sample_role": "ocr",
            }
        ]
    )
    frame_assignments = pd.DataFrame(
        [
            {
                "scene_id": 0,
                "frame_number": 10,
                "seconds": 0.5,
                "frame_path": str(frame_path),
                "sample_role": "ocr",
                "camera_id": "cam_0",
                "cluster_id": 0,
            }
        ]
    )
    mock_result = OcrReadability(
        verdict=OcrReadabilityVerdict.READABLE,
        words=["PLAYER"],
        detections=[
            OcrDetection(
                text="PLAYER",
                confidence=0.9,
                bbox=np.array([10, 10, 100, 40]),
            )
        ],
        text_candidates=[],
        vlm_crops=[],
        overlay_readable=True,
        needs_vlm=False,
        reasons=[],
    )
    config = PipelineConfig(output_dir=tmp_path, ocr_prefetch_workers=0)
    out = tmp_path / "frame_ocr.csv"

    with (
        patch(
            "broadcast_pipeline.ocr_runner.load_image",
            return_value=np.zeros((8, 8, 3), dtype=np.uint8),
        ),
        patch(
            "broadcast_pipeline.ocr_runner.assess_readability_from_bgr",
            return_value=mock_result,
        ),
    ):
        result = run_segment_ocr(
            config,
            frame_index,
            frame_assignments,
            output_path=out,
        )

    assert "detections_json" in result.columns
    detections = json.loads(result.iloc[0]["detections_json"])
    assert detections[0]["text"] == "PLAYER"
    assert detections[0]["bbox"] == [10, 10, 100, 40]


def test_run_segment_ocr_prefetch_submits_load_image(tmp_path):
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    paths = []
    for idx, frame_number in enumerate((10, 20)):
        frame_path = frame_dir / f"frame_{idx}.jpg"
        frame_path.write_bytes(b"not-a-real-jpeg")
        paths.append(frame_path)

    frame_index = pd.DataFrame(
        [
            {
                "scene_id": 0,
                "frame_number": frame_number,
                "seconds": float(frame_number) / 20.0,
                "frame_path": str(path),
                "sample_role": "ocr",
            }
            for frame_number, path in zip((10, 20), paths)
        ]
    )
    frame_assignments = pd.DataFrame(
        [
            {
                "scene_id": 0,
                "frame_number": frame_number,
                "seconds": float(frame_number) / 20.0,
                "frame_path": str(path),
                "sample_role": "ocr",
                "camera_id": "cam_0",
                "cluster_id": 0,
            }
            for frame_number, path in zip((10, 20), paths)
        ]
    )
    mock_result = OcrReadability(
        verdict=OcrReadabilityVerdict.NO_TEXT,
        words=[],
        detections=[],
        text_candidates=[],
        vlm_crops=[],
        overlay_readable=False,
        needs_vlm=False,
        reasons=[],
    )
    config = PipelineConfig(output_dir=tmp_path, ocr_prefetch_workers=2)
    out = tmp_path / "frame_ocr.csv"
    load_calls: list[str] = []

    def fake_load(path):
        load_calls.append(str(path))
        return np.zeros((8, 8, 3), dtype=np.uint8)

    with (
        patch("broadcast_pipeline.ocr_runner.load_image", side_effect=fake_load),
        patch(
            "broadcast_pipeline.ocr_runner.assess_readability_from_bgr",
            return_value=mock_result,
        ),
    ):
        run_segment_ocr(
            config,
            frame_index,
            frame_assignments,
            output_path=out,
        )

    assert len(load_calls) == 2
    assert str(paths[0]) in load_calls
    assert str(paths[1]) in load_calls
