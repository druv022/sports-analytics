from __future__ import annotations

import json

import pandas as pd

from broadcast_pipeline.config import PipelineConfig
from broadcast_pipeline.text_enrich import enrich_ocr_observations


def _frame_row(
    *,
    scene_id: int,
    frame_number: int,
    camera_id: str,
    detections: list[dict],
    words: list[str] | None = None,
) -> dict:
    return {
        "scene_id": scene_id,
        "frame_number": frame_number,
        "seconds": float(frame_number),
        "camera_id": camera_id,
        "words_json": json.dumps(words if words is not None else [d["text"] for d in detections]),
        "detections_json": json.dumps(detections),
        "verdict": "readable",
        "used_unk": False,
    }


def test_enrich_carries_forward_missing_region_across_adjacent_same_camera():
    config = PipelineConfig(readability_size_multiplier=1.25)
    frame_ocr = pd.DataFrame(
        [
            _frame_row(
                scene_id=0,
                frame_number=10,
                camera_id="cam_0",
                detections=[
                    {
                        "text": "PLAYER",
                        "confidence": 0.9,
                        "bbox": [100, 100, 200, 140],
                        "source": "ocr",
                    }
                ],
            ),
            _frame_row(
                scene_id=0,
                frame_number=20,
                camera_id="cam_0",
                detections=[],
                words=[],
            ),
        ]
    )

    enriched = enrich_ocr_observations(config, frame_ocr)
    second = json.loads(enriched.loc[enriched["frame_number"] == 20, "detections_json"].iloc[0])
    assert len(second) == 1
    assert second[0]["text"] == "PLAYER"
    assert second[0]["source"] == "carried"
    assert "PLAYER" in json.loads(enriched.loc[enriched["frame_number"] == 20, "words_json"].iloc[0])


def test_enrich_does_not_carry_across_camera_change():
    config = PipelineConfig(readability_size_multiplier=1.25)
    frame_ocr = pd.DataFrame(
        [
            _frame_row(
                scene_id=0,
                frame_number=10,
                camera_id="cam_0",
                detections=[
                    {
                        "text": "PLAYER",
                        "confidence": 0.9,
                        "bbox": [100, 100, 200, 140],
                        "source": "ocr",
                    }
                ],
            ),
            _frame_row(
                scene_id=0,
                frame_number=20,
                camera_id="cam_1",
                detections=[],
                words=[],
            ),
        ]
    )

    enriched = enrich_ocr_observations(config, frame_ocr)
    second = json.loads(enriched.loc[enriched["frame_number"] == 20, "detections_json"].iloc[0])
    assert second == []


def test_enrich_readability_labels_use_size_multiplier():
    config = PipelineConfig(readability_size_multiplier=1.25)
    frame_ocr = pd.DataFrame(
        [
            _frame_row(
                scene_id=0,
                frame_number=10,
                camera_id="cam_0",
                detections=[
                    {
                        "text": "BIG",
                        "confidence": 0.9,
                        "bbox": [0, 0, 200, 100],
                        "source": "ocr",
                    },
                    {
                        "text": "SMALL",
                        "confidence": 0.9,
                        "bbox": [0, 0, 50, 20],
                        "source": "ocr",
                    },
                ],
            )
        ]
    )

    enriched = enrich_ocr_observations(config, frame_ocr)
    detections = json.loads(enriched.iloc[0]["detections_json"])
    labels = {det["text"]: det["readability_label"] for det in detections}
    assert labels["BIG"] == "good"
    assert labels["SMALL"] == "partial"


def test_enrich_passthrough_without_detections_column():
    config = PipelineConfig()
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
    enriched = enrich_ocr_observations(config, frame_ocr)
    assert "detections_json" not in enriched.columns
    assert enriched.iloc[0]["words_json"] == '["PLAYER"]'
