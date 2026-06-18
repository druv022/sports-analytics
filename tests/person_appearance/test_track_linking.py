from __future__ import annotations

import numpy as np

from src.person_appearance.config import AppearanceConfig
from src.person_appearance.track_linking import link_tracks, select_dominant_track
from src.person_appearance.types import FrameAppearance, PersonDetection


def _frame(scene_id: int, frame_number: int, bboxes: list[tuple[int, int, int, int]]) -> FrameAppearance:
    detections = tuple(
        PersonDetection(
            bbox_xyxy=bbox,
            confidence=0.9,
            mask=_rect_mask(bbox),
        )
        for bbox in bboxes
    )
    return FrameAppearance(
        scene_id=scene_id,
        frame_number=frame_number,
        frame_path=f"scene_{scene_id}_frame_{frame_number}.jpg",
        person_count=len(bboxes),
        person_colors=(),
        confidence=0.9,
        status="ok",
        detections=detections,
    )


def _rect_mask(bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def test_link_tracks_across_frames():
    config = AppearanceConfig()
    frames = [
        _frame(0, 0, [(20, 20, 80, 120)]),
        _frame(0, 1, [(22, 22, 82, 122)]),
    ]
    tracks = link_tracks(frames, config)
    assert len(tracks) == 1
    assert tracks[0].frame_count == 2


def test_select_dominant_track_consistent_prefers_more_frames():
    config = AppearanceConfig(dominant_track_policy="consistent")
    frames = [
        _frame(0, 0, [(20, 20, 60, 100), (100, 20, 160, 140)]),
        _frame(0, 1, [(22, 22, 62, 102)]),
        _frame(0, 2, [(24, 24, 64, 104)]),
    ]
    tracks = link_tracks(frames, config)
    dominant = select_dominant_track(tracks, config.dominant_track_policy)
    assert dominant is not None
    assert dominant.frame_count == 3


def test_select_dominant_track_biggest_by_area():
    config = AppearanceConfig(dominant_track_policy="biggest")
    frames = [
        _frame(0, 0, [(20, 20, 60, 100), (100, 20, 180, 180)]),
        _frame(0, 1, [(100, 20, 180, 180)]),
    ]
    tracks = link_tracks(frames, config)
    dominant = select_dominant_track(tracks, config.dominant_track_policy)
    assert dominant is not None
    assert dominant.median_mask_area() > 60 * 80
