"""Lightweight bbox IoU tracking for dominant person selection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.person_appearance.config import AppearanceConfig, DominantTrackPolicy
from src.person_appearance.types import FrameAppearance, PersonDetection


@dataclass
class TrackObservation:
    frame: FrameAppearance
    detection: PersonDetection
    mask_area: int


@dataclass
class PersonTrack:
    track_id: int
    observations: list[TrackObservation] = field(default_factory=list)

    @property
    def frame_count(self) -> int:
        return len(self.observations)

    def median_mask_area(self) -> int:
        if not self.observations:
            return 0
        areas = [obs.mask_area for obs in self.observations]
        return int(np.median(areas))


def bbox_iou(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return float(inter / (area_a + area_b - inter + 1e-6))


def _mask_area(detection: PersonDetection) -> int:
    if detection.mask is None:
        x1, y1, x2, y2 = detection.bbox_xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)
    return int(np.sum(detection.mask > 0))


def link_tracks(
    frames: list[FrameAppearance],
    config: AppearanceConfig,
) -> list[PersonTrack]:
    """Greedy IoU linking across consecutive frames sorted by frame number."""
    ordered = sorted(frames, key=lambda frame: frame.frame_number)
    tracks: list[PersonTrack] = []
    next_id = 0

    for frame_idx, frame in enumerate(ordered):
        detections = list(frame.detections)
        if not detections:
            continue

        if frame_idx == 0:
            for det in detections:
                tracks.append(
                    PersonTrack(
                        track_id=next_id,
                        observations=[
                            TrackObservation(frame=frame, detection=det, mask_area=_mask_area(det))
                        ],
                    )
                )
                next_id += 1
            continue

        prev_frame_number = ordered[frame_idx - 1].frame_number
        open_tracks = [
            track
            for track in tracks
            if track.observations[-1].frame.frame_number == prev_frame_number
        ]
        unmatched = list(range(len(detections)))
        matches: list[tuple[float, int, int]] = []

        for track_idx, track in enumerate(open_tracks):
            last_det = track.observations[-1].detection
            for det_idx in unmatched:
                iou = bbox_iou(last_det.bbox_xyxy, detections[det_idx].bbox_xyxy)
                if iou >= config.track_match_iou:
                    matches.append((iou, track_idx, det_idx))

        matches.sort(reverse=True)
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for _, track_idx, det_idx in matches:
            if track_idx in used_tracks or det_idx in used_dets:
                continue
            det = detections[det_idx]
            open_tracks[track_idx].observations.append(
                TrackObservation(frame=frame, detection=det, mask_area=_mask_area(det))
            )
            used_tracks.add(track_idx)
            used_dets.add(det_idx)

        for det_idx, det in enumerate(detections):
            if det_idx in used_dets:
                continue
            tracks.append(
                PersonTrack(
                    track_id=next_id,
                    observations=[
                        TrackObservation(frame=frame, detection=det, mask_area=_mask_area(det))
                    ],
                )
            )
            next_id += 1

    return tracks


def select_dominant_track(
    tracks: list[PersonTrack],
    policy: DominantTrackPolicy,
) -> PersonTrack | None:
    if not tracks:
        return None
    if policy == "biggest":
        return max(tracks, key=lambda track: track.median_mask_area())
    return max(
        tracks,
        key=lambda track: (track.frame_count, track.median_mask_area()),
    )
