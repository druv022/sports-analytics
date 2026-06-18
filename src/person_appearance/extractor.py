"""Frame and scene appearance extraction."""

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np
import pandas as pd

from src.person_appearance.color_profile import (
    collect_masked_bgr_pixels,
    mask_pixel_count,
    primary_color_from_detection,
    primary_color_from_pixels,
)
from src.person_appearance.config import AppearanceConfig
from src.person_appearance.segmenter import PersonSegmenter, create_segmenter
from src.person_appearance.signature import appearance_signature_string
from src.person_appearance.track_linking import link_tracks, select_dominant_track
from src.person_appearance.types import FrameAppearance, PersonDetection, SceneAppearance


def _frame_status(person_count: int, mean_conf: float, config: AppearanceConfig) -> str:
    if person_count == 0:
        return "no_person"
    if mean_conf < config.min_confidence:
        return "low_conf"
    return "ok"


def _pick_biggest_detection(
    image: np.ndarray,
    detections: list[tuple[tuple[int, int, int, int], float, np.ndarray | None]],
    config: AppearanceConfig,
) -> tuple[tuple[int, int, int, int], float, np.ndarray | None] | None:
    if not detections:
        return None
    return max(
        detections,
        key=lambda item: mask_pixel_count(image, item[0], item[2], config),
    )


def analyze_frame(
    image: np.ndarray,
    *,
    scene_id: int,
    frame_number: int,
    frame_path: str,
    config: AppearanceConfig,
    segmenter: PersonSegmenter,
) -> FrameAppearance:
    raw_detections = segmenter.detect(image)
    detections: list[PersonDetection] = []

    for bbox, confidence, mask in raw_detections:
        detections.append(
            PersonDetection(
                bbox_xyxy=bbox,
                confidence=confidence,
                mask=mask,
                clothing_color=None,
            )
        )

    person_count = len(detections)
    mean_conf = (
        float(np.mean([d.confidence for d in detections])) if detections else 0.0
    )
    status = _frame_status(person_count, mean_conf, config)

    primary_label = ""
    primary_bgr: tuple[int, int, int] | None = None
    biggest = _pick_biggest_detection(image, raw_detections, config)
    if biggest is not None:
        bbox, _, mask = biggest
        result = primary_color_from_detection(image, bbox, mask, config)
        primary_label = result.label
        primary_bgr = result.bgr

    person_colors = (primary_label,) if primary_label else ()

    return FrameAppearance(
        scene_id=scene_id,
        frame_number=frame_number,
        frame_path=frame_path,
        person_count=person_count,
        person_colors=person_colors,
        confidence=mean_conf,
        status=status,  # type: ignore[arg-type]
        primary_bgr=primary_bgr,
        detections=tuple(detections),
    )


def _scene_primary_from_track(
    track,
    config: AppearanceConfig,
) -> tuple[str, tuple[int, int, int] | None]:
    pixel_chunks: list[np.ndarray] = []
    for obs in track.observations:
        image = load_frame_image(obs.frame.frame_path)
        if image is None:
            continue
        pixels = collect_masked_bgr_pixels(
            image,
            obs.detection.bbox_xyxy,
            obs.detection.mask,
            config,
        )
        if pixels.size:
            pixel_chunks.append(pixels)

    if not pixel_chunks:
        return "", None

    pooled = np.vstack(pixel_chunks)
    result = primary_color_from_pixels(pooled, config)
    return result.label, result.bgr


def build_scene_appearances(
    frame_appearances: list[FrameAppearance],
    scene_type_lookup: dict[int, str],
    config: AppearanceConfig,
) -> list[SceneAppearance]:
    scenes: list[SceneAppearance] = []
    by_scene: dict[int, list[FrameAppearance]] = {}
    for frame in frame_appearances:
        by_scene.setdefault(frame.scene_id, []).append(frame)

    for scene_id, frames in sorted(by_scene.items()):
        count_votes = Counter(frame.person_count for frame in frames)
        person_count = count_votes.most_common(1)[0][0]
        confidences = [frame.confidence for frame in frames if frame.person_count > 0]
        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        statuses = [frame.status for frame in frames]
        if all(s == "no_person" for s in statuses):
            status = "no_person"
        elif mean_conf < config.min_confidence and person_count > 0:
            status = "low_conf"
        elif "low_conf" in statuses and person_count > 0:
            status = "low_conf"
        else:
            status = "ok" if person_count > 0 else "no_person"

        tracks = link_tracks(frames, config)
        dominant = select_dominant_track(tracks, config.dominant_track_policy)
        primary_label = ""
        primary_bgr: tuple[int, int, int] | None = None
        dominant_track_frames = 0
        dominant_track_median_area = 0

        if dominant is not None:
            primary_label, primary_bgr = _scene_primary_from_track(dominant, config)
            dominant_track_frames = dominant.frame_count
            dominant_track_median_area = dominant.median_mask_area()

        if not primary_label and status != "no_person":
            frame_labels = [frame.person_colors[0] for frame in frames if frame.person_colors]
            if frame_labels:
                primary_label = Counter(frame_labels).most_common(1)[0][0]

        person_colors = (primary_label,) if primary_label else ()
        signature = appearance_signature_string(primary_label)

        scenes.append(
            SceneAppearance(
                scene_id=scene_id,
                scene_type=scene_type_lookup.get(scene_id, "closeup"),
                person_count=person_count,
                person_colors=person_colors,
                appearance_signature=signature,
                confidence=mean_conf,
                status=status,  # type: ignore[arg-type]
                primary_bgr=primary_bgr,
                dominant_track_frames=dominant_track_frames,
                dominant_track_median_area=dominant_track_median_area,
            )
        )
    return scenes


def frame_appearances_to_dataframe(frames: list[FrameAppearance]) -> pd.DataFrame:
    rows = []
    for frame in frames:
        rows.append(
            {
                "scene_id": frame.scene_id,
                "frame_number": frame.frame_number,
                "frame_path": frame.frame_path,
                "person_count": frame.person_count,
                "person_colors_json": list(frame.person_colors),
                "primary_bgr_json": list(frame.primary_bgr) if frame.primary_bgr else [],
                "confidence": round(frame.confidence, 4),
                "status": frame.status,
            }
        )
    return pd.DataFrame(rows)


def scene_appearances_to_dataframe(scenes: list[SceneAppearance]) -> pd.DataFrame:
    rows = []
    for scene in scenes:
        rows.append(
            {
                "scene_id": scene.scene_id,
                "scene_type": scene.scene_type,
                "person_count": scene.person_count,
                "person_colors_json": list(scene.person_colors),
                "appearance_signature": scene.appearance_signature,
                "primary_bgr_json": list(scene.primary_bgr) if scene.primary_bgr else [],
                "dominant_track_frames": scene.dominant_track_frames,
                "dominant_track_median_area": scene.dominant_track_median_area,
                "confidence": round(scene.confidence, 4),
                "status": scene.status,
            }
        )
    return pd.DataFrame(rows).sort_values("scene_id")


def load_frame_image(frame_path: str) -> np.ndarray | None:
    image = cv2.imread(frame_path)
    return image


def default_segmenter(config: AppearanceConfig) -> PersonSegmenter:
    return create_segmenter(config)
