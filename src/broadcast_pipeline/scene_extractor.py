from __future__ import annotations

import json
from pathlib import Path

import cv2
import pandas as pd
from scenedetect import ContentDetector, SceneManager, open_video

from broadcast_pipeline.config import PipelineConfig
from broadcast_pipeline.progress import ProgressTracker, log_info
from broadcast_pipeline.scenes_io import load_scenes
from broadcast_pipeline.types import FrameRecord, Scene, VideoMeta


def camera_sample_frames(start_frame: int, end_frame: int, n: int) -> list[int]:
    last_frame = end_frame - 1
    if last_frame < start_frame:
        return [start_frame]
    if n <= 1:
        return [start_frame]
    if n == 2:
        return [start_frame, last_frame]
    step = (last_frame - start_frame) / (n - 1)
    frames = [start_frame + int(round(step * i)) for i in range(n)]
    seen: set[int] = set()
    result: list[int] = []
    for frame in frames:
        clamped = max(start_frame, min(last_frame, frame))
        if clamped not in seen:
            seen.add(clamped)
            result.append(clamped)
    return result


def ocr_sample_frames(
    start_frame: int,
    end_frame: int,
    fps: float,
    ocr_hz: float,
) -> list[int]:
    last_frame = end_frame - 1
    if last_frame < start_frame:
        return []
    if ocr_hz <= 0 or fps <= 0:
        return []
    step = max(1, int(round(fps / ocr_hz)))
    return list(range(start_frame, last_frame + 1, step))


def detect_scenes(video_path: Path, threshold: float):
    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video, show_progress=True)
    return video, scene_manager.get_scene_list()


def build_extraction_targets(
    scene_list,
    meta: VideoMeta,
    config: PipelineConfig,
) -> tuple[list[Scene], dict[int, list[dict]]]:
    scenes: list[Scene] = []
    targets: dict[int, list[dict]] = {}

    for scene_idx, (start, end) in enumerate(scene_list):
        start_frame = start.frame_num
        end_frame = end.frame_num
        scenes.append(
            Scene(
                scene_id=scene_idx,
                start_frame=start_frame,
                end_frame=end_frame,
                start_sec=start.get_seconds(),
                end_sec=end.get_seconds(),
            )
        )

        camera_frames = camera_sample_frames(
            start_frame, end_frame, config.camera_samples_per_scene
        )
        ocr_frames = ocr_sample_frames(
            start_frame, end_frame, meta.fps, config.ocr_samples_per_sec
        )
        role_by_frame: dict[int, set[str]] = {}
        for fn in camera_frames:
            role_by_frame.setdefault(fn, set()).add("camera")
        for fn in ocr_frames:
            role_by_frame.setdefault(fn, set()).add("ocr")

        frames_dir = config.output_dir / "frames"
        for frame_num, roles in role_by_frame.items():
            rel_path = frames_dir / f"scene_{scene_idx}_frame_{frame_num}.jpg"
            for role in roles:
                targets.setdefault(frame_num, []).append(
                    {
                        "scene_id": scene_idx,
                        "frame_number": frame_num,
                        "frame_path": str(rel_path),
                        "sample_role": role,
                    }
                )

    return scenes, targets


def extract_scenes_and_frames(
    config: PipelineConfig,
    meta: VideoMeta,
) -> tuple[list[Scene], list[FrameRecord], pd.DataFrame]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = config.output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    log_info("  Detecting scene cuts (PySceneDetect progress below)...")
    video, scene_list = detect_scenes(config.video_path, config.detector_threshold)
    log_info(f"  Detected {len(scene_list)} scene(s)")
    scenes, targets = build_extraction_targets(scene_list, meta, config)
    n_target_frames = len(targets)
    log_info(f"  Extracting {n_target_frames} unique frame(s) from video")

    try:
        video.reset()
        rows: list[dict] = []
        frame_num = 0
        progress = ProgressTracker(meta.frame_count, "Video scan", step_pct=10)
        while True:
            frame = video.read()
            if frame is False:
                break
            if frame_num in targets:
                pos = video.position
                for meta_row in targets[frame_num]:
                    path = Path(meta_row["frame_path"])
                    if not path.exists():
                        if not cv2.imwrite(str(path), frame):
                            raise RuntimeError(f"Failed to write {path}")
                    rows.append(
                        {
                            **meta_row,
                            "timecode": pos.get_timecode(),
                            "seconds": pos.seconds,
                            "height": frame.shape[0],
                            "width": frame.shape[1],
                        }
                    )
            frame_num += 1
            progress.advance()
    finally:
        if hasattr(video, "close"):
            video.close()
        del video, scene_list, targets

    df = pd.DataFrame(rows).sort_values(["scene_id", "sample_role", "frame_number"])
    frame_records = [
        FrameRecord(
            scene_id=int(r.scene_id),
            frame_number=int(r.frame_number),
            seconds=float(r.seconds),
            frame_path=Path(r.frame_path),
            sample_role=r.sample_role,
        )
        for r in df.itertuples(index=False)
    ]
    return scenes, frame_records, df


def save_scenes(scenes: list[Scene], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "scene_id": s.scene_id,
            "start_frame": s.start_frame,
            "end_frame": s.end_frame,
            "start_sec": s.start_sec,
            "end_sec": s.end_sec,
        }
        for s in scenes
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
