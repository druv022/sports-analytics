from __future__ import annotations

import json
from pathlib import Path

import cv2
from scenedetect import open_video

from broadcast_pipeline.types import VideoMeta


def _probe_opencv(path: Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return fps, frame_count, width, height


def probe_video(path: str | Path) -> VideoMeta:
    video_path = Path(path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cv_fps, cv_frames, width, height = _probe_opencv(video_path)

    sd_fps = cv_fps
    sd_frames = cv_frames
    try:
        video = open_video(str(video_path))
        if hasattr(video, "frame_rate") and video.frame_rate:
            sd_fps = float(video.frame_rate)
        if hasattr(video, "duration") and video.duration and sd_fps > 0:
            sd_frames = int(video.duration.get_frames())
    except Exception:
        pass

    fps = sd_fps if sd_fps > 0 else cv_fps
    frame_count = max(sd_frames, cv_frames, 1)
    fps_source = "scenedetect" if sd_fps > 0 else "opencv"
    if abs(sd_fps - cv_fps) > 0.5 and cv_fps > 0:
        fps_source = "scenedetect+opencv"

    duration_sec = frame_count / fps if fps > 0 else 0.0
    return VideoMeta(
        path=video_path.resolve(),
        fps=fps,
        frame_count=frame_count,
        duration_sec=duration_sec,
        width=width,
        height=height,
        fps_source=fps_source,
    )


def save_video_meta(meta: VideoMeta, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "path": str(meta.path),
        "fps": meta.fps,
        "frame_count": meta.frame_count,
        "duration_sec": meta.duration_sec,
        "width": meta.width,
        "height": meta.height,
        "fps_source": meta.fps_source,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_video_meta(path: Path) -> VideoMeta:
    data = json.loads(path.read_text(encoding="utf-8"))
    return VideoMeta(
        path=Path(data["path"]),
        fps=float(data["fps"]),
        frame_count=int(data["frame_count"]),
        duration_sec=float(data["duration_sec"]),
        width=int(data["width"]),
        height=int(data["height"]),
        fps_source=str(data.get("fps_source", "scenedetect")),
    )
