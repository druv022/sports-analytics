from pathlib import Path

import cv2
import pandas as pd
from scenedetect import ContentDetector, SceneManager, open_video

VIDEO_PATH = "data/Untitled.mp4"
OUTPUT_DIR = Path("data/scene_samples")
CSV_PATH = Path("data/scene_samples.csv")
DETECTOR_THRESHOLD = 27.0


def scene_frame_numbers(start, end):
    """Return (image_idx, frame_num) for start, middle, and end of a scene."""
    last_frame = end.frame_num - 1
    candidates = [
        (0, start.frame_num),
        (1, (start.frame_num + last_frame) // 2),
        (2, last_frame),
    ]
    seen = set()
    result = []
    for image_idx, frame_num in candidates:
        if frame_num not in seen:
            seen.add(frame_num)
            result.append((image_idx, frame_num))
    return result


def detect_scenes(video_path: str):
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=DETECTOR_THRESHOLD))
    scene_manager.detect_scenes(video, show_progress=True)
    return video, scene_manager.get_scene_list()


def build_targets(scene_list):
    """Map frame_num -> metadata rows to write when that frame is read."""
    targets = {}
    for scene_idx, (start, end) in enumerate(scene_list):
        for image_idx, frame_num in scene_frame_numbers(start, end):
            rel_path = OUTPUT_DIR / f"scene_{scene_idx}_frame_{frame_num}.jpg"
            targets.setdefault(frame_num, []).append(
                {
                    "scene_id": scene_idx,
                    "image_idx": image_idx,
                    "frame_number": frame_num,
                    "frame_path": str(rel_path),
                }
            )
    return targets


def extract_frames(video, targets):
    """Single linear pass through the video — much faster than repeated seek()."""
    video.reset()
    rows = []
    frame_num = 0

    while True:
        frame = video.read()
        if frame is False:
            break

        if frame_num in targets:
            pos = video.position
            for meta in targets[frame_num]:
                path = Path(meta["frame_path"])
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f"Failed to write {path}")

                rows.append(
                    {
                        **meta,
                        "timecode": pos.get_timecode(),
                        "seconds": pos.seconds,
                        "height": frame.shape[0],
                        "width": frame.shape[1],
                    }
                )

        frame_num += 1

    return rows


def extract_frames_from_video(video_path: str = VIDEO_PATH):
    video_path = str(video_path)
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    video, scene_list = detect_scenes(video_path)
    targets = build_targets(scene_list)
    rows = extract_frames(video, targets)

    df_scene_samples = pd.DataFrame(rows).sort_values(["scene_id", "image_idx"])
    df_scene_samples.to_csv(CSV_PATH, index=False)
    print(f"Saved {len(df_scene_samples)} scene samples to {OUTPUT_DIR}")
    print(f"Wrote metadata to {CSV_PATH}")


if __name__ == "__main__":
    extract_frames_from_video(VIDEO_PATH)