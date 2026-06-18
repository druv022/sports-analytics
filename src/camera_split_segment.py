from pathlib import Path

from broadcast_pipeline.config import PipelineConfig
from broadcast_pipeline.scene_extractor import extract_scenes_and_frames, save_scenes
from broadcast_pipeline.video_meta import probe_video, save_video_meta

VIDEO_PATH = "data/Untitled.mp4"
OUTPUT_DIR = Path("data/scene_samples")
CSV_PATH = Path("data/scene_samples.csv")
DETECTOR_THRESHOLD = 27.0


def extract_frames_from_video(video_path: str = VIDEO_PATH):
    config = PipelineConfig(
        video_path=Path(video_path),
        output_dir=Path("data/pipeline_legacy"),
        detector_threshold=DETECTOR_THRESHOLD,
        camera_samples_per_scene=3,
        ocr_samples_per_sec=0.0,
    )
    meta = probe_video(config.video_path)
    scenes, _, frame_index = extract_scenes_and_frames(config, meta)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    legacy_rows = []
    for row in frame_index[frame_index["sample_role"] == "camera"].itertuples(index=False):
        scene_id = int(getattr(row, "scene_id"))
        frame_number = int(getattr(row, "frame_number"))
        legacy_name = OUTPUT_DIR / f"scene_{scene_id}_frame_{frame_number}.jpg"
        src = Path(getattr(row, "frame_path"))
        if src.resolve() != legacy_name.resolve():
            legacy_name.write_bytes(src.read_bytes())
        image_idx = 0
        camera_frames = frame_index[
            (frame_index["scene_id"] == scene_id) & (frame_index["sample_role"] == "camera")
        ].sort_values("frame_number")["frame_number"].tolist()
        if frame_number in camera_frames:
            image_idx = camera_frames.index(frame_number)
        legacy_rows.append(
            {
                "scene_id": scene_id,
                "image_idx": image_idx,
                "frame_number": frame_number,
                "frame_path": str(legacy_name),
                "timecode": getattr(row, "timecode"),
                "seconds": float(getattr(row, "seconds")),
                "height": int(getattr(row, "height")),
                "width": int(getattr(row, "width")),
            }
        )

    import pandas as pd

    df_scene_samples = pd.DataFrame(legacy_rows).sort_values(["scene_id", "image_idx"])
    df_scene_samples.to_csv(CSV_PATH, index=False)
    save_video_meta(meta, config.output_dir / "video_meta.json")
    save_scenes(scenes, config.output_dir / "scenes.json")
    print(f"Saved {len(df_scene_samples)} scene samples to {OUTPUT_DIR}")
    print(f"Wrote metadata to {CSV_PATH}")


if __name__ == "__main__":
    extract_frames_from_video(VIDEO_PATH)
