from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd

from broadcast_pipeline.config import PipelineConfig
from broadcast_pipeline.progress import ProgressTracker
from src.camera_assignemnt.scene_classifier.classifier import classify_scene, resolve_scene_mlp
from src.camera_assignemnt.scene_classifier.config import Config as SceneConfig


def _project_root(output_dir: Path) -> Path:
    return output_dir.resolve().parent.parent


def classify_scenes(config: PipelineConfig, frame_index: pd.DataFrame) -> pd.DataFrame:
    scene_config = SceneConfig()
    project_root = _project_root(config.output_dir)
    resolve_scene_mlp(scene_config, project_root=project_root)
    camera_df = frame_index[frame_index["sample_role"] == "camera"].copy()
    frame_votes: list[dict] = []
    progress = ProgressTracker(len(camera_df), "Scene classification")

    for row in camera_df.itertuples(index=False):
        frame_path = Path(getattr(row, "frame_path"))
        image = cv2.imread(str(frame_path))
        if image is None:
            progress.advance()
            continue
        scene_type, court_ratio, _ = classify_scene(
            image, scene_config, project_root=project_root
        )
        frame_votes.append(
            {
                "scene_id": int(getattr(row, "scene_id")),
                "frame_number": int(getattr(row, "frame_number")),
                "scene_type": scene_type,
                "court_ratio": court_ratio,
            }
        )
        progress.advance()

    if not frame_votes:
        return pd.DataFrame(columns=["scene_id", "scene_type", "vote_counts_json"])

    votes_df = pd.DataFrame(frame_votes)
    scene_rows: list[dict] = []
    for scene_id, group in votes_df.groupby("scene_id"):
        counts = Counter(group["scene_type"].tolist())
        winner = counts.most_common(1)[0][0]
        scene_rows.append(
            {
                "scene_id": int(scene_id),
                "scene_type": winner,
                "vote_counts_json": json.dumps(dict(counts)),
            }
        )

    return pd.DataFrame(scene_rows).sort_values("scene_id")
