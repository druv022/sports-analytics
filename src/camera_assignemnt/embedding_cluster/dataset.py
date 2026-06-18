"""Load scene samples for clustering (no ground-truth labels)."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import pandas as pd

from typing import Literal

from src.camera_assignemnt.embedding_cluster.models import Frame, SceneSample

SampleFilter = Literal["middle", "camera", "all"]

_SCENE_PATTERN = re.compile(r"scene_(\d+)_frame_(\d+)", re.IGNORECASE)
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _parse_scene_id(path: Path) -> str | None:
    match = _SCENE_PATTERN.search(path.stem)
    return match.group(1) if match else None


def _parse_frame_number(path: Path) -> int:
    match = _SCENE_PATTERN.search(path.stem)
    return int(match.group(2)) if match else 0


def load_image(path: str | Path) -> Frame:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def resolve_sample_frame(sample: SceneSample) -> Frame:
    """Return the cached frame or load it from disk on demand."""
    if sample.frame is not None:
        return sample.frame
    return load_image(sample.frame_path)


def _filter_rows(
    df: pd.DataFrame,
    sample_filter: SampleFilter,
    middle_image_idx: int,
) -> pd.DataFrame:
    if sample_filter == "middle":
        return _middle_row_per_scene(df, middle_image_idx)
    if sample_filter == "camera":
        if "sample_role" in df.columns:
            filtered = df[df["sample_role"] == "camera"]
            if not filtered.empty:
                sort_cols = ["scene_id", "frame_number"] if "frame_number" in filtered.columns else ["scene_id"]
                return filtered.sort_values(sort_cols)
        return _middle_row_per_scene(df, middle_image_idx)
    if sample_filter == "all":
        sort_cols = ["scene_id", "frame_number"] if "frame_number" in df.columns else ["scene_id"]
        return df.sort_values(sort_cols)
    raise ValueError(f"Unknown sample_filter: {sample_filter!r}")


def _middle_row_per_scene(df: pd.DataFrame, middle_image_idx: int) -> pd.DataFrame:
    if "image_idx" in df.columns:
        middle = df[df["image_idx"] == middle_image_idx]
        if not middle.empty:
            return middle.sort_values("scene_id")
        return df.sort_values(["scene_id", "image_idx"]).groupby("scene_id", as_index=False).nth(1)
    return df.sort_values("scene_id").drop_duplicates("scene_id", keep="first")


def load_scene_samples_from_csv(
    metadata_csv: str | Path,
    samples_dir: str | Path | None = None,
    middle_image_idx: int = 1,
    load_frames: bool = True,
    sample_filter: SampleFilter = "middle",
) -> list[SceneSample]:
    """Load scene sample frames from metadata CSV."""
    csv_path = Path(metadata_csv)
    root = csv_path.parent.parent if samples_dir is None else Path(samples_dir).parent
    base_dir = Path(samples_dir) if samples_dir else csv_path.parent

    df = pd.read_csv(csv_path)
    required = {"scene_id", "frame_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata CSV missing columns: {sorted(missing)}")

    df = _filter_rows(df, sample_filter, middle_image_idx)
    samples: list[SceneSample] = []

    for scene_idx, row in enumerate(df.itertuples(index=False)):
        frame_path = Path(getattr(row, "frame_path"))
        if not frame_path.is_absolute():
            candidates = [
                frame_path,
                base_dir / frame_path.name,
                root / frame_path,
            ]
            resolved = next((p for p in candidates if p.exists()), frame_path)
            frame_path = resolved

        scene_id = str(getattr(row, "scene_id"))
        image_idx = int(getattr(row, "image_idx", middle_image_idx))
        frame = load_image(frame_path) if load_frames else None
        samples.append(
            SceneSample(
                scene_idx=scene_idx,
                scene_id=scene_id,
                image_idx=image_idx,
                frame_path=str(frame_path),
                frame=frame,
            )
        )

    return samples


def load_scene_samples_from_dir(
    samples_dir: str | Path,
    middle_image_idx: int = 1,
    load_frames: bool = True,
) -> list[SceneSample]:
    """Discover scenes from filenames and pick the middle frame per scene_id."""
    directory = Path(samples_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Samples directory not found: {directory}")

    by_scene: dict[str, list[Path]] = {}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        scene_id = _parse_scene_id(path)
        if scene_id is None:
            continue
        by_scene.setdefault(scene_id, []).append(path)

    samples: list[SceneSample] = []
    for scene_idx, scene_id in enumerate(sorted(by_scene, key=lambda s: int(s))):
        paths = sorted(by_scene[scene_id], key=_parse_frame_number)
        if len(paths) == 1:
            chosen = paths[0]
            image_idx = 0
        else:
            target_idx = min(middle_image_idx, len(paths) - 1)
            chosen = paths[target_idx]
            image_idx = target_idx

        frame = load_image(chosen) if load_frames else None
        samples.append(
            SceneSample(
                scene_idx=scene_idx,
                scene_id=scene_id,
                image_idx=image_idx,
                frame_path=str(chosen),
                frame=frame,
            )
        )

    return samples


def load_scene_samples(
    samples_dir: str | Path = "data/scene_samples",
    metadata_csv: str | Path | None = "data/scene_samples.csv",
    middle_image_idx: int = 1,
    load_frames: bool = True,
    sample_filter: SampleFilter = "middle",
) -> list[SceneSample]:
    """Prefer metadata CSV when present; otherwise scan the samples directory."""
    csv_path = Path(metadata_csv) if metadata_csv else None
    if csv_path and csv_path.exists():
        return load_scene_samples_from_csv(
            csv_path,
            samples_dir=samples_dir,
            middle_image_idx=middle_image_idx,
            load_frames=load_frames,
            sample_filter=sample_filter,
        )
    return load_scene_samples_from_dir(
        samples_dir,
        middle_image_idx=middle_image_idx,
        load_frames=load_frames,
    )
