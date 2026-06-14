"""Tests for scene sample loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.camera_assignemnt.approach_4.dataset import (
    load_scene_samples,
    load_scene_samples_from_dir,
)

ROOT = Path(__file__).resolve().parents[4]
SAMPLES_DIR = ROOT / "data" / "scene_samples"
METADATA_CSV = ROOT / "data" / "scene_samples.csv"


@pytest.mark.skipif(not SAMPLES_DIR.exists(), reason="scene_samples directory missing")
def test_load_scene_samples_from_dir_picks_middle_frame():
    samples = load_scene_samples_from_dir(SAMPLES_DIR, middle_image_idx=1, load_frames=False)
    assert samples
    scene_ids = [s.scene_id for s in samples]
    assert len(scene_ids) == len(set(scene_ids))


@pytest.mark.skipif(not METADATA_CSV.exists(), reason="scene_samples.csv missing")
def test_load_scene_samples_prefers_csv():
    samples = load_scene_samples(
        samples_dir=SAMPLES_DIR,
        metadata_csv=METADATA_CSV,
        middle_image_idx=1,
        load_frames=False,
    )
    assert samples
    assert all(s.frame_path for s in samples)


@pytest.mark.skipif(not METADATA_CSV.exists(), reason="scene_samples.csv missing")
def test_middle_image_idx_is_one():
    samples = load_scene_samples(
        samples_dir=SAMPLES_DIR,
        metadata_csv=METADATA_CSV,
        middle_image_idx=1,
        load_frames=False,
    )
    assert all(s.image_idx == 1 for s in samples)
