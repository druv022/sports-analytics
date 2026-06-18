"""Tests for folder-based scene classification dataset loading."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.camera_assignemnt.scene_classifier.config import Config
from src.camera_assignemnt.scene_classifier.classifier import (
    folder_to_label,
    load_classification_dataset,
    parse_scene_id,
)


def test_parse_scene_id() -> None:
    assert parse_scene_id("scene_12_frame_3456.jpg") == "12"
    assert parse_scene_id("other_name.jpg") == "other_name"


def test_folder_to_label_mapping() -> None:
    assert folder_to_label("full") == "full_court"
    assert folder_to_label("closs-ups") == "closeup"


def test_folder_to_label_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown classification folder"):
        folder_to_label("unknown-folder")


def test_load_classification_dataset(tmp_path: Path) -> None:
    full_dir = tmp_path / "full"
    close_dir = tmp_path / "closs-ups"
    full_dir.mkdir()
    close_dir.mkdir()

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[:, :] = (180, 120, 60)
    cv2.imwrite(str(full_dir / "scene_1_frame_10.jpg"), frame)

    close_frame = np.zeros((120, 160, 3), dtype=np.uint8)
    close_frame[:, :] = (180, 200, 220)
    cv2.imwrite(str(close_dir / "scene_2_frame_20.jpg"), close_frame)

    config = Config(histogram_bins=8)
    dataset = load_classification_dataset(tmp_path, config)

    assert len(dataset) == 2
    assert dataset.X.shape == (2, 6 * 2 * config.histogram_bins)
    assert set(dataset.y.tolist()) == {"full_court", "closeup"}
    assert set(dataset.groups.tolist()) == {"1", "2"}


def test_load_classification_dataset_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_classification_dataset(tmp_path / "missing", Config())


def test_load_classification_dataset_empty_dir(tmp_path: Path) -> None:
    (tmp_path / "full").mkdir(parents=True)
    dataset = load_classification_dataset(tmp_path, Config())
    assert len(dataset) == 0
    assert dataset.X.shape == (0, 0)
