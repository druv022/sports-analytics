"""Shared fixtures for camera assignment tests."""

from __future__ import annotations

import numpy as np
import pytest

from src.camera_assignemnt.approach_1.config import Config


@pytest.fixture
def config() -> Config:
    """Default pipeline config with deterministic random seed."""
    return Config(ransac_seed=42, dbscan_min_samples=2)


def make_solid_frame(
    height: int = 1080,
    width: int = 1920,
    bgr: tuple[int, int, int] = (120, 80, 40),
) -> np.ndarray:
    """Create a solid-colour BGR frame."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = bgr
    return frame


def make_hard_court_frame(height: int = 480, width: int = 640) -> np.ndarray:
    """Synthetic hard court: mostly blue-green in HSV hard range."""
    return make_solid_frame(height, width, bgr=(180, 120, 60))


def make_closeup_frame(height: int = 480, width: int = 640) -> np.ndarray:
    """Synthetic close-up: skin-tone dominant, little court colour."""
    return make_solid_frame(height, width, bgr=(180, 200, 220))
