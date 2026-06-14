"""Tests for multi-region HSV histogram features."""

from __future__ import annotations

import cv2
import numpy as np

from src.camera_assignemnt.approach_1.config import Config
from src.camera_assignemnt.approach_1.classifier import multi_region_hsv_histogram, region_rois


def make_distinct_colour_frame(
    height: int,
    width: int,
    hue_offset: int,
) -> np.ndarray:
    """Frame with distinct HSV hue for clustering separation."""
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    hsv[:, :, 0] = hue_offset % 180
    hsv[:, :, 1] = 200
    hsv[:, :, 2] = 200
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_histogram_feature_shape() -> None:
    config = Config(histogram_bins=16)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    feat = multi_region_hsv_histogram(frame, config)
    assert feat.shape == (6 * 2 * 16,)
    assert feat.dtype.name == "float32"


def test_histogram_is_normalised_per_region() -> None:
    config = Config(histogram_bins=8)
    frame = make_distinct_colour_frame(240, 320, hue_offset=60)
    feat = multi_region_hsv_histogram(frame, config)
    bins = config.histogram_bins
    for i in range(6):
        region = feat[i * 2 * bins : (i + 1) * 2 * bins]
        assert region.sum() > 0


def test_region_rois_cover_frame() -> None:
    rois = region_rois(900, 600)
    assert set(rois.keys()) == {
        "full",
        "top_left",
        "top_right",
        "bottom_left",
        "bottom_right",
        "center",
    }
    x0, y0, x1, y1 = rois["center"]
    assert x1 - x0 == 300
    assert y1 - y0 == 200
