from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.person_appearance.color_profile import (
    collect_masked_bgr_pixels,
    primary_color_from_detection,
    quantize_lab,
)
from src.person_appearance.config import AppearanceConfig


def _solid_bgr_patch(color_bgr: tuple[int, int, int], size: int = 80) -> np.ndarray:
    patch = np.zeros((size, size, 3), dtype=np.uint8)
    patch[:, :] = color_bgr
    return patch


def test_quantize_lab_red_and_blue():
    config = AppearanceConfig()
    red_cv = np.array([53.0 * 255 / 100, 80.0 + 128, 67.0 + 128], dtype=np.float32)
    blue_cv = np.array([32.0 * 255 / 100, 79.0 + 128, -108.0 + 128], dtype=np.float32)
    assert quantize_lab(red_cv, config) == "red"
    assert quantize_lab(blue_cv, config) == "blue"


def test_histogram_red_masked_patch():
    config = AppearanceConfig(color_method="histogram_hsv")
    image = _solid_bgr_patch((20, 20, 255))
    bbox = (10, 10, 70, 70)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[10:70, 10:70] = 255
    result = primary_color_from_detection(image, bbox, mask, config)
    assert result.label in {"red", "orange"}


def test_mask_excludes_outside_pixels():
    config = AppearanceConfig()
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[20:80, 20:50] = (0, 0, 255)
    image[20:80, 50:80] = (255, 0, 0)
    bbox = (20, 20, 80, 80)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:50] = 255
    pixels = collect_masked_bgr_pixels(image, bbox, mask, config)
    assert pixels.shape[0] > 0
    assert np.all(pixels[:, 2] > 200)
    assert np.all(pixels[:, 0] < 50)


def test_neutral_for_gray_mask_pixels():
    config = AppearanceConfig(color_method="histogram_hsv")
    image = _solid_bgr_patch((128, 128, 128))
    bbox = (10, 10, 70, 70)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[10:70, 10:70] = 255
    result = primary_color_from_detection(image, bbox, mask, config)
    assert result.label in {"neutral", "white", "black"}
