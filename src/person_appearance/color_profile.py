"""Clothing color extraction with mask-gated histogram and lighting normalization."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from src.person_appearance.config import AppearanceConfig, PALETTE_LAB_CENTROIDS


@dataclass(frozen=True)
class PrimaryColorResult:
    label: str
    bgr: tuple[int, int, int]


def _gray_world_balance(bgr: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if bgr.size == 0:
        return bgr
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    if float(np.median(hsv[:, :, 1])) > 60:
        return bgr
    result = bgr.astype(np.float32)
    means = result.reshape(-1, 3).mean(axis=0)
    gray = float(means.mean())
    for channel in range(3):
        if means[channel] > 1e-3:
            result[:, :, channel] *= gray / means[channel]
    return np.clip(result, 0, 255).astype(np.uint8)


def _opencv_lab_to_standard(lab: NDArray[np.float32]) -> NDArray[np.float32]:
    l_cv, a_cv, b_cv = lab
    return np.array(
        [l_cv * 100.0 / 255.0, a_cv - 128.0, b_cv - 128.0],
        dtype=np.float32,
    )


def _delta_e(lab_a: NDArray[np.float32], lab_b: NDArray[np.float32]) -> float:
    return float(np.linalg.norm(lab_a - lab_b))


def colors_match(label_a: str, label_b: str, tolerance: float) -> bool:
    if not label_a or not label_b:
        return False
    if label_a == label_b:
        return True
    centroid_a = np.array(PALETTE_LAB_CENTROIDS[label_a], dtype=np.float32)
    centroid_b = np.array(PALETTE_LAB_CENTROIDS[label_b], dtype=np.float32)
    return _delta_e(centroid_a, centroid_b) < tolerance


def quantize_lab(lab_cv: NDArray[np.float32], config: AppearanceConfig) -> str:
    lab = _opencv_lab_to_standard(lab_cv)
    best_label = config.palette[0]
    best_dist = float("inf")
    for label in config.palette:
        centroid = np.array(PALETTE_LAB_CENTROIDS[label], dtype=np.float32)
        dist = _delta_e(lab, centroid)
        if dist < best_dist:
            best_dist = dist
            best_label = label
    return best_label


def quantize_bgr(bgr: tuple[int, int, int], config: AppearanceConfig) -> str:
    pixel = np.array([[list(bgr)]], dtype=np.uint8)
    lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    return quantize_lab(lab, config)


def _region_bounds(
    bbox_xyxy: tuple[int, int, int, int],
    config: AppearanceConfig,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    if config.color_mask_region == "full":
        return x1, y1, x2, y2
    h = max(1, y2 - y1)
    ty1 = y1 + int(h * config.torso_y_start)
    ty2 = y1 + int(h * config.torso_y_end)
    ty2 = max(ty1 + 1, min(ty2, y2))
    return x1, ty1, x2, ty2


def _prepare_mask(
    mask: NDArray[np.uint8] | None,
    region: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    config: AppearanceConfig,
) -> NDArray[np.uint8]:
    x1, y1, x2, y2 = region
    h_img, w_img = image_shape
    x1 = max(0, min(x1, w_img))
    x2 = max(0, min(x2, w_img))
    y1 = max(0, min(y1, h_img))
    y2 = max(0, min(y2, h_img))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0), dtype=np.uint8)

    if mask is None:
        return np.ones((y2 - y1, x2 - x1), dtype=np.uint8) * 255

    mask_crop = mask[y1:y2, x1:x2]
    if mask_crop.size == 0:
        return np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)

    binary = (mask_crop > 0).astype(np.uint8) * 255
    if config.mask_erode_px > 0:
        kernel = np.ones((config.mask_erode_px, config.mask_erode_px), dtype=np.uint8)
        binary = cv2.erode(binary, kernel, iterations=1)
    return binary


def collect_masked_bgr_pixels(
    image: NDArray[np.uint8],
    bbox_xyxy: tuple[int, int, int, int],
    mask: NDArray[np.uint8] | None,
    config: AppearanceConfig,
) -> NDArray[np.uint8]:
    """Return Nx3 BGR pixels inside the person mask (mask pixels only)."""
    region = _region_bounds(bbox_xyxy, config)
    x1, y1, x2, y2 = region
    h_img, w_img = image.shape[:2]
    x1 = max(0, min(x1, w_img))
    x2 = max(0, min(x2, w_img))
    y1 = max(0, min(y1, h_img))
    y2 = max(0, min(y2, h_img))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 3), dtype=np.uint8)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    balanced = _gray_world_balance(crop)
    mask_crop = _prepare_mask(mask, region, image.shape[:2], config)
    if mask_crop.size == 0 or not (mask_crop > 0).any():
        return np.zeros((0, 3), dtype=np.uint8)

    pixel_mask = mask_crop > 0
    pixels = balanced[pixel_mask]
    if pixels.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    return pixels.reshape(-1, 3).astype(np.uint8)


def mask_pixel_count(
    image: NDArray[np.uint8],
    bbox_xyxy: tuple[int, int, int, int],
    mask: NDArray[np.uint8] | None,
    config: AppearanceConfig,
) -> int:
    return int(collect_masked_bgr_pixels(image, bbox_xyxy, mask, config).shape[0])


def primary_color_from_pixels(
    bgr_pixels: NDArray[np.uint8],
    config: AppearanceConfig,
) -> PrimaryColorResult:
    if bgr_pixels.size == 0:
        return PrimaryColorResult("neutral", (128, 128, 128))

    if config.color_method == "median_lab":
        return _primary_color_median_lab(bgr_pixels, config)
    return _primary_color_histogram_hsv(bgr_pixels, config)


def _primary_color_median_lab(
    bgr_pixels: NDArray[np.uint8],
    config: AppearanceConfig,
) -> PrimaryColorResult:
    hsv = cv2.cvtColor(bgr_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    sat = hsv[:, 1].astype(np.float32) / 255.0
    colored = sat >= config.min_saturation
    pixels = bgr_pixels[colored] if colored.any() else bgr_pixels
    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).astype(np.float32).reshape(
        -1, 3
    )
    median_lab_cv = np.median(lab, axis=0).astype(np.float32)
    label = quantize_lab(median_lab_cv, config)
    median_bgr = np.median(pixels, axis=0).astype(np.int32)
    return PrimaryColorResult(label, (int(median_bgr[0]), int(median_bgr[1]), int(median_bgr[2])))


def _primary_color_histogram_hsv(
    bgr_pixels: NDArray[np.uint8],
    config: AppearanceConfig,
) -> PrimaryColorResult:
    hsv = cv2.cvtColor(bgr_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    sat = hsv[:, 1].astype(np.float32) / 255.0
    colored = sat >= config.min_saturation
    sample = bgr_pixels[colored] if colored.any() else bgr_pixels
    hsv_sample = cv2.cvtColor(sample.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hues = hsv_sample[:, 0].astype(np.int32)
    weights = hsv_sample[:, 1].astype(np.float32) / 255.0
    if weights.sum() <= 0:
        weights = np.ones(len(hues), dtype=np.float32)

    hist, _ = np.histogram(
        hues,
        bins=config.histogram_h_bins,
        range=(0, 180),
        weights=weights,
    )
    peak_bin = int(np.argmax(hist))
    bin_low = peak_bin * 180 // config.histogram_h_bins
    bin_high = (peak_bin + 1) * 180 // config.histogram_h_bins
    in_bin = (hues >= bin_low) & (hues < bin_high)
    bin_pixels = sample[in_bin] if in_bin.any() else sample
    median_bgr = np.median(bin_pixels, axis=0).astype(np.int32)
    bgr_tuple = (int(median_bgr[0]), int(median_bgr[1]), int(median_bgr[2]))
    label = quantize_bgr(bgr_tuple, config)
    return PrimaryColorResult(label, bgr_tuple)


def primary_color_from_detection(
    image: NDArray[np.uint8],
    bbox_xyxy: tuple[int, int, int, int],
    mask: NDArray[np.uint8] | None,
    config: AppearanceConfig,
) -> PrimaryColorResult:
    pixels = collect_masked_bgr_pixels(image, bbox_xyxy, mask, config)
    return primary_color_from_pixels(pixels, config)


def dominant_clothing_color(
    image: NDArray[np.uint8],
    bbox_xyxy: tuple[int, int, int, int],
    mask: NDArray[np.uint8] | None,
    config: AppearanceConfig,
) -> str:
    return primary_color_from_detection(image, bbox_xyxy, mask, config).label
