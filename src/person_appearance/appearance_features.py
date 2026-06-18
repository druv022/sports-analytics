"""Soft appearance features for full-court camera clustering."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.person_appearance.config import PRIMARY_PALETTE
from src.person_appearance.types import SceneAppearance

MAX_PERSON_COUNT_BUCKET = 6


def build_appearance_feature_vector(
    person_count: int,
    colors: tuple[str, ...],
) -> NDArray[np.float32]:
    count_vec = np.zeros(MAX_PERSON_COUNT_BUCKET + 1, dtype=np.float32)
    bucket = min(max(person_count, 0), MAX_PERSON_COUNT_BUCKET)
    count_vec[bucket] = 1.0

    palette_hist = np.zeros(len(PRIMARY_PALETTE), dtype=np.float32)
    for color in colors:
        if color in PRIMARY_PALETTE:
            palette_hist[PRIMARY_PALETTE.index(color)] += 1.0
    if palette_hist.sum() > 0:
        palette_hist /= palette_hist.sum()

    return np.concatenate([count_vec, palette_hist]).astype(np.float32)


def appearance_features_from_scenes(
    appearances: list[SceneAppearance],
) -> dict[str, NDArray[np.float32]]:
    return {
        str(app.scene_id): build_appearance_feature_vector(app.person_count, app.person_colors)
        for app in appearances
        if app.scene_type == "full_court"
    }


def blend_appearance_features(
    features: NDArray[np.float32],
    scene_ids: list[str],
    appearance_by_scene: dict[str, NDArray[np.float32]],
    *,
    weight: float,
) -> NDArray[np.float32]:
    if weight <= 0 or not appearance_by_scene or features.size == 0:
        return features

    app_dim = next(iter(appearance_by_scene.values())).shape[0]
    blended_rows: list[NDArray[np.float32]] = []
    for idx, scene_id in enumerate(scene_ids):
        visual = features[idx].astype(np.float32)
        visual_norm = visual / (float(np.linalg.norm(visual)) + 1e-8)
        app = appearance_by_scene.get(scene_id)
        if app is None:
            app_norm = np.zeros(app_dim, dtype=np.float32)
            effective_weight = 0.0
        else:
            app_norm = app / (float(np.linalg.norm(app)) + 1e-8)
            effective_weight = weight
        row = np.concatenate(
            [
                visual_norm * (1.0 - effective_weight),
                app_norm * effective_weight,
            ]
        ).astype(np.float32)
        blended_rows.append(row)
    return np.stack(blended_rows, axis=0)
