from __future__ import annotations

import numpy as np

from src.person_appearance.appearance_features import (
    blend_appearance_features,
    build_appearance_feature_vector,
)


def test_build_appearance_feature_vector_shape():
    vec = build_appearance_feature_vector(2, ("blue", "white"))
    assert vec.shape[0] == 7 + 9


def test_blend_appearance_features_changes_full_court_rows():
    features = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    appearance = {"1": build_appearance_feature_vector(1, ("blue",))}
    blended = blend_appearance_features(
        features,
        ["1", "2"],
        appearance,
        weight=0.2,
    )
    assert blended.shape[0] == 2
    assert blended.shape[1] > features.shape[1]
    assert not np.allclose(blended[0], blended[1])
