from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.person_appearance.config import AppearanceConfig
from src.person_appearance.signature import (
    appearance_signature_string,
    build_compatibility_components,
    normalize_signature,
    signatures_compatible,
)
from src.person_appearance.types import SceneAppearance


def _scene(
    scene_id: int,
    count: int,
    color: str,
    *,
    signature: str | None = None,
) -> SceneAppearance:
    sig = signature if signature is not None else color
    return SceneAppearance(
        scene_id=scene_id,
        scene_type="closeup",
        person_count=count,
        person_colors=(color,) if color else (),
        appearance_signature=sig,
        confidence=0.9,
        status="ok" if color else "no_person",
    )


@pytest.fixture
def config() -> AppearanceConfig:
    return AppearanceConfig()


def test_appearance_signature_string_is_color_only():
    assert appearance_signature_string("red") == "red"
    assert appearance_signature_string("") == ""


def test_normalize_legacy_signature():
    assert normalize_signature("2:red,black") == "red"
    assert normalize_signature("red") == "red"


def test_same_color_compatible_regardless_of_count(config: AppearanceConfig):
    a = _scene(1, 1, "blue")
    b = _scene(2, 4, "blue")
    assert signatures_compatible(a, b, config)


def test_different_color_incompatible(config: AppearanceConfig):
    a = _scene(1, 2, "blue")
    b = _scene(2, 2, "red")
    assert not signatures_compatible(a, b, config)


def test_empty_signature_incompatible(config: AppearanceConfig):
    a = _scene(1, 0, "")
    b = _scene(2, 1, "blue")
    assert not signatures_compatible(a, b, config)


def test_build_compatibility_components_groups_by_color(config: AppearanceConfig):
    scenes = [
        _scene(1, 2, "blue"),
        _scene(2, 3, "blue"),
        _scene(3, 1, "red"),
    ]
    components = build_compatibility_components(scenes, config)
    assert components[1] == components[2]
    assert components[3] != components[1]
