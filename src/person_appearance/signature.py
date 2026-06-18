"""Appearance signature building and color-only compatibility."""

from __future__ import annotations

from src.person_appearance.color_profile import colors_match
from src.person_appearance.config import AppearanceConfig
from src.person_appearance.types import SceneAppearance


def normalize_signature(raw: str) -> str:
    """Parse legacy count-prefixed signatures (e.g. 2:red,black -> red)."""
    if not raw:
        return ""
    if ":" not in raw:
        return raw.strip()
    colors_part = raw.split(":", 1)[1].strip()
    if not colors_part:
        return ""
    return colors_part.split(",")[0].strip()


def appearance_signature_string(primary_color: str) -> str:
    return primary_color.strip() if primary_color else ""


def _primary_color_from_scene(
    appearance: SceneAppearance | str | tuple[str, ...],
) -> str:
    if isinstance(appearance, SceneAppearance):
        if appearance.appearance_signature:
            return normalize_signature(appearance.appearance_signature)
        if appearance.person_colors:
            return appearance.person_colors[0]
        return ""
    if isinstance(appearance, str):
        return normalize_signature(appearance)
    if appearance:
        return str(appearance[0])
    return ""


def signatures_compatible(
    a: SceneAppearance | str | tuple[str, ...],
    b: SceneAppearance | str | tuple[str, ...],
    config: AppearanceConfig,
) -> bool:
    color_a = _primary_color_from_scene(a)
    color_b = _primary_color_from_scene(b)
    if not color_a or not color_b:
        return False
    return colors_match(color_a, color_b, config.color_tolerance)


def _find_parent(parent: dict[int, int], node: int) -> int:
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def _union(parent: dict[int, int], a: int, b: int) -> None:
    root_a = _find_parent(parent, a)
    root_b = _find_parent(parent, b)
    if root_a != root_b:
        parent[root_b] = root_a


def build_compatibility_components(
    appearances: list[SceneAppearance],
    config: AppearanceConfig,
    *,
    eligible_scene_ids: set[int] | None = None,
) -> dict[int, int]:
    """Return scene_id -> component_id via union-find over compatible appearances."""
    eligible = [
        app
        for app in appearances
        if eligible_scene_ids is None or app.scene_id in eligible_scene_ids
    ]
    if not eligible:
        return {}

    scene_ids = [app.scene_id for app in eligible]
    parent = {scene_id: scene_id for scene_id in scene_ids}

    for i, left in enumerate(eligible):
        for right in eligible[i + 1 :]:
            if signatures_compatible(left, right, config):
                _union(parent, left.scene_id, right.scene_id)

    roots = {_find_parent(parent, scene_id) for scene_id in scene_ids}
    root_to_component = {root: idx for idx, root in enumerate(sorted(roots))}
    return {scene_id: root_to_component[_find_parent(parent, scene_id)] for scene_id in scene_ids}
