"""Person detection and clothing-color appearance signatures."""

from src.person_appearance.config import AppearanceConfig
from src.person_appearance.extractor import analyze_frame, build_scene_appearances
from src.person_appearance.signature import (
    appearance_signature_string,
    build_compatibility_components,
    normalize_signature,
    signatures_compatible,
)
from src.person_appearance.types import FrameAppearance, PersonDetection, SceneAppearance

__all__ = [
    "AppearanceConfig",
    "FrameAppearance",
    "PersonDetection",
    "SceneAppearance",
    "analyze_frame",
    "appearance_signature_string",
    "build_compatibility_components",
    "build_scene_appearances",
    "normalize_signature",
    "signatures_compatible",
]
