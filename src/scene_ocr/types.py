"""Data types for scene OCR results."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from numpy.typing import NDArray


class OcrReadabilityVerdict(str, Enum):
    NO_TEXT = "no_text"
    READABLE = "readable"
    NEEDS_VLM = "needs_vlm"


@dataclass(frozen=True)
class OcrDetection:
    """A single line-level text detection from the image."""

    text: str
    confidence: float
    bbox: NDArray


@dataclass(frozen=True)
class TextBlobCandidate:
    """A region that likely contains text (detector box or contrast blob)."""

    bbox: NDArray
    area: float
    source: Literal["detector", "contrast", "overlay_region"]


@dataclass(frozen=True)
class OcrReadability:
    """Readability assessment for an image."""

    verdict: OcrReadabilityVerdict
    words: list[str]
    detections: list[OcrDetection]
    text_candidates: list[TextBlobCandidate]
    vlm_crops: list[NDArray]
    overlay_readable: bool
    needs_vlm: bool
    reasons: list[str]
