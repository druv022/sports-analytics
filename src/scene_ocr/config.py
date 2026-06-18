"""Configuration for scene OCR extraction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OcrConfig:
    """Settings for RapidOCR-based word extraction."""

    min_confidence: float = 0.5
    use_angle_cls: bool = True
    dedupe: bool = True
    preprocess: bool = True
    scale: float = 2.0
    clahe: bool = True
    box_thresh: float = 0.3
    unclip_ratio: float = 1.8
    engine_text_score: float = 0.3
    use_cuda: bool | None = None
    rec_batch_num: int | None = None
    cls_batch_num: int | None = None

    def resolved_rec_batch_num(self, backend: str) -> int:
        if self.rec_batch_num is not None:
            return self.rec_batch_num
        if backend == "cuda":
            return 16
        if backend == "coreml":
            return 8
        return 6

    def resolved_cls_batch_num(self, backend: str) -> int:
        if self.cls_batch_num is not None:
            return self.cls_batch_num
        if backend == "cuda":
            return 16
        if backend == "coreml":
            return 8
        return 6


@dataclass(frozen=True)
class OpenAIVlmConfig:
    """Settings for OpenAI vision-language OCR escalation."""

    model: str = "gpt-4o"
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    max_tokens: int = 256
    temperature: float = 0.0
    jpeg_quality: int = 85
    base_url: str | None = None


@dataclass(frozen=True)
class OverlayRegion:
    """Normalized rectangular region of interest (x0, y0, x1, y1) in 0..1."""

    name: str
    x0: float
    y0: float
    x1: float
    y1: float


DEFAULT_OVERLAY_REGIONS: tuple[OverlayRegion, ...] = (
    OverlayRegion("score_bug", 0.0, 0.82, 0.45, 1.0),
    OverlayRegion("logo", 0.85, 0.0, 1.0, 0.12),
    OverlayRegion("sponsor_band", 0.0, 0.08, 0.78, 0.62),
)


@dataclass(frozen=True)
class ReadabilityConfig:
    """Settings for text presence and VLM escalation gating."""

    confidence_readable: float = 0.65
    confidence_weak: float = 0.30
    min_box_height_px: int = 12
    min_blob_area: float = 1000.0
    blob_threshold: int = 200
    max_vlm_crops: int = 3
    crop_pad_ratio: float = 0.12
    max_crop_dim_px: int = 512
    overlay_regions: tuple[OverlayRegion, ...] = DEFAULT_OVERLAY_REGIONS
    crop_iou_dedupe: float = 0.5
