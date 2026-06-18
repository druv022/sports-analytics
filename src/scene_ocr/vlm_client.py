"""Pluggable VLM OCR clients (protocol, no-op stub, OpenAI vision)."""

from __future__ import annotations

import base64
import json
import os
import re
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from src.scene_ocr.config import OpenAIVlmConfig

DEFAULT_VLM_PROMPT = (
    "List every visible word or letter in this image, including partially hidden "
    'or cropped characters. Return JSON: {"tokens": [...]}'
)

_OPENAI_AVAILABLE = False
OpenAI: Any = None

try:
    from openai import OpenAI

    _OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[misc, assignment]

_TOKEN_JSON_RE = re.compile(r"\{[^{}]*\"(?:tokens|words)\"[^{}]*\}", re.DOTALL)


class VlmOcrClient(Protocol):
    def read_text(self, crops: list[NDArray], prompt: str) -> list[str]: ...


class NullVlmClient:
    """No-op VLM client; returns empty results."""

    def read_text(self, crops: list[NDArray], prompt: str = DEFAULT_VLM_PROMPT) -> list[str]:
        return []


def require_openai() -> None:
    if not _OPENAI_AVAILABLE:
        raise ImportError(
            "openai is required for OpenAI VLM OCR. "
            "Install with: pip install -e '.[ocr-vlm]'"
        )


def bgr_to_data_url(bgr: NDArray, jpeg_quality: int = 85) -> str:
    """Encode a BGR image as a JPEG data URL for the OpenAI vision API."""
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise ValueError("Failed to encode crop as JPEG")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def parse_vlm_tokens(content: str) -> list[str]:
    """Parse model output into a flat token list."""
    text = content.strip()
    if not text:
        return []

    candidates = [text]
    match = _TOKEN_JSON_RE.search(text)
    if match:
        candidates.insert(0, match.group(0))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            raw = data.get("tokens") or data.get("words") or []
            if isinstance(raw, list):
                return [str(token).strip() for token in raw if str(token).strip()]

    return [token for token in re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", text)]


class OpenAIVlmClient:
    """Call OpenAI vision models on OCR escalation crops."""

    def __init__(self, config: OpenAIVlmConfig | None = None) -> None:
        require_openai()
        self.config = config or OpenAIVlmConfig()
        api_key = self.config.api_key or os.environ.get(self.config.api_key_env)
        if not api_key:
            raise ValueError(
                f"OpenAI API key not set. Pass OpenAIVlmConfig(api_key=...) or set "
                f"{self.config.api_key_env}."
            )
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url
        self._client = OpenAI(**client_kwargs)

    def read_text(
        self,
        crops: list[NDArray],
        prompt: str = DEFAULT_VLM_PROMPT,
    ) -> list[str]:
        """Run vision OCR on each crop and return tokens from all crops."""
        words: list[str] = []
        for crop in crops:
            if crop.size == 0:
                continue
            bgr = np.asarray(crop, dtype=np.uint8)
            if bgr.ndim != 3 or bgr.shape[2] != 3:
                raise ValueError(f"Expected HxWx3 BGR crop, got shape {bgr.shape}")

            response = self._client.chat.completions.create(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": bgr_to_data_url(bgr, self.config.jpeg_quality),
                                },
                            },
                        ],
                    }
                ],
            )
            content = response.choices[0].message.content or ""
            words.extend(parse_vlm_tokens(content))
        return words
