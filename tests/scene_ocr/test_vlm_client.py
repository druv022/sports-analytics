"""Tests for VLM OCR clients."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.scene_ocr.config import OpenAIVlmConfig
from src.scene_ocr.vlm_client import (
    OpenAIVlmClient,
    bgr_to_data_url,
    parse_vlm_tokens,
    require_openai,
)


def test_parse_vlm_tokens_json():
    content = '{"tokens": ["CHASE", "H", "E"]}'
    assert parse_vlm_tokens(content) == ["CHASE", "H", "E"]


def test_parse_vlm_tokens_words_key():
    content = '{"words": ["SINNER", "ITA"]}'
    assert parse_vlm_tokens(content) == ["SINNER", "ITA"]


def test_parse_vlm_tokens_fallback_regex():
    content = "Some prose with CHASE visible"
    assert "CHASE" in parse_vlm_tokens(content)


def test_bgr_to_data_url():
    image = np.full((32, 64, 3), 128, dtype=np.uint8)
    url = bgr_to_data_url(image)
    assert url.startswith("data:image/jpeg;base64,")


def test_require_openai_raises_when_missing():
    with patch("src.scene_ocr.vlm_client._OPENAI_AVAILABLE", False):
        with pytest.raises(ImportError, match="openai is required"):
            require_openai()


def test_openai_vlm_client_reads_tokens(monkeypatch):
    pytest.importorskip("openai")

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content='{"tokens": ["CHASE"]}'))]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("src.scene_ocr.vlm_client.OpenAI", return_value=mock_client):
        client = OpenAIVlmClient(OpenAIVlmConfig(api_key="test-key"))
        crop = np.full((40, 80, 3), 255, dtype=np.uint8)
        cv2.putText(crop, "X", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        words = client.read_text([crop])

    assert words == ["CHASE"]
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o"
    assert call_kwargs["response_format"] == {"type": "json_object"}


def test_openai_vlm_client_requires_api_key():
    pytest.importorskip("openai")

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="OpenAI API key not set"):
            OpenAIVlmClient(OpenAIVlmConfig(api_key=None))
