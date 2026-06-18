#!/usr/bin/env python3
"""Run lightweight scene OCR on a single image and print detected words.

OpenAI VLM escalation example:
  pip install -e ".[ocr,ocr-vlm]"
  export OPENAI_API_KEY=sk-...
  python scripts/run_ocr.py --image frame.jpg --assess --openai-vlm --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scene_ocr.config import OcrConfig, OpenAIVlmConfig, ReadabilityConfig
from src.scene_ocr.extractor import extract_words
from src.scene_ocr.readability import assess_readability
from src.scene_ocr.vlm_client import OpenAIVlmClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Path to input image")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum line confidence to keep (default: 0.5)",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Keep duplicate words (case-insensitive)",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Disable CLAHE + upscaling (faster, worse on small scoreboard text)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="Upscale factor when preprocessing is enabled (default: 2.0)",
    )
    parser.add_argument(
        "--assess",
        action="store_true",
        help="Assess readability and report VLM escalation verdict",
    )
    parser.add_argument(
        "--openai-vlm",
        action="store_true",
        help="Escalate unread crops to OpenAI vision (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-4o",
        help="OpenAI vision model when --openai-vlm is set (default: gpt-4o)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of one word per line",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ocr_config = OcrConfig(
        min_confidence=args.min_confidence,
        dedupe=not args.no_dedupe,
        preprocess=not args.no_preprocess,
        scale=args.scale,
    )

    if args.assess:
        vlm_client = None
        if args.openai_vlm:
            vlm_client = OpenAIVlmClient(OpenAIVlmConfig(model=args.openai_model))

        result = assess_readability(
            args.image,
            ocr_config,
            ReadabilityConfig(),
            vlm_client=vlm_client,
        )
        if args.json:
            payload = {
                "image": str(args.image.resolve()),
                "verdict": result.verdict.value,
                "overlay_readable": result.overlay_readable,
                "needs_vlm": result.needs_vlm,
                "words": result.words,
                "reasons": result.reasons,
                "vlm_crop_count": len(result.vlm_crops),
            }
            if args.openai_vlm:
                payload["openai_model"] = args.openai_model
            print(json.dumps(payload, indent=2))
        else:
            print(f"verdict: {result.verdict.value}")
            print(f"overlay_readable: {result.overlay_readable}")
            print(f"needs_vlm: {result.needs_vlm}")
            for reason in result.reasons:
                print(f"reason: {reason}")
            for word in result.words:
                print(word)
        return 0

    words = extract_words(args.image, ocr_config)
    if args.json:
        payload = {"image": str(args.image.resolve()), "words": words}
        print(json.dumps(payload, indent=2))
    else:
        for word in words:
            print(word)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
