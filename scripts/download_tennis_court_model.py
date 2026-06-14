#!/usr/bin/env python3
"""Download TennisCourtDetector pretrained weights."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "models" / "tennis_court_detector.pth"

# Google Drive file id from TennisCourtDetector README
GDRIVE_FILE_ID = "1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG"


def download_with_gdown(out_path: Path) -> bool:
    try:
        import gdown
    except ImportError:
        return False

    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    gdown.download(url, str(out_path), quiet=False)
    return out_path.is_file() and out_path.stat().st_size > 1000


def download_with_requests(out_path: Path) -> bool:
    try:
        import urllib.request
    except ImportError:
        return False

    url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, out_path)
    except Exception:
        return False
    return out_path.is_file() and out_path.stat().st_size > 1000


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TennisCourtDetector weights.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.output.is_file() and args.output.stat().st_size > 1000:
        print(f"Model already exists: {args.output}")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("Attempting download from Google Drive...")
    if download_with_gdown(args.output) or download_with_requests(args.output):
        print(f"Saved model to {args.output}")
        return

    print(
        "\nAutomatic download failed. Download manually:\n"
        "  https://drive.google.com/file/d/1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG/view\n"
        f"Save as: {args.output}\n\n"
        "Or install gdown and retry:\n"
        "  pip install gdown\n"
        "  python scripts/download_tennis_court_model.py",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
