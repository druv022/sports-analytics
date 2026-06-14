from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_KEYPOINTS_JSON = ROOT / "data" / "court_reference" / "keypoints.json"


@dataclass(frozen=True)
class CourtReferenceData:
    reference_image: Path
    reference_size: tuple[int, int]
    dimension_keypoints: np.ndarray
    tcd_keypoints: np.ndarray
    H_dim_to_tcd: np.ndarray
    H_tcd_to_dim: np.ndarray
    labels: tuple[str, ...]


@lru_cache(maxsize=1)
def load_court_reference(path: str | Path | None = None) -> CourtReferenceData:
    json_path = Path(path) if path is not None else DEFAULT_KEYPOINTS_JSON
    if not json_path.is_file():
        raise FileNotFoundError(
            f"Court reference keypoints not found: {json_path}. "
            "Run scripts/annotate_court_keypoints.py first."
        )

    data = json.loads(json_path.read_text(encoding="utf-8"))
    keypoints = sorted(data["keypoints"], key=lambda k: k["index"])

    dimension = np.array([k["dimension"] for k in keypoints], dtype=np.float32)
    tcd = np.array([k["tcd"] for k in keypoints], dtype=np.float32)
    labels = tuple(k["label"] for k in keypoints)

    ref_image = ROOT / data["reference_image"]
    w, h = data["reference_size"]

    H_dim_to_tcd = np.array(data["H_dim_to_tcd"], dtype=np.float64)
    H_tcd_to_dim = np.array(data["H_tcd_to_dim"], dtype=np.float64)

    return CourtReferenceData(
        reference_image=ref_image,
        reference_size=(int(h), int(w)),
        dimension_keypoints=dimension,
        tcd_keypoints=tcd,
        H_dim_to_tcd=H_dim_to_tcd,
        H_tcd_to_dim=H_tcd_to_dim,
        labels=labels,
    )


def dimension_line_segments(ref: CourtReferenceData) -> list[tuple[np.ndarray, np.ndarray]]:
    """Court line segments in dimension reference coordinates."""
    k = ref.dimension_keypoints
    pairs = [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (8, 9),
        (10, 11),
        (12, 13),
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),
        (8, 10),
        (9, 11),
    ]
    return [(k[i], k[j]) for i, j in pairs]


def build_dimension_line_mask(ref: CourtReferenceData, thickness: int = 3) -> np.ndarray:
    """Binary mask of reference court lines on the dimension diagram."""
    h, w = ref.reference_size
    mask = np.zeros((h, w), dtype=np.uint8)
    for p1, p2 in dimension_line_segments(ref):
        cv2.line(
            mask,
            tuple(p1.astype(int)),
            tuple(p2.astype(int)),
            255,
            thickness,
            cv2.LINE_AA,
        )
    return mask
