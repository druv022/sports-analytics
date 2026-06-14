#!/usr/bin/env python3
"""Click-annotate 14 court keypoints on Court_dimension.png and save bridge JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from third_party.tennis_court_detector.court_reference import CourtReference

DEFAULT_IMAGE = ROOT / "data" / "Court_dimension.png"
DEFAULT_OUTPUT = ROOT / "data" / "court_reference" / "keypoints.json"

LABELS = [
    "baseline_top_left",
    "baseline_top_right",
    "baseline_bottom_left",
    "baseline_bottom_right",
    "left_inner_top",
    "left_inner_bottom",
    "right_inner_top",
    "right_inner_bottom",
    "top_inner_left",
    "top_inner_right",
    "bottom_inner_left",
    "bottom_inner_right",
    "middle_line_top",
    "middle_line_bottom",
]


def build_json(dimension_points: list[list[float]], image_path: Path) -> dict:
    tcd_ref = CourtReference()
    tcd_points = [[float(x), float(y)] for x, y in tcd_ref.key_points]

    dim_arr = np.array(dimension_points, dtype=np.float32)
    tcd_arr = np.array(tcd_points, dtype=np.float32)
    H_dim_to_tcd, _ = cv2.findHomography(dim_arr, tcd_arr, cv2.RANSAC, 3.0)
    H_tcd_to_dim = np.linalg.inv(H_dim_to_tcd)

    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]

    keypoints = []
    for i, label in enumerate(LABELS):
        keypoints.append(
            {
                "index": i,
                "label": label,
                "dimension": dimension_points[i],
                "tcd": tcd_points[i],
            }
        )

    return {
        "reference_image": str(image_path.relative_to(ROOT)),
        "reference_size": [w, h],
        "keypoints": keypoints,
        "H_dim_to_tcd": H_dim_to_tcd.tolist(),
        "H_tcd_to_dim": H_tcd_to_dim.tolist(),
    }


def interactive_annotate(image_path: Path, output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    points: list[list[float]] = []
    vis = image.copy()

    def _cb(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or len(points) >= 14:
            return
        points.append([float(x), float(y)])
        label = LABELS[len(points) - 1]
        cv2.circle(vis, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(
            vis,
            f"{len(points)}:{label[:12]}",
            (x + 10, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow("Annotate court keypoints", vis)

    cv2.imshow("Annotate court keypoints", vis)
    cv2.setMouseCallback("Annotate court keypoints", _cb)
    print("Click 14 keypoints in order (see LABELS in script). Press any key when done.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) != 14:
        raise RuntimeError(f"Expected 14 points, got {len(points)}")

    data = build_json(points, image_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate court keypoints on reference diagram.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--recompute-bridge",
        action="store_true",
        help="Recompute bridge matrices from existing keypoints JSON without clicking.",
    )
    args = parser.parse_args()

    if args.recompute_bridge:
        data = json.loads(args.output.read_text(encoding="utf-8"))
        pts = [k["dimension"] for k in data["keypoints"]]
        rebuilt = build_json(pts, ROOT / data["reference_image"])
        args.output.write_text(json.dumps(rebuilt, indent=2), encoding="utf-8")
        print(f"Recomputed bridge matrices in {args.output}")
        return

    interactive_annotate(args.image, args.output)


if __name__ == "__main__":
    main()
