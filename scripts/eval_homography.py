#!/usr/bin/env python3
"""Evaluate keypoint-based homography on full-court frames."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.camera_assignemnt.approach_2.rectangle_detector import load_image  # noqa: E402
from src.camera_assignemnt.approach_3.config import HomographyConfig  # noqa: E402
from src.camera_assignemnt.approach_3.homography_projector import (  # noqa: E402
    draw_backprojected_lines,
    estimate_homography,
    map_scene_to_reference,
    stack_overlays_on_reference,
)
from src.camera_assignemnt.approach_3.keypoint_detector import (  # noqa: E402
    detect_court_keypoints,
    draw_detected_keypoints,
)

DEFAULT_GT_CSV = ROOT / "data" / "GT_scene_samples.csv"
DEFAULT_REFERENCE = ROOT / "data" / "Court_dimension.png"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "evaluation" / "homography"
DEFAULT_JSON = ROOT / "data" / "evaluation" / "homography_eval.json"


def load_full_court_samples(gt_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(gt_csv)
    keep = [c for c in df.columns if c in {"scene_id", "image_idx", "frame_path", "scene_type"}]
    df = df[keep].copy()
    df["scene_type"] = df["scene_type"].astype(str).str.strip()
    return df[df["scene_type"] == "full_court"].reset_index(drop=True)


def _resize_to_width(image: np.ndarray, target_width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w == target_width:
        return image
    scale = target_width / w
    new_h = max(1, int(h * scale))
    return cv2.resize(image, (target_width, new_h), interpolation=cv2.INTER_AREA)


def build_grid(
    panels: list[tuple[str, np.ndarray]],
    cols: int = 4,
    cell_width: int = 420,
    gap: int = 8,
    label_height: int = 28,
) -> np.ndarray:
    if not panels:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    resized = [_resize_to_width(img, cell_width) for _, img in panels]
    cell_h = max(img.shape[0] for img in resized) + label_height
    n = len(panels)
    rows = math.ceil(n / cols)

    canvas_h = rows * cell_h + (rows + 1) * gap
    canvas_w = cols * cell_width + (cols + 1) * gap
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for idx, ((label, _), img) in enumerate(zip(panels, resized)):
        row, col = divmod(idx, cols)
        x0 = gap + col * (cell_width + gap)
        y0 = gap + row * (cell_h + gap)
        canvas[y0 + label_height : y0 + label_height + img.shape[0], x0 : x0 + img.shape[1]] = img
        cv2.putText(
            canvas,
            label,
            (x0 + 4, y0 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return canvas


def evaluate_homography(
    gt_csv: Path = DEFAULT_GT_CSV,
    reference_path: Path = DEFAULT_REFERENCE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    homography_config: HomographyConfig | None = None,
    max_samples: int | None = None,
) -> dict:
    homography_config = homography_config or HomographyConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = load_image(reference_path)
    samples = load_full_court_samples(gt_csv)
    if max_samples is not None:
        samples = samples.head(max_samples)

    per_frame_dir = output_dir / "per_frame"
    per_frame_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    successful_warps: list[np.ndarray] = []
    exact_warps: list[np.ndarray] = []
    grid_panels: list[tuple[str, np.ndarray]] = []
    missing_images: list[str] = []

    for _, sample in samples.iterrows():
        frame_path = Path(str(sample["frame_path"]))
        label = frame_path.stem

        if not frame_path.is_file():
            missing_images.append(str(frame_path))
            rows.append({
                "frame_path": str(frame_path),
                "scene_id": int(sample["scene_id"]),
                "image_idx": int(sample["image_idx"]),
                "success": False,
                "exact": False,
                "message": "missing_image",
            })
            continue

        scene = load_image(frame_path)
        warped, overlay, result = map_scene_to_reference(scene, reference, homography_config)

        row = {
            "frame_path": str(frame_path),
            "scene_id": int(sample["scene_id"]),
            "image_idx": int(sample["image_idx"]),
            "success": result.success,
            "exact": result.exact,
            "reproj_error_px": result.reproj_error,
            "line_alignment_error_px": result.line_alignment_error_px,
            "reference_line_error_px": result.reference_line_error_px,
            "inlier_count": result.inlier_count,
            "n_keypoints_detected": result.n_keypoints_detected,
            "message": result.message,
        }
        rows.append(row)

        if result.success:
            detection = detect_court_keypoints(scene, homography_config)
            kps_vis = draw_detected_keypoints(scene, detection)
            if result.H is not None:
                bp_vis = draw_backprojected_lines(scene, result.H)
            else:
                bp_vis = scene.copy()

            cv2.imwrite(str(per_frame_dir / f"{label}_warped.jpg"), warped)
            cv2.imwrite(str(per_frame_dir / f"{label}_overlay.jpg"), overlay)
            cv2.imwrite(str(per_frame_dir / f"{label}_keypoints.jpg"), kps_vis)
            cv2.imwrite(str(per_frame_dir / f"{label}_backproj.jpg"), bp_vis)

            successful_warps.append(warped)
            status = "exact" if result.exact else "approx"
            grid_panels.append((f"{label} ({status})", overlay))
            if result.exact:
                exact_warps.append(warped)
        else:
            fail_vis = reference.copy()
            cv2.putText(
                fail_vis,
                f"FAIL: {result.message}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            grid_panels.append((f"{label} (fail)", fail_vis))

    composite = stack_overlays_on_reference(
        reference,
        exact_warps or successful_warps,
        alpha=homography_config.composite_layer_alpha,
    )
    grid = build_grid(grid_panels)

    composite_path = output_dir / "all_mapped_on_reference.jpg"
    grid_path = output_dir / "grid_overlays.jpg"
    cv2.imwrite(str(composite_path), composite)
    cv2.imwrite(str(grid_path), grid)

    n_total = len(rows)
    n_success = sum(1 for r in rows if r["success"])
    n_exact = sum(1 for r in rows if r.get("exact"))
    line_errors = [
        r["line_alignment_error_px"]
        for r in rows
        if r["success"] and r["line_alignment_error_px"] != float("inf")
    ]
    ref_errors = [
        r["reference_line_error_px"]
        for r in rows
        if r["success"] and r.get("reference_line_error_px", float("inf")) != float("inf")
    ]

    summary = {
        "method": "keypoint_detector",
        "reference_path": str(reference_path),
        "gt_csv": str(gt_csv),
        "max_line_error_px": homography_config.max_line_error_px,
        "n_total": n_total,
        "n_success": n_success,
        "n_exact": n_exact,
        "success_rate": n_success / n_total if n_total else 0.0,
        "exact_rate": n_exact / n_total if n_total else 0.0,
        "mean_line_alignment_px": float(np.mean(line_errors)) if line_errors else None,
        "median_line_alignment_px": float(np.median(line_errors)) if line_errors else None,
        "mean_reference_line_error_px": float(np.mean(ref_errors)) if ref_errors else None,
        "median_reference_line_error_px": float(np.median(ref_errors)) if ref_errors else None,
        "missing_images": missing_images,
        "outputs": {
            "composite": str(composite_path),
            "grid": str(grid_path),
            "per_frame_dir": str(per_frame_dir),
        },
        "frames": rows,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate keypoint homography on full-court frames."
    )
    parser.add_argument("--gt-csv", type=Path, default=DEFAULT_GT_CSV)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-line-error", type=float, default=5.0)
    parser.add_argument("--no-refine-kps", action="store_true")
    parser.add_argument("--no-tcd-repair", action="store_true")
    args = parser.parse_args()

    config = HomographyConfig(
        max_line_error_px=args.max_line_error,
        use_refine_kps=not args.no_refine_kps,
        use_tcd_homography_repair=not args.no_tcd_repair,
    )

    summary = evaluate_homography(
        gt_csv=args.gt_csv,
        reference_path=args.reference,
        output_dir=args.output_dir,
        homography_config=config,
        max_samples=args.max_samples,
    )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"\nHomography eval: {summary['n_success']}/{summary['n_total']} success, "
        f"{summary['n_exact']}/{summary['n_total']} exact "
        f"(line error <= {config.max_line_error_px}px)"
    )
    if summary["median_line_alignment_px"] is not None:
        print(f"Median line alignment: {summary['median_line_alignment_px']:.2f} px")
    if summary["median_reference_line_error_px"] is not None:
        print(f"Median reference-space error: {summary['median_reference_line_error_px']:.2f} px")
    print(f"Composite: {summary['outputs']['composite']}")
    print(f"JSON:      {args.json_out}")


if __name__ == "__main__":
    main()
