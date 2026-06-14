#!/usr/bin/env python3
"""Evaluate saved cluster assignments against GT_scene_samples.csv (eval only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.camera_assignemnt.approach_4.config import (  # noqa: E402
    EMBEDDING_OUTPUT_SLUGS,
    resolve_method_and_backend,
)
from src.camera_assignemnt.approach_4.evaluate import evaluate_against_gt  # noqa: E402

DEFAULT_GT_CSV = ROOT / "data" / "GT_scene_samples.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "evaluation"

METHOD_CHOICES = ("hsv", "embedding", "ensemble", *sorted(EMBEDDING_OUTPUT_SLUGS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=METHOD_CHOICES,
        default="hsv",
        help="Clustering approach (hsv, ensemble, embedding, or an embedding slug "
        "such as resnet50 / dinov2_vits14).",
    )
    parser.add_argument(
        "--backend",
        choices=("resnet50", "dinov2_vits14"),
        default="resnet50",
        help="Embedding backend when --method=embedding.",
    )
    parser.add_argument(
        "--assignments-csv",
        type=Path,
        default=None,
        help="Defaults to data/evaluation/camera_assignments_{slug}.csv for the method.",
    )
    parser.add_argument("--gt-csv", type=Path, default=DEFAULT_GT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _, _, slug = resolve_method_and_backend(args.method, args.backend)
    assignments_csv = args.assignments_csv or (
        args.output_dir / f"camera_assignments_{slug}.csv"
    )

    report = evaluate_against_gt(
        assignments_csv=assignments_csv,
        gt_csv=args.gt_csv,
        root=ROOT,
    )
    report["method_slug"] = slug

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"camera_clustering_{slug}_gt.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    if report.get("n_evaluated", 0) == 0:
        print(report.get("error", "No rows evaluated."))
        print(f"Assignments: {assignments_csv}")
        print(f"Wrote {out_path}")
        return 1

    metrics = report["metrics"]
    print(f"Method: {slug}")
    print(f"Assignments: {assignments_csv}")
    print(f"Evaluated {report['n_evaluated']} / {report['n_gt_scenes']} GT scenes")
    print(f"Hungarian accuracy: {metrics['hungarian_accuracy']:.3f}")
    print(f"Macro F1: {metrics['macro_f1']:.3f}")
    print(f"ARI: {metrics['adjusted_rand_index']:.3f}")
    print(f"V-measure: {metrics['v_measure']:.3f}")
    print(f"Purity: {metrics['cluster_purity']:.3f}")
    print(f"Noise rate: {metrics['noise_rate']:.3f}")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
