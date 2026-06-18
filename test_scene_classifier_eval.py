#!/usr/bin/env python3
"""Evaluate classify_scene on scene_samples against GT_scene_samples.csv."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.camera_assignemnt.scene_classifier.classifier import (  # noqa: E402
    _load_cached_scene_mlp,
    classify_scene,
)
from src.camera_assignemnt.scene_classifier.config import Config  # noqa: E402

DEFAULT_GT_CSV = ROOT / "data" / "GT_scene_samples.csv"
DEFAULT_OUTPUT = ROOT / "data" / "evaluation" / "scene_classifier_eval.json"
DEFAULT_MISCLASSIFIED_DIR = ROOT / "data" / "evaluation" / "misclassified"
SCENE_TYPES = ("closeup", "full_court")


def load_gt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = [c for c in df.columns if c in {"scene_id", "image_idx", "frame_path", "scene_type"}]
    df = df[keep].copy()
    df["scene_type"] = df["scene_type"].astype(str).str.strip()
    df = df[df["scene_type"].isin(SCENE_TYPES)]
    return df


def misclassified_filename(source: Path, expected: str, predicted: str) -> str:
    """Build a filename with ground-truth and prediction appended."""
    return f"{source.stem}_gt-{expected}_pred-{predicted}{source.suffix}"


def save_misclassified_images(
    misclassified: list[dict],
    output_dir: Path,
) -> list[str]:
    """Copy misclassified frames into output_dir with error details in the name."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    for row in misclassified:
        source = Path(row["frame_path"])
        dest = output_dir / misclassified_filename(
            source,
            str(row["expected"]),
            str(row["predicted"]),
        )
        shutil.copy2(source, dest)
        saved.append(str(dest))

    return saved


def evaluate_scene_classifier(
    gt_csv: Path = DEFAULT_GT_CSV,
    config: Config | None = None,
    misclassified_dir: Path | None = DEFAULT_MISCLASSIFIED_DIR,
) -> dict:
    if config is None:
        config = Config()

    _load_cached_scene_mlp.cache_clear()
    raw_gt = pd.read_csv(gt_csv)
    n_excluded_partial_gt = int(
        (raw_gt["scene_type"].astype(str).str.strip() == "partial_court").sum()
    )
    gt = load_gt(gt_csv)

    y_true: list[str] = []
    y_pred: list[str] = []
    rows: list[dict] = []
    missing_images: list[str] = []

    for _, sample in gt.iterrows():
        frame_path = Path(str(sample["frame_path"]))
        if not frame_path.is_file():
            missing_images.append(str(frame_path))
            continue

        frame = cv2.imread(str(frame_path))
        if frame is None:
            missing_images.append(str(frame_path))
            continue

        predicted, court_ratio, _ = classify_scene(frame, config)
        expected = str(sample["scene_type"])

        y_true.append(expected)
        y_pred.append(predicted)
        rows.append(
            {
                "scene_id": int(sample["scene_id"]),
                "image_idx": int(sample["image_idx"]),
                "frame_path": str(frame_path),
                "expected": expected,
                "predicted": predicted,
                "court_ratio": court_ratio,
                "correct": expected == predicted,
            }
        )

    if not y_true:
        raise RuntimeError(
            f"No evaluable frames found for {gt_csv}. "
            f"Missing images: {len(missing_images)}"
        )

    labels = sorted(set(y_true) | set(y_pred))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    misclassified = [row for row in rows if not row["correct"]]
    saved_misclassified: list[str] = []
    if misclassified_dir is not None and misclassified:
        saved_misclassified = save_misclassified_images(misclassified, misclassified_dir)

    return {
        "gt_csv": str(gt_csv),
        "model_path": config.scene_mlp_path,
        "n_gt_rows": len(raw_gt),
        "n_excluded_partial_gt": n_excluded_partial_gt,
        "n_evaluated": len(rows),
        "n_missing_images": len(missing_images),
        "missing_images": missing_images,
        "labels": labels,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "misclassified": misclassified,
        "misclassified_dir": str(misclassified_dir) if misclassified_dir else None,
        "saved_misclassified": saved_misclassified,
    }


def print_report(results: dict) -> None:
    print(f"GT CSV: {results['gt_csv']}")
    print(f"Model:  {results['model_path']}")
    print(
        f"Evaluated: {results['n_evaluated']} / {results['n_gt_rows']} "
        f"(excluded partial_court GT: {results['n_excluded_partial_gt']}, "
        f"missing images: {results['n_missing_images']})"
    )
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"Macro F1: {results['macro_f1']:.3f}")

    print("\nConfusion matrix (rows=true, cols=pred):")
    print("Labels:", results["labels"])
    for row in results["confusion_matrix"]:
        print(" ", row)

    print("\nPer-class metrics:")
    report = results["classification_report"]
    for label in results["labels"]:
        stats = report.get(label, {})
        if isinstance(stats, dict):
            print(
                f"  {label}: precision={stats.get('precision', 0):.3f} "
                f"recall={stats.get('recall', 0):.3f} "
                f"f1={stats.get('f1-score', 0):.3f} "
                f"support={stats.get('support', 0)}"
            )

    misclassified = results["misclassified"]
    if misclassified:
        print(f"\nMisclassified frames ({len(misclassified)}):")
        for row in misclassified:
            print(
                f"  scene_{row['scene_id']} idx={row['image_idx']} "
                f"{Path(row['frame_path']).name}: "
                f"true={row['expected']} pred={row['predicted']} "
                f"court_ratio={row['court_ratio']:.3f}"
            )
    else:
        print("\nNo misclassified frames.")

    saved = results.get("saved_misclassified", [])
    if saved:
        print(f"\nSaved {len(saved)} misclassified images to {results['misclassified_dir']}")
        for path in saved[:5]:
            print(f"  {Path(path).name}")
        if len(saved) > 5:
            print(f"  ... and {len(saved) - 5} more")


def test_scene_classifier_against_gt() -> None:
    """Pytest entry point: run GT evaluation and require a trained model."""
    model_path = Path(Config().scene_mlp_path)
    if not model_path.is_file():
        import pytest

        pytest.skip(f"Scene MLP model not found at {model_path}")

    results = evaluate_scene_classifier()
    assert results["n_evaluated"] > 0
    assert 0.0 <= results["accuracy"] <= 1.0
    print(
        f"\nscene_classifier GT eval: "
        f"accuracy={results['accuracy']:.3f} "
        f"macro_f1={results['macro_f1']:.3f} "
        f"n={results['n_evaluated']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gt-csv",
        type=Path,
        default=DEFAULT_GT_CSV,
        help=f"Ground-truth CSV (default: {DEFAULT_GT_CSV})",
    )
    parser.add_argument(
        "--model",
        default="models/scene_mlp.joblib",
        help="Path to trained scene MLP (default: models/scene_mlp.joblib)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Write metrics JSON (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip writing JSON report",
    )
    parser.add_argument(
        "--misclassified-dir",
        type=Path,
        default=DEFAULT_MISCLASSIFIED_DIR,
        help=(
            "Directory for misclassified frame copies with gt/pred in the filename "
            f"(default: {DEFAULT_MISCLASSIFIED_DIR}). Use --no-save-misclassified to skip."
        ),
    )
    parser.add_argument(
        "--no-save-misclassified",
        action="store_true",
        help="Do not copy misclassified frames to disk",
    )
    args = parser.parse_args()

    config = Config(scene_mlp_path=str(args.model))
    if not Path(config.scene_mlp_path).is_file():
        raise SystemExit(
            f"Scene MLP model not found at {config.scene_mlp_path}. "
            "Train one with: python scripts/train_scene_classifier.py"
        )

    misclassified_dir = None if args.no_save_misclassified else args.misclassified_dir
    results = evaluate_scene_classifier(
        gt_csv=args.gt_csv,
        config=config,
        misclassified_dir=misclassified_dir,
    )
    print_report(results)

    if not args.no_json:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote metrics to {args.output}")


if __name__ == "__main__":
    main()
