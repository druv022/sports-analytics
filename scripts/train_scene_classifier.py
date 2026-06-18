#!/usr/bin/env python3
"""Train the scene-type MLP classifier from folder-labeled images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python scripts/train_scene_classifier.py` from repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.camera_assignemnt.scene_classifier.classifier import (
    load_classification_dataset,
    train_from_classification_dir,
)
from src.camera_assignemnt.scene_classifier.config import Config


def _print_cv_summary(results: dict) -> None:
    print(f"Samples: {results['n_samples']}  Groups: {results['n_groups']}")
    print(f"CV folds: {results['n_splits']}")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"Macro F1: {results['macro_f1']:.3f}")
    print("Confusion matrix (rows=true, cols=pred):")
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

    print("\nFold metrics:")
    for fold in results["fold_metrics"]:
        print(
            f"  fold {fold['fold']}: "
            f"train={fold['n_train']} test={fold['n_test']} "
            f"acc={fold['accuracy']:.3f} macro_f1={fold['macro_f1']:.3f}"
        )

    misclassified = results.get("misclassified", [])
    if misclassified:
        print("\nMisclassified CV holdout frames:")
        for row in misclassified:
            print(
                f"  {Path(row['path']).name}: "
                f"true={row['true_label']} pred={row['predicted']} "
                f"conf={row['confidence']:.3f}"
            )
    else:
        print("\nNo misclassified CV holdout frames.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="data/classification",
        help="Root folder with class subdirectories (default: data/classification)",
    )
    parser.add_argument(
        "--output",
        default="models/scene_mlp.joblib",
        help="Path to save the trained model (default: models/scene_mlp.joblib)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of grouped CV folds (default: 5)",
    )
    parser.add_argument(
        "--metrics-json",
        default="",
        help="Optional path to write CV metrics as JSON",
    )
    args = parser.parse_args()

    config = Config()
    dataset = load_classification_dataset(args.data_dir, config)
    if len(dataset) == 0:
        raise SystemExit(f"No labeled images found under {args.data_dir}")

    print(f"Loaded {len(dataset)} samples from {args.data_dir}")
    print(f"Labels: { {label: int((dataset.y == label).sum()) for label in sorted(set(dataset.y.tolist()))} }")

    results = train_from_classification_dir(
        data_dir=args.data_dir,
        output_path=args.output,
        config=config,
        cv_folds=args.cv_folds,
    )

    _print_cv_summary(results)
    print(f"\nSaved model to {results['model_path']}")

    if args.metrics_json:
        metrics_path = Path(args.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            key: value
            for key, value in results.items()
            if key != "classification_report"
        }
        serializable["classification_report"] = results["classification_report"]
        metrics_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
