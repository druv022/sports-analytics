#!/usr/bin/env python3
"""Tune ensemble member weights and voting thresholds on a random GT dev split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.camera_assignemnt.embedding_cluster.config import (  # noqa: E402
    DEFAULT_ENSEMBLE_TUNING_PATH,
    ClusterConfig,
    EmbeddingConfig,
    PipelineConfig,
)
from src.camera_assignemnt.embedding_cluster.ensemble_tune import (  # noqa: E402
    save_ensemble_tuning,
    tune_ensemble,
)

DEFAULT_GT_CSV = ROOT / "data" / "GT_scene_samples.csv"
DEFAULT_METADATA_CSV = ROOT / "data" / "scene_samples.csv"
DEFAULT_SAMPLES_DIR = ROOT / "data" / "scene_samples"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-csv", type=Path, default=DEFAULT_GT_CSV)
    parser.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES_DIR)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--output", type=Path, default=ROOT / DEFAULT_ENSEMBLE_TUNING_PATH)
    parser.add_argument("--tune-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "models")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PipelineConfig(
        method="ensemble",
        samples_dir=str(args.samples_dir),
        metadata_csv=str(args.metadata_csv),
        cluster=ClusterConfig(),
        embedding=EmbeddingConfig(
            device=args.device,
            weights_dir=str(args.weights_dir),
        ),
    )

    report = tune_ensemble(
        config,
        gt_csv=args.gt_csv,
        tune_size=args.tune_size,
        random_state=args.seed,
    )
    out_path = save_ensemble_tuning(report, args.output)

    tune_metrics = report["ensemble_metrics_tune"]
    holdout_metrics = report["ensemble_metrics_holdout"]
    print(f"Wrote {out_path}")
    print(f"Tune scenes: {len(report['tune_scene_ids'])}  Hold-out: {len(report['holdout_scene_ids'])}")
    print(f"Member weights: {report['member_weights']}")
    print(f"Link threshold: {report['link_threshold']:.2f}  Noise threshold: {report['noise_threshold']:.2f}")
    print(
        "Tune metrics: "
        f"acc={tune_metrics['hungarian_accuracy']:.3f} "
        f"noise={tune_metrics['noise_rate']:.3f} "
        f"objective={tune_metrics['objective']:.3f}"
    )
    print(
        "Hold-out metrics: "
        f"acc={holdout_metrics['hungarian_accuracy']:.3f} "
        f"noise={holdout_metrics['noise_rate']:.3f} "
        f"objective={holdout_metrics['objective']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
