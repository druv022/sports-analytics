#!/usr/bin/env python3
"""Cluster scene_samples by visual similarity (HSV baseline or embeddings)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.camera_assignemnt.embedding_cluster.config import (  # noqa: E402
    EMBEDDING_OUTPUT_SLUGS,
    ClusterConfig,
    EmbeddingConfig,
    PipelineConfig,
    resolve_method_and_backend,
)
from src.camera_assignemnt.embedding_cluster.pipeline import assign_cameras  # noqa: E402
from src.camera_assignemnt.embedding_cluster.summarize import summarize_clusters  # noqa: E402
from src.camera_assignemnt.embedding_cluster.visualize import (  # noqa: E402
    plot_tsne,
    plotting_available,
    save_cluster_montages,
)

DEFAULT_SAMPLES_DIR = ROOT / "data" / "scene_samples"
DEFAULT_METADATA_CSV = ROOT / "data" / "scene_samples.csv"
DEFAULT_EVAL_DIR = ROOT / "data" / "evaluation"
DEFAULT_VERIFY_DIR = ROOT / "data" / "verification"
METHOD_CHOICES = ("hsv", "embedding", "ensemble", *sorted(EMBEDDING_OUTPUT_SLUGS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=METHOD_CHOICES,
        default="hsv",
        help="Clustering approach: hsv, ensemble, embedding, or an embedding slug "
        "(resnet50, dinov2_vits14).",
    )
    parser.add_argument(
        "--backend",
        choices=("resnet50", "dinov2_vits14"),
        default="resnet50",
        help="Embedding backend (embedding method only).",
    )
    parser.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES_DIR)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--verify-dir", type=Path, default=DEFAULT_VERIFY_DIR)
    parser.add_argument("--dbscan-eps", type=float, default=None)
    parser.add_argument("--no-auto-eps", action="store_true")
    parser.add_argument(
        "--auto-eps",
        action="store_true",
        help="Auto-calibrate eps from k-NN elbow (default for embedding).",
    )
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--temporal-window", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--no-viz", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pipeline_method, backend, _slug = resolve_method_and_backend(args.method, args.backend)

    cluster_cfg = ClusterConfig(
        pca_components=args.pca_components,
        dbscan_eps=args.dbscan_eps,
        auto_eps=args.auto_eps
        or (pipeline_method == "embedding" and not args.no_auto_eps)
        or (pipeline_method == "ensemble" and not args.no_auto_eps),
        temporal_window=args.temporal_window,
    )
    embedding_cfg = EmbeddingConfig(
        backend=backend,
        device=args.device,
        weights_dir=str(args.weights_dir),
    )
    config = PipelineConfig(
        method=pipeline_method,
        samples_dir=str(args.samples_dir),
        metadata_csv=str(args.metadata_csv),
        cluster=cluster_cfg,
        embedding=embedding_cfg,
    )

    output = assign_cameras(config)
    if not output.results or output.reduced_matrix is None or output.dbscan_eps is None:
        print("No scenes found to cluster.")
        return 1

    summary = summarize_clusters(
        output.results,
        output.reduced_matrix,
        output.dbscan_eps,
        method=output.method,
        temporal_window=cluster_cfg.temporal_window,
    )
    if output.ensemble_member_stats is not None:
        summary["ensemble"] = {
            "link_threshold": output.ensemble_vote_threshold,
            "noise_threshold": output.ensemble_noise_threshold,
            "member_weights": output.ensemble_member_weights,
            "members": output.ensemble_member_stats,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"camera_clustering_{output.method}.json"
    assignments_path = args.output_dir / f"camera_assignments_{output.method}.csv"

    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    pd.DataFrame(summary["assignments"]).to_csv(assignments_path, index=False)

    print(f"Method: {output.method}")
    print(f"Scenes: {summary['n_scenes']}  Clusters: {summary['n_clusters']}  Noise: {summary['n_noise']}")
    if output.method == "ensemble":
        print(f"Link threshold: {summary['dbscan_eps']:.3f}")
        ensemble_info = summary.get("ensemble", {})
        weights = ensemble_info.get("member_weights", {})
        if weights:
            print(f"Member weights: {weights}")
        print(f"Noise threshold: {ensemble_info.get('noise_threshold', 'n/a')}")
        for member, stats in ensemble_info.get("members", {}).items():
            print(
                f"  {member}: clusters={stats['n_clusters']} "
                f"noise={stats['n_noise']} param={stats.get('cluster_param', 'n/a')}"
            )
    else:
        print(f"Cluster param: {summary['dbscan_eps']:.4f}")
    metrics = summary["metrics"]
    print(
        "Metrics: "
        f"silhouette={metrics['silhouette']} "
        f"davies_bouldin={metrics['davies_bouldin']} "
        f"temporal_coherence={metrics['temporal_coherence']:.3f} "
        f"noise_rate={metrics['noise_rate']:.3f}"
    )
    print(f"Wrote {summary_path}")
    print(f"Wrote {assignments_path}")

    if not args.no_viz:
        if not plotting_available():
            print(
                "Skipping visualization: matplotlib is not installed. "
                "Run: pip install matplotlib  (or pip install -e '.')"
            )
        else:
            labels = [r.cluster_id for r in output.results]
            tsne_path = args.verify_dir / f"cluster_tsne_{output.method}.png"
            plot_tsne(
                output.reduced_matrix,
                labels,
                tsne_path,
                title=f"Camera clusters ({output.method})",
            )
            montage_dir = args.verify_dir / f"clusters_{output.method}"
            montages = save_cluster_montages(output.results, montage_dir)
            print(f"Wrote {tsne_path}")
            print(f"Wrote {len(montages)} cluster montages to {montage_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
