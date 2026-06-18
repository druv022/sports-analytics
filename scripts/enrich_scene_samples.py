#!/usr/bin/env python3
"""Merge embedding_cluster camera assignments into scene_samples.csv for all frames."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_ASSIGNMENTS = ROOT / "data" / "evaluation" / "camera_assignments_ensemble.csv"
DEFAULT_SAMPLES = ROOT / "data" / "scene_samples.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--samples-csv", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--out", type=Path, default=None, help="Output CSV (default: overwrite samples-csv)")
    parser.add_argument(
        "--assignment-source",
        default="embedding_cluster_ensemble",
        help="Value written to assignment_source column",
    )
    parser.add_argument(
        "--refresh-scene-type",
        action="store_true",
        help="Re-classify scene_type per scene using scene_classifier middle frame",
    )
    return parser.parse_args()


def _classify_scenes(samples_df: pd.DataFrame) -> pd.Series:
    import cv2

    from src.camera_assignemnt.scene_classifier.classifier import classify_scene
    from src.camera_assignemnt.scene_classifier.config import Config

    config = Config()
    scene_types: dict[int, str] = {}

    middle = samples_df[samples_df["image_idx"] == 1]
    if middle.empty:
        middle = samples_df.groupby("scene_id", as_index=False).nth(1)

    for row in middle.itertuples(index=False):
        frame_path = Path(getattr(row, "frame_path"))
        if not frame_path.is_absolute():
            frame_path = ROOT / frame_path
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        scene_type, _, _ = classify_scene(image, config)
        scene_types[int(getattr(row, "scene_id"))] = scene_type

    return samples_df["scene_id"].map(scene_types)


def enrich_samples(
    samples_csv: Path,
    assignments_csv: Path,
    assignment_source: str,
    refresh_scene_type: bool = False,
) -> pd.DataFrame:
    samples = pd.read_csv(samples_csv)
    assignments = pd.read_csv(assignments_csv)

    required_assign = {"scene_id", "camera_id"}
    missing = required_assign - set(assignments.columns)
    if missing:
        raise ValueError(f"Assignments CSV missing columns: {sorted(missing)}")

    assign_cols = ["scene_id", "camera_id"]
    if "cluster_id" in assignments.columns:
        assign_cols.append("cluster_id")

    lookup = assignments[assign_cols].drop_duplicates("scene_id")
    lookup["scene_id"] = lookup["scene_id"].astype(int)
    samples["scene_id"] = samples["scene_id"].astype(int)

    merged = samples.drop(columns=[c for c in ("camera_id", "cluster_id", "assignment_source") if c in samples.columns])
    merged = merged.merge(lookup, on="scene_id", how="left")

    if merged["camera_id"].isna().any():
        missing_ids = merged.loc[merged["camera_id"].isna(), "scene_id"].unique()
        raise ValueError(f"Missing camera_id for scene_ids: {missing_ids.tolist()}")

    merged["assignment_source"] = assignment_source

    if refresh_scene_type:
        merged["scene_type"] = _classify_scenes(merged)
        merged["scene_type"] = merged.groupby("scene_id")["scene_type"].transform("first")

    return merged


def main() -> int:
    args = parse_args()
    if not args.assignments.is_file():
        print(f"Assignments not found: {args.assignments}")
        print("Run: python scripts/run_camera_clustering.py --method ensemble")
        return 1
    if not args.samples_csv.is_file():
        print(f"Samples CSV not found: {args.samples_csv}")
        return 1

    out_path = args.out or args.samples_csv
    enriched = enrich_samples(
        args.samples_csv,
        args.assignments,
        args.assignment_source,
        refresh_scene_type=args.refresh_scene_type,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out_path, index=False)

    n_scenes = enriched["scene_id"].nunique()
    n_cameras = enriched["camera_id"].nunique()
    print(f"Wrote {len(enriched)} rows ({n_scenes} scenes) -> {out_path}")
    print(f"Unique camera_id labels: {n_cameras}")
    print(enriched["camera_id"].value_counts().head(10).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
