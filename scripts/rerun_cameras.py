#!/usr/bin/env python3
"""Re-run cameras stage only and compare metrics to baseline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from broadcast_pipeline.camera_assignment import assign_cameras_multi_frame  # noqa: E402
from broadcast_pipeline.camera_vlm_qa import run_camera_collage_vlm_qa  # noqa: E402
from broadcast_pipeline.config import PipelineConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Re-running cameras overwrites scene_assignments.csv and related artifacts. "
            "If using main.py with resume=True, delete scene_assignments.csv first or run "
            "main.py --from-step cameras without --resume."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "pipeline")
    parser.add_argument("--gt-csv", type=Path, default=ROOT / "data" / "GT_scene_samples.csv")
    parser.add_argument("--baseline-dir", type=Path, default=None)
    parser.add_argument("--no-collages", action="store_true")
    parser.add_argument("--no-analysis", action="store_true")
    parser.add_argument("--vlm-qa", action="store_true", help="Run optional VLM collage QA")
    return parser.parse_args()


def _load_metrics(path: Path) -> dict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("supervised_metrics", payload)


def _print_delta(baseline: dict | None, current: dict | None, scene_path: Path) -> None:
    if baseline is None or current is None:
        print("Baseline or current metrics missing; skipping delta.")
        return
    print("\n=== Metrics delta (current vs baseline) ===")
    for key in ("hungarian_accuracy", "noise_rate", "cluster_purity", "singleton_camera_rate"):
        b = baseline.get(key)
        c = current.get(key)
        if b is None or c is None:
            continue
        print(f"  {key}: {c:.3f} (was {b:.3f}, Δ {c - b:+.3f})")

    if scene_path.is_file():
        import pandas as pd

        scenes = pd.read_csv(scene_path)
        cam1 = int((scenes["camera_id"] == "cam_1").sum())
        print(f"  scenes on cam_1: {cam1}")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    baseline_dir = args.baseline_dir or (output_dir / "baseline")

    config = PipelineConfig(output_dir=output_dir)
    if args.vlm_qa:
        config.camera_vlm_collage_qa = True

    import pandas as pd

    frame_index = pd.read_csv(config.artifact("frame_index"))
    scene_assignments, frame_assignments = assign_cameras_multi_frame(config, frame_index)
    scene_assignments.to_csv(config.artifact("scene_assignments"), index=False)
    frame_assignments.to_csv(config.artifact("frame_assignments"), index=False)

    if not args.no_collages:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "render_camera_collages.py"),
                "--output-dir",
                str(output_dir),
            ],
            check=True,
        )

    if config.camera_vlm_collage_qa:
        report = run_camera_collage_vlm_qa(config)
        print(f"VLM QA: {len(report.get('cameras', []))} multi-scene collages checked")

    analysis_path = output_dir / "camera_assignment_analysis.json"
    if not args.no_analysis:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "analyze_camera_assignment.py"),
                "--output-dir",
                str(output_dir),
                "--gt-csv",
                str(args.gt_csv),
            ],
            check=True,
        )

    baseline_metrics = _load_metrics(baseline_dir / "camera_assignment_analysis.json")
    current_metrics = _load_metrics(analysis_path)
    _print_delta(baseline_metrics, current_metrics, config.artifact("scene_assignments"))

    print(f"\nWrote {config.artifact('scene_assignments')}")
    print(f"Wrote {config.artifact('frame_assignments')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
