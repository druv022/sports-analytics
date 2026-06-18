#!/usr/bin/env python3
"""Deep analysis of camera assignments vs ground truth and internal consistency."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.camera_assignemnt.embedding_cluster.evaluate import (  # noqa: E402
    evaluate_against_gt,
    hungarian_mapped_accuracy,
    load_gt_eval_rows,
)

DEFAULT_GT = ROOT / "data" / "GT_scene_samples.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "pipeline"
BASELINE_DIR = ROOT / "data" / "evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gt-csv", type=Path, default=DEFAULT_GT)
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Defaults to <output-dir>/camera_assignment_analysis.json",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=None,
        help="Defaults to <output-dir>/camera_assignment_analysis.md",
    )
    return parser.parse_args()


def _parse_votes(raw: str) -> dict[str, int]:
    try:
        return ast.literal_eval(str(raw).replace("'", '"'))
    except (SyntaxError, ValueError):
        return {}


def _scene_type_by_id(gt_csv: Path) -> dict[str, str]:
    gt = pd.read_csv(gt_csv)
    return (
        gt.assign(scene_id=gt["scene_id"].astype(str))
        .groupby("scene_id")["scene_type"]
        .first()
        .astype(str)
        .to_dict()
    )


def _load_baseline_metrics(gt_csv: Path) -> list[dict]:
    rows: list[dict] = []
    if not BASELINE_DIR.is_dir():
        return rows
    for path in sorted(BASELINE_DIR.glob("camera_clustering_*_gt.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        slug = path.stem.replace("camera_clustering_", "").replace("_gt", "")
        metrics = payload.get("metrics", {})
        rows.append(
            {
                "method": slug,
                "n_evaluated": payload.get("n_evaluated"),
                "hungarian_accuracy": metrics.get("hungarian_accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "ari": metrics.get("adjusted_rand_index"),
                "v_measure": metrics.get("v_measure"),
                "purity": metrics.get("cluster_purity"),
                "noise_rate": metrics.get("noise_rate"),
            }
        )
    return rows


def _confusion_table(merged: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for gt_cam, group in merged.groupby("camera_id"):
        pred_counts = Counter(group["mapped_pred"].tolist())
        rows.append(
            {
                "gt_camera": gt_cam,
                "support": int(len(group)),
                "predictions": dict(sorted(pred_counts.items())),
            }
        )
    return sorted(rows, key=lambda item: (-item["support"], item["gt_camera"]))


def _singleton_camera_stats(scene_pred: pd.DataFrame) -> dict:
    counts = scene_pred.groupby("camera_id")["scene_id"].nunique()
    n_cameras = int(len(counts))
    n_singletons = int((counts == 1).sum())
    rate = float(n_singletons / n_cameras) if n_cameras else 0.0
    return {
        "singleton_camera_count": n_singletons,
        "pred_unique_cameras": n_cameras,
        "singleton_camera_rate": rate,
    }


def _reconcile_split_scene_count(scene_pred: pd.DataFrame) -> int:
    split_count = 0
    for row in scene_pred.itertuples(index=False):
        votes = _parse_votes(getattr(row, "camera_vote_counts_json", ""))
        if not votes:
            continue
        winner = max(votes.items(), key=lambda item: item[1])[0]
        if str(row.camera_id) != winner:
            split_count += 1
    return split_count


def _temporal_analysis(scene_df: pd.DataFrame) -> dict:
    ordered = scene_df.sort_values("scene_id", key=lambda s: s.astype(int))
    cameras = ordered["pred_camera_id"].tolist()
    gt_cameras = ordered["camera_id"].tolist()
    mapped = ordered["mapped_pred"].tolist()
    switches_pred = sum(a != b for a, b in zip(cameras, cameras[1:]))
    switches_gt = sum(a != b for a, b in zip(gt_cameras, gt_cameras[1:]))
    switches_mapped = sum(a != b for a, b in zip(mapped, mapped[1:]))
    return {
        "n_scenes": len(ordered),
        "pred_camera_switches": switches_pred,
        "gt_camera_switches": switches_gt,
        "mapped_camera_switches": switches_mapped,
        "pred_unique_cameras": int(ordered["pred_camera_id"].nunique()),
        "gt_unique_cameras": int(ordered["camera_id"].nunique()),
    }


def _frame_level_analysis(
    gt_csv: Path,
    frame_assignments: pd.DataFrame,
    cluster_mapping: dict[int, str],
    root: Path,
) -> dict:
    gt = pd.read_csv(gt_csv)
    gt["scene_id"] = gt["scene_id"].astype(str)
    camera_frames = frame_assignments[frame_assignments["sample_role"] == "camera"].copy()
    camera_frames["scene_id"] = camera_frames["scene_id"].astype(str)

    matched = gt.merge(
        camera_frames.rename(columns={"camera_id": "pred_camera_id"}),
        on=["scene_id", "frame_number"],
        how="inner",
        suffixes=("_gt", "_pred"),
    )
    if matched.empty:
        return {"n_matched_frames": 0}

    def map_cluster(cluster_id: int) -> str:
        if int(cluster_id) < 0:
            return "unknown"
        return cluster_mapping.get(int(cluster_id), "unknown")

    matched["mapped_pred"] = matched["cluster_id"].map(map_cluster)
    matched["correct_mapped"] = matched["camera_id"] == matched["mapped_pred"]
    matched["correct_raw_pred"] = matched["camera_id"] == matched["pred_camera_id"]

    by_idx = (
        matched.groupby("image_idx")["correct_mapped"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "accuracy", "count": "n_frames"})
    )

    return {
        "n_matched_frames": int(len(matched)),
        "n_scenes_with_matches": int(matched["scene_id"].nunique()),
        "hungarian_mapped_accuracy": float(matched["correct_mapped"].mean()),
        "raw_label_accuracy": float(matched["correct_raw_pred"].mean()),
        "by_image_idx": by_idx.to_dict(orient="records"),
        "noise_rate": float((matched["cluster_id"] < 0).mean()),
    }


def _intra_scene_disagreement(
    scene_assignments: pd.DataFrame,
    frame_assignments: pd.DataFrame,
) -> dict:
    """Vote-json reflects pre-majority frame disagreement; frame_assignments does not."""
    rows: list[dict] = []
    for row in scene_assignments.itertuples(index=False):
        votes = _parse_votes(getattr(row, "camera_vote_counts_json", ""))
        unique_cameras = sorted(votes.keys())
        rows.append(
            {
                "scene_id": int(row.scene_id),
                "n_frames": int(sum(int(v) for v in votes.values())),
                "unique_camera_ids": len(unique_cameras),
                "camera_vote_counts": votes,
                "winner": str(row.camera_id),
                "unanimous": len(votes) == 1,
            }
        )

    df = pd.DataFrame(rows)
    mixed = df[df["unique_camera_ids"] > 1]

    # frame_assignments only stores scene-propagated labels; cluster spread is not preserved.
    camera_frames = frame_assignments[frame_assignments["sample_role"] == "camera"]
    propagated_unique = (
        camera_frames.groupby("scene_id")["camera_id"].nunique().gt(1).sum()
        if not camera_frames.empty
        else 0
    )

    return {
        "n_scenes": int(len(df)),
        "unanimous_scenes": int(df["unanimous"].sum()),
        "scenes_with_mixed_frame_votes": int(len(mixed)),
        "mixed_vote_scenes": mixed.sort_values("scene_id").to_dict(orient="records"),
        "note": (
            "frame_assignments.csv stores scene-level labels on every frame; "
            "use camera_vote_counts_json for pre-majority disagreement."
        ),
        "propagated_frame_label_disagreement_scenes": int(propagated_unique),
    }


def _scene_classifier_vs_gt(
    scene_types_path: Path,
    gt_scene_types: dict[str, str],
    eval_scene_ids: set[str],
) -> dict:
    if not scene_types_path.is_file():
        return {"available": False}

    pred_types = pd.read_csv(scene_types_path)
    pred_types["scene_id"] = pred_types["scene_id"].astype(str)
    pred_types = pred_types[pred_types["scene_id"].isin(eval_scene_ids)].copy()
    pred_types["gt_scene_type"] = pred_types["scene_id"].map(gt_scene_types)
    pred_types["correct"] = pred_types["scene_type"] == pred_types["gt_scene_type"]

    by_type: dict[str, dict] = {}
    for gt_type, group in pred_types.groupby("gt_scene_type"):
        by_type[gt_type] = {
            "support": int(len(group)),
            "accuracy": float(group["correct"].mean()),
        }

    return {
        "available": True,
        "n_scenes": int(len(pred_types)),
        "accuracy": float(pred_types["correct"].mean()),
        "by_gt_scene_type": by_type,
        "mismatches": pred_types[~pred_types["correct"]][
            ["scene_id", "scene_type", "gt_scene_type"]
        ].to_dict(orient="records"),
    }


def _vote_confidence_vs_accuracy(scene_df: pd.DataFrame) -> dict:
    work = scene_df.copy()
    work["votes"] = work["camera_vote_counts_json"].map(_parse_votes)
    work["winner_votes"] = work.apply(
        lambda row: row["votes"].get(row["pred_camera_id"], 0), axis=1
    )
    work["total_votes"] = work["votes"].map(lambda votes: sum(int(v) for v in votes.values()))
    work["vote_share"] = work["winner_votes"] / work["total_votes"].replace(0, np.nan)

    bins = [0.0, 0.6, 0.8, 1.0]
    labels = ["<60%", "60-79%", "80-100%"]
    work["confidence_bin"] = pd.cut(work["vote_share"], bins=bins, labels=labels, include_lowest=True)

    by_bin: dict[str, dict] = {}
    for label, group in work.groupby("confidence_bin", observed=False):
        if group.empty:
            continue
        by_bin[str(label)] = {
            "n_scenes": int(len(group)),
            "accuracy": float(group["correct"].mean()),
        }

    return {
        "mean_vote_share": float(work["vote_share"].mean()),
        "unanimous_scenes": int((work["vote_share"] == 1.0).sum()),
        "accuracy_by_vote_share": by_bin,
        "low_confidence_incorrect": work[(work["vote_share"] < 0.6) & (~work["correct"])][
            ["scene_id", "camera_id", "pred_camera_id", "vote_share", "camera_vote_counts_json"]
        ].to_dict(orient="records"),
    }


def analyze(output_dir: Path, gt_csv: Path) -> dict:
    root = ROOT
    scene_path = output_dir / "scene_assignments.csv"
    frame_path = output_dir / "frame_assignments.csv"
    scene_types_path = output_dir / "scene_types.csv"

    supervised = evaluate_against_gt(scene_path, gt_csv, root=root)
    gt_scene_types = _scene_type_by_id(gt_csv)
    gt_mid = load_gt_eval_rows(gt_csv, root=root)
    scene_pred = pd.read_csv(scene_path)
    frame_assignments = pd.read_csv(frame_path)

    scene_pred["scene_id"] = scene_pred["scene_id"].astype(str)
    merged = gt_mid.merge(
        scene_pred.rename(columns={"camera_id": "pred_camera_id"}),
        on="scene_id",
    )

    cluster_mapping = {
        int(k): str(v) for k, v in supervised.get("cluster_to_camera_mapping", {}).items()
    }
    y_true = merged["camera_id"].to_numpy()
    y_clusters = merged["cluster_id"].astype(int).to_numpy()
    _, _ = hungarian_mapped_accuracy(y_true, y_clusters)
    merged["mapped_pred"] = [
        cluster_mapping.get(int(c), "unknown") if int(c) >= 0 else "unknown"
        for c in y_clusters
    ]
    merged["correct"] = merged["camera_id"] == merged["mapped_pred"]
    merged["gt_scene_type"] = merged["scene_id"].map(gt_scene_types)
    merged["pred_noise"] = merged["cluster_id"] < 0

    scene_detail = merged[
        [
            "scene_id",
            "gt_scene_type",
            "camera_id",
            "pred_camera_id",
            "cluster_id",
            "mapped_pred",
            "correct",
            "pred_noise",
            "camera_vote_counts_json",
        ]
    ].sort_values("scene_id", key=lambda s: s.astype(int))

    accuracy_by_scene_type: dict[str, dict] = {}
    for scene_type, group in merged.groupby("gt_scene_type"):
        accuracy_by_scene_type[scene_type] = {
            "support": int(len(group)),
            "accuracy": float(group["correct"].mean()),
            "noise_rate": float(group["pred_noise"].mean()),
        }

    noise_rows = merged[merged["pred_noise"]]
    noise_summary = {
        "n_scenes": int(len(noise_rows)),
        "gt_camera_distribution": Counter(noise_rows["camera_id"].tolist()),
        "pred_camera_distribution": Counter(noise_rows["pred_camera_id"].tolist()),
        "scenes": noise_rows[
            ["scene_id", "camera_id", "pred_camera_id", "camera_vote_counts_json"]
        ].to_dict(orient="records"),
    }

    pipeline_metrics = supervised.get("metrics", {})
    singleton_stats = _singleton_camera_stats(scene_pred)
    reconcile_splits = _reconcile_split_scene_count(scene_pred)
    pipeline_metrics = {
        **pipeline_metrics,
        **singleton_stats,
        "reconcile_split_scenes": reconcile_splits,
    }
    pipeline_row = {
        "method": "pipeline_scene_assignments",
        "n_evaluated": supervised.get("n_evaluated"),
        "hungarian_accuracy": pipeline_metrics.get("hungarian_accuracy"),
        "macro_f1": pipeline_metrics.get("macro_f1"),
        "ari": pipeline_metrics.get("adjusted_rand_index"),
        "v_measure": pipeline_metrics.get("v_measure"),
        "purity": pipeline_metrics.get("cluster_purity"),
        "noise_rate": pipeline_metrics.get("noise_rate"),
        "singleton_camera_rate": pipeline_metrics.get("singleton_camera_rate"),
    }
    baselines = _load_baseline_metrics(gt_csv)
    comparison = [pipeline_row] + baselines
    comparison.sort(
        key=lambda row: (
            row.get("hungarian_accuracy") is None,
            -(row.get("hungarian_accuracy") or 0.0),
        )
    )

    return {
        "output_dir": str(output_dir),
        "gt_csv": str(gt_csv),
        "coverage": {
            "n_gt_scenes": supervised.get("n_gt_scenes"),
            "n_pipeline_scenes": int(scene_pred["scene_id"].nunique()),
            "n_evaluated": supervised.get("n_evaluated"),
            "missing_gt_scene_ids": sorted(
                set(gt_mid["scene_id"]) - set(scene_pred["scene_id"].astype(str)),
                key=int,
            ),
        },
        "supervised_metrics": pipeline_metrics,
        "cluster_to_camera_mapping": supervised.get("cluster_to_camera_mapping"),
        "accuracy_by_gt_scene_type": accuracy_by_scene_type,
        "confusion_by_gt_camera": _confusion_table(merged),
        "temporal": _temporal_analysis(merged),
        "frame_level": _frame_level_analysis(gt_csv, frame_assignments, cluster_mapping, root),
        "intra_scene_disagreement": _intra_scene_disagreement(scene_pred, frame_assignments),
        "scene_classifier_vs_gt": _scene_classifier_vs_gt(
            scene_types_path,
            gt_scene_types,
            set(merged["scene_id"]),
        ),
        "vote_confidence": _vote_confidence_vs_accuracy(merged),
        "noise_analysis": noise_summary,
        "method_comparison": comparison,
        "scene_detail": scene_detail.to_dict(orient="records"),
        "incorrect_scenes": scene_detail[~scene_detail["correct"]].to_dict(orient="records"),
    }


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_markdown(report: dict) -> str:
    metrics = report["supervised_metrics"]
    coverage = report["coverage"]
    lines = [
        "# Camera Assignment Deep Analysis",
        "",
        f"**Output dir:** `{report['output_dir']}`  ",
        f"**GT:** `{report['gt_csv']}`  ",
        f"**Evaluated scenes:** {coverage['n_evaluated']} / {coverage['n_gt_scenes']} GT scenes "
        f"({coverage['n_pipeline_scenes']} in pipeline)",
        "",
        "## Executive summary",
        "",
        _md_table(
            ["Metric", "Value"],
            [
                ["Hungarian accuracy", f"{metrics.get('hungarian_accuracy', 0):.1%}"],
                ["Macro F1", f"{metrics.get('macro_f1', 0):.3f}"],
                ["Adjusted Rand Index", f"{metrics.get('adjusted_rand_index', 0):.3f}"],
                ["V-measure", f"{metrics.get('v_measure', 0):.3f}"],
                ["Cluster purity", f"{metrics.get('cluster_purity', 0):.1%}"],
                ["Noise rate (cluster_id = -1)", f"{metrics.get('noise_rate', 0):.1%}"],
                ["Singleton camera rate", f"{metrics.get('singleton_camera_rate', 0):.1%}"],
                ["Reconcile split scenes", str(metrics.get("reconcile_split_scenes", 0))],
            ],
        ),
        "",
        "## Accuracy by GT scene type",
        "",
    ]

    scene_rows = []
    for scene_type, stats in sorted(report["accuracy_by_gt_scene_type"].items()):
        scene_rows.append(
            [
                scene_type,
                str(stats["support"]),
                f"{stats['accuracy']:.1%}",
                f"{stats['noise_rate']:.1%}",
            ]
        )
    lines.append(_md_table(["GT scene type", "Scenes", "Accuracy", "Noise rate"], scene_rows))

    frame = report["frame_level"]
    lines.extend(
        [
            "",
            "## Frame-level analysis (GT frames matched in pipeline camera samples)",
            "",
            _md_table(
                ["Metric", "Value"],
                [
                    ["Matched GT frames", str(frame.get("n_matched_frames", 0))],
                    ["Hungarian-mapped frame accuracy", f"{frame.get('hungarian_mapped_accuracy', 0):.1%}"],
                    ["Raw label frame accuracy", f"{frame.get('raw_label_accuracy', 0):.1%}"],
                    ["Frame noise rate", f"{frame.get('noise_rate', 0):.1%}"],
                ],
            ),
        ]
    )

    intra = report["intra_scene_disagreement"]
    lines.extend(
        [
            "",
            "## Intra-scene frame disagreement (from vote counts, before majority vote)",
            "",
            f"- Unanimous scenes (all 5 frames agree): **{intra['unanimous_scenes']}** / {intra['n_scenes']}",
            f"- Scenes with mixed frame votes: **{intra['scenes_with_mixed_frame_votes']}**",
            f"- ({intra['note']})",
        ]
    )
    if intra.get("mixed_vote_scenes"):
        lines.append("")
        lines.append("Mixed-vote scenes:")
        for row in intra["mixed_vote_scenes"]:
            lines.append(
                f"- Scene {row['scene_id']}: winner `{row['winner']}` from {row['camera_vote_counts']}"
            )

    classifier = report["scene_classifier_vs_gt"]
    if classifier.get("available"):
        lines.extend(
            [
                "",
                "## Scene-type classifier vs GT",
                "",
                f"Overall accuracy: **{classifier['accuracy']:.1%}** on evaluated scenes",
            ]
        )
        cls_rows = [
            [gt_type, str(stats["support"]), f"{stats['accuracy']:.1%}"]
            for gt_type, stats in sorted(classifier["by_gt_scene_type"].items())
        ]
        lines.append("")
        lines.append(_md_table(["GT scene type", "Scenes", "Classifier accuracy"], cls_rows))

    vote = report["vote_confidence"]
    lines.extend(
        [
            "",
            "## Vote confidence vs Hungarian accuracy",
            "",
            f"- Mean winning vote share: **{vote['mean_vote_share']:.1%}**",
            f"- Unanimous scenes (5/5): **{vote['unanimous_scenes']}**",
        ]
    )
    vote_rows = [
        [label, str(stats["n_scenes"]), f"{stats['accuracy']:.1%}"]
        for label, stats in vote.get("accuracy_by_vote_share", {}).items()
    ]
    if vote_rows:
        lines.append("")
        lines.append(_md_table(["Vote share bin", "Scenes", "Accuracy"], vote_rows))

    temporal = report["temporal"]
    lines.extend(
        [
            "",
            "## Temporal churn",
            "",
            _md_table(
                ["Metric", "Value"],
                [
                    ["GT camera switches", str(temporal["gt_camera_switches"])],
                    ["Predicted camera switches", str(temporal["pred_camera_switches"])],
                    ["Mapped-pred switches", str(temporal["mapped_camera_switches"])],
                    ["Unique GT cameras", str(temporal["gt_unique_cameras"])],
                    ["Unique predicted cameras", str(temporal["pred_unique_cameras"])],
                    [
                        "Singleton cameras",
                        str(metrics.get("singleton_camera_count", "n/a")),
                    ],
                ],
            ),
        ]
    )

    lines.extend(["", "## Method comparison (Hungarian accuracy)", ""])
    cmp_rows = []
    for row in report["method_comparison"]:
        acc = row.get("hungarian_accuracy")
        cmp_rows.append(
            [
                row["method"],
                str(row.get("n_evaluated", "")),
                f"{acc:.1%}" if acc is not None else "n/a",
                f"{row.get('noise_rate', 0):.1%}" if row.get("noise_rate") is not None else "n/a",
            ]
        )
    lines.append(_md_table(["Method", "N scenes", "Accuracy", "Noise"], cmp_rows))

    lines.extend(["", "## Incorrect scenes", ""])
    for row in report["incorrect_scenes"]:
        lines.append(
            f"- Scene {row['scene_id']} ({row['gt_scene_type']}): "
            f"GT `{row['camera_id']}` → mapped `{row['mapped_pred']}` "
            f"(pred `{row['pred_camera_id']}`, cluster {row['cluster_id']}, votes {row['camera_vote_counts_json']})"
        )

    if coverage["missing_gt_scene_ids"]:
        lines.extend(
            [
                "",
                "## Missing GT scenes",
                "",
                "GT scenes not present in pipeline output: "
                + ", ".join(coverage["missing_gt_scene_ids"]),
            ]
        )

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    report_json = args.report_json or (args.output_dir / "camera_assignment_analysis.json")
    report_md = args.report_md or (args.output_dir / "camera_assignment_analysis.md")

    report = analyze(args.output_dir, args.gt_csv)

    def _json_default(obj: object) -> object:
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, Counter):
            return dict(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    report_json.write_text(
        json.dumps(report, indent=2, default=_json_default),
        encoding="utf-8",
    )
    report_md.write_text(render_markdown(report), encoding="utf-8")

    metrics = report["supervised_metrics"]
    print(f"Wrote {report_json}")
    print(f"Wrote {report_md}")
    print(
        f"Evaluated {report['coverage']['n_evaluated']} scenes | "
        f"Hungarian accuracy {metrics.get('hungarian_accuracy', 0):.1%} | "
        f"noise {metrics.get('noise_rate', 0):.1%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
