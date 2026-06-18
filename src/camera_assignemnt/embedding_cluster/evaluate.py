"""Ground-truth evaluation for saved cluster assignments (eval only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    v_measure_score,
)

DEFAULT_GT_COLUMNS = ("scene_id", "image_idx", "frame_path", "camera_id")


def load_gt_eval_rows(
    gt_csv: str | Path,
    middle_image_idx: int = 1,
    root: Path | None = None,
) -> pd.DataFrame:
    """Load GT rows for evaluation; middle frame per scene only."""
    df = pd.read_csv(gt_csv)
    keep = [c for c in DEFAULT_GT_COLUMNS if c in df.columns]
    df = df[keep].copy()
    df["scene_id"] = df["scene_id"].astype(str)
    df["camera_id"] = df["camera_id"].astype(str).str.strip()

    if "image_idx" in df.columns:
        middle = df[df["image_idx"] == middle_image_idx]
        if not middle.empty:
            df = middle
        else:
            df = df.sort_values(["scene_id", "image_idx"]).groupby("scene_id", as_index=False).nth(1)

    if root is not None and "frame_path" in df.columns:
        exists_mask = df["frame_path"].apply(lambda p: Path(p).exists() or (root / p).exists())
        df = df[exists_mask].copy()

    return df.sort_values("scene_id").reset_index(drop=True)


def load_predictions(assignments_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(assignments_csv)
    required = {"scene_id", "camera_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Assignments CSV missing columns: {sorted(missing)}")
    df["scene_id"] = df["scene_id"].astype(str)
    return df


def join_predictions_to_gt(
    predictions: pd.DataFrame,
    gt: pd.DataFrame,
) -> pd.DataFrame:
    """Inner join on scene_id."""
    pred_cols = ["scene_id", "camera_id"]
    if "cluster_id" in predictions.columns:
        pred_cols.append("cluster_id")
    if "frame_path" in predictions.columns:
        pred_cols.append("frame_path")

    merged = gt.merge(
        predictions[pred_cols].rename(columns={"camera_id": "pred_camera_id"}),
        on="scene_id",
        how="inner",
        suffixes=("_gt", "_pred"),
    )
    return merged


def _cluster_labels_from_predictions(merged: pd.DataFrame) -> NDArray[np.int64]:
    if "cluster_id" in merged.columns:
        return merged["cluster_id"].astype(int).to_numpy()
    labels = []
    mapping: dict[str, int] = {}
    next_id = 0
    for cam in merged["pred_camera_id"]:
        key = str(cam)
        if key == "unknown":
            labels.append(-1)
            continue
        if key not in mapping:
            mapping[key] = next_id
            next_id += 1
        labels.append(mapping[key])
    return np.array(labels, dtype=np.int64)


def hungarian_mapped_accuracy(
    y_true: NDArray,
    y_pred_clusters: NDArray[np.int64],
) -> tuple[float, dict[int, str]]:
    """Map cluster IDs to GT camera labels via Hungarian algorithm."""
    true_labels = np.array([str(x) for x in y_true])
    pred_labels = y_pred_clusters.astype(int)

    unique_true = sorted(set(true_labels))
    unique_pred = sorted(set(pred_labels))

    if len(unique_pred) == 0 or len(unique_true) == 0:
        return 0.0, {}

    cost = np.zeros((len(unique_pred), len(unique_true)), dtype=np.int64)
    for i, pred in enumerate(unique_pred):
        mask = pred_labels == pred
        pred_true = true_labels[mask]
        for j, true in enumerate(unique_true):
            cost[i, j] = np.sum(pred_true != true)

    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {unique_pred[i]: unique_true[j] for i, j in zip(row_ind, col_ind)}

    mapped = []
    for pred in pred_labels:
        if pred < 0:
            mapped.append("unknown")
        else:
            mapped.append(mapping.get(pred, "unknown"))

    accuracy = float(np.mean(np.array(mapped) == true_labels))
    return accuracy, mapping


def cluster_purity(y_true: NDArray, y_pred_clusters: NDArray[np.int64]) -> float:
    total = len(y_true)
    if total == 0:
        return 0.0

    correct = 0
    for cluster in set(y_pred_clusters):
        if cluster < 0:
            continue
        mask = y_pred_clusters == cluster
        counts = pd.Series(y_true[mask]).value_counts()
        correct += int(counts.iloc[0])
    return correct / total


def evaluate_against_gt(
    assignments_csv: str | Path,
    gt_csv: str | Path,
    middle_image_idx: int = 1,
    root: Path | None = None,
) -> dict:
    """Compute supervised metrics by joining saved predictions to GT."""
    root = root or Path(assignments_csv).resolve().parents[2]
    gt = load_gt_eval_rows(gt_csv, middle_image_idx=middle_image_idx, root=root)
    predictions = load_predictions(assignments_csv)
    merged = join_predictions_to_gt(predictions, gt)

    if merged.empty:
        return {
            "gt_csv": str(gt_csv),
            "assignments_csv": str(assignments_csv),
            "n_gt_scenes": len(gt),
            "n_evaluated": 0,
            "error": "No overlapping scene_id rows between predictions and GT.",
        }

    y_true = merged["camera_id"].to_numpy()
    y_clusters = _cluster_labels_from_predictions(merged)

    accuracy, mapping = hungarian_mapped_accuracy(y_true, y_clusters)

    mapped_preds = []
    for pred in y_clusters:
        if pred < 0:
            mapped_preds.append("unknown")
        else:
            mapped_preds.append(mapping.get(int(pred), "unknown"))

    macro_f1 = float(
        f1_score(y_true, mapped_preds, average="macro", zero_division=0)
    )
    ari = float(adjusted_rand_score(y_true, y_clusters))
    v_measure = float(v_measure_score(y_true, y_clusters))
    purity = float(cluster_purity(y_true, y_clusters))
    noise_rate = float(np.mean(y_clusters == -1))

    per_camera: dict[str, dict] = {}
    for cam in sorted(set(y_true)):
        mask = y_true == cam
        cam_clusters = y_clusters[mask]
        cam_mapped = np.array(mapped_preds)[mask]
        per_camera[cam] = {
            "support": int(mask.sum()),
            "accuracy": float(np.mean(cam_mapped == cam)),
            "noise_rate": float(np.mean(cam_clusters == -1)),
        }

    return {
        "gt_csv": str(gt_csv),
        "assignments_csv": str(assignments_csv),
        "n_gt_scenes": len(gt),
        "n_evaluated": len(merged),
        "cluster_to_camera_mapping": {str(k): v for k, v in mapping.items()},
        "metrics": {
            "hungarian_accuracy": accuracy,
            "macro_f1": macro_f1,
            "adjusted_rand_index": ari,
            "v_measure": v_measure,
            "cluster_purity": purity,
            "noise_rate": noise_rate,
        },
        "per_camera": per_camera,
        "evaluated_rows": merged[
            ["scene_id", "camera_id", "pred_camera_id"]
            + (["cluster_id"] if "cluster_id" in merged.columns else [])
        ].to_dict(orient="records"),
    }
