"""Tune weighted ensemble voting on a random GT dev split."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.camera_assignemnt.embedding_cluster.config import (
    DEFAULT_ENSEMBLE_TUNING_PATH,
    EnsembleConfig,
    PipelineConfig,
)
from src.camera_assignemnt.embedding_cluster.dataset import load_scene_samples
from src.camera_assignemnt.embedding_cluster.evaluate import hungarian_mapped_accuracy, load_gt_eval_rows
from src.camera_assignemnt.embedding_cluster.ensemble import vote_cluster_assignments
from src.camera_assignemnt.embedding_cluster.models import ClusterResult, SceneSample

LINK_THRESHOLDS = (0.4, 0.5, 0.6, 0.67)
NOISE_THRESHOLDS = (0.4, 0.5, 0.6, 0.7, 0.8)
WEIGHT_FLOOR = 0.05


def split_gt_scene_ids(
    gt_csv: str | Path,
    tune_size: int = 20,
    random_state: int = 42,
    middle_image_idx: int = 1,
) -> tuple[list[str], list[str]]:
    """Split GT scene IDs into tune and hold-out sets."""
    gt = load_gt_eval_rows(gt_csv, middle_image_idx=middle_image_idx)
    scene_ids = gt["scene_id"].astype(str).tolist()
    if tune_size >= len(scene_ids):
        raise ValueError(f"tune_size ({tune_size}) must be smaller than GT scenes ({len(scene_ids)})")

    rng = np.random.default_rng(random_state)
    tune_ids = sorted(rng.choice(scene_ids, size=tune_size, replace=False).tolist())
    holdout_ids = sorted(set(scene_ids) - set(tune_ids))
    return tune_ids, holdout_ids


def _scene_indices(samples: list[SceneSample], scene_ids: list[str]) -> NDArray[np.int64]:
    id_to_idx = {sample.scene_id: idx for idx, sample in enumerate(samples)}
    missing = [scene_id for scene_id in scene_ids if scene_id not in id_to_idx]
    if missing:
        raise ValueError(f"GT scene_ids missing from loaded samples: {missing[:5]}")
    return np.array([id_to_idx[scene_id] for scene_id in scene_ids], dtype=np.int64)


def score_member_on_indices(
    labels: NDArray[np.int64],
    y_true: NDArray,
    indices: NDArray[np.int64],
) -> dict[str, float]:
    """Score one member labeling on selected scene indices."""
    subset_labels = labels[indices]
    subset_true = y_true[indices]
    accuracy, _ = hungarian_mapped_accuracy(subset_true, subset_labels)
    noise_rate = float(np.mean(subset_labels < 0))
    combined = accuracy * (1.0 - noise_rate)
    return {
        "hungarian_accuracy": float(accuracy),
        "noise_rate": noise_rate,
        "combined_score": float(combined),
    }


def compute_member_weights(member_scores: dict[str, dict[str, float]]) -> dict[str, float]:
    """Derive normalized member weights from tune-set scores."""
    raw = {
        name: max(stats["combined_score"], 0.0)
        for name, stats in member_scores.items()
    }
    total = sum(raw.values())
    if total <= 0:
        equal = 1.0 / len(raw)
        return {name: equal for name in raw}

    weights = {name: value / total for name, value in raw.items()}
    floored = {name: max(WEIGHT_FLOOR, weight) for name, weight in weights.items()}
    total = sum(floored.values())
    return {name: weight / total for name, weight in floored.items()}


def evaluate_ensemble_on_indices(
    samples: list[SceneSample],
    base_results: list[ClusterResult],
    labelings: list[NDArray[np.int64]],
    member_names: list[str],
    reduced: NDArray[np.float32],
    y_true_full: NDArray,
    indices: NDArray[np.int64],
    member_weights: dict[str, float],
    link_threshold: float,
    noise_threshold: float,
    temporal_window: int,
) -> dict[str, float]:
    """Evaluate weighted ensemble on a subset of scene indices."""
    voted = vote_cluster_assignments(
        base_results,
        labelings,
        member_names,
        reduced,
        member_weights=member_weights,
        link_threshold=link_threshold,
        noise_threshold=noise_threshold,
        temporal_window=temporal_window,
    )
    voted_labels = np.array([r.cluster_id for r in voted], dtype=np.int64)
    subset_labels = voted_labels[indices]
    subset_true = y_true_full[indices]
    accuracy, _ = hungarian_mapped_accuracy(subset_true, subset_labels)
    noise_rate = float(np.mean(subset_labels < 0))
    objective = float(accuracy - 0.3 * noise_rate)
    return {
        "hungarian_accuracy": float(accuracy),
        "noise_rate": noise_rate,
        "objective": objective,
    }


def grid_search_thresholds(
    samples: list[SceneSample],
    base_results: list[ClusterResult],
    labelings: list[NDArray[np.int64]],
    member_names: list[str],
    reduced: NDArray[np.float32],
    gt_by_scene: dict[str, str],
    tune_indices: NDArray[np.int64],
    member_weights: dict[str, float],
    temporal_window: int,
) -> tuple[float, float, dict[str, float]]:
    """Pick link/noise thresholds maximizing tune-set objective."""
    y_true_full = np.array([gt_by_scene[s.scene_id] for s in samples], dtype=object)
    best_link = LINK_THRESHOLDS[0]
    best_noise = NOISE_THRESHOLDS[0]
    best_metrics = {"objective": -1.0}

    for link_threshold in LINK_THRESHOLDS:
        for noise_threshold in NOISE_THRESHOLDS:
            metrics = evaluate_ensemble_on_indices(
                samples,
                base_results,
                labelings,
                member_names,
                reduced,
                y_true_full,
                tune_indices,
                member_weights,
                link_threshold,
                noise_threshold,
                temporal_window,
            )
            if metrics["objective"] > best_metrics["objective"]:
                best_link = link_threshold
                best_noise = noise_threshold
                best_metrics = metrics

    return best_link, best_noise, best_metrics


def tune_ensemble(
    config: PipelineConfig,
    gt_csv: str | Path,
    tune_size: int = 20,
    random_state: int = 42,
) -> dict:
    """Run ensemble member clusterings and tune weights/thresholds on a GT dev split."""
    gt_csv = Path(gt_csv)
    tune_ids, holdout_ids = split_gt_scene_ids(
        gt_csv,
        tune_size=tune_size,
        random_state=random_state,
        middle_image_idx=config.cluster.middle_image_idx,
    )

    gt = load_gt_eval_rows(gt_csv, middle_image_idx=config.cluster.middle_image_idx)
    gt_by_scene = dict(zip(gt["scene_id"].astype(str), gt["camera_id"].astype(str)))

    samples = load_scene_samples(
        samples_dir=config.samples_dir,
        metadata_csv=config.metadata_csv,
        middle_image_idx=config.cluster.middle_image_idx,
        load_frames=True,
    )
    from src.camera_assignemnt.embedding_cluster.pipeline import run_ensemble_member_labelings

    labelings, member_names, reduced, base_results, member_meta = run_ensemble_member_labelings(
        samples,
        config,
    )

    tune_indices = _scene_indices(samples, tune_ids)
    holdout_indices = _scene_indices(samples, holdout_ids)
    y_true_full = np.array([gt_by_scene.get(s.scene_id, "") for s in samples], dtype=object)

    member_scores: dict[str, dict[str, float]] = {}
    for name, labels in zip(member_names, labelings):
        member_scores[name] = score_member_on_indices(labels, y_true_full, tune_indices)

    member_weights = compute_member_weights(member_scores)
    link_threshold, noise_threshold, tune_metrics = grid_search_thresholds(
        samples,
        base_results,
        labelings,
        member_names,
        reduced,
        gt_by_scene,
        tune_indices,
        member_weights,
        config.cluster.temporal_window,
    )

    holdout_metrics = evaluate_ensemble_on_indices(
        samples,
        base_results,
        labelings,
        member_names,
        reduced,
        y_true_full,
        holdout_indices,
        member_weights,
        link_threshold,
        noise_threshold,
        config.cluster.temporal_window,
    )

    return {
        "gt_csv": str(gt_csv),
        "tune_sample_size": tune_size,
        "random_state": random_state,
        "tune_scene_ids": tune_ids,
        "holdout_scene_ids": holdout_ids,
        "member_weights": member_weights,
        "link_threshold": link_threshold,
        "noise_threshold": noise_threshold,
        "member_scores_tune": member_scores,
        "ensemble_metrics_tune": tune_metrics,
        "ensemble_metrics_holdout": holdout_metrics,
        "member_cluster_meta": member_meta,
    }


def save_ensemble_tuning(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return path


def load_ensemble_tuning(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def resolve_ensemble_settings(ensemble_cfg: EnsembleConfig, root: Path | None = None) -> EnsembleConfig:
    """Merge explicit config with saved tuning artifact when weights are unset."""
    if ensemble_cfg.member_weights is not None:
        return ensemble_cfg

    tuning_path = Path(ensemble_cfg.tuning_path)
    if not tuning_path.is_absolute() and root is not None:
        tuning_path = root / tuning_path

    report = load_ensemble_tuning(tuning_path)
    if report is None:
        equal = 1.0 / len(ensemble_cfg.members)
        return EnsembleConfig(
            members=ensemble_cfg.members,
            member_weights={name: equal for name in ensemble_cfg.members},
            link_threshold=ensemble_cfg.link_threshold,
            noise_threshold=ensemble_cfg.noise_threshold,
            vote_threshold=ensemble_cfg.vote_threshold,
            tuning_path=ensemble_cfg.tuning_path,
            tune_sample_size=ensemble_cfg.tune_sample_size,
            tune_random_state=ensemble_cfg.tune_random_state,
        )

    return EnsembleConfig(
        members=ensemble_cfg.members,
        member_weights={k: float(v) for k, v in report["member_weights"].items()},
        link_threshold=float(report.get("link_threshold", ensemble_cfg.link_threshold)),
        noise_threshold=float(report.get("noise_threshold", ensemble_cfg.noise_threshold)),
        vote_threshold=ensemble_cfg.vote_threshold,
        tuning_path=ensemble_cfg.tuning_path,
        tune_sample_size=int(report.get("tune_sample_size", ensemble_cfg.tune_sample_size)),
        tune_random_state=int(report.get("random_state", ensemble_cfg.tune_random_state)),
    )
