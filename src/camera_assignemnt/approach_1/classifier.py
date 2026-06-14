"""Scene classification via multi-region HSV histograms and an MLP."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import cv2
import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.camera_assignemnt.approach_1.config import COURT_HSV_RANGES, Config
from src.camera_assignemnt.approach_1.models import Frame, SceneType

RegionFn = Callable[[int, int], tuple[int, int, int, int]]

REGION_ORDER: tuple[str, ...] = (
    "full",
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
    "center",
)

FOLDER_LABEL_MAP: dict[str, SceneType] = {
    "full": "full_court",
    "closs-ups": "closeup",
    "closeup": "closeup",
    "close-ups": "closeup",
}

_SCENE_ID_PATTERN = re.compile(r"scene_(\d+)_frame", re.IGNORECASE)
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _region_full(_width: int, _height: int) -> tuple[int, int, int, int]:
    return 0, 0, _width, _height


def _region_top_left(width: int, height: int) -> tuple[int, int, int, int]:
    return 0, 0, width // 2, height // 2


def _region_top_right(width: int, height: int) -> tuple[int, int, int, int]:
    return width // 2, 0, width, height // 2


def _region_bottom_left(width: int, height: int) -> tuple[int, int, int, int]:
    return 0, height // 2, width // 2, height


def _region_bottom_right(width: int, height: int) -> tuple[int, int, int, int]:
    return width // 2, height // 2, width, height


def _region_center(width: int, height: int) -> tuple[int, int, int, int]:
    return width // 3, height // 3, 2 * width // 3, 2 * height // 3


REGION_BUILDERS: dict[str, RegionFn] = {
    "full": _region_full,
    "top_left": _region_top_left,
    "top_right": _region_top_right,
    "bottom_left": _region_bottom_left,
    "bottom_right": _region_bottom_right,
    "center": _region_center,
}


def region_rois(width: int, height: int) -> dict[str, tuple[int, int, int, int]]:
    """Return pixel ROIs ``(x0, y0, x1, y1)`` for each named region."""
    return {name: builder(width, height) for name, builder in REGION_BUILDERS.items()}


def _histogram_for_crop(
    hsv: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    bins: int,
) -> NDArray[np.float32]:
    crop = hsv[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros(2 * bins, dtype=np.float32)

    hist_h = cv2.calcHist([crop], [0], None, [bins], [0, 180]).flatten()
    hist_s = cv2.calcHist([crop], [1], None, [bins], [0, 256]).flatten()
    feat = np.concatenate([hist_h, hist_s]).astype(np.float32)
    total = float(feat.sum())
    return feat if total <= 0 else feat / total


def multi_region_hsv_histogram(frame: Frame, config: Config) -> NDArray[np.float32]:
    """Concatenated multi-region H+S histogram (192-d with default bins=16)."""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    bins = config.histogram_bins
    weights = config.hsv_region_weights

    parts: list[NDArray[np.float32]] = []
    for name in REGION_ORDER:
        x0, y0, x1, y1 = REGION_BUILDERS[name](width, height)
        region_hist = _histogram_for_crop(hsv, x0, y0, x1, y1, bins)
        parts.append(region_hist * weights.get(name, 1.0))

    return np.concatenate(parts).astype(np.float32)


@dataclass(frozen=True)
class ClassificationSample:
    path: Path
    label: SceneType
    scene_id: str
    histogram: NDArray[np.float32]


@dataclass(frozen=True)
class ClassificationDataset:
    samples: tuple[ClassificationSample, ...]

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(s.path for s in self.samples)

    @property
    def X(self) -> NDArray[np.float32]:
        if not self.samples:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack([s.histogram for s in self.samples], axis=0)

    @property
    def y(self) -> NDArray[np.str_]:
        return np.array([s.label for s in self.samples])

    @property
    def groups(self) -> NDArray[np.str_]:
        return np.array([s.scene_id for s in self.samples])

    def __len__(self) -> int:
        return len(self.samples)


def parse_scene_id(filename: str) -> str:
    match = _SCENE_ID_PATTERN.search(filename)
    if match is None:
        return Path(filename).stem
    return match.group(1)


def folder_to_label(folder_name: str) -> SceneType:
    key = folder_name.strip().lower()
    if key not in FOLDER_LABEL_MAP:
        raise ValueError(
            f"Unknown classification folder {folder_name!r}; "
            f"expected one of {sorted(FOLDER_LABEL_MAP)}"
        )
    return FOLDER_LABEL_MAP[key]


def load_classification_dataset(
    data_dir: str | Path,
    config: Config | None = None,
) -> ClassificationDataset:
    """Load labeled images from ``data_dir/<label_folder>/`` with HSV histograms."""
    if config is None:
        config = Config()

    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Classification directory not found: {root}")

    samples: list[ClassificationSample] = []
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue

        label = folder_to_label(class_dir.name)
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in _IMAGE_SUFFIXES:
                continue

            frame = cv2.imread(str(image_path))
            if frame is None:
                raise ValueError(f"Failed to read image: {image_path}")

            samples.append(
                ClassificationSample(
                    path=image_path,
                    label=label,
                    scene_id=parse_scene_id(image_path.name),
                    histogram=multi_region_hsv_histogram(frame, config),
                )
            )

    return ClassificationDataset(samples=tuple(samples))


def build_mlp_pipeline(config: Config) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(128,),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=config.ransac_seed,
                ),
            ),
        ]
    )


def train_scene_mlp(
    X: NDArray[np.float32],
    y: NDArray[np.str_],
    config: Config,
) -> Pipeline:
    pipeline = build_mlp_pipeline(config)
    pipeline.fit(X, y)
    return pipeline


def save_scene_mlp(pipeline: Pipeline, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output)


def load_scene_mlp(path: str | Path) -> Pipeline:
    return joblib.load(path)


def predict_scene_type(
    histogram: NDArray[np.float32],
    pipeline: Pipeline,
) -> SceneType:
    features = np.asarray(histogram, dtype=np.float32).reshape(1, -1)
    label = pipeline.predict(features)[0]
    return label  # type: ignore[return-value]


def _misclassified_rows(
    dataset: ClassificationDataset,
    y_true: NDArray[np.str_],
    y_pred: NDArray[np.str_],
    y_proba: NDArray[np.float64],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample, true_label, pred_label, proba_row in zip(
        dataset.samples, y_true, y_pred, y_proba, strict=True
    ):
        if true_label == pred_label:
            continue
        rows.append(
            {
                "path": str(sample.path),
                "true_label": str(true_label),
                "predicted": str(pred_label),
                "confidence": float(np.max(proba_row)),
            }
        )
    return rows


def cross_validate_scene_mlp(
    dataset: ClassificationDataset,
    config: Config,
    n_splits: int = 5,
) -> dict[str, Any]:
    if len(dataset) == 0:
        raise ValueError("Cannot cross-validate an empty dataset")

    X, y, groups = dataset.X, dataset.y, dataset.groups
    unique_groups = np.unique(groups)
    n_splits = min(n_splits, len(unique_groups))
    if n_splits < 2:
        raise ValueError(
            f"Need at least 2 scene groups for cross-validation, got {len(unique_groups)}"
        )

    pipeline = build_mlp_pipeline(config)
    splitter = GroupKFold(n_splits=n_splits)
    splits = list(splitter.split(X, y, groups=groups))

    y_pred = cross_val_predict(pipeline, X, y, cv=splits)
    y_proba = cross_val_predict(pipeline, X, y, cv=splits, method="predict_proba")

    labels = sorted(set(y.tolist()))
    report = classification_report(
        y, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    fold_metrics: list[dict[str, Any]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        fold_y_true = y[test_idx]
        fold_y_pred = y_pred[test_idx]
        fold_labels = sorted(set(fold_y_true.tolist()))
        fold_metrics.append(
            {
                "fold": fold_idx,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "accuracy": float(accuracy_score(fold_y_true, fold_y_pred)),
                "macro_f1": float(
                    f1_score(
                        fold_y_true,
                        fold_y_pred,
                        average="macro",
                        labels=fold_labels,
                        zero_division=0,
                    )
                ),
            }
        )

    return {
        "n_samples": len(dataset),
        "n_groups": int(len(unique_groups)),
        "n_splits": n_splits,
        "labels": labels,
        "accuracy": float(accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro", labels=labels, zero_division=0)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y, y_pred, labels=labels).tolist(),
        "fold_metrics": fold_metrics,
        "misclassified": _misclassified_rows(dataset, y, y_pred, y_proba),
    }


def train_from_classification_dir(
    data_dir: str | Path,
    output_path: str | Path,
    config: Config | None = None,
    cv_folds: int = 5,
) -> dict[str, Any]:
    if config is None:
        config = Config()

    dataset = load_classification_dataset(data_dir, config)
    cv_results = cross_validate_scene_mlp(dataset, config, n_splits=cv_folds)
    save_scene_mlp(train_scene_mlp(dataset.X, dataset.y, config), output_path)

    cv_results["model_path"] = str(output_path)
    cv_results["n_train_samples"] = len(dataset)
    return cv_results


def compute_court_mask(frame: Frame, config: Config) -> tuple[float, np.ndarray]:
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lo, hi = COURT_HSV_RANGES[config.surface]
    court_mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
    court_ratio = float(court_mask.sum()) / 255.0 / (height * width)
    return court_ratio, court_mask


def _classify_by_court_ratio(court_ratio: float, config: Config) -> SceneType:
    if court_ratio > config.full_court_ratio:
        return "full_court"
    return "closeup"


@lru_cache(maxsize=1)
def _load_cached_scene_mlp(model_path: str) -> Pipeline:
    return load_scene_mlp(model_path)


def classify_scene(
    frame: Frame,
    config: Config,
) -> tuple[SceneType, float, np.ndarray]:
    """Classify a scene; uses MLP when model exists, else court-ratio fallback."""
    court_ratio, court_mask = compute_court_mask(frame, config)

    model_path = Path(config.scene_mlp_path)
    if not model_path.is_file():
        return _classify_by_court_ratio(court_ratio, config), court_ratio, court_mask

    pipeline = _load_cached_scene_mlp(str(model_path.resolve()))
    histogram = multi_region_hsv_histogram(frame, config)
    scene_type = predict_scene_type(histogram, pipeline)
    return scene_type, court_ratio, court_mask
