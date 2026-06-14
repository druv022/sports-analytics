"""Visualization helpers for cluster inspection."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from src.camera_assignemnt.approach_4.models import ClusterResult

try:
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    _PLOT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLOT_AVAILABLE = False


def require_plotting() -> None:
    if not _PLOT_AVAILABLE:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib  (or pip install -e '.')"
        )


def plotting_available() -> bool:
    return _PLOT_AVAILABLE


def plot_tsne(
    reduced: NDArray[np.float32],
    labels: NDArray[np.int64] | list[int],
    save_path: str | Path,
    title: str = "Cluster t-SNE",
    gt_labels: list[str] | None = None,
) -> Path:
    """Save a 2-D t-SNE scatter plot coloured by cluster (and optional GT)."""
    require_plotting()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    labels_arr = np.array(labels, dtype=np.int64)
    n_samples = len(reduced)
    if n_samples < 2:
        return save_path

    perplexity = min(30, max(2, n_samples - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=0,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(reduced)

    n_plots = 2 if gt_labels else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    scatter = axes[0].scatter(coords[:, 0], coords[:, 1], c=labels_arr, cmap="tab20", s=30)
    axes[0].set_title(title)
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")
    fig.colorbar(scatter, ax=axes[0], label="cluster_id")

    if gt_labels:
        gt_numeric = {label: idx for idx, label in enumerate(sorted(set(gt_labels)))}
        gt_colors = np.array([gt_numeric[label] for label in gt_labels])
        scatter_gt = axes[1].scatter(
            coords[:, 0], coords[:, 1], c=gt_colors, cmap="tab20", s=30
        )
        axes[1].set_title(f"{title} — GT camera_id")
        axes[1].set_xlabel("t-SNE 1")
        axes[1].set_ylabel("t-SNE 2")
        fig.colorbar(scatter_gt, ax=axes[1], label="camera_id")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _resize_thumb(image: np.ndarray, max_size: int = 160) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max_size / max(h, w)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)))


def save_cluster_montages(
    results: list[ClusterResult],
    save_dir: str | Path,
    max_per_cluster: int = 8,
    thumb_size: int = 160,
) -> list[str]:
    """Save per-cluster contact sheets for manual inspection."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    by_cluster: dict[int, list[ClusterResult]] = {}
    for result in results:
        by_cluster.setdefault(result.cluster_id, []).append(result)

    saved: list[str] = []
    for cluster_id in sorted(by_cluster):
        members = by_cluster[cluster_id][:max_per_cluster]
        thumbs = []
        for member in members:
            image = cv2.imread(member.frame_path)
            if image is None:
                continue
            thumb = _resize_thumb(image, thumb_size)
            label = f"id{member.scene_id}"
            cv2.putText(
                thumb,
                label,
                (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            thumbs.append(thumb)

        if not thumbs:
            continue

        row = cv2.hconcat(thumbs)
        out_path = save_dir / f"cluster_{cluster_id}_montage.jpg"
        cv2.imwrite(str(out_path), row)
        saved.append(str(out_path))

    return saved
