"""Configuration for embedding-based camera clustering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Backend = Literal["resnet50", "dinov2_vits14"]
FeatureMethod = Literal["hsv", "embedding", "ensemble"]
Clusterer = Literal["hdbscan", "dbscan"]
OutputSlug = Literal["hsv", "resnet50", "dinov2_vits14", "ensemble"]

EMBEDDING_OUTPUT_SLUGS: frozenset[str] = frozenset({"resnet50", "dinov2_vits14"})

DEFAULT_ENSEMBLE_TUNING_PATH = "data/evaluation/ensemble_tuning.json"


def output_method_slug(
    method: str,
    embedding: EmbeddingConfig | None = None,
) -> str:
    """Return the canonical output filename suffix for a clustering approach."""
    if method in EMBEDDING_OUTPUT_SLUGS:
        return method
    if method == "hsv":
        return "hsv"
    if method == "ensemble":
        return "ensemble"
    if method == "embedding":
        cfg = embedding or EmbeddingConfig()
        return cfg.backend
    raise ValueError(f"Unknown clustering method: {method!r}")


def resolve_method_and_backend(
    method: str,
    backend: Backend | str = "resnet50",
) -> tuple[FeatureMethod | str, Backend, str]:
    """Normalize CLI method/backend into pipeline method, backend, and output slug."""
    if method in EMBEDDING_OUTPUT_SLUGS:
        return "embedding", method, method  # type: ignore[return-value]
    slug = output_method_slug(method, EmbeddingConfig(backend=backend))  # type: ignore[arg-type]
    return method, backend, slug  # type: ignore[return-value]


@dataclass
class EnsembleConfig:
    """Settings for HSV + ResNet50 + DINOv2 ViT-S/14 voting ensemble."""

    members: tuple[str, ...] = ("hsv", "resnet50", "dinov2_vits14")
    member_weights: dict[str, float] | None = None
    link_threshold: float = 0.5
    noise_threshold: float = 0.6
    vote_threshold: float = 2 / 3
    tuning_path: str = DEFAULT_ENSEMBLE_TUNING_PATH
    tune_sample_size: int = 20
    tune_random_state: int = 42


@dataclass
class ClusterConfig:
    """Shared clustering hyper-parameters for HSV and embedding features."""

    pca_components: int = 64
    use_standard_scaler: bool = True
    reduce_pca: bool = True
    normalize_l2: bool = False
    clusterer: Clusterer = "hdbscan"
    dbscan_eps: float | None = None
    dbscan_min_samples: int = 2
    dbscan_metric: str = "euclidean"
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int = 2
    hdbscan_cluster_selection_epsilon: float = 0.0
    temporal_window: int = 4
    middle_image_idx: int = 1
    apply_temporal: bool = True
    auto_eps: bool = True
    eps_elbow_quantile: float = 0.9
    random_state: int = 0


@dataclass
class EmbeddingConfig:
    """Feature extraction settings for deep embedding backend."""

    backend: Backend = "resnet50"
    device: str = "cpu"
    batch_size: int = 32
    weights_dir: str = "models"


@dataclass
class PipelineConfig:
    """End-to-end pipeline paths and method selection."""

    method: FeatureMethod = "hsv"
    samples_dir: str = "data/scene_samples"
    metadata_csv: str = "data/scene_samples.csv"
    cluster: ClusterConfig | None = None
    embedding: EmbeddingConfig | None = None
    ensemble: EnsembleConfig | None = None
    appearance_features_by_scene: dict[str, list[float]] | None = None
    appearance_feature_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.cluster is None:
            self.cluster = ClusterConfig()
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
        if self.ensemble is None:
            self.ensemble = EnsembleConfig()
