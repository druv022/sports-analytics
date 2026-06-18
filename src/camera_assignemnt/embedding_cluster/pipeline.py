"""End-to-end camera clustering pipeline."""

from __future__ import annotations

import gc

import numpy as np
from numpy.typing import NDArray

from src.camera_assignemnt.scene_classifier.classifier import multi_region_hsv_histogram
from src.camera_assignemnt.scene_classifier.config import Config as SceneClassifierConfig
from src.camera_assignemnt.embedding_cluster.cluster import cluster_features
from src.camera_assignemnt.embedding_cluster.config import (
    ClusterConfig,
    EmbeddingConfig,
    PipelineConfig,
    output_method_slug,
)
from src.camera_assignemnt.embedding_cluster.dataset import load_scene_samples, resolve_sample_frame
from src.camera_assignemnt.embedding_cluster.embedder import (
    default_device,
    dinov2_batch_size,
    extract_features_batch,
    load_model,
    release_model,
)
from src.camera_assignemnt.embedding_cluster.ensemble import member_summary, vote_cluster_assignments
from src.camera_assignemnt.embedding_cluster.ensemble_tune import resolve_ensemble_settings
from src.camera_assignemnt.embedding_cluster.models import ClusterResult, PipelineOutput, SceneSample
from src.camera_assignemnt.embedding_cluster.summarize import summarize_clusters
from src.person_appearance.appearance_features import blend_appearance_features

ENSEMBLE_EMBEDDING_BACKENDS = {
    "resnet50": "resnet50",
    "dinov2_vits14": "dinov2_vits14",
}

# Per-backend clustering presets (tuned on scene_samples).
EMBEDDING_CLUSTER_PRESETS: dict[str, dict] = {
    "resnet50": {
        "clusterer": "dbscan",
        "auto_eps": True,
        "dbscan_min_samples": 2,
    },
    "dinov2_vits14": {
        "clusterer": "hdbscan",
        "hdbscan_min_cluster_size": 3,
        "hdbscan_min_samples": 2,
        "auto_eps": False,
    },
}


def _build_results(samples: list[SceneSample]) -> list[ClusterResult]:
    return [
        ClusterResult(
            scene_idx=s.scene_idx,
            scene_id=s.scene_id,
            frame_path=s.frame_path,
        )
        for s in samples
    ]


def _copy_clusterer_fields(source: ClusterConfig, **overrides) -> ClusterConfig:
    base = {
        "pca_components": source.pca_components,
        "use_standard_scaler": source.use_standard_scaler,
        "reduce_pca": source.reduce_pca,
        "normalize_l2": source.normalize_l2,
        "clusterer": source.clusterer,
        "dbscan_eps": source.dbscan_eps,
        "dbscan_min_samples": source.dbscan_min_samples,
        "dbscan_metric": source.dbscan_metric,
        "hdbscan_min_cluster_size": source.hdbscan_min_cluster_size,
        "hdbscan_min_samples": source.hdbscan_min_samples,
        "hdbscan_cluster_selection_epsilon": source.hdbscan_cluster_selection_epsilon,
        "temporal_window": source.temporal_window,
        "middle_image_idx": source.middle_image_idx,
        "apply_temporal": source.apply_temporal,
        "auto_eps": source.auto_eps,
        "eps_elbow_quantile": source.eps_elbow_quantile,
        "random_state": source.random_state,
    }
    base.update(overrides)
    return ClusterConfig(**base)


def _hsv_style_cluster_config(cluster_cfg: ClusterConfig) -> ClusterConfig:
    return _copy_clusterer_fields(
        cluster_cfg,
        use_standard_scaler=False,
        reduce_pca=False,
        dbscan_eps=cluster_cfg.dbscan_eps if cluster_cfg.dbscan_eps is not None else 0.35,
        dbscan_metric="euclidean",
        auto_eps=False,
    )


def _embedding_style_cluster_config(
    cluster_cfg: ClusterConfig,
    backend: str = "resnet50",
) -> ClusterConfig:
    """Return backend-specific clustering preset for deep embeddings."""
    preset = EMBEDDING_CLUSTER_PRESETS.get(backend, EMBEDDING_CLUSTER_PRESETS["resnet50"])
    return _copy_clusterer_fields(
        cluster_cfg,
        use_standard_scaler=False,
        reduce_pca=True,
        normalize_l2=True,
        dbscan_metric="cosine",
        **preset,
    )


def extract_hsv_features(samples: list[SceneSample]) -> NDArray[np.float32]:
    config = SceneClassifierConfig()
    features = []
    for sample in samples:
        features.append(multi_region_hsv_histogram(resolve_sample_frame(sample), config))
    return np.stack(features, axis=0).astype(np.float32)


def extract_embedding_features(
    samples: list[SceneSample],
    config: EmbeddingConfig | None = None,
    model=None,
) -> tuple[NDArray[np.float32], object | None]:
    if config is None:
        config = EmbeddingConfig()
    if config.device == "cpu":
        try:
            config.device = default_device()
        except ImportError:
            pass

    loaded_here = model is None
    if loaded_here:
        model = load_model(config)

    try:
        features = extract_features_batch(samples, model, config)
    finally:
        if loaded_here:
            release_model(model)
    return features, None if loaded_here else model


def _maybe_blend_appearance(
    features: NDArray[np.float32],
    samples: list[SceneSample],
    pipeline_cfg: PipelineConfig | None,
) -> NDArray[np.float32]:
    if pipeline_cfg is None:
        return features
    if (
        pipeline_cfg.appearance_feature_weight <= 0
        or not pipeline_cfg.appearance_features_by_scene
    ):
        return features
    appearance_map = {
        scene_id: np.asarray(vector, dtype=np.float32)
        for scene_id, vector in pipeline_cfg.appearance_features_by_scene.items()
    }
    scene_ids = [str(sample.scene_id) for sample in samples]
    return blend_appearance_features(
        features,
        scene_ids,
        appearance_map,
        weight=pipeline_cfg.appearance_feature_weight,
    )


def cluster_samples(
    samples: list[SceneSample],
    cluster_cfg: ClusterConfig,
    method: str,
    embedding_cfg: EmbeddingConfig | None = None,
    apply_temporal: bool | None = None,
    pipeline_cfg: PipelineConfig | None = None,
) -> tuple[list[ClusterResult], NDArray[np.float32], float]:
    """Run one base clustering pass (HSV or embedding)."""
    results = _build_results(samples)

    if method == "hsv":
        raw_features = extract_hsv_features(samples)
        active_cfg = _hsv_style_cluster_config(cluster_cfg)
    elif method == "embedding":
        if embedding_cfg is None:
            raise ValueError("embedding_cfg is required for embedding clustering")
        raw_features, _ = extract_embedding_features(samples, embedding_cfg)
        active_cfg = _embedding_style_cluster_config(cluster_cfg, embedding_cfg.backend)
    else:
        raise ValueError(f"Unsupported base clustering method: {method!r}")

    raw_features = _maybe_blend_appearance(raw_features, samples, pipeline_cfg)
    use_temporal = cluster_cfg.apply_temporal if apply_temporal is None else apply_temporal
    return cluster_features(results, raw_features, active_cfg, apply_temporal=use_temporal)


def _embedding_config_for_member(
    member: str,
    base: EmbeddingConfig,
) -> EmbeddingConfig:
    backend = ENSEMBLE_EMBEDDING_BACKENDS.get(member)
    if backend is None:
        raise ValueError(f"Unknown ensemble embedding member: {member!r}")
    device = base.device
    if device == "cpu":
        try:
            device = default_device()
        except ImportError:
            pass
    if backend == "dinov2_vits14":
        batch_size = dinov2_batch_size(device)
    else:
        batch_size = base.batch_size
        if str(device).split(":")[0] == "cuda" and batch_size == 32:
            batch_size = 64
    return EmbeddingConfig(
        backend=backend,
        device=device,
        batch_size=batch_size,
        weights_dir=base.weights_dir,
    )


def run_ensemble_member_labelings(
    samples: list[SceneSample],
    config: PipelineConfig,
    apply_temporal: bool | None = None,
) -> tuple[
    list[NDArray[np.int64]],
    list[str],
    NDArray[np.float32],
    list[ClusterResult],
    dict[str, dict],
]:
    """Run each ensemble member clustering and return labelings plus metadata."""
    ensemble_cfg = config.ensemble
    base_results = _build_results(samples)
    labelings: list[NDArray[np.int64]] = []
    member_names: list[str] = []
    reduced_for_viz: NDArray[np.float32] | None = None
    member_meta: dict[str, dict] = {}

    for member in ensemble_cfg.members:
        if member == "hsv":
            print("  Ensemble member: HSV", flush=True)
            results, reduced, cluster_param = cluster_samples(
                samples, config.cluster, "hsv", apply_temporal=apply_temporal, pipeline_cfg=config
            )
        elif member in ENSEMBLE_EMBEDDING_BACKENDS:
            print(f"  Ensemble member: {member}", flush=True)
            emb_cfg = _embedding_config_for_member(member, config.embedding)
            results, reduced, cluster_param = cluster_samples(
                samples,
                config.cluster,
                "embedding",
                emb_cfg,
                apply_temporal=apply_temporal,
                pipeline_cfg=config,
            )
            if member == "resnet50":
                reduced_for_viz = reduced
        else:
            raise ValueError(f"Unsupported ensemble member: {member!r}")

        labelings.append(np.array([r.cluster_id for r in results], dtype=np.int64))
        member_names.append(member)
        member_meta[member] = {
            "cluster_param": cluster_param,
            "n_clusters": int(len(set(labelings[-1]) - {-1})),
            "n_noise": int(np.sum(labelings[-1] < 0)),
        }
        if member in ENSEMBLE_EMBEDDING_BACKENDS:
            gc.collect()
            try:
                import torch

                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except ImportError:
                pass

    if reduced_for_viz is None and labelings:
        _, reduced_for_viz, _ = cluster_samples(
            samples,
            config.cluster,
            "embedding",
            _embedding_config_for_member("resnet50", config.embedding),
            apply_temporal=apply_temporal,
            pipeline_cfg=config,
        )

    return labelings, member_names, reduced_for_viz, base_results, member_meta


def assign_cameras_ensemble(
    config: PipelineConfig,
    samples: list[SceneSample] | None = None,
    apply_temporal: bool | None = None,
) -> PipelineOutput:
    """Run HSV, ResNet50, and DINOv2 clusterings with weighted voting."""
    if samples is None:
        samples = load_scene_samples(
            samples_dir=config.samples_dir,
            metadata_csv=config.metadata_csv,
            middle_image_idx=config.cluster.middle_image_idx,
            load_frames=True,
        )
    if not samples:
        return PipelineOutput(method="ensemble")

    from pathlib import Path

    root = Path(config.metadata_csv).resolve().parent.parent
    ensemble_cfg = resolve_ensemble_settings(config.ensemble, root=root)

    labelings, member_names, reduced_for_viz, base_results, member_meta = (
        run_ensemble_member_labelings(samples, config, apply_temporal=apply_temporal)
    )

    voted_results = vote_cluster_assignments(
        base_results,
        labelings,
        member_names,
        reduced_for_viz,
        member_weights=ensemble_cfg.member_weights,
        link_threshold=ensemble_cfg.link_threshold,
        noise_threshold=ensemble_cfg.noise_threshold,
        temporal_window=config.cluster.temporal_window,
    )

    stats = member_summary(labelings, member_names)
    for name, meta in member_meta.items():
        stats[name]["cluster_param"] = meta["cluster_param"]

    return PipelineOutput(
        results=voted_results,
        reduced_matrix=reduced_for_viz,
        dbscan_eps=ensemble_cfg.link_threshold,
        method="ensemble",
        ensemble_vote_threshold=ensemble_cfg.link_threshold,
        ensemble_noise_threshold=ensemble_cfg.noise_threshold,
        ensemble_member_weights=ensemble_cfg.member_weights,
        ensemble_member_stats=stats,
        ensemble_member_labelings=labelings,
        ensemble_member_names=member_names,
    )


def assign_cameras(
    config: PipelineConfig | None = None,
    samples: list[SceneSample] | None = None,
    apply_temporal: bool | None = None,
) -> PipelineOutput:
    """Run clustering on scene_samples and return camera assignments."""
    if config is None:
        config = PipelineConfig()

    if config.method == "ensemble":
        return assign_cameras_ensemble(config, samples=samples, apply_temporal=apply_temporal)

    cluster_cfg = config.cluster
    if samples is None:
        samples = load_scene_samples(
            samples_dir=config.samples_dir,
            metadata_csv=config.metadata_csv,
            middle_image_idx=config.cluster.middle_image_idx,
            load_frames=True,
        )
    if not samples:
        return PipelineOutput(method=output_method_slug(config.method, config.embedding))

    if config.method == "hsv":
        results, reduced, eps = cluster_samples(
            samples, cluster_cfg, "hsv", apply_temporal=apply_temporal, pipeline_cfg=config
        )
    elif config.method == "embedding":
        results, reduced, eps = cluster_samples(
            samples,
            cluster_cfg,
            "embedding",
            config.embedding,
            apply_temporal=apply_temporal,
            pipeline_cfg=config,
        )
    else:
        raise ValueError(f"Unknown method: {config.method!r}")

    method_slug = output_method_slug(config.method, config.embedding)
    single_labeling = np.array([r.cluster_id for r in results], dtype=np.int64)
    return PipelineOutput(
        results=results,
        reduced_matrix=reduced,
        dbscan_eps=eps,
        method=method_slug,
        ensemble_member_labelings=[single_labeling],
        ensemble_member_names=[method_slug],
    )


def run_pipeline(config: PipelineConfig | None = None) -> dict:
    """Run pipeline and return unsupervised summary dict."""
    output = assign_cameras(config)
    if config is None:
        config = PipelineConfig()
    if not output.results or output.reduced_matrix is None or output.dbscan_eps is None:
        return summarize_clusters([], np.empty((0, 0), dtype=np.float32), 0.0, config.method)

    summary = summarize_clusters(
        output.results,
        output.reduced_matrix,
        output.dbscan_eps,
        method=output.method,
        temporal_window=config.cluster.temporal_window,
    )
    if output.ensemble_member_stats is not None:
        summary["ensemble"] = {
            "link_threshold": output.ensemble_vote_threshold,
            "noise_threshold": output.ensemble_noise_threshold,
            "member_weights": output.ensemble_member_weights,
            "members": output.ensemble_member_stats,
        }
    return summary
