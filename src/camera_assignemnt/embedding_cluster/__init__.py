"""Camera clustering via visual embeddings or HSV histogram baseline."""

from src.camera_assignemnt.embedding_cluster.config import ClusterConfig, EmbeddingConfig, PipelineConfig
from src.camera_assignemnt.embedding_cluster.evaluate import evaluate_against_gt
from src.camera_assignemnt.embedding_cluster.pipeline import assign_cameras, run_pipeline
from src.camera_assignemnt.embedding_cluster.summarize import summarize_clusters

__all__ = [
    "ClusterConfig",
    "EmbeddingConfig",
    "PipelineConfig",
    "assign_cameras",
    "run_pipeline",
    "summarize_clusters",
    "evaluate_against_gt",
]
