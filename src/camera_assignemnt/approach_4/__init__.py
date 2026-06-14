"""Camera clustering via visual embeddings or HSV histogram baseline."""

from src.camera_assignemnt.approach_4.config import ClusterConfig, EmbeddingConfig, PipelineConfig
from src.camera_assignemnt.approach_4.evaluate import evaluate_against_gt
from src.camera_assignemnt.approach_4.pipeline import assign_cameras, run_pipeline
from src.camera_assignemnt.approach_4.summarize import summarize_clusters

__all__ = [
    "ClusterConfig",
    "EmbeddingConfig",
    "PipelineConfig",
    "assign_cameras",
    "run_pipeline",
    "summarize_clusters",
    "evaluate_against_gt",
]
