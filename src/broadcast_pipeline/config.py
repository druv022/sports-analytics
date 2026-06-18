from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

PipelineStep = Literal[
    "all",
    "meta",
    "extract",
    "filter",
    "appearance",
    "cameras",
    "ocr",
    "reference",
    "enrich",
    "associate",
    "aggregate",
]

STAGE_ORDER: tuple[PipelineStep, ...] = (
    "meta",
    "extract",
    "filter",
    "appearance",
    "cameras",
    "ocr",
    "reference",
    "enrich",
    "associate",
    "aggregate",
)


@dataclass
class PipelineConfig:
    video_path: Path = field(default_factory=lambda: Path("data/Untitled.mp4"))
    output_dir: Path = field(default_factory=lambda: Path("data/pipeline"))
    reference_csv: Path | None = None
    detector_threshold: float = 27.0
    camera_samples_per_scene: int = 5
    ocr_samples_per_sec: float = 2.0
    ensemble_method: str = "ensemble"
    enable_vlm: bool = False
    fast_cameras: bool = False
    ocr_scale: float = 1.5
    ocr_preprocess: bool = True
    ocr_rec_batch: int | None = None
    ocr_cls_batch: int | None = None
    ocr_prefetch_workers: int = 2
    ocr_gc_interval: int = 100
    ocr_csv_flush_interval: int = 10
    unk_token: str = "UNK"
    association_min_match_chars: int = 3
    association_min_reference_coverage: float = 0.6
    association_min_prefix_coverage: float = 0.5
    association_min_token_coverage: float = 0.5
    enrich_enabled: bool = True
    enrich_region_iou: float = 0.3
    readability_size_multiplier: float = 1.25
    default_new_text_approved: bool = True
    accelerator: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    persist_camera_debug: bool = True
    camera_scene_temporal_fill: bool = True
    camera_min_vote_share: float | None = 0.6
    camera_merge_closeup_clusters: bool = True
    camera_merge_similarity_threshold: float = 0.70
    camera_merge_max_group_size: int = 3
    camera_reconcile_min_split_size: int = 2
    camera_reconcile_reuse_labels: bool = True
    camera_vlm_collage_qa: bool = False
    appearance_enabled: bool = True
    appearance_model_path: Path | None = None
    appearance_imgsz: int = 640
    appearance_min_confidence: float = 0.5
    appearance_color_tolerance: float = 18.0
    appearance_count_slack: int = 1
    appearance_min_sequence_match: int = 2
    appearance_color_method: Literal["median_lab", "histogram_hsv"] = "histogram_hsv"
    appearance_color_mask_region: Literal["torso", "full"] = "torso"
    appearance_dominant_track_policy: Literal["biggest", "consistent"] = "consistent"
    appearance_track_match_iou: float = 0.3
    appearance_mask_erode_px: int = 1
    appearance_histogram_h_bins: int = 32
    camera_appearance_reconcile: bool = True
    from_step: PipelineStep = "all"
    to_step: PipelineStep | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.video_path = Path(self.video_path)
        if self.reference_csv is None:
            self.reference_csv = self.output_dir / "approved_text_reference.csv"
        else:
            self.reference_csv = Path(self.reference_csv)

    def artifact(self, name: str) -> Path:
        paths = {
            "video_meta": self.output_dir / "video_meta.json",
            "scenes": self.output_dir / "scenes.json",
            "frame_index": self.output_dir / "frame_index.csv",
            "scene_types": self.output_dir / "scene_types.csv",
            "frame_appearance": self.output_dir / "frame_appearance.csv",
            "scene_appearance": self.output_dir / "scene_appearance.csv",
            "scene_assignments": self.output_dir / "scene_assignments.csv",
            "frame_assignments": self.output_dir / "frame_assignments.csv",
            "frame_camera_results": self.output_dir / "frame_camera_results.csv",
            "camera_clustering_debug": self.output_dir / "camera_clustering_debug.npz",
            "camera_merge_log": self.output_dir / "camera_merge_log.json",
            "camera_vlm_qa": self.output_dir / "camera_vlm_qa.json",
            "frame_ocr": self.output_dir / "frame_ocr.csv",
            "frame_ocr_enriched": self.output_dir / "frame_ocr_enriched.csv",
            "reference": self.reference_csv,
            "frame_text_associated": self.output_dir / "frame_text_associated.csv",
            "dropped_text": self.output_dir / "dropped_text.csv",
            "aggregated_complete": self.output_dir / "aggregated_complete.csv",
            "aggregated_partial": self.output_dir / "aggregated_partial.csv",
            "pipeline_summary": self.output_dir / "pipeline_summary.json",
        }
        return paths[name]

    def frames_camera_dir(self) -> Path:
        return self.output_dir / "frames" / "camera"

    def frames_ocr_dir(self) -> Path:
        return self.output_dir / "frames" / "ocr"
