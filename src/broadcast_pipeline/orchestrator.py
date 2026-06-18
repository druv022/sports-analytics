from __future__ import annotations

import gc
from pathlib import Path

import pandas as pd

from broadcast_pipeline.artifacts import resolve_stage_range, should_skip_stage, validate_stage_inputs
from broadcast_pipeline.camera_assignment import assign_cameras_multi_frame
from broadcast_pipeline.config import PipelineConfig, PipelineStep
from broadcast_pipeline.preflight import preflight
from broadcast_pipeline.progress import log_skip, log_stage_done, log_stage_start
from broadcast_pipeline.types import PipelineSummary


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def run_stages(
    config: PipelineConfig,
    from_step: PipelineStep,
    until_step: PipelineStep,
    *,
    log_range: bool = True,
) -> PipelineSummary:
    """Run pipeline stages from *from_step* through *until_step* (inclusive)."""
    if from_step == "all":
        raise ValueError("run_stages requires an explicit from_step, not 'all'")
    if until_step == "all":
        raise ValueError("run_stages requires an explicit until_step, not 'all'")

    slice_stages = resolve_stage_range(from_step=from_step, to_step=until_step)
    if log_range:
        print(f"run_stages: {' → '.join(slice_stages)}")

    saved_from = config.from_step
    saved_to = config.to_step
    try:
        config.from_step = from_step
        config.to_step = until_step
        return run_pipeline(config)
    finally:
        config.from_step = saved_from
        config.to_step = saved_to


def run_pipeline(config: PipelineConfig) -> PipelineSummary:
    stages = resolve_stage_range(config.from_step, config.to_step)
    range_label = (
        f"{stages[0]}→{stages[-1]}" if len(stages) > 1 else stages[0]
    )
    log_stage_start("preflight", detail=f"{len(stages)} stage(s): {range_label}")
    preflight(config, stages)
    if config.from_step != "all":
        validate_stage_inputs(config, config.from_step)
    log_stage_done("preflight")

    summary = PipelineSummary(output_dir=config.output_dir)
    meta = None
    scenes = []
    frame_index = pd.DataFrame()
    scene_types = pd.DataFrame()
    scene_assignments = pd.DataFrame()
    frame_assignments = pd.DataFrame()
    frame_ocr = pd.DataFrame()
    reference = pd.DataFrame()
    associated = pd.DataFrame()
    dropped = pd.DataFrame()
    n_new_reference = 0

    if "meta" in stages:
        from broadcast_pipeline.video_meta import load_video_meta, probe_video, save_video_meta

        meta_path = config.artifact("video_meta")
        if should_skip_stage(config, "meta", meta_path):
            log_skip("meta", detail=str(meta_path.name))
            meta = load_video_meta(meta_path)
        else:
            log_stage_start("meta", detail=config.video_path.name)
            meta = probe_video(config.video_path)
            save_video_meta(meta, meta_path)
            log_stage_done(
                "meta",
                detail=(
                    f"{meta.duration_sec:.1f}s, {meta.fps:.2f} fps, "
                    f"{meta.frame_count} frames, {meta.width}x{meta.height}"
                ),
            )
        summary.video_meta = meta
        summary.artifacts["video_meta"] = meta_path

    if "extract" in stages:
        from broadcast_pipeline.scene_extractor import (
            extract_scenes_and_frames,
            load_scenes,
            save_scenes,
        )

        if meta is None:
            meta = load_video_meta(config.artifact("video_meta"))
        frame_index_path = config.artifact("frame_index")
        scenes_path = config.artifact("scenes")
        if should_skip_stage(config, "extract", frame_index_path) and scenes_path.is_file():
            log_skip("extract", detail=f"{scenes_path.name}, {frame_index_path.name}")
            frame_index = _load_csv(frame_index_path)
            scenes = load_scenes(scenes_path)
        else:
            log_stage_start("extract", detail=config.video_path.name)
            scenes, _, frame_index = extract_scenes_and_frames(config, meta)
            frame_index.to_csv(frame_index_path, index=False)
            save_scenes(scenes, scenes_path)
            log_stage_done(
                "extract",
                detail=f"{len(scenes)} scenes, {len(frame_index)} sampled frames",
            )
        summary.n_scenes = len(scenes)
        summary.artifacts["frame_index"] = frame_index_path
        summary.artifacts["scenes"] = scenes_path
        gc.collect()

    if "filter" in stages:
        from broadcast_pipeline.scene_filter import classify_scenes

        if frame_index.empty:
            frame_index = _load_csv(config.artifact("frame_index"))
        scene_types_path = config.artifact("scene_types")
        if should_skip_stage(config, "filter", scene_types_path):
            log_skip("filter", detail=scene_types_path.name)
            scene_types = _load_csv(scene_types_path)
        else:
            log_stage_start("filter", detail=f"{len(frame_index['scene_id'].unique())} scenes")
            scene_types = classify_scenes(config, frame_index)
            scene_types.to_csv(scene_types_path, index=False)
            log_stage_done("filter", detail=f"{len(scene_types)} scene types assigned")
        summary.artifacts["scene_types"] = scene_types_path
        gc.collect()

    if "appearance" in stages:
        from broadcast_pipeline.appearance_filter import analyze_scenes

        if frame_index.empty:
            frame_index = _load_csv(config.artifact("frame_index"))
        if scene_types.empty and config.artifact("scene_types").is_file():
            scene_types = _load_csv(config.artifact("scene_types"))
        frame_appearance_path = config.artifact("frame_appearance")
        scene_appearance_path = config.artifact("scene_appearance")
        if (
            not config.appearance_enabled
            or (
                should_skip_stage(config, "appearance", scene_appearance_path)
                and frame_appearance_path.is_file()
            )
        ):
            if config.appearance_enabled:
                log_skip(
                    "appearance",
                    detail=f"{scene_appearance_path.name}, {frame_appearance_path.name}",
                )
            else:
                log_skip("appearance", detail="disabled")
        else:
            log_stage_start(
                "appearance",
                detail=f"{len(frame_index['scene_id'].unique())} scenes",
            )
            frame_appearance, scene_appearance = analyze_scenes(config, frame_index)
            frame_appearance.to_csv(frame_appearance_path, index=False)
            scene_appearance.to_csv(scene_appearance_path, index=False)
            log_stage_done(
                "appearance",
                detail=f"{len(scene_appearance)} scene appearance signatures",
            )
        summary.artifacts["frame_appearance"] = frame_appearance_path
        summary.artifacts["scene_appearance"] = scene_appearance_path
        gc.collect()

    if "cameras" in stages:
        gc.collect()
        if frame_index.empty:
            frame_index = _load_csv(config.artifact("frame_index"))
        scene_assignments_path = config.artifact("scene_assignments")
        frame_assignments_path = config.artifact("frame_assignments")
        if (
            should_skip_stage(config, "cameras", scene_assignments_path)
            and frame_assignments_path.is_file()
        ):
            log_skip(
                "cameras",
                detail=f"{scene_assignments_path.name}, {frame_assignments_path.name}",
            )
            scene_assignments = _load_csv(scene_assignments_path)
            frame_assignments = _load_csv(frame_assignments_path)
        else:
            method = "hsv" if config.fast_cameras else config.ensemble_method
            n_camera_samples = len(frame_index[frame_index["sample_role"] == "camera"])
            log_stage_start("cameras", detail=f"method={method}, {n_camera_samples} samples")
            scene_assignments, frame_assignments = assign_cameras_multi_frame(
                config, frame_index
            )
            scene_assignments.to_csv(scene_assignments_path, index=False)
            frame_assignments.to_csv(frame_assignments_path, index=False)
            n_cameras = scene_assignments["camera_id"].nunique()
            log_stage_done("cameras", detail=f"{n_cameras} camera(s) across {len(scene_assignments)} scenes")
        summary.n_cameras = scene_assignments["camera_id"].nunique()
        summary.artifacts["scene_assignments"] = scene_assignments_path
        summary.artifacts["frame_assignments"] = frame_assignments_path
        frame_camera_results_path = config.artifact("frame_camera_results")
        if frame_camera_results_path.is_file():
            summary.artifacts["frame_camera_results"] = frame_camera_results_path
        debug_path = config.artifact("camera_clustering_debug")
        if debug_path.is_file():
            summary.artifacts["camera_clustering_debug"] = debug_path

    if "ocr" in stages:
        from broadcast_pipeline.ocr_runner import ocr_is_complete, run_segment_ocr

        if frame_index.empty:
            frame_index = _load_csv(config.artifact("frame_index"))
        if frame_assignments.empty:
            frame_assignments = _load_csv(config.artifact("frame_assignments"))
        frame_ocr_path = config.artifact("frame_ocr")
        if (
            config.resume
            and should_skip_stage(config, "ocr", frame_ocr_path)
            and ocr_is_complete(frame_ocr_path, frame_index)
        ):
            log_skip("ocr", detail=frame_ocr_path.name)
            frame_ocr = _load_csv(frame_ocr_path)
        else:
            if (
                frame_ocr_path.is_file()
                and ocr_is_complete(frame_ocr_path, frame_index)
                and not config.resume
            ):
                frame_ocr_path.unlink()
            n_ocr_frames = len(frame_index[frame_index["sample_role"] == "ocr"])
            log_stage_start("ocr", detail=f"{n_ocr_frames} frames")
            frame_ocr = run_segment_ocr(
                config,
                frame_index,
                frame_assignments,
                output_path=frame_ocr_path,
            )
            log_stage_done("ocr", detail=f"{len(frame_ocr)} OCR observations stored")
        summary.n_frames = len(frame_ocr)
        summary.artifacts["frame_ocr"] = frame_ocr_path

    if "reference" in stages:
        from broadcast_pipeline.text_reference import update_text_reference

        if frame_ocr.empty:
            frame_ocr = _load_csv(config.artifact("frame_ocr"))
        reference_path = config.reference_csv
        log_stage_start("reference", detail=str(reference_path.name))
        reference, n_new_reference = update_text_reference(config, frame_ocr)
        log_stage_done(
            "reference",
            detail=f"{len(reference)} entries, {n_new_reference} new",
        )
        summary.n_new_reference_text = n_new_reference
        summary.artifacts["reference"] = reference_path

    if "enrich" in stages:
        from broadcast_pipeline.text_enrich import (
            enrich_is_complete,
            enrich_ocr_observations,
        )

        if frame_ocr.empty:
            frame_ocr = _load_csv(config.artifact("frame_ocr"))
        if frame_assignments.empty:
            frame_assignments = _load_csv(config.artifact("frame_assignments"))
        enriched_path = config.artifact("frame_ocr_enriched")
        if (
            config.resume
            and should_skip_stage(config, "enrich", enriched_path)
            and enrich_is_complete(enriched_path, frame_ocr)
        ):
            log_skip("enrich", detail=enriched_path.name)
        else:
            log_stage_start("enrich")
            enriched = enrich_ocr_observations(config, frame_ocr, frame_assignments)
            enriched.to_csv(enriched_path, index=False)
            log_stage_done("enrich", detail=f"{len(enriched)} enriched OCR rows")
        summary.artifacts["frame_ocr_enriched"] = enriched_path

    if "associate" in stages:
        from broadcast_pipeline.text_reference import _load_reference
        from broadcast_pipeline.text_associate import associate_text
        from broadcast_pipeline.text_enrich import load_processed_frame_ocr

        frame_ocr = load_processed_frame_ocr(config)
        if reference.empty:
            reference = _load_reference(config.reference_csv)
        associated_path = config.artifact("frame_text_associated")
        dropped_path = config.artifact("dropped_text")
        if should_skip_stage(config, "associate", associated_path) and dropped_path.is_file():
            log_skip(
                "associate",
                detail=f"{associated_path.name}, {dropped_path.name}",
            )
            associated = _load_csv(associated_path)
            dropped = _load_csv(dropped_path)
        else:
            log_stage_start("associate")
            raw_for_prov = None
            if config.artifact("frame_ocr_enriched").is_file():
                raw_for_prov = _load_csv(config.artifact("frame_ocr"))
            associated, dropped = associate_text(
                config, frame_ocr, reference, raw_frame_ocr=raw_for_prov
            )
            associated.to_csv(associated_path, index=False)
            dropped.to_csv(dropped_path, index=False)
            log_stage_done(
                "associate",
                detail=f"{len(associated)} associated, {len(dropped)} dropped",
            )
        summary.artifacts["frame_text_associated"] = associated_path
        summary.artifacts["dropped_text"] = dropped_path

    if "aggregate" in stages:
        from broadcast_pipeline.aggregator import (
            aggregate_text_timeline,
            load_aggregate_inputs,
            write_pipeline_summary,
        )
        from broadcast_pipeline.text_enrich import load_processed_frame_ocr
        from broadcast_pipeline.text_reference import _load_reference

        if associated.empty:
            associated = _load_csv(config.artifact("frame_text_associated"))
        frame_ocr = load_processed_frame_ocr(config)
        if reference.empty:
            reference = _load_reference(config.reference_csv)
        if meta is None or not scenes:
            meta, scenes = load_aggregate_inputs(config)

        log_stage_start("aggregate")
        complete_df, partial_df, agg_summary = aggregate_text_timeline(
            config, associated, frame_ocr, meta, scenes
        )
        complete_path = config.artifact("aggregated_complete")
        partial_path = config.artifact("aggregated_partial")
        complete_df.to_csv(complete_path, index=False)
        partial_df.to_csv(partial_path, index=False)

        summary.n_scenes = agg_summary.n_scenes
        summary.n_frames = agg_summary.n_frames
        summary.n_cameras = agg_summary.n_cameras
        summary.artifacts["aggregated_complete"] = complete_path
        summary.artifacts["aggregated_partial"] = partial_path

        if reference.empty:
            reference = _load_reference(config.reference_csv)
        write_pipeline_summary(
            config,
            summary,
            reference,
            n_new_reference,
            complete_df=complete_df,
            partial_df=partial_df,
        )
        summary.artifacts["pipeline_summary"] = config.artifact("pipeline_summary")
        log_stage_done(
            "aggregate",
            detail=(
                f"{len(complete_df)} complete rows, "
                f"{len(partial_df)} partial rows"
            ),
        )

    return summary
