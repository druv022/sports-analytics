from __future__ import annotations

from pathlib import Path

from broadcast_pipeline.config import STAGE_ORDER, PipelineConfig, PipelineStep

STAGE_REQUIRES: dict[PipelineStep, tuple[str, ...]] = {
    "meta": (),
    "extract": ("video_meta",),
    "filter": ("frame_index",),
    "appearance": ("frame_index", "scene_types"),
    "cameras": ("frame_index",),
    "ocr": ("frame_index", "frame_assignments"),
    "reference": ("frame_ocr",),
    "enrich": ("frame_ocr",),
    "associate": ("frame_ocr", "reference", "frame_assignments"),
    "aggregate": ("frame_text_associated", "scenes", "video_meta"),
}


def resolve_stage_range(
    from_step: PipelineStep = "all",
    to_step: PipelineStep | None = None,
) -> tuple[PipelineStep, ...]:
    """Return the inclusive stage slice to execute."""
    start_idx = 0 if from_step == "all" else STAGE_ORDER.index(from_step)
    if to_step is None or to_step == "all":
        end_idx = len(STAGE_ORDER) - 1
    else:
        if to_step not in STAGE_ORDER:
            raise ValueError(f"Unknown pipeline step: {to_step!r}")
        end_idx = STAGE_ORDER.index(to_step)

    if start_idx > end_idx:
        start_name = STAGE_ORDER[start_idx]
        end_name = STAGE_ORDER[end_idx]
        raise ValueError(
            f"from_step {from_step!r} ({start_name}) is after to_step {to_step!r} ({end_name})"
        )

    return STAGE_ORDER[start_idx : end_idx + 1]


def validate_stage_inputs(config: PipelineConfig, step: PipelineStep) -> None:
    if step == "all":
        return
    missing: list[str] = []
    for artifact_key in STAGE_REQUIRES.get(step, ()):
        path = config.artifact(artifact_key)
        if not path.is_file():
            missing.append(str(path))
    if step in ("meta", "extract") and not config.video_path.is_file():
        missing.append(str(config.video_path))
    if missing:
        raise FileNotFoundError(
            f"Cannot run from step {step!r}; missing inputs: {', '.join(missing)}"
        )


def should_skip_stage(
    config: PipelineConfig,
    step: PipelineStep,
    output_path: Path | None = None,
) -> bool:
    if not config.resume or output_path is None:
        return False
    return output_path.is_file()
