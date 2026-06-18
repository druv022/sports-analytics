from __future__ import annotations

import importlib

from broadcast_pipeline.config import STAGE_ORDER, PipelineConfig, PipelineStep
from src.accelerator.install import check_gpu_stack, warn_gpu_stack_gaps
from src.camera_assignemnt.scene_classifier.classifier import resolve_scene_mlp
from src.camera_assignemnt.scene_classifier.config import Config as SceneConfig
from src.camera_assignemnt.scene_classifier.models import SceneMlpError

_STAGE_DEPS: dict[PipelineStep, tuple[str, ...]] = {
    "meta": (),
    "extract": ("scenedetect", "cv2", "pandas"),
    "filter": ("cv2", "pandas"),
    "cameras": ("numpy", "pandas", "sklearn"),
    "ocr": ("cv2", "pandas"),
    "reference": ("pandas",),
    "enrich": ("pandas",),
    "associate": ("pandas",),
    "aggregate": ("pandas",),
}


def _check_import(module: str) -> str | None:
    try:
        importlib.import_module(module)
        return None
    except ImportError:
        hints = {
            "rapidocr": 'pip install -e ".[ocr]"',
            "torch": 'pip install -e ".[embedding]"',
            "openai": 'pip install -e ".[ocr-vlm]"',
        }
        return hints.get(module, f"pip install {module}")


def preflight(config: PipelineConfig, stages: tuple[PipelineStep, ...]) -> None:
    modules: set[str] = set()
    for stage in stages:
        modules.update(_STAGE_DEPS.get(stage, ()))
    if config.enable_vlm:
        modules.add("openai")
    if config.ensemble_method == "ensemble" and "cameras" in stages:
        modules.update({"torch", "torchvision"})
    if "ocr" in stages:
        modules.add("rapidocr")
    errors: list[str] = []
    for module in sorted(modules):
        hint = _check_import(module)
        if hint:
            errors.append(f"{module}: {hint}")
    if errors:
        raise ImportError("Missing dependencies:\n" + "\n".join(errors))

    if "filter" in stages:
        scene_config = SceneConfig()
        project_root = config.output_dir.resolve().parent.parent
        try:
            resolve_scene_mlp(scene_config, project_root=project_root)
        except SceneMlpError as exc:
            raise SceneMlpError(
                f"Preflight failed for filter stage: {exc}"
            ) from exc

    if config.accelerator in ("auto", "cuda"):
        needs_gpu_check = (
            ("ocr" in stages)
            or (config.ensemble_method == "ensemble" and "cameras" in stages)
        )
        if needs_gpu_check:
            report = check_gpu_stack(config.accelerator)
            warn_gpu_stack_gaps(report, config.accelerator)
