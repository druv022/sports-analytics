"""Accelerator device resolution and GPU stack diagnostics."""

from src.accelerator.device import (
    Accelerator,
    OcrBackend,
    log_accelerator_summary,
    resolve_ocr_backend,
    resolve_ocr_use_cuda,
    resolve_torch_device,
    warn_once,
)
from src.accelerator.install import (
    GpuStackReport,
    check_gpu_stack,
    gpu_install_hints,
    warn_gpu_stack_gaps,
)

__all__ = [
    "Accelerator",
    "GpuStackReport",
    "OcrBackend",
    "check_gpu_stack",
    "gpu_install_hints",
    "log_accelerator_summary",
    "resolve_ocr_backend",
    "resolve_ocr_use_cuda",
    "resolve_torch_device",
    "warn_gpu_stack_gaps",
    "warn_once",
]
