"""GPU stack diagnostics and install hints (no automatic pip installs)."""



from __future__ import annotations



import importlib

import importlib.metadata

import sys

from dataclasses import dataclass

from typing import TYPE_CHECKING



from src.accelerator.device import (

    Accelerator,

    CUDA_TORCH_INSTALL_HINT,

    OCR_COREML_INSTALL_HINT,

    OCR_CUDA_INSTALL_HINT,

    resolve_ocr_backend,

    resolve_torch_device,

    warn_once,

)



if TYPE_CHECKING:

    pass





@dataclass(frozen=True)

class GpuStackReport:

    torch_installed: bool

    torch_cuda: bool

    torch_cuda_device: str | None

    torch_mps: bool

    onnxruntime_installed: bool

    ort_providers: tuple[str, ...]

    rapidocr_version: str | None

    resolved_torch_device: str

    resolved_ocr_backend: str



    @property

    def resolved_ocr_cuda(self) -> bool:

        return self.resolved_ocr_backend != "cpu"



    @property

    def ocr_cuda_missing(self) -> bool:

        return (

            self.onnxruntime_installed

            and sys.platform != "darwin"

            and "CUDAExecutionProvider" not in self.ort_providers

        )



    @property

    def ocr_coreml_missing(self) -> bool:

        return (

            sys.platform == "darwin"

            and self.onnxruntime_installed

            and "CoreMLExecutionProvider" not in self.ort_providers

        )



    @property

    def torch_cuda_missing(self) -> bool:

        return self.torch_installed and not self.torch_cuda





def check_gpu_stack(accelerator: Accelerator = "auto") -> GpuStackReport:

    """Inspect the active environment for GPU acceleration readiness."""

    torch_installed = importlib.util.find_spec("torch") is not None

    torch_cuda = False

    torch_cuda_device: str | None = None

    torch_mps = False



    if torch_installed:

        import torch



        torch_cuda = bool(torch.cuda.is_available())

        if torch_cuda:

            torch_cuda_device = torch.cuda.get_device_name(0)

        torch_mps = bool(

            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

        )



    ort_providers: tuple[str, ...] = ()

    onnxruntime_installed = importlib.util.find_spec("onnxruntime") is not None

    if onnxruntime_installed:

        import onnxruntime as ort



        ort_providers = tuple(ort.get_available_providers())



    rapidocr_version: str | None = None

    if importlib.util.find_spec("rapidocr") is not None:

        rapidocr_version = importlib.metadata.version("rapidocr")



    return GpuStackReport(

        torch_installed=torch_installed,

        torch_cuda=torch_cuda,

        torch_cuda_device=torch_cuda_device,

        torch_mps=torch_mps,

        onnxruntime_installed=onnxruntime_installed,

        ort_providers=ort_providers,

        rapidocr_version=rapidocr_version,

        resolved_torch_device=resolve_torch_device(accelerator),

        resolved_ocr_backend=resolve_ocr_backend(accelerator),

    )





def gpu_install_hints(report: GpuStackReport) -> list[str]:

    """Return copy-paste install commands for detected GPU gaps."""

    hints: list[str] = []

    if report.torch_cuda_missing:

        hints.append(f"CUDA torch: {CUDA_TORCH_INSTALL_HINT}")

    if report.ocr_cuda_missing:

        hints.append('OCR GPU (NVIDIA): pip install -e ".[ocr-gpu]"')

    if report.ocr_coreml_missing:

        hints.append(f"OCR CoreML (macOS): {OCR_COREML_INSTALL_HINT}")

    if not report.torch_installed:

        hints.append('Camera embeddings: pip install -e ".[embedding]"')

    if report.rapidocr_version is None:

        hints.append('OCR: pip install -e ".[ocr]" or pip install "rapidocr>=3.8" onnxruntime')

    return hints





def warn_gpu_stack_gaps(

    report: GpuStackReport,

    accelerator: Accelerator = "auto",

) -> None:

    """Emit warnings for GPU gaps when accelerator requests or could use CUDA/CoreML."""

    if accelerator not in ("auto", "cuda", "mps"):

        return



    if accelerator == "cuda" and report.torch_cuda_missing:

        warn_once(

            "preflight_torch_cuda_missing",

            f"CUDA torch is not available for {sys.executable}; "

            f"embeddings will use {report.resolved_torch_device}. "

            f"Install with: {CUDA_TORCH_INSTALL_HINT}",

        )



    if report.ocr_cuda_missing and accelerator in ("auto", "cuda"):

        warn_once(

            "preflight_ocr_cuda_missing",

            "CUDAExecutionProvider is not available; OCR will run on CPU. "

            f"Install with: {OCR_CUDA_INSTALL_HINT}",

        )



    if report.ocr_coreml_missing and accelerator in ("auto", "mps"):

        warn_once(

            "preflight_ocr_coreml_missing",

            "CoreMLExecutionProvider is not available; OCR will run on CPU. "

            f"Install with: {OCR_COREML_INSTALL_HINT}",

        )

