"""Resolve torch and ONNX OCR accelerators with graceful CPU/MPS/CoreML fallback."""



from __future__ import annotations



import sys

import warnings

from typing import Literal



Accelerator = Literal["auto", "cuda", "mps", "cpu"]

OcrBackend = Literal["cuda", "coreml", "cpu"]



_WARNED: set[str] = set()



CUDA_TORCH_INSTALL_HINT = (

    "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"

)

OCR_CUDA_INSTALL_HINT = (

    "pip uninstall -y onnxruntime onnxruntime-gpu && pip install onnxruntime-gpu"

)

OCR_COREML_INSTALL_HINT = (

    "pip install onnxruntime  # standard macOS wheel includes CoreMLExecutionProvider"

)





def warn_once(key: str, message: str) -> None:

    """Emit a RuntimeWarning at most once per process for the given key."""

    if key in _WARNED:

        return

    _WARNED.add(key)

    warnings.warn(message, RuntimeWarning, stacklevel=3)





def _torch_cuda_available() -> bool:

    try:

        import torch



        return bool(torch.cuda.is_available())

    except ImportError:

        return False





def _torch_mps_available() -> bool:

    try:

        import torch



        return bool(

            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

        )

    except ImportError:

        return False





def _auto_torch_device() -> str:

    if _torch_cuda_available():

        return "cuda"

    if _torch_mps_available():

        return "mps"

    return "cpu"





def _cuda_execution_provider_available() -> bool:

    try:

        import onnxruntime as ort



        return "CUDAExecutionProvider" in ort.get_available_providers()

    except ImportError:

        return False





def _coreml_execution_provider_available() -> bool:

    try:

        import onnxruntime as ort



        return "CoreMLExecutionProvider" in ort.get_available_providers()

    except ImportError:

        return False





def resolve_torch_device(preferred: Accelerator = "auto") -> str:

    """Pick a torch device string, warning and falling back when unavailable."""

    if preferred == "auto":

        return _auto_torch_device()



    if preferred == "cuda":

        if _torch_cuda_available():

            return "cuda"

        fallback = _auto_torch_device()

        warn_once(

            "torch_cuda_unavailable",

            "CUDA is not available for torch; camera embeddings will run on "

            f"{fallback} instead. "

            f"Install CUDA torch with: {CUDA_TORCH_INSTALL_HINT}",

        )

        return fallback



    if preferred == "mps":

        if _torch_mps_available():

            return "mps"

        warn_once(

            "torch_mps_unavailable",

            f"MPS is not available for {sys.executable}; "

            "camera embeddings will run on CPU instead.",

        )

        return "cpu"



    return "cpu"





def resolve_ocr_backend(preferred: Accelerator = "auto") -> OcrBackend:

    """Pick the ONNX execution provider for RapidOCR (CUDA, CoreML, or CPU)."""

    if preferred == "cpu":

        return "cpu"



    if preferred in ("cuda", "auto") and _cuda_execution_provider_available():

        return "cuda"



    if preferred in ("auto", "mps") and _coreml_execution_provider_available():

        return "coreml"



    if preferred == "cuda":

        warn_once(

            "ocr_cuda_ep_missing",

            "onnxruntime-gpu / CUDAExecutionProvider is not available for "

            f"{sys.executable}; OCR will run on CPU. "

            f"Install with: {OCR_CUDA_INSTALL_HINT}",

        )

    elif preferred == "auto" and _torch_cuda_available():

        warn_once(

            "ocr_cuda_ep_missing_on_gpu_host",

            "torch CUDA is available but CUDAExecutionProvider is not; "

            f"OCR will run on CPU. Install with: {OCR_CUDA_INSTALL_HINT}",

        )

    elif preferred in ("auto", "mps") and sys.platform == "darwin":

        warn_once(

            "ocr_coreml_ep_missing",

            "CoreMLExecutionProvider is not available; OCR will run on CPU. "

            f"Install with: {OCR_COREML_INSTALL_HINT}",

        )

    return "cpu"





def resolve_ocr_use_cuda(preferred: Accelerator = "auto") -> bool:

    """Return True when RapidOCR should use a hardware ONNX provider (CUDA or CoreML)."""

    return resolve_ocr_backend(preferred) != "cpu"





def log_accelerator_summary(torch_device: str, ocr_backend: OcrBackend | bool) -> None:

    """Log resolved accelerator choices (informational)."""

    try:

        from broadcast_pipeline.progress import log_info

    except ImportError:

        return

    if isinstance(ocr_backend, bool):

        ocr_label = "GPU" if ocr_backend else "CPU"

    else:

        ocr_label = {"cuda": "CUDA", "coreml": "CoreML", "cpu": "CPU"}[ocr_backend]

    log_info(f"  Accelerator: torch={torch_device}, OCR={ocr_label}")
    if ocr_label == "CPU" and torch_device == "cuda":
        log_info(
            "  Hint: OCR fell back to CPU while torch uses CUDA — "
            "install onnxruntime-gpu (pip install -e '.[ocr-gpu]')"
        )

