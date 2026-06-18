"""Pretrained vision model feature extraction."""

from __future__ import annotations

import os
import ssl
import urllib.request
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.camera_assignemnt.embedding_cluster.config import EmbeddingConfig
from src.camera_assignemnt.embedding_cluster.dataset import resolve_sample_frame
from src.camera_assignemnt.embedding_cluster.models import Frame, SceneSample

try:
    import torch
    import torchvision.models as tv_models
    import torchvision.transforms as T
    from PIL import Image

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]

RESNET50_WEIGHTS_NAME = "resnet50-11ad3fa6.pth"
RESNET50_WEIGHTS_URL = f"https://download.pytorch.org/models/{RESNET50_WEIGHTS_NAME}"
DINOV2_INPUT_SIZE = 518
DINOV2_INPUT_SIZE_MAC = 224

RESNET_PREPROCESS = None
DINOV2_PREPROCESS = None


def _dinov2_compose(input_size: int):
    import torchvision.transforms as transforms

    return transforms.Compose(
        [
            transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


if _TORCH_AVAILABLE:
    RESNET_PREPROCESS = T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    DINOV2_PREPROCESS = _dinov2_compose(DINOV2_INPUT_SIZE)


def require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "torch and torchvision are required for embedding extraction. "
            "Install with: pip install -e '.[embedding]'"
        )


def default_device() -> str:
    require_torch()
    from src.accelerator.device import resolve_torch_device

    return resolve_torch_device("auto")


def dinov2_batch_size(device: str | torch.device) -> int:
    """Pick a DINOv2 batch size that avoids OOM on large ViT inputs."""
    device_type = str(device).split(":")[0]
    if device_type == "cuda":
        return 32
    if device_type == "mps":
        return 4
    return 2


def dinov2_input_size(device: str | torch.device) -> int:
    """Use a smaller DINOv2 crop on CPU/MPS to avoid ViT attention OOM."""
    device_type = str(device).split(":")[0]
    if device_type == "cuda":
        return DINOV2_INPUT_SIZE
    return DINOV2_INPUT_SIZE_MAC


def release_model(model) -> None:
    """Drop a feature extractor and return cached accelerator memory."""
    if not _TORCH_AVAILABLE or model is None:
        return
    import gc

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def configure_ssl_certs() -> None:
    """Use certifi CA bundle when the system store is missing (common on macOS)."""
    try:
        import certifi

        bundle = certifi.where()
        os.environ.setdefault("SSL_CERT_FILE", bundle)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", bundle)
    except ImportError:
        pass


def _ssl_context() -> ssl.SSLContext:
    configure_ssl_certs()
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download_file(url: str, dest: Path) -> Path:
    """Download a file with an explicit SSL context."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    request = urllib.request.Request(url, headers={"User-Agent": "cv-problem/0.1"})
    try:
        with urllib.request.urlopen(request, context=_ssl_context(), timeout=120) as resp:
            dest.write_bytes(resp.read())
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download model weights from {url}. "
            f"Save the file manually to {dest} or set embedding.weights_dir. "
            "On macOS with python.org Python, run: "
            "'/Applications/Python 3.13/Install Certificates.command'"
        ) from exc
    return dest


def resnet50_weight_candidates(config: EmbeddingConfig) -> list[Path]:
    """Search paths for cached ResNet-50 weights."""
    candidates: list[Path] = []
    if config.weights_dir:
        candidates.append(Path(config.weights_dir) / RESNET50_WEIGHTS_NAME)
    if _TORCH_AVAILABLE:
        candidates.append(Path(torch.hub.get_dir()) / "checkpoints" / RESNET50_WEIGHTS_NAME)
    candidates.append(Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / RESNET50_WEIGHTS_NAME)
    return candidates


def resolve_resnet50_weights(config: EmbeddingConfig) -> Path:
    """Return a local path to ResNet-50 weights, downloading if needed."""
    for path in resnet50_weight_candidates(config):
        if path.exists() and path.stat().st_size > 0:
            return path

    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required to download ResNet-50 weights.")

    cache_path = Path(torch.hub.get_dir()) / "checkpoints" / RESNET50_WEIGHTS_NAME
    return download_file(RESNET50_WEIGHTS_URL, cache_path)


def load_resnet50_extractor(config: EmbeddingConfig, device: torch.device):
    """Build ResNet-50 feature extractor from local or downloaded weights."""
    weights_path = resolve_resnet50_weights(config)
    base = tv_models.resnet50(weights=None)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    base.load_state_dict(state)
    return torch.nn.Sequential(
        *list(base.children())[:-1],
        torch.nn.Flatten(),
    ).to(device).eval()


def load_dinov2_vits14(config: EmbeddingConfig, device: torch.device):
    """Load DINOv2 ViT-S/14 from torch hub."""
    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14",
        pretrained=True,
        trust_repo=True,
    )
    model = model.to(device).eval()
    if device.type in {"mps", "cuda"}:
        model = model.half()
    return model


def load_model(config: EmbeddingConfig):
    """Load a pretrained feature extractor."""
    require_torch()
    configure_ssl_certs()
    device = torch.device(config.device or default_device())

    if config.backend == "dinov2_vits14":
        model = load_dinov2_vits14(config, device)
    elif config.backend == "resnet50":
        model = load_resnet50_extractor(config, device)
    else:
        raise ValueError(f"Unknown backend: {config.backend!r}")

    return model


def preprocess_frame(frame: Frame, config: EmbeddingConfig):
    """Convert BGR numpy frame to a normalised model input tensor."""
    require_torch()
    import cv2

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    if config.backend == "dinov2_vits14":
        device = config.device or default_device()
        return _dinov2_compose(dinov2_input_size(device))(pil).unsqueeze(0)
    return RESNET_PREPROCESS(pil).unsqueeze(0)


def extract_features_batch(
    samples: list[SceneSample],
    model,
    config: EmbeddingConfig,
) -> NDArray[np.float32]:
    """Extract feature vectors for all scene samples."""
    require_torch()
    import gc

    device = next(model.parameters()).device
    batch_size = config.batch_size
    if config.backend == "dinov2_vits14":
        batch_size = min(batch_size, dinov2_batch_size(device))
    all_feats: list[NDArray[np.float32]] = []
    use_autocast = (
        config.backend == "dinov2_vits14"
        and str(device).split(":")[0] in {"mps", "cuda"}
    )

    device_type = str(device).split(":")[0]
    is_cuda = device_type == "cuda"

    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        tensors = []
        for sample in batch:
            tensors.append(preprocess_frame(resolve_sample_frame(sample), config))

        batch_tensor = torch.cat(tensors, dim=0).to(device)
        if use_autocast:
            batch_tensor = batch_tensor.half()
        del tensors
        with torch.inference_mode():
            if use_autocast:
                feats = model(batch_tensor)
            else:
                feats = model(batch_tensor)
            if isinstance(feats, tuple):
                feats = feats[0]
            if feats.ndim > 2:
                feats = feats.reshape(feats.shape[0], -1)
            all_feats.append(feats.float().detach().cpu().numpy().astype(np.float32))
        del batch_tensor, feats
        is_last_batch = start + batch_size >= len(samples)
        if config.backend == "dinov2_vits14":
            gc.collect()
            if device_type == "mps":
                torch.mps.empty_cache()
            elif is_cuda and is_last_batch:
                torch.cuda.empty_cache()

    if not all_feats:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack(all_feats)
