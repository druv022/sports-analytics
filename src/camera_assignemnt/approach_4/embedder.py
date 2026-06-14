"""Pretrained vision model feature extraction."""

from __future__ import annotations

import os
import ssl
import urllib.request
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.camera_assignemnt.approach_4.config import EmbeddingConfig
from src.camera_assignemnt.approach_4.models import Frame, SceneSample

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

RESNET_PREPROCESS = None
DINOV2_PREPROCESS = None
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
    DINOV2_PREPROCESS = T.Compose(
        [
            T.Resize(DINOV2_INPUT_SIZE, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(DINOV2_INPUT_SIZE),
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "torch and torchvision are required for embedding extraction. "
            "Install with: pip install -e '.[embedding]'"
        )


def default_device() -> str:
    require_torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


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
    return model.to(device).eval()


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
        return DINOV2_PREPROCESS(pil).unsqueeze(0)
    return RESNET_PREPROCESS(pil).unsqueeze(0)


def extract_features_batch(
    samples: list[SceneSample],
    model,
    config: EmbeddingConfig,
) -> NDArray[np.float32]:
    """Extract feature vectors for all scene samples."""
    require_torch()
    device = next(model.parameters()).device
    batch_size = config.batch_size
    all_feats: list[NDArray[np.float32]] = []

    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        tensors = []
        for sample in batch:
            if sample.frame is None:
                raise ValueError(f"Sample has no loaded frame: {sample.frame_path}")
            tensors.append(preprocess_frame(sample.frame, config))

        batch_tensor = torch.cat(tensors, dim=0).to(device)
        with torch.no_grad():
            feats = model(batch_tensor)
            if isinstance(feats, tuple):
                feats = feats[0]
            if feats.ndim > 2:
                feats = feats.reshape(feats.shape[0], -1)
            all_feats.append(feats.cpu().numpy().astype(np.float32))

    if not all_feats:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack(all_feats)
