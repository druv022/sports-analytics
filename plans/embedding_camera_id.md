# Camera ID via Global Visual Embeddings

## Table of Contents

1. [Overview](#1-overview)
2. [Why Visual Embeddings Work for Camera ID](#2-why-visual-embeddings-work-for-camera-id)
3. [Pipeline Architecture](#3-pipeline-architecture)
4. [Model Selection](#4-model-selection)
5. [Implementation](#5-implementation)
   - [Dependencies](#51-dependencies)
   - [Configuration](#52-configuration)
   - [Data Model](#53-data-model)
   - [Step 1 — Model Loading](#54-step-1--model-loading)
   - [Step 2 — Frame Preprocessing](#55-step-2--frame-preprocessing)
   - [Step 3 — Feature Extraction](#56-step-3--feature-extraction)
   - [Step 4 — Batch Processing All Scenes](#57-step-4--batch-processing-all-scenes)
   - [Step 5 — PCA Dimensionality Reduction](#58-step-5--pca-dimensionality-reduction)
   - [Step 6 — EPS Calibration via k-NN Distance](#59-step-6--eps-calibration-via-k-nn-distance)
   - [Step 7 — DBSCAN Clustering](#510-step-7--dbscan-clustering)
   - [Step 8 — Temporal Fill](#511-step-8--temporal-fill)
   - [Step 9 — Optional Hybrid Mode](#512-step-9--optional-hybrid-mode)
   - [Step 10 — Main Pipeline](#513-step-10--main-pipeline)
   - [Step 11 — Debug Utilities](#514-step-11--debug-utilities)
6. [Integration with PySceneDetect](#6-integration-with-pyscenedetect)
7. [Known Limitations](#7-known-limitations)
8. [Tuning Guide](#8-tuning-guide)

---

## 1. Overview

This module assigns camera IDs by extracting **global visual embeddings**
from each scene's representative frame using a pretrained deep vision model,
then clustering those embeddings to find camera groups.

Unlike the VP and homography approaches, embeddings require no court to be
visible. A player close-up still carries camera-specific information in its
background (crowd colour, lighting angle, ambient haze) that appears as a
consistent pattern across all close-up shots from the same camera. This
makes embeddings the only method that can produce a direct camera assignment
for every scene type without a temporal fill fallback.

---

## 2. Why Visual Embeddings Work for Camera ID

Every frame captured from a specific camera position carries an implicit
fingerprint in its global visual statistics:

- **Crowd background colour and texture** — the section of stands visible
  behind the player differs by camera angle.
- **Lighting direction** — the shadow pattern on the court and player
  clothing is determined by the angle between the light source and the
  camera.
- **Court surface proportion and hue** — how much court is visible and
  from what angle shifts the overall colour distribution.
- **Out-of-court environment** — sponsors, architecture, and sky visible
  at the frame edges vary by camera position.

A deep model pretrained on a large image corpus has already learned to
represent these properties in its feature space. Scenes from the same
camera cluster naturally in that space without any domain-specific tuning.

This is the only method in our pipeline that:
- Works on close-up player shots with no court visible
- Requires no hand-crafted geometry
- Requires no labelled training data from the specific broadcast
- Needs no knowledge of the court dimensions or surface type

---

## 3. Pipeline Architecture

```
PySceneDetect scenes
        │
        ▼
  Middle frame per scene
        │
        ▼
  Resize + normalise             ← same preprocessing for every scene
  (224×224, ImageNet stats)
        │
        ▼
  Forward pass through            ← DINO ViT-S/8 (primary)
  pretrained model                  or ResNet-50 (fallback / CPU)
        │
        ▼
  Global feature vector           ← 384D (DINO ViT-S) or 2048D (ResNet-50)
  per scene
        │
        ▼
  Stack all scene features
  into matrix (N × D)
        │
        ▼
  PCA → 64 dimensions             ← reduces DBSCAN distance sensitivity
        │
        ▼
  k-NN distance plot /            ← estimate eps from data distribution
  automatic elbow detection
        │
        ▼
  DBSCAN clustering               ← no need to specify number of cameras
        │
        ▼
  Optional: temporal fill         ← handles noise-labelled scenes
        │
        ▼
  Camera ID per scene
```

---

## 4. Model Selection

Three options in ascending order of quality and computational cost:

| Model | Feature dim | Input | GPU req. | Notes |
|---|---|---|---|---|
| ResNet-50 | 2048 | 224×224 | Optional | Widely available; good baseline |
| DINO ViT-S/16 | 384 | 224×224 | Recommended | Best self-supervised, fast |
| DINO ViT-S/8 | 384 | 224×224 | Recommended | Highest quality; 4× slower than /16 |

**Recommended default:** DINO ViT-S/16. It produces significantly richer
scene-level features than ResNet because DINO's self-supervised objective
explicitly encourages global scene understanding (the CLS token attends
to the whole image, not just local texture).

**CPU constraint:** Use ResNet-50. The global average pool output at 2048D
is strong enough for camera clustering, especially after PCA reduction to
64D.

**No internet access:** DINO requires downloading weights from torch.hub
on first use. Cache them offline with `torch.hub.load(... force_reload=False)`
or use `torchvision`'s ResNet-50 with local weights.

---

## 5. Implementation

### 5.1 Dependencies

```
torch          >= 2.0
torchvision    >= 0.15
numpy          >= 1.24
scikit-learn   >= 1.3
opencv-python  >= 4.8
Pillow         >= 9.0
```

```bash
pip install torch torchvision numpy scikit-learn opencv-python Pillow
```

For GPU inference (strongly recommended for DINO):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

### 5.2 Configuration

```python
from dataclasses import dataclass, field
from typing import Literal, Optional

Backend = Literal["dino_vits16", "dino_vits8", "resnet50"]


@dataclass
class EmbeddingConfig:
    # Model backend
    backend: Backend = "dino_vits16"

    # Compute device — "cuda" if GPU available, else "cpu"
    device: str = "cpu"

    # PCA output dimensionality
    # 64 is a good default: preserves >90% variance for typical broadcast
    # footage while making DBSCAN fast and the distance metric stable.
    pca_components: int = 64

    # DBSCAN parameters
    # eps here is in PCA-reduced space, not pixel space.
    # Calibrate using suggest_eps() on your footage (§5.9).
    dbscan_eps:         float = 5.0
    dbscan_min_samples: int   = 2

    # Temporal fill window (scenes) — used only for noise-labelled scenes
    temporal_window: int = 4

    # Batch size for GPU forward passes
    # Reduce if GPU runs out of memory (try 16 or 8).
    batch_size: int = 32
```

---

### 5.3 Data Model

```python
import numpy as np
from dataclasses import dataclass

Frame = np.ndarray   # BGR uint8, shape (H, W, 3)


@dataclass
class EmbeddingResult:
    scene_idx:  int
    embedding:  Optional[np.ndarray] = None   # raw feature vector (D,)
    reduced:    Optional[np.ndarray] = None   # PCA-reduced (pca_components,)
    camera_id:  Optional[str]        = None
```

---

### 5.4 Step 1 — Model Loading

**Why `eval()` and `no_grad()`?**
Calling `model.eval()` disables dropout and batch normalisation updates
that are only needed during training. Combined with `torch.no_grad()` at
inference time, this halves memory usage and speeds up the forward pass by
~30%.

**Why `torch.hub.load` for DINO and not a weights file?**
DINO is not distributed through `torchvision.models` as of this writing.
`torch.hub` downloads the model definition and weights from Facebook
Research's GitHub repository and caches them locally. Set
`trust_repo=True` to suppress the interactive prompt.

```python
import torch
import torchvision.models as tv_models


def load_model(config: EmbeddingConfig) -> torch.nn.Module:
    """
    Load and return the feature extraction model.

    The returned model accepts a (B, 3, 224, 224) tensor and outputs
    (B, D) feature vectors where D depends on the backend:
      dino_vits16 / dino_vits8 → 384
      resnet50                 → 2048
    """
    device = torch.device(config.device)

    if config.backend in ("dino_vits16", "dino_vits8"):
        model = torch.hub.load(
            "facebookresearch/dino:main",
            config.backend,        # e.g. "dino_vits16"
            pretrained=True,
            trust_repo=True,
        )
        # DINO's forward() returns the CLS token embedding directly
        model = model.to(device).eval()

    elif config.backend == "resnet50":
        base = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
        # Remove the final FC layer; keep global average pool
        # Output shape: (B, 2048, 1, 1) → squeeze to (B, 2048)
        model = torch.nn.Sequential(
            *list(base.children())[:-1],
            torch.nn.Flatten(),
        ).to(device).eval()

    else:
        raise ValueError(f"Unknown backend: {config.backend!r}")

    return model
```

---

### 5.5 Step 2 — Frame Preprocessing

**Why ImageNet normalisation?**
The pretrained models were trained with pixel values normalised by the
ImageNet channel mean and standard deviation. Applying the same
normalisation at inference ensures the input distribution matches what the
model saw during training. Skipping this step typically degrades feature
quality by 10–20%.

**Why centre-crop to 224×224?**
ViT models expect a fixed spatial resolution. Centre-cropping at 256→224
keeps the most informative central content and discards edges more likely
to contain broadcast overlays (score boxes, sponsor bugs, lower-third text)
that would vary by broadcaster rather than by camera position.

```python
import torchvision.transforms as T
from PIL import Image


# Standard ImageNet preprocessing shared across all backends
PREPROCESS = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


def preprocess_frame(frame: Frame) -> torch.Tensor:
    """
    Convert a BGR numpy frame to a normalised (1, 3, 224, 224) tensor.

    OpenCV loads images as BGR; PIL and PyTorch expect RGB.
    """
    import cv2
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return PREPROCESS(pil).unsqueeze(0)   # (1, 3, 224, 224)
```

---

### 5.6 Step 3 — Feature Extraction

Single-frame extraction used during debugging or when batch processing
is not available.

```python
def extract_features_single(
    frame:  Frame,
    model:  torch.nn.Module,
    config: EmbeddingConfig,
) -> np.ndarray:
    """
    Extract a (D,) feature vector from one frame.

    Use extract_features_batch for efficiency over many frames.
    """
    device = torch.device(config.device)
    tensor = preprocess_frame(frame).to(device)

    with torch.no_grad():
        feat = model(tensor)   # (1, D)

    return feat.squeeze(0).cpu().numpy()   # (D,)
```

---

### 5.7 Step 4 — Batch Processing All Scenes

**Why batch processing?**
Running each frame through the model individually wastes GPU bandwidth.
Batching groups frames into tensors that saturate the GPU's parallel
compute units, cutting total inference time by 4–8×.

**Why not one giant batch?**
GPU memory is finite. A batch of 256 frames at 224×224 in float32 requires
~600 MB just for the input tensor, before model activations. The default
`batch_size=32` stays well within 4 GB GPU memory for all three backends.

```python
from typing import List


def extract_features_batch(
    frames: List[Frame],
    model:  torch.nn.Module,
    config: EmbeddingConfig,
) -> np.ndarray:
    """
    Extract features from all frames using batched GPU inference.

    Returns
    -------
    np.ndarray of shape (N, D) where N = len(frames) and D is the
    model's feature dimension (384 for DINO ViT-S, 2048 for ResNet-50).

    Progress
    --------
    Prints a progress indicator every 10 batches. Remove or replace
    with tqdm if preferred.
    """
    device   = torch.device(config.device)
    all_feats: List[np.ndarray] = []
    n        = len(frames)
    bs       = config.batch_size

    for start in range(0, n, bs):
        batch_frames = frames[start : start + bs]
        tensors      = [preprocess_frame(f) for f in batch_frames]
        batch_tensor = torch.cat(tensors, dim=0).to(device)   # (B, 3, 224, 224)

        with torch.no_grad():
            feats = model(batch_tensor)   # (B, D)

        all_feats.append(feats.cpu().numpy())

        if (start // bs) % 10 == 0:
            print(f"  Embedding: {min(start + bs, n)}/{n} scenes")

    return np.vstack(all_feats)   # (N, D)
```

---

### 5.8 Step 5 — PCA Dimensionality Reduction

**Why PCA before DBSCAN?**
DBSCAN uses Euclidean distance, which degrades in high-dimensional spaces
due to the "curse of dimensionality": in 384D or 2048D, all pairwise
distances concentrate around the same value, making it hard to distinguish
cluster members from noise. PCA projects the features onto the principal
axes of variation — the directions that capture most inter-camera
differences — and discards directions dominated by noise.

**Why 64 components specifically?**
In typical broadcast footage, the major sources of variation are:
court-vs-crowd proportion, lighting angle, zoom level, background colour,
and a handful of camera-position-specific textures. These rarely require
more than 20–30 principal components to explain. 64 is conservative and
preserves any broadcast-specific dimensions that exceed this baseline.
You can check how much variance is explained by printing
`pca.explained_variance_ratio_.cumsum()` and choosing the component count
where cumulative variance first exceeds 0.90.

```python
from sklearn.decomposition import PCA


def reduce_dimensions(
    features:       np.ndarray,
    n_components:   int,
    random_state:   int = 0,
) -> tuple[np.ndarray, PCA]:
    """
    Fit PCA on all scene features and return reduced features + fitted PCA.

    Parameters
    ----------
    features    : (N, D) raw embedding matrix
    n_components: target dimensionality after PCA
    random_state: for reproducibility

    Returns
    -------
    reduced : (N, n_components) float32 array
    pca     : fitted sklearn PCA instance (keep for projecting new frames)

    Diagnostics
    -----------
    After calling this function, check:
        cumvar = pca.explained_variance_ratio_.cumsum()
        print(f"Variance explained by {n_components} PCs: {cumvar[-1]:.1%}")
    If below 85%, increase n_components. If above 98%, decrease it.
    """
    pca     = PCA(n_components=n_components, random_state=random_state)
    reduced = pca.fit_transform(features).astype(np.float32)

    cumvar = pca.explained_variance_ratio_.cumsum()
    print(f"PCA: {n_components} components explain {cumvar[-1]:.1%} of variance")

    return reduced, pca
```

---

### 5.9 Step 6 — EPS Calibration via k-NN Distance

**Why calibrate eps from data rather than hard-coding it?**
The embedding distances depend on:
- Which model is used (DINO distances are very different from ResNet distances)
- The specific broadcast footage (different levels of visual diversity)
- The PCA reduction (distances are in PCA space, not raw embedding space)

A hard-coded eps of 5.0 is only a starting guess. The k-NN elbow method
plots the sorted distance to each point's k-th nearest neighbour. In a
dataset with natural clusters, this curve is flat within clusters and then
rises sharply at the cluster boundary — the elbow. The eps at the elbow
separates intra-cluster distances from inter-cluster ones.

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np


def suggest_eps(
    features: np.ndarray,
    k:        int = 5,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate a good DBSCAN eps using the k-th nearest neighbour distance plot.

    Parameters
    ----------
    features : (N, d) reduced feature matrix
    k        : neighbour count (use DBSCAN min_samples value)

    Returns
    -------
    suggested_eps : float — eps at the detected elbow
    indices       : sorted indices (for plotting)
    k_dists       : sorted k-th NN distances (for plotting)

    How to use
    ----------
    eps, idxs, dists = suggest_eps(reduced_features, k=config.dbscan_min_samples)
    # Optional: plot to verify
    import matplotlib.pyplot as plt
    plt.plot(dists); plt.axhline(eps, color='r', linestyle='--')
    plt.xlabel('Scene index (sorted)'); plt.ylabel(f'{k}-NN distance')
    plt.title('DBSCAN eps calibration'); plt.show()
    # Then set config.dbscan_eps = eps (or adjust from the plot)
    """
    nbrs     = NearestNeighbors(n_neighbors=k + 1).fit(features)
    dists, _ = nbrs.kneighbors(features)
    k_dists  = np.sort(dists[:, -1])    # k-th NN distance per point

    # Elbow detection: index of maximum second derivative (sharpest turn)
    d2         = np.diff(k_dists, n=2)
    elbow_idx  = int(np.argmax(d2)) + 2   # +2 for double diff offset
    suggested  = float(k_dists[elbow_idx])

    print(f"Suggested eps: {suggested:.4f}  (elbow at index {elbow_idx}/{len(k_dists)})")
    return suggested, np.arange(len(k_dists)), k_dists
```

---

### 5.10 Step 7 — DBSCAN Clustering

**Why DBSCAN and not K-means or Agglomerative?**
- **K-means** requires specifying the number of cameras, which we do not know.
- **Agglomerative clustering** also requires either a cluster count or a
  distance threshold, and does not naturally handle noise points.
- **DBSCAN** discovers the number of clusters from density, labels isolated
  outliers as noise (label = -1), and makes no assumption about cluster
  shape. In practice, embedding clusters for broadcast cameras are
  approximately spherical in PCA space, so DBSCAN works well.

```python
from sklearn.cluster import DBSCAN
from collections import Counter


def cluster_embeddings(
    results:        list[EmbeddingResult],
    reduced_matrix: np.ndarray,
    config:         EmbeddingConfig,
) -> list[EmbeddingResult]:
    """
    Apply DBSCAN to the PCA-reduced embedding matrix and write camera_id
    to each EmbeddingResult.

    Parameters
    ----------
    results        : list of EmbeddingResult (one per scene)
    reduced_matrix : (N, pca_components) float32 from reduce_dimensions
    config         : EmbeddingConfig

    Notes
    -----
    Unlike the VP and homography pipelines, embeddings are extracted for
    every scene including close-ups. DBSCAN therefore has input for all N
    scenes and does not need to skip any. Scenes labelled -1 (noise) are
    unusual scenes (highly distinctive one-off shots) and are passed to
    temporal_fill.
    """
    labels = DBSCAN(
        eps=config.dbscan_eps,
        min_samples=config.dbscan_min_samples,
    ).fit_predict(reduced_matrix)

    for result, lbl in zip(results, labels):
        result.reduced  = reduced_matrix[result.scene_idx]
        result.camera_id = f"cam_{lbl}" if lbl >= 0 else None

    n_clusters = len(set(labels) - {-1})
    n_noise    = (labels == -1).sum()
    print(f"DBSCAN: {n_clusters} camera clusters, {n_noise} noise points")

    return results
```

---

### 5.11 Step 8 — Temporal Fill

In the embedding pipeline, temporal fill handles only the noise-labelled
scenes (DBSCAN label = -1). These are scenes that were visually distinct
enough from all clusters to be excluded — for example, a unique replay
angle or a scoreboard cut.

```python
def temporal_fill(
    results: list[EmbeddingResult],
    window:  int,
) -> list[EmbeddingResult]:
    """
    Assign camera_id to noise-labelled scenes using majority vote over
    ±window neighbouring scenes that have a confirmed camera_id.

    Identical algorithm to camera_id_assignment.py §6.10.
    """
    cam_ids = [r.camera_id for r in results]

    for i, r in enumerate(results):
        if r.camera_id is not None:
            continue
        start     = max(0, i - window)
        end       = min(len(cam_ids), i + window + 1)
        neighbors = [
            cam_ids[j] for j in range(start, end)
            if j != i and cam_ids[j] not in (None, "unknown")
        ]
        r.camera_id = (
            Counter(neighbors).most_common(1)[0][0]
            if neighbors else "unknown"
        )

    return results
```

---

### 5.12 Step 9 — Optional Hybrid Mode

The VP (or homography) pipeline is geometrically grounded and assigns
cameras based on court pose, while embeddings assign cameras based on
overall visual appearance. These two signals are complementary:

- On court-visible scenes with clean geometry, VP/H is precise.
- On close-up scenes, embeddings are the only signal.

The hybrid mode uses whichever signal is available and trusts the geometric
signal more when both agree.

```python
def hybrid_assign(
    geo_ids:   list[Optional[str]],   # from VP or homography pipeline
    emb_ids:   list[Optional[str]],   # from embedding pipeline
) -> list[str]:
    """
    Merge two camera ID lists, preferring the geometric assignment when
    available and falling back to the embedding assignment otherwise.

    Priority:
      1. Geometric (VP/homography) if not None and not 'unknown'
      2. Embedding if geometric is unavailable
      3. 'unknown' if both are unavailable

    This allows the embedding pipeline to cover close-up scenes where
    geometric methods produce None, while the more precise geometric
    assignment is kept for full-court scenes.
    """
    merged = []
    for geo, emb in zip(geo_ids, emb_ids):
        if geo is not None and geo != "unknown":
            merged.append(geo)
        elif emb is not None and emb != "unknown":
            merged.append(emb)
        else:
            merged.append("unknown")
    return merged
```

**Caveat on label alignment**
Both pipelines produce cluster labels independently (cam_0, cam_1, …) but
with no guarantee of consistent numbering. cam_0 from the VP pipeline may
correspond to cam_2 from the embedding pipeline. Before merging, align
the labels by matching the most common embedding label to the most common
geometric label in the overlap set:

```python
def align_labels(
    ref_ids:  list[str],   # geometric (reference)
    emb_ids:  list[str],   # embedding (to be realigned)
) -> list[str]:
    """
    Remap embedding camera IDs to match geometric camera IDs.
    Uses majority vote over scenes where both pipelines produced a result.
    """
    # Find overlap scenes
    mapping: dict[str, Counter] = {}
    for geo, emb in zip(ref_ids, emb_ids):
        if geo not in (None, "unknown") and emb not in (None, "unknown"):
            mapping.setdefault(emb, Counter())[geo] += 1

    # Build remapping: each embedding label → most frequent geometric label
    remap = {emb: ctr.most_common(1)[0][0] for emb, ctr in mapping.items()}

    return [remap.get(e, e) for e in emb_ids]
```

---

### 5.13 Step 10 — Main Pipeline

```python
from typing import Optional


def assign_cameras(
    middle_frames: list[Frame],
    config:        Optional[EmbeddingConfig] = None,
    model:         Optional[torch.nn.Module] = None,
) -> tuple[list[str], list[EmbeddingResult], PCA]:
    """
    Full embedding-based camera ID assignment pipeline.

    Parameters
    ----------
    middle_frames : one BGR frame per detected scene
    config        : EmbeddingConfig; defaults to EmbeddingConfig() if None
    model         : pretrained model; loaded from config.backend if None.
                    Pass an existing model to avoid reloading between calls.

    Returns
    -------
    camera_ids : list[str] — "cam_0", "cam_1", ..., "unknown" per scene
    results    : list[EmbeddingResult] — full per-scene detail
    pca        : fitted PCA instance (for projecting new frames)

    Steps
    -----
    1. Load model (if not provided)
    2. Extract features for all frames (batched)
    3. PCA → 64 dimensions
    4. Auto-suggest eps from k-NN distribution
    5. DBSCAN cluster
    6. Temporal fill for noise scenes
    """
    if config is None:
        config = EmbeddingConfig()
    if not middle_frames:
        return [], [], None

    # Step 1: Load model
    if model is None:
        print(f"Loading model: {config.backend} on {config.device}")
        model = load_model(config)

    # Step 2: Extract raw embeddings for all scenes
    print(f"Extracting embeddings for {len(middle_frames)} scenes...")
    raw_features = extract_features_batch(middle_frames, model, config)
    # raw_features: (N, D)

    # Step 3: PCA reduction
    reduced, pca = reduce_dimensions(raw_features, config.pca_components)
    # reduced: (N, pca_components)

    # Build result objects
    results = [
        EmbeddingResult(scene_idx=i, embedding=raw_features[i], reduced=reduced[i])
        for i in range(len(middle_frames))
    ]

    # Step 4: Calibrate eps if still at default
    if config.dbscan_eps == EmbeddingConfig().dbscan_eps:
        suggested_eps, _, _ = suggest_eps(reduced, k=config.dbscan_min_samples)
        print(f"Auto-calibrated eps: {suggested_eps:.4f}  "
              f"(default was {config.dbscan_eps})")
        config = EmbeddingConfig(
            **{**config.__dict__, "dbscan_eps": suggested_eps}
        )

    # Step 5: DBSCAN cluster
    results = cluster_embeddings(results, reduced, config)

    # Step 6: Temporal fill for noise-labelled scenes
    results = temporal_fill(results, config.temporal_window)

    camera_ids = [r.camera_id or "unknown" for r in results]
    return camera_ids, results, pca


def summarise(
    results: list[EmbeddingResult],
    raw_features: Optional[np.ndarray] = None,
) -> dict:
    """Diagnostic summary for calibration and debugging."""
    total      = len(results)
    by_camera  = Counter(r.camera_id for r in results)
    n_resolved = sum(1 for r in results if r.camera_id not in (None, "unknown"))

    out = {
        "total":        total,
        "by_camera":    dict(by_camera),
        "resolved_rate": n_resolved / max(total, 1),
        "unknown_rate":  by_camera.get("unknown", 0) / max(total, 1),
    }
    return out
```

---

### 5.14 Step 11 — Debug Utilities

```python
def nearest_neighbours_of_scene(
    scene_idx: int,
    results:   list[EmbeddingResult],
    k:         int = 5,
) -> list[tuple[int, float, str]]:
    """
    Return the k most visually similar scenes to scene_idx.

    Useful for verifying that scenes from the same camera are genuinely
    close in embedding space, and that scenes from different cameras are
    separated.

    Returns list of (scene_idx, distance, camera_id) tuples sorted by
    ascending distance.
    """
    query  = results[scene_idx].reduced
    others = [
        (r.scene_idx,
         float(np.linalg.norm(query - r.reduced)),
         r.camera_id or "unknown")
        for r in results
        if r.reduced is not None and r.scene_idx != scene_idx
    ]
    others.sort(key=lambda x: x[1])
    return others[:k]


def visualise_clusters_2d(
    results:  list[EmbeddingResult],
    output_path: str = "cluster_viz.png",
) -> None:
    """
    Project PCA-reduced embeddings to 2D via UMAP and save a colour-coded
    scatter plot. Requires the `umap-learn` package.

    Each point is one scene; colour encodes camera_id. Tight, well-separated
    clusters indicate good embedding quality for this footage. Overlapping
    clusters indicate cameras that are visually too similar to separate
    reliably.

    Install: pip install umap-learn
    """
    try:
        import umap
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("Install umap-learn and matplotlib to use this function.")
        return

    reduced = np.vstack([r.reduced for r in results if r.reduced is not None])
    ids     = [r.camera_id or "unknown" for r in results if r.reduced is not None]
    unique  = sorted(set(ids))
    colours = {u: cm.tab10(i / max(len(unique) - 1, 1)) for i, u in enumerate(unique)}

    proj = umap.UMAP(n_components=2, random_state=0).fit_transform(reduced)

    fig, ax = plt.subplots(figsize=(10, 7))
    for uid in unique:
        mask = [i for i, x in enumerate(ids) if x == uid]
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   c=[colours[uid]], label=uid, alpha=0.7, s=20)

    ax.legend(title="Camera")
    ax.set_title("Embedding clusters (UMAP 2D projection)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved cluster visualisation → {output_path}")
    plt.close()


def export_cluster_montage(
    results:     list[EmbeddingResult],
    frames:      list[Frame],
    camera_id:   str,
    output_path: str,
    max_frames:  int = 16,
) -> None:
    """
    Save a grid of representative frames from one camera cluster.

    Useful for sanity-checking that a cluster corresponds to a single
    coherent camera view rather than an accidental visual grouping.
    """
    import cv2, math

    cluster_frames = [
        frames[r.scene_idx]
        for r in results
        if r.camera_id == camera_id
    ][:max_frames]

    if not cluster_frames:
        print(f"No frames for {camera_id}")
        return

    n    = len(cluster_frames)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    H, W = cluster_frames[0].shape[:2]
    thumb_w, thumb_h = 320, 180
    canvas = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)

    for i, frame in enumerate(cluster_frames):
        thumb = cv2.resize(frame, (thumb_w, thumb_h))
        r, c  = divmod(i, cols)
        canvas[r*thumb_h:(r+1)*thumb_h, c*thumb_w:(c+1)*thumb_w] = thumb

    cv2.putText(canvas, camera_id, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imwrite(output_path, canvas)
    print(f"Saved montage for {camera_id} ({n} frames) → {output_path}")
```

---

## 6. Integration with PySceneDetect

```python
import cv2
import numpy as np
from scenedetect import detect, ContentDetector

video_path = "us_open_match.mp4"
scenes     = detect(video_path, ContentDetector(threshold=27.0))
cap        = cv2.VideoCapture(video_path)


def read_mid(scene, cap):
    mid = (scene[0].get_frames() + scene[1].get_frames()) // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    _, frame = cap.read()
    return frame


frames = [read_mid(s, cap) for s in scenes]

# ── Configure ─────────────────────────────────────────────────────────────────

config = EmbeddingConfig(
    backend         = "dino_vits16",
    device          = "cuda" if torch.cuda.is_available() else "cpu",
    pca_components  = 64,
    dbscan_eps      = 5.0,     # will be auto-calibrated
    dbscan_min_samples = 2,
    batch_size      = 32,
)

# ── Run pipeline ──────────────────────────────────────────────────────────────

camera_ids, results, pca = assign_cameras(frames, config)

for i, (scene, cam_id) in enumerate(zip(scenes, camera_ids)):
    print(f"Scene {i:03d}  [{scene[0].get_timecode()} → {scene[1].get_timecode()}]"
          f"  →  {cam_id}")

print("\n=== Summary ===")
for k, v in summarise(results).items():
    print(f"  {k}: {v}")

# ── Optional: UMAP visualisation ─────────────────────────────────────────────
# pip install umap-learn matplotlib first
# visualise_clusters_2d(results, "us_open_clusters.png")

# ── Optional: Montage per cluster ────────────────────────────────────────────
for cam in set(camera_ids) - {"unknown"}:
    export_cluster_montage(results, frames, cam, f"montage_{cam}.jpg")

# ── Optional: Hybrid with geometric pipeline ─────────────────────────────────
# from camera_id_assignment import assign_cameras as vp_assign, Config as VPConfig
# vp_ids  = vp_assign(frames, VPConfig(surface="hard"))
# emb_ids_aligned = align_labels(vp_ids, camera_ids)
# final_ids = hybrid_assign(vp_ids, emb_ids_aligned)

cap.release()
```

Expected output:

```
Loading model: dino_vits16 on cuda
Extracting embeddings for 310 scenes...
  Embedding: 32/310 scenes
  Embedding: 64/310 scenes
  ...
PCA: 64 components explain 93.4% of variance
Suggested eps: 4.82  (default was 5.0)
DBSCAN: 3 camera clusters, 14 noise points

Scene 000  [00:00:00.000 → 00:00:04.200]  →  cam_0
Scene 001  [00:00:04.200 → 00:00:09.800]  →  cam_0
Scene 002  [00:00:09.800 → 00:00:10.500]  →  cam_1
...

=== Summary ===
  total: 310
  by_camera: {'cam_0': 196, 'cam_1': 78, 'cam_2': 32, 'unknown': 4}
  resolved_rate: 0.987
  unknown_rate: 0.013
```

---

## 7. Known Limitations

**Visually similar cameras will merge into one cluster**
Two cameras at different positions but capturing visually similar frames
(e.g., two baseline cameras on opposite ends of the court with nearly
identical stands behind them) will appear as one cluster. This is a
fundamental limit of appearance-based methods. The VP or homography
pipeline would distinguish them geometrically — hybrid mode (§5.12)
resolves this by deferring to geometric assignments on court-visible scenes.

**Broadcast graphic overlays add noise**
Score boxes, sponsor watermarks, and player stats overlays appear at fixed
positions in the frame but may change their content between scenes. If an
overlay appears on scenes from multiple cameras, it can slightly pull their
embeddings together. Using `CenterCrop(224)` from a 256×256 resize
reduces this by focusing on the inner part of the frame, but it does not
eliminate overlays in the centre.

**Replay segments may cluster with real cameras**
Slow-motion replays are re-encoded from real broadcast cameras and look
visually similar to their source camera's live feed. A replay from cam_0
will cluster with cam_0, producing a correct-looking but technically
incorrect assignment (the replay segment is not a live broadcast moment).
Pre-filter replays using the slow-motion detection method before running
the embedding pipeline.

**First-run latency for DINO**
`torch.hub.load` downloads the DINO weights on the first call (~85 MB).
In an air-gapped environment, use the `--hub-dir` or set
`torch.hub.set_dir('/path/to/local/cache')` and pre-download weights
manually. The ResNet-50 backend uses `torchvision`'s built-in weights
and does not require internet access after the initial `pip install`.

**Auto-calibrated eps can be wrong for very short matches**
The k-NN elbow detection assumes enough scenes exist per camera to form
a meaningful density. For a match with fewer than 30 scenes total, the
elbow may not be clearly visible. In this case, set `dbscan_eps` manually
from the k-NN distance plot and use `dbscan_min_samples=2`.

---

## 8. Tuning Guide

**`backend`**
Start with `dino_vits16` if a GPU is available — it is the most reliable.
Fall back to `resnet50` for CPU-only environments. Never use `dino_vits8`
on CPU; the processing time is prohibitive.

**`pca_components`**
After running the pipeline, check:
```python
cumvar = pca.explained_variance_ratio_.cumsum()
print(f"Variance at 64 PCs: {cumvar[63]:.1%}")
```
- If below 85%: increase to 96 or 128.
- If above 98%: decrease to 32 for faster DBSCAN.

**`dbscan_eps`**
Always run `suggest_eps()` on a new broadcast before setting this manually.
The auto-calibration is good but benefits from human review of the k-NN
distance plot:
- A steep elbow at low distance → tight clusters, set eps just above the
  elbow.
- A gradual curve → cameras are visually spread; set eps at the inflection
  point but accept that some scenes may be misassigned.

**`dbscan_min_samples`**
Use 2 for short clips (< 100 scenes). Use 3–5 for full match footage
(300+ scenes) to suppress noise points from genuinely one-off shots.

**`batch_size`**
Reduce if you see a CUDA out-of-memory error. A safe value for any GPU
with ≥ 4 GB VRAM is 16. For the ResNet-50 backend on CPU, batch size
makes no speed difference.

**Evaluating cluster quality without ground truth**
Run `export_cluster_montage` for each camera cluster and visually inspect:
- Good cluster: all thumbnails show the same camera angle with consistent
  background.
- Over-split: two clusters for what looks like the same camera view —
  increase eps.
- Under-split: one cluster contains both baseline and sideline shots —
  decrease eps or switch to the hybrid mode which enforces geometric
  separation on court-visible scenes.
