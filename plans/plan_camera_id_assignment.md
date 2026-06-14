# Camera ID Assignment — Tennis Broadcast Analysis

## Table of Contents

1. [Overview](#1-overview)
2. [Valid Use Cases and Assumptions](#2-valid-use-cases-and-assumptions)
3. [Why the Naive VP Approach Breaks for Tennis](#3-why-the-naive-vp-approach-breaks-for-tennis)
4. [Pipeline Architecture](#4-pipeline-architecture)
5. [Design Decisions](#5-design-decisions)
6. [Implementation](#6-implementation)
   - [Dependencies](#61-dependencies)
   - [Configuration](#62-configuration)
   - [Data Model](#63-data-model)
   - [Step 1 — Scene Classifier](#64-step-1--scene-classifier)
   - [Step 2 — White-Line Detection](#65-step-2--white-line-detection)
   - [Step 3 — Vanishing Point Estimation](#66-step-3--vanishing-point-estimation)
   - [Step 4 — HSV Histogram Fallback](#67-step-4--hsv-histogram-fallback)
   - [Step 5 — Per-Scene Feature Extraction](#68-step-5--per-scene-feature-extraction)
   - [Step 6 — DBSCAN Clustering](#69-step-6--dbscan-clustering)
   - [Step 7 — Temporal Fill](#610-step-7--temporal-fill)
   - [Step 8 — Main Entry Point](#611-step-8--main-entry-point)
   - [Step 9 — Debug Utilities](#612-step-9--debug-utilities)
7. [Integration with PySceneDetect](#7-integration-with-pyscenedetect)
8. [Known Limitations](#8-known-limitations)
9. [Tuning Guide](#9-tuning-guide)

---

## 1. Overview

This module assigns a camera identifier to each scene detected by
PySceneDetect in a tennis broadcast video. The output is a list of strings —
one per input scene — such as `['cam_0', 'cam_0', 'cam_1', 'unknown', ...]`.

The approach is based on **vanishing point (VP) geometry**: parallel lines on
the court (sidelines, baselines, service lines) converge to a point in the
2D image whose location encodes the camera's pose. Scenes from the same
fixed camera produce the same VP; clustering VPs across scenes therefore
recovers the camera groupings.

Tennis requires a **scene-classification layer** before VP estimation because
a large fraction of broadcast scenes contain player close-ups with no court
visible. Attempting VP on these scenes produces garbage that silently
contaminates the clusters.

---

## 2. Valid Use Cases and Assumptions

The pipeline is designed for — and only reliable for — the following scenario:

| Condition | Required |
|---|---|
| Camera positions are fixed (gantry or hard-mounted) | Yes |
| Wide-angle court-dominant shots are the majority | Yes |
| Court has clearly painted white lines | Yes |
| Zoom level per camera is approximately constant | Yes |
| Replays are either absent or pre-filtered | Recommended |
| Number of cameras is unknown in advance | Handled (DBSCAN) |
| Close-up player shots appear between court shots | Handled |

**It will degrade or fail for:**

- Cameras that pan or tilt actively during play (VP shifts with every frame).
- Hawk-Eye / virtual-camera replays (no real court geometry).
- Night matches under variable floodlighting (HSV court mask drifts).
- Multiple cameras at nearly identical positions (VP clusters overlap).
- Very short matches or matches with very few scenes per camera
  (`min_samples` filtering in DBSCAN will drop minority cameras).

---

## 3. Why the Naive VP Approach Breaks for Tennis

A naive pipeline — take mid-frame, mask field color, detect Hough lines, 
compute VP, cluster — has several structural problems when applied to tennis:

### 3.1 Scene diversity is high

A typical tennis broadcast mixes:
- ~60% baseline/wide court shots (VP is reliable)
- ~15% player close-ups (no court, VP is meaningless)
- ~10% partial-court sideline shots (VP is marginal)
- ~10% replays and Hawk-Eye (no real geometry)
- ~5% crowd, coach, umpire cuts (no court at all)

Feeding all of these into a single VP pipeline without a gating step
contaminates every cluster.

### 3.2 Field-color masking is the wrong primitive for tennis

Masking for court surface color requires knowing and hand-tuning the HSV
range per tournament (clay ≠ hard ≠ grass) and is disrupted by shadow,
lighting changes, and wet surfaces. Tennis has a better primitive: **white
lines**. They exist on every surface, are high-contrast, and are invariant
across tournaments. We detect white (high-V, low-S in HSV) instead.

### 3.3 Line angle filtering has an arctan2 ambiguity bug

`cv2.HoughLinesP` returns `(x1, y1, x2, y2)` with endpoints in arbitrary
order. `np.arctan2(y2-y1, x2-x1)` therefore returns either `θ` or `θ+π`
for the same physical line depending on endpoint ordering. A naive filter
`abs(angle) < 40°` misses near-horizontal lines whose direction vector
points left (`angle ≈ π`).

The fix is to **fold** angles into `[0, π/2]` before any threshold:

```python
angles = np.arctan2(dy, dx)
a      = np.abs(angles)
folded = np.where(a > np.pi / 2, np.pi - a, a)
```

### 3.4 Normalisation is resolution-invariant, not zoom-invariant

Dividing VP pixel coordinates by frame width and height removes the effect
of encoding resolution (1080p vs 4K from the same lens), but does **not**
remove zoom. When focal length changes, the VP shifts away from the
principal point in proportion to the focal length change, and dividing by
the fixed frame size does not cancel this.

For broadcasts with significant zoom variation on a single camera, treat
scenes at very different zoom levels as potential false splits and widen
`dbscan_eps` to absorb the shift.

---

## 4. Pipeline Architecture

```
PySceneDetect scenes
        │
        ▼
  [middle frame per scene]
        │
        ▼
  ┌─────────────────────┐
  │   Scene classifier  │  ← measures court visibility ratio
  └─────────────────────┘
     │           │           │
  >35%       10–35%        <10%
(full court)(partial)   (close-up)
     │           │           │
     ▼           ▼           ▼
White-line    HSV histo-  (deferred —
  VP (RANSAC) gram         temporal fill)
     │           │
     └─────┬─────┘
           │
           ▼
    DBSCAN clustering
    (VP space, 2D)
           │
           ▼
    Temporal fill
    (close-ups inherit
     neighbour camera)
           │
           ▼
  camera_id per scene
```

---

## 5. Design Decisions

### Why scene classification first?

A VP estimated from a player close-up is not a camera fingerprint — it is
noise. Running DBSCAN on a mix of real VPs and close-up noise makes clusters
unstable. The classifier gates the VP path to scenes where the geometry is
trustworthy.

### Why white lines instead of court color?

White lines are invariant across clay, hard, and grass courts. They appear
in every full-court scene and are easily isolated with a single HSV range
(high V, low S). Court color requires per-tournament HSV tuning and is
disrupted by shadow and lighting changes. White lines are the more robust
and general primitive.

### Why RANSAC for VP estimation?

Hough lines on a sports broadcast frame always include outliers: advertising
board edges, crowd boundaries, and net posts produce edges that survive the
white-line mask. RANSAC selects the largest inlier set (the actual court
lines) while discarding the noise without needing to pre-label which lines
are real.

### Why DBSCAN over K-means?

The number of cameras is not known in advance and varies per broadcast.
DBSCAN discovers the number of clusters from the data, labels clear outliers
as noise (label `-1`, mapped to `'unknown'`) rather than forcing them into
the nearest cluster, and handles irregularly shaped clusters that can arise
when cameras pan slightly.

### Why two-pass (cluster then temporal fill)?

Close-up scenes have no geometric signal. Trying to include them in
geometric clustering is circular. The two-pass structure keeps the geometry
clean in pass 1 and then uses the broadcast rhythm (director cuts from
camera A to a reaction shot and back to camera A) to recover the camera
assignment in pass 2.

---

## 6. Implementation

### 6.1 Dependencies

```
opencv-python >= 4.8
numpy         >= 1.24
scikit-learn  >= 1.3
```

Install:

```bash
pip install opencv-python numpy scikit-learn
```

---

### 6.2 Configuration

```python
from dataclasses import dataclass
from typing import Literal

Surface = Literal["clay", "hard", "grass"]

# HSV ranges for the court surface (not the white lines)
# Used only by the scene classifier to measure court visibility ratio
COURT_HSV_RANGES: dict[str, tuple] = {
    "hard":  ((90,  60,  60), (130, 255, 255)),  # blue hard court (US Open / AO)
    "clay":  ((5,   80,  80), (20,  255, 255)),  # clay red-orange (Roland Garros)
    "grass": ((35,  40,  40), (85,  255, 255)),  # grass green (Wimbledon)
}


@dataclass
class Config:
    # Tournament surface — determines the court visibility mask
    surface: Surface = "hard"

    # Scene classification thresholds (fraction of frame pixels)
    full_court_ratio:    float = 0.35   # > this → full_court
    partial_court_ratio: float = 0.10   # > this → partial_court, else closeup

    # Hough line detection parameters
    hough_threshold:  int = 50   # accumulator votes to accept a line
    hough_min_length: int = 40   # minimum segment length in pixels
    hough_max_gap:    int = 8    # maximum collinear gap to bridge

    # Minimum accepted lines before attempting VP estimation
    min_lines_for_vp: int = 5

    # RANSAC vanishing point
    ransac_iterations: int   = 200
    ransac_inlier_px:  float = 5.0  # pixel distance from line to VP to count as inlier

    # DBSCAN clustering (in normalised VP coordinates [0,1] × [0,1])
    dbscan_eps:         float = 0.06
    dbscan_min_samples: int   = 2

    # Temporal fill — neighbourhood radius in number of scenes
    temporal_window: int = 4

    # Histogram bins per channel (H and S), total feature = 2 × bins
    histogram_bins: int = 16
```

---

### 6.3 Data Model

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

SceneType = Literal["full_court", "partial_court", "closeup"]
Frame     = np.ndarray  # BGR uint8, shape (H, W, 3)


@dataclass
class SceneResult:
    """All information extracted for a single scene."""
    scene_idx:   int
    scene_type:  SceneType
    court_ratio: float                         # fraction of frame that is court

    # Set when VP estimation succeeds (full/partial court scenes)
    vp: Optional[tuple[float, float]] = None   # (x_norm, y_norm), both in [0, 1]

    # Set when VP fails but court is partially visible
    histogram: Optional[np.ndarray] = None     # normalised 32-d vector

    # Filled by clustering and temporal passes
    camera_id: Optional[str] = None            # e.g. "cam_0", "cam_1", "unknown"
```

---

### 6.4 Step 1 — Scene Classifier

Measures what fraction of the frame is court surface using a per-surface
HSV mask. This fraction determines which processing branch the scene enters.

The court mask returned here is reused in downstream steps (white-line
detection, histogram), avoiding a second colour conversion.

```python
import cv2
import numpy as np


def classify_scene(
    frame: Frame,
    config: Config,
) -> tuple[SceneType, float, np.ndarray]:
    """
    Classify a scene by court visibility and return the court mask.

    Returns
    -------
    scene_type  : "full_court" | "partial_court" | "closeup"
    court_ratio : float — fraction of frame pixels classified as court surface
    court_mask  : uint8 binary mask (0 / 255), shape (H, W)

    Thresholds
    ----------
    >35% → full_court    VP estimation is reliable
    >10% → partial_court VP is marginal; histogram used as fallback
    ≤10% → closeup       No geometric signal; temporal fill used
    """
    H, W = frame.shape[:2]
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lo, hi     = COURT_HSV_RANGES[config.surface]
    court_mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
    court_ratio = float(court_mask.sum()) / 255.0 / (H * W)

    if court_ratio > config.full_court_ratio:
        scene_type: SceneType = "full_court"
    elif court_ratio > config.partial_court_ratio:
        scene_type = "partial_court"
    else:
        scene_type = "closeup"

    return scene_type, court_ratio, court_mask
```

---

### 6.5 Step 2 — White-Line Detection

Detects white court line segments within the visible court region.

White lines (high brightness, low saturation in HSV) are the correct
primitive for tennis because they exist on every surface and require no
per-tournament colour tuning. We restrict detection to pixels adjacent
to the court surface mask to exclude advertising boards, player clothing,
and net edges.

```python
def detect_court_lines(
    frame: Frame,
    court_mask: np.ndarray,
    config: Config,
) -> Optional[np.ndarray]:
    """
    Detect white court line segments inside the visible court region.

    Returns
    -------
    np.ndarray of shape (N, 4) with columns (x1, y1, x2, y2), or None
    if no lines pass the Hough threshold.

    Why dilate the court mask?
    --------------------------
    The white line pixels sit on the boundary of the court surface, not
    strictly inside it. Dilating by 20 × 20 ensures line pixels that touch
    the court edge are included while still excluding advertising boards
    and crowd areas that are far from the court.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # White: high brightness (V > 185), low saturation (S < 40)
    white_mask = cv2.inRange(hsv, (0, 0, 185), (180, 40, 255))

    # Restrict to pixels near the court surface
    kernel        = np.ones((20, 20), np.uint8)
    court_dilated = cv2.dilate(court_mask, kernel)
    lines_region  = cv2.bitwise_and(white_mask, court_dilated)

    edges = cv2.Canny(lines_region, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=config.hough_threshold,
        minLineLength=config.hough_min_length,
        maxLineGap=config.hough_max_gap,
    )

    return None if lines is None else lines[:, 0, :]  # shape (N, 4)
```

---

### 6.6 Step 3 — Vanishing Point Estimation

Estimates the primary vanishing point from the detected line segments using
RANSAC. Works in homogeneous coordinates so VP can be outside the image frame.

**Critical fix — angle ambiguity:**
`cv2.HoughLinesP` returns endpoints in arbitrary order, so
`arctan2(y2-y1, x2-x1)` can return either `θ` or `θ+π` for the same
physical line. The `_fold_angles` helper maps all angles to `[0, π/2]`
before any direction-based filtering.

```python
def _line_to_hom(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """Two endpoints → homogeneous line [a, b, c] where ax + by + c = 0."""
    return np.cross([x1, y1, 1.0], [x2, y2, 1.0])


def _fold_angles(pts: np.ndarray) -> np.ndarray:
    """
    Fold line angles from arctan2 range [-π, π] into [0, π/2].

    arctan2 assigns opposite signs to the same physical line depending on
    which endpoint is labelled (x1,y1). Folding collapses that ambiguity
    before any angle-threshold filtering.
    """
    angles = np.arctan2(pts[:, 3] - pts[:, 1], pts[:, 2] - pts[:, 0])
    a      = np.abs(angles)
    return np.where(a > np.pi / 2, np.pi - a, a)


def estimate_vanishing_point(
    line_pts: np.ndarray,
    config: Config,
    frame_shape: tuple[int, int],
) -> Optional[tuple[float, float]]:
    """
    Estimate the primary VP via RANSAC over detected line segments.

    Parameters
    ----------
    line_pts    : array (N, 4) — one row per segment (x1, y1, x2, y2)
    config      : Config
    frame_shape : (H, W) for normalisation

    Returns
    -------
    (vp_x_norm, vp_y_norm) — VP coordinates normalised by frame (W, H)
    None if:
      — fewer than min_lines_for_vp segments supplied
      — every line pair is parallel (no finite intersection found)
      — the best RANSAC hypothesis has fewer than 3 inliers (unreliable)

    Normalisation note
    ------------------
    Dividing by (W, H) makes coordinates resolution-independent but NOT
    zoom-invariant. A camera that zooms in will shift the VP in normalised
    space. For moderate zoom variation, widen dbscan_eps to 0.08–0.12.
    For extreme zoom variation, treat scenes from the same camera at
    different zoom levels as separate clusters and merge them manually.

    Inlier distance note
    --------------------
    The inlier test measures the distance from the candidate VP to the
    *infinite line* through each segment, not to the segment itself. For
    segments shorter than ~60 px this can accept false inliers whose
    extension passes near the VP. The 40 px minLineLength filter
    partially mitigates this; RANSAC voting further suppresses the effect.
    """
    if len(line_pts) < config.min_lines_for_vp:
        return None

    H, W      = frame_shape
    hom_lines = [_line_to_hom(*pt) for pt in line_pts]
    n         = len(hom_lines)
    rng       = np.random.default_rng(seed=0)

    best_vp, best_count = None, 0

    for _ in range(config.ransac_iterations):
        i, j = rng.choice(n, size=2, replace=False)
        pt   = np.cross(hom_lines[i], hom_lines[j])

        # w ≈ 0 means the two lines are parallel → intersection at infinity
        # (occurs when camera looks straight along the court axis)
        if abs(pt[2]) < 1e-7:
            continue

        vp    = np.array([pt[0] / pt[2], pt[1] / pt[2]])
        count = sum(
            abs(l[0] * vp[0] + l[1] * vp[1] + l[2])
            / (np.hypot(l[0], l[1]) + 1e-8)
            < config.ransac_inlier_px
            for l in hom_lines
        )

        if count > best_count:
            best_vp, best_count = vp, count

    if best_vp is None or best_count < 3:
        return None

    return float(best_vp[0] / W), float(best_vp[1] / H)
```

---

### 6.7 Step 4 — HSV Histogram Fallback

When the court is partially visible but too few line segments are detected
for a reliable VP, a compact HSV histogram of the court region acts as a
camera signature. Different camera positions see different balances of
shadow, surface shading, and court-to-background ratio that show up as
distinct hue/saturation distributions.

This histogram is used for partial-court scenes where VP estimation returns
`None`. It is a weaker signal than VP — histograms for two cameras at
similar angles may overlap — but it is better than discarding the scene.

```python
def court_hsv_histogram(
    frame: Frame,
    court_mask: np.ndarray,
    bins: int = 16,
) -> np.ndarray:
    """
    Compact normalised HSV histogram of the visible court pixels.

    Returns a (2 × bins,) float32 vector (H channel then S channel).
    The vector is L1-normalised so that lighting intensity does not dominate.
    """
    hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], court_mask, [bins], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], court_mask, [bins], [0, 256]).flatten()
    feat   = np.concatenate([hist_h, hist_s]).astype(np.float32)
    return feat / (feat.sum() + 1e-8)
```

---

### 6.8 Step 5 — Per-Scene Feature Extraction

Routes each scene through the appropriate branch based on its type and
extracts the best available feature.

```python
def extract_feature(
    frame: Frame,
    scene_idx: int,
    config: Config,
) -> SceneResult:
    """
    Classify one scene and extract the strongest available camera feature.

    Branch logic
    ------------
    full_court  → attempt VP; histogram if VP fails
    partial     → attempt VP; histogram if VP fails
    closeup     → no feature; deferred to temporal fill

    Both full and partial scenes attempt VP first because partial scenes
    sometimes have enough long white line segments. The histogram is
    reserved as a fallback, not a primary.
    """
    scene_type, court_ratio, court_mask = classify_scene(frame, config)
    result = SceneResult(
        scene_idx=scene_idx,
        scene_type=scene_type,
        court_ratio=court_ratio,
    )

    if scene_type == "closeup":
        return result  # no geometric signal available

    # Attempt VP from white lines
    line_pts = detect_court_lines(frame, court_mask, config)
    if line_pts is not None and len(line_pts) >= config.min_lines_for_vp:
        result.vp = estimate_vanishing_point(line_pts, config, frame.shape[:2])

    # If VP failed, store histogram as fallback
    if result.vp is None:
        result.histogram = court_hsv_histogram(frame, court_mask, config.histogram_bins)

    return result
```

---

### 6.9 Step 6 — DBSCAN Clustering

Clusters scenes with valid VP estimates into camera groups. Operates only
on the 2D normalised VP space `(vp_x_norm, vp_y_norm)`.

Scenes without a VP (close-ups, failed VP estimation) are left with
`camera_id = None` and resolved in the temporal fill pass.

```python
from sklearn.cluster import DBSCAN


def cluster_by_vp(
    results: list[SceneResult],
    config: Config,
) -> list[SceneResult]:
    """
    Assign camera IDs to scenes that have a valid VP using DBSCAN.

    Why DBSCAN over K-means
    -----------------------
    - No need to specify the number of cameras in advance.
    - Scenes whose VP is far from any cluster are labelled noise (-1)
      and mapped to 'unknown' rather than being forced into a wrong cluster.
    - Handles slight VP spread from minor camera movement within a game.

    Cluster labels
    --------------
    DBSCAN returns integer labels: 0, 1, 2, ... are valid clusters; -1 is
    noise. We map these to "cam_0", "cam_1", ... and "unknown" respectively.
    """
    geo = [(r.scene_idx, r.vp) for r in results if r.vp is not None]
    if not geo:
        return results

    idxs, vps  = zip(*geo)
    vp_matrix  = np.array(vps, dtype=np.float32)
    labels     = DBSCAN(
        eps=config.dbscan_eps,
        min_samples=config.dbscan_min_samples,
    ).fit_predict(vp_matrix)

    idx_to_label = dict(zip(idxs, labels))
    for r in results:
        lbl = idx_to_label.get(r.scene_idx)
        if lbl is not None:
            r.camera_id = f"cam_{lbl}" if lbl >= 0 else "unknown"

    return results
```

---

### 6.10 Step 7 — Temporal Fill

Assigns camera IDs to all remaining scenes (close-ups, noise outliers,
VP estimation failures) using the most common camera label found in a
window of ±`temporal_window` adjacent scenes.

This works because the broadcast rhythm is: *baseline camera → player
reaction → baseline camera*. The neighbourhood vote recovers the
surrounding camera consistently.

```python
from collections import Counter


def temporal_fill(
    results: list[SceneResult],
    window: int,
) -> list[SceneResult]:
    """
    Fill camera_id for unresolved scenes using temporal neighbourhood voting.

    Algorithm
    ---------
    For each scene with camera_id = None (or "unknown"):
      1. Collect camera IDs from the ±window neighbours that have a
         non-None, non-"unknown" label.
      2. Assign the most common label (majority vote).
      3. If no neighbours have a resolved label, assign "unknown".

    Chaining risk
    -------------
    A run of consecutive close-ups longer than `window` scenes will have
    no resolved neighbours on one side. These scenes inherit from the
    resolved side only — which is usually correct but can fail at
    broadcast transitions (e.g., post-match ceremonies).

    If 'unknown' appears in the final output it signals:
      - An isolated close-up run with no surrounding court-view scenes.
      - A set of scenes that DBSCAN labelled as noise (VP too variable).
      These can be reviewed manually or excluded from logo impression counts.
    """
    cam_ids = [r.camera_id for r in results]

    for i, r in enumerate(results):
        if r.camera_id is not None:
            continue

        start     = max(0, i - window)
        end       = min(len(cam_ids), i + window + 1)
        neighbors = [
            cam_ids[j]
            for j in range(start, end)
            if j != i and cam_ids[j] not in (None, "unknown")
        ]

        r.camera_id = (
            Counter(neighbors).most_common(1)[0][0]
            if neighbors
            else "unknown"
        )

    return results
```

---

### 6.11 Step 8 — Main Entry Point

```python
def assign_cameras(
    middle_frames: list[Frame],
    config: Optional[Config] = None,
) -> list[str]:
    """
    Full camera ID assignment pipeline for a tennis broadcast.

    Parameters
    ----------
    middle_frames : list of BGR frames, one per detected scene.
                    Use the midpoint frame from each PySceneDetect scene.
    config        : Config — defaults to Config() if not provided.
                    Set `surface` to match the tournament court type.

    Returns
    -------
    List[str] — one camera ID per input frame.
    Values are "cam_0", "cam_1", ... for identified cameras, or "unknown"
    when no reliable assignment could be made.

    Example
    -------
    >>> cfg    = Config(surface="clay")
    >>> ids    = assign_cameras(frames, cfg)
    >>> for i, (scene, cam) in enumerate(zip(scenes, ids)):
    ...     print(f"Scene {i}: {cam}")
    Scene 0: cam_0
    Scene 1: cam_0
    Scene 2: cam_1
    Scene 3: unknown
    """
    if config is None:
        config = Config()

    if not middle_frames:
        return []

    # Pass 1: extract features for each scene
    results: list[SceneResult] = [
        extract_feature(frame, i, config)
        for i, frame in enumerate(middle_frames)
    ]

    # Pass 2: cluster scenes with valid VP estimates
    results = cluster_by_vp(results, config)

    # Pass 3: temporal fill for unresolved scenes
    results = temporal_fill(results, config.temporal_window)

    return [r.camera_id or "unknown" for r in results]


def summarise(results: list[SceneResult]) -> dict:
    """
    Return a summary dict useful for calibration and debugging.

    Keys
    ----
    total         : total scene count
    by_type       : Counter of SceneType strings
    by_camera     : Counter of camera_id strings (includes 'unknown')
    vp_success_rate : fraction of non-closeup scenes that produced a VP
    unknown_rate  : fraction of scenes assigned 'unknown'
    """
    total        = len(results)
    by_type      = Counter(r.scene_type for r in results)
    by_camera    = Counter(r.camera_id  for r in results)
    non_closeup  = [r for r in results if r.scene_type != "closeup"]
    vp_success   = sum(1 for r in non_closeup if r.vp is not None)

    return {
        "total":            total,
        "by_type":          dict(by_type),
        "by_camera":        dict(by_camera),
        "vp_success_rate":  vp_success / max(len(non_closeup), 1),
        "unknown_rate":     by_camera.get("unknown", 0) / max(total, 1),
    }
```

---

### 6.12 Step 9 — Debug Utilities

Use these to visually verify that line detection and VP estimation are
working correctly on your specific broadcast before running the full pipeline.

```python
def draw_debug_frame(
    frame: Frame,
    result: SceneResult,
    line_pts: Optional[np.ndarray] = None,
) -> Frame:
    """
    Overlay scene type, court ratio, detected lines, and VP on a frame.

    Returns a copy of the frame with annotations. Does not modify the
    original.

    Usage
    -----
    >>> result   = extract_feature(frame, 0, config)
    >>> lines    = detect_court_lines(frame, court_mask, config)
    >>> annotated = draw_debug_frame(frame, result, lines)
    >>> cv2.imwrite("debug.jpg", annotated)
    """
    out    = frame.copy()
    H, W   = out.shape[:2]
    YELLOW = (0, 255, 255)
    GREEN  = (0, 255, 0)
    RED    = (0, 0, 255)

    # Draw detected line segments
    if line_pts is not None:
        for x1, y1, x2, y2 in line_pts:
            cv2.line(out, (x1, y1), (x2, y2), GREEN, 1)

    # Draw vanishing point
    if result.vp is not None:
        vx = int(result.vp[0] * W)
        vy = int(result.vp[1] * H)
        cv2.circle(out, (vx, vy), 8, RED, -1)
        cv2.circle(out, (vx, vy), 12, RED, 2)

    # Overlay text
    info = (
        f"type={result.scene_type}  "
        f"ratio={result.court_ratio:.2f}  "
        f"vp={result.vp}  "
        f"cam={result.camera_id}"
    )
    cv2.putText(out, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 1)

    return out


def export_debug_video(
    cap: cv2.VideoCapture,
    scenes: list,
    results: list[SceneResult],
    output_path: str,
    config: Optional[Config] = None,
) -> None:
    """
    Write a debug video where each scene's middle frame is annotated.

    Parameters
    ----------
    cap         : open cv2.VideoCapture for the original video
    scenes      : list of (start_timecode, end_timecode) from PySceneDetect
    results     : output of the full assign_cameras pipeline (SceneResult list)
    output_path : path for the output MP4 file
    config      : Config used during assignment

    The output is a slideshow: one annotated frame per scene, each held for
    1 second. Useful for quickly reviewing all assignments without watching
    the full match.
    """
    if config is None:
        config = Config()

    fps    = 1
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (W, H)
    )

    for scene, result in zip(scenes, results):
        start_frame = scene[0].get_frames()
        end_frame   = scene[1].get_frames()
        mid_frame   = (start_frame + end_frame) // 2

        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ok, frame = cap.read()
        if not ok:
            continue

        _, _, court_mask = classify_scene(frame, config)
        line_pts         = detect_court_lines(frame, court_mask, config)
        annotated        = draw_debug_frame(frame, result, line_pts)
        writer.write(annotated)

    writer.release()
```

---

## 7. Integration with PySceneDetect

```python
import cv2
import numpy as np
from scenedetect import detect, ContentDetector, open_video

# ── Detect scenes ─────────────────────────────────────────────────────────────

video_path = "roland_garros_match.mp4"
scenes     = detect(video_path, ContentDetector(threshold=27.0))

# ── Extract middle frame from each scene ─────────────────────────────────────

cap = cv2.VideoCapture(video_path)


def read_middle_frame(scene: tuple, cap: cv2.VideoCapture) -> np.ndarray:
    start_f = scene[0].get_frames()
    end_f   = scene[1].get_frames()
    mid_f   = (start_f + end_f) // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_f)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {mid_f} from {video_path}")
    return frame


middle_frames = [read_middle_frame(s, cap) for s in scenes]

# ── Run camera assignment ─────────────────────────────────────────────────────

config   = Config(surface="clay")           # Roland Garros → clay
cam_ids  = assign_cameras(middle_frames, config)

# ── Print results ─────────────────────────────────────────────────────────────

for i, (scene, cam_id) in enumerate(zip(scenes, cam_ids)):
    start = scene[0].get_timecode()
    end   = scene[1].get_timecode()
    print(f"Scene {i:03d}  [{start} → {end}]  →  {cam_id}")

# ── Summary ───────────────────────────────────────────────────────────────────

results = [extract_feature(f, i, config) for i, f in enumerate(middle_frames)]
results = cluster_by_vp(results, config)
results = temporal_fill(results, config.temporal_window)
summary = summarise(results)

print("\n=== Assignment summary ===")
for k, v in summary.items():
    print(f"  {k}: {v}")

# ── Optional: export debug video ─────────────────────────────────────────────

export_debug_video(cap, scenes, results, "debug_cam_assignment.mp4", config)
cap.release()
```

Expected output format:

```
Scene 000  [00:00:00.000 → 00:00:04.200]  →  cam_0
Scene 001  [00:00:04.200 → 00:00:09.850]  →  cam_0
Scene 002  [00:00:09.850 → 00:00:10.500]  →  unknown
Scene 003  [00:00:10.500 → 00:00:14.100]  →  cam_1
...

=== Assignment summary ===
  total: 312
  by_type: {'full_court': 184, 'partial_court': 62, 'closeup': 66}
  by_camera: {'cam_0': 198, 'cam_1': 74, 'cam_2': 28, 'unknown': 12}
  vp_success_rate: 0.74
  unknown_rate: 0.038
```

---

## 8. Known Limitations

### VP at infinity for end-on cameras

When a camera looks straight along the court axis (e.g., a camera positioned
directly behind the baseline), the sidelines appear nearly parallel in the
image. Their intersection is at or near infinity: `pt[2] ≈ 0` in homogeneous
coordinates. The RANSAC loop skips these intersections, reducing the inlier
pool. If the camera has a strong end-on alignment, VP estimation may return
`None` even for full-court scenes. That scene falls through to temporal fill.

Mitigation: add a second VP branch for near-vertical lines (those running
toward the camera), which would converge to a different VP well within the
frame for an end-on camera.

### Zoom changes

`vp_x_norm = vp_x / W` removes resolution differences but not zoom. When
focal length doubles, `vp_x` doubles relative to the principal point, and
normalising by the fixed `W` does not cancel this. Scenes from the same
camera at very different zoom levels may appear as separate DBSCAN clusters.

Mitigation: widen `dbscan_eps` (0.08–0.12) to absorb moderate zoom drift,
or detect zoom-change scenes separately using optical flow magnitude.

### Camera pan within a scene

If the camera pans during the scene, the mid-frame VP is not representative
of the scene as a whole. Frequent panning will spread the VP cloud for that
camera beyond `dbscan_eps`, causing DBSCAN to either split it into multiple
clusters or label all its scenes as noise.

Mitigation: use the first and last quarters of the scene as additional
sample frames and take the median of the three VPs before clustering.

### Consecutive close-ups longer than `temporal_window`

A run of `> 2 × temporal_window` consecutive close-up scenes will have no
resolved neighbour on at least one side. Those scenes will be assigned
`'unknown'`. This commonly occurs during changeovers (player sits for 90
seconds) and medical time-outs.

Mitigation: increase `temporal_window` for these segments, or detect
changeover scenes by their duration and handle them as a special class.

### Hawk-Eye replay scenes

Hawk-Eye (virtual camera) replays do not correspond to any real broadcast
camera. They may or may not produce a VP, but if they do it will not match
any real camera cluster. DBSCAN will label them as noise (`'unknown'`).
This is the correct behaviour. Filter them out before logo impression
analysis — no court logos are visible in Hawk-Eye replays.

### HSV range drift

The `COURT_HSV_RANGES` constants are approximate. Evening matches,
overcast conditions, stadium roof closures, or different camera exposures
can shift the apparent court hue by 5–10 HSV units. If the court visibility
ratio is systematically low for your footage, tune the ranges for that
specific broadcast.

---

## 9. Tuning Guide

### Choosing `surface`

| Tournament | Surface value |
|---|---|
| Australian Open | `"hard"` |
| Roland Garros | `"clay"` |
| Wimbledon | `"grass"` |
| US Open | `"hard"` |
| Indoor hardcourt | `"hard"` (may need HSV adjustment) |

If the `vp_success_rate` from `summarise()` is below 0.5 for full-court
scenes, the court mask is likely too restrictive. Print
`cv2.imshow("court_mask", court_mask)` on a representative frame and check
coverage; then widen the HSV range.

### Choosing `dbscan_eps`

| Situation | Suggested eps |
|---|---|
| Fully static cameras, no zoom | 0.04 – 0.06 |
| Cameras with minor pan | 0.07 – 0.10 |
| Significant zoom variation | 0.10 – 0.15 |

Start at 0.06. If `summarise()` shows more camera IDs than expected, reduce
eps. If fewer than expected (cameras merged), increase it.

### Choosing `dbscan_min_samples`

Use `min_samples=2` for short matches or tournaments where individual cameras
appear infrequently. Use `min_samples=3` or higher for long matches to
reduce false positives from stray close-up VP estimates.

### Choosing `temporal_window`

Average scene length in your footage divided by 2 is a good starting point.
For a match with average scene length 6 seconds and scenes around 4–5 seconds,
`window=4` is appropriate. Increase to 6–8 if changeover sequences produce
many `'unknown'` scenes.

### Choosing `hough_min_length`

Scale with the expected pixel length of court line segments:

| Zoom level | Suggested min_length |
|---|---|
| Wide baseline (full court visible) | 50–80 |
| Medium (half court) | 35–50 |
| Tight (near-court partial) | 25–35 |

Increasing `min_length` reduces false lines from player shadows and net posts
but also reduces line count on partially visible courts.

### Checking `vp_success_rate`

A healthy value is 0.65–0.85 for a match with typical camera coverage.
Below 0.5 indicates:
- HSV ranges are too restrictive (court mask covers too little)
- `min_lines_for_vp` is too high for the available segments
- `hough_min_length` is too long for the zoom level

Above 0.90 may indicate false positives from advertising boards or player
clothing contaminating the white-line mask. Check `detect_court_lines`
output visually with `draw_debug_frame`.
