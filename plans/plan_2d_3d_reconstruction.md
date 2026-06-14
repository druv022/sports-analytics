# 2D → 3D Court Reconstruction

Implementation target: `src/2D_3D/`

## Table of Contents

1. [Overview](#1-overview)
2. [Valid Use Cases and Assumptions](#2-valid-use-cases-and-assumptions)
3. [Why This Fits the Existing Codebase](#3-why-this-fits-the-existing-codebase)
4. [Pipeline Architecture](#4-pipeline-architecture)
5. [Design Decisions](#5-design-decisions)
6. [Implementation Plan](#6-implementation-plan)
   - [Module Layout](#61-module-layout)
   - [Step 1 — World Court Model](#62-step-1--world-court-model)
   - [Step 2 — 2D Keypoint Extraction](#63-step-2--2d-keypoint-extraction)
   - [Step 3 — 2D–3D Correspondences](#64-step-3--2d3d-correspondences)
   - [Step 4 — Camera Intrinsics and Pose (PnP)](#65-step-4--camera-intrinsics-and-pose-pnp)
   - [Step 5 — 3D Geometry Assembly](#66-step-5--3d-geometry-assembly)
   - [Step 6 — Export and Visualization](#67-step-6--export-and-visualization)
   - [Step 7 — Evaluation](#68-step-7--evaluation)
   - [Step 8 — CLI Entry Point](#69-step-8--cli-entry-point)
7. [Integration with Existing Pipelines](#7-integration-with-existing-pipelines)
8. [Known Limitations](#8-known-limitations)
9. [Milestones and Acceptance Criteria](#9-milestones-and-acceptance-criteria)
10. [Future Extensions](#10-future-extensions)

---

## 1. Overview

Build a **metric 3D model of a tennis court** from a single 2D broadcast frame.

The output is not a generic neural depth map — it is a **geometrically grounded court model** whose line layout matches ITF regulations and whose camera pose is recoverable from detected court keypoints. A single frame is sufficient because the court is a **known planar structure**; depth comes from calibration, not from learned monocular depth.

Primary deliverables per input frame:

| Output | Description |
|---|---|
| `CourtModel3D` | Named 3D line segments + court surface mesh in world metres |
| `CameraPose` | Rotation, translation, intrinsics, reprojection error |
| `scene_overlay.jpg` | 2D verification: projected 3D lines drawn on the frame |
| `court_model.glb` / `.obj` | Exportable mesh for Blender, Three.js, etc. |

---

## 2. Valid Use Cases and Assumptions

| Condition | Required |
|---|---|
| Full or near-full court visible in frame | Yes (same gate as homography eval) |
| Court white lines detectable | Yes |
| Reference diagram available (`data/Court_dimension.png`) | Yes |
| Camera is static within a scene (no pan/tilt during shot) | Yes |
| Metric accuracy needed for court lines | Yes |
| Metric accuracy needed for players / background | No (out of scope v1) |

**It will degrade or fail for:**

- Close-ups with fewer than 4 reliable court intersections visible.
- Heavy occlusion (player covering key intersections).
- Hawk-Eye / virtual replays (no real camera geometry).
- Strong lens distortion without calibration (broadcast lenses are usually mild; v1 uses zero-distortion approximation).

---

## 3. Why This Fits the Existing Codebase

The project already solves the hardest 2D sub-problems:

```
Reference diagram (2D)          Broadcast frame (2D)
        │                                │
        ▼                                ▼
 approach_2: detect_rectangles    approach_3: detect_white_lines
        │                                │
        └──────────► homography ◄────────┘
                   (ref ↔ scene 2D)
```

Homography maps the **reference diagram plane** to the **image plane**. That is exactly the projection of a **world plane (z = 0)** when the camera has unknown pose. The missing piece is:

1. Assign **world coordinates (metres)** to reference keypoints.
2. Estimate **camera intrinsics + extrinsics** via PnP instead of only a 3×3 homography.
3. **Lift** reference geometry into 3D and re-project for validation.

Re-use list:

| Existing module | Re-use in 2D_3D |
|---|---|
| `approach_2.rectangle_detector` | Reference rectangle / corner extraction |
| `approach_3.homography_projector` | Scene keypoints, corner matching, RANSAC pattern |
| `scripts/eval_homography.py` | Full-court sample loader, eval harness pattern |
| `data/Court_dimension.png` | Canonical 2D template |
| `data/GT_scene_samples.csv` | Frames labelled `full_court` for eval |

---

## 4. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT: scene_frame.jpg                   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
┌───────────────────┐                         ┌───────────────────┐
│ Reference branch  │                         │ Scene branch      │
│ (once, cached)    │                         │ (per frame)       │
├───────────────────┤                         ├───────────────────┤
│ detect_rectangles │                         │ detect_white_lines│
│ on Court_dim.png  │                         │ cluster intersects│
│ → ref keypoints   │                         │ → scene keypoints │
└─────────┬─────────┘                         └─────────┬─────────┘
          │                                             │
          ▼                                             │
┌───────────────────┐                                   │
│ WorldCourtModel   │  ref_px → world_xyz (z=0)        │
│ (ITF dimensions)  │◄──────────────────────────────────┤
└─────────┬─────────┘                                   │
          │                                             │
          └──────────────────┬──────────────────────────┘
                             ▼
                  ┌─────────────────────┐
                  │ Match ref ↔ scene   │
                  │ (reuse homography   │
                  │  matching logic)    │
                  └──────────┬──────────┘
                             ▼
                  ┌─────────────────────┐
                  │ solvePnPRansac      │
                  │ → R, t, K (init)    │
                  └──────────┬──────────┘
                             ▼
                  ┌─────────────────────┐
                  │ Build CourtModel3D  │
                  │ lines + surface mesh│
                  └──────────┬──────────┘
                             ▼
                  ┌─────────────────────┐
                  │ Export + visualize  │
                  └─────────────────────┘
```

Data flow summary:

1. **Reference keypoints** live in diagram pixel space.
2. **World model** maps diagram pixels → `(X, Y, 0)` in metres.
3. **Scene keypoints** live in image pixel space.
4. **PnP** finds camera pose such that `world_xyz` projects to `scene_uv`.
5. **3D model** is the world court template (independent of the frame); the frame only determines how we view it.

---

## 5. Design Decisions

### 5.1 Planar court first (not monocular depth)

Tennis courts are flat. v1 models the playing surface as `z = 0` and all lines as 3D polylines on that plane (net posts optionally at `z = net_height` in v2). This avoids MiDaS / Depth Anything, which produce relative depth without metric scale.

### 5.2 World origin and axes

Use a standard ITF-aligned frame:

- **Origin**: center of the court (intersection of center mark and center service line).
- **+X**: doubles sideline direction (width).
- **+Y**: baseline-to-baseline direction (length).
- **+Z**: up.

All dimensions from ITF Rules of Tennis ( singles/doubles alleys, service boxes, center mark, etc.).

### 5.3 Reference diagram → world mapping

`Court_dimension.png` is a 2D schematic, not to scale in pixels. Build an affine (or piecewise) map from diagram `(u, v)` to world `(X, Y)` by anchoring four outer corners:

| Diagram corner | World coordinate |
|---|---|
| Near-left baseline × sideline | `(-half_width, -half_length, 0)` |
| Near-right | `(+half_width, -half_length, 0)` |
| Far-left | `(-half_width, +half_length, 0)` |
| Far-right | `(+half_width, +half_length, 0)` |

Inner lines (service boxes, center mark) are **derived analytically** from ITF dimensions, then optionally snapped to nearest detected reference rectangle corner for sub-pixel alignment.

### 5.4 Camera intrinsics

Broadcast cameras are unknown. v1 strategy (in order):

1. **Default pinhole**: `fx = fy = 1.2 × max(w, h)`, principal point at image center. Good enough for line overlay validation.
2. **FOV prior by scene type** (optional): baseline cameras ≈ 25–35° horizontal FOV; tune from eval set.
3. **Refine with PnP**: if ≥ 6 points, optionally run `cv2.calibrateCamera` on a held-out set of frames from the same camera cluster (future, ties to camera_id pipeline).

Distortion: assume zero for v1 (`k1 = k2 = p1 = p2 = 0`).

### 5.5 Minimum correspondences

| Points | Capability |
|---|---|
| 4 | Pose with fixed intrinsics (PnP); fragile |
| 6+ | RANSAC-stable pose |
| 8+ | Optional intrinsic refinement |

Reuse homography's inlier count and reprojection error as quality gates.

### 5.6 Output format

- **Lines**: list of `(name, (X0,Y0,Z0), (X1,Y1,Z1))` in metres.
- **Surface**: single quad or subdivided rectangle mesh for court colour.
- **Export**: `.obj` (simple, no deps) + optional `.glb` via `trimesh` if added to requirements.

---

## 6. Implementation Plan

### 6.1 Module Layout

```
src/2D_3D/
├── __init__.py
├── config.py              # ReconstructionConfig dataclass
├── world_model.py         # ITF dimensions, ref_px → world_xyz
├── correspondences.py     # Match ref/scene keypoints → 2D-3D pairs
├── camera_pose.py         # Intrinsics defaults, solvePnPRansac, project
├── mesh_builder.py        # CourtModel3D assembly
├── exporter.py            # OBJ / GLB writers
├── visualizer.py          # 2D overlay + optional Open3D viewer
└── pipeline.py            # Orchestrates end-to-end ReconstructionResult
```

Scripts:

```
scripts/reconstruct_court_3d.py   # CLI: one frame → outputs
scripts/eval_3d_reconstruction.py # Batch eval on full_court GT samples
```

Tests:

```
tests/2D_3D/
├── test_world_model.py
├── test_camera_pose.py
└── test_pipeline_synthetic.py    # Synthetic camera, known pose round-trip
```

---

### 6.2 Step 1 — World Court Model

**File**: `world_model.py`

**Responsibilities**:

- Define `ITF_COURT` constants (metres):

  | Feature | Value (m) |
  |---|---|
  | Doubles width | 10.97 |
  | Singles width | 8.23 |
  | Length (baseline to baseline) | 23.77 |
  | Service line from net | 6.40 |
  | Center mark length | 0.10 |
  | Line width | 0.05 (visualisation only) |

- `WorldCourtModel` dataclass:
  - `keypoints: dict[str, np.ndarray]` — named 3D points
  - `lines: list[CourtLine3D]` — named segments
  - `surface_corners: np.ndarray` — `(4, 3)` doubles court corners

- `build_world_model_from_reference(ref_rects, ref_shape, config) -> WorldCourtModel`
  - Detect / cache reference rectangles once.
  - Map each named diagram keypoint to world XYZ.
  - Emit full line set (baselines, sidelines, service lines, center service line, center mark).

- `diagram_px_to_world(px, ref_shape, config) -> np.ndarray`
  - Bilinear or homography from diagram outer quad to world outer quad.

**Acceptance**: Given `Court_dimension.png`, print all keypoint names with world coords; assert outer corners match ITF dimensions within 1 mm.

---

### 6.3 Step 2 — 2D Keypoint Extraction

**File**: reuse via thin wrapper in `correspondences.py`

**Responsibilities**:

- `extract_reference_keypoints(reference_path, config) -> np.ndarray, dict`
  - Call `detect_rectangles` with existing `default_ref_detect_config()` from `eval_homography.py`.
  - `collect_all_corners(ref_rects)`.

- `extract_scene_keypoints(scene_image, config) -> np.ndarray`
  - Call `detect_white_lines`, `find_line_intersections`, cluster — same as `homography_projector`.

No duplication: import from `approach_2` and `approach_3`, do not fork line detection.

---

### 6.4 Step 3 — 2D–3D Correspondences

**File**: `correspondences.py`

**Responsibilities**:

- Reuse `match_corners_by_position` from homography (normalised diagram position ↔ normalised scene position).
- For each matched pair `(ref_px, scene_px)`:
  - `world_xyz = diagram_px_to_world(ref_px)`
  - Store `Correspondence(image_uv=scene_px, world_xyz=world_xyz, name=...)`.

- `CorrespondenceSet` dataclass:
  - `points_2d: (N, 2)`
  - `points_3d: (N, 3)`
  - `names: list[str]`
  - `match_costs: (N,)`

- Quality filter: drop matches with cost above `config.max_match_cost` (mirror homography threshold).

**Acceptance**: On `scene_25_frame_13090.jpg`, obtain ≥ 8 correspondences with the same matches homography uses.

---

### 6.5 Step 4 — Camera Intrinsics and Pose (PnP)

**File**: `camera_pose.py`

**Responsibilities**:

- `estimate_intrinsics(image_shape, config) -> K` — 3×3 matrix.

- `estimate_pose(correspondences, K, config) -> CameraPose`:
  ```python
  ok, rvec, tvec, inliers = cv2.solvePnPRansac(
      objectPoints=points_3d,
      imagePoints=points_2d,
      cameraMatrix=K,
      distCoeffs=zeros,
      ...
  )
  ```
  - Convert `rvec, tvec` to `R, t` (world → camera).
  - Compute mean reprojection error over inliers.

- `project_world_points(xyz, CameraPose) -> uv` — for overlay and eval.

- `CameraPose` dataclass: `K, R, t, rvec, tvec, reproj_error, inlier_count, success`.

**Synthetic test**: Place a virtual camera at known `(R, t)`, project world court corners, add noise, recover pose — error < 1° rotation, < 0.5% translation scale.

---

### 6.6 Step 5 — 3D Geometry Assembly

**File**: `mesh_builder.py`

**Responsibilities**:

- `CourtModel3D` dataclass:
  - `lines: list[CourtLine3D]`
  - `vertices, faces` for court surface (optional line tubes for thick rendering)
  - `world_model: WorldCourtModel`

- `build_court_model(world_model, config) -> CourtModel3D`
  - Pure function of world model (frame-independent geometry).
  - Line width for mesh: `config.line_width_m` (default 0.05 m).

- Optional v1.1: `texture_coords` from inverse projection of scene image onto court plane (homography-based UV from approach_3 `H_inv`).

---

### 6.7 Step 6 — Export and Visualization

**File**: `exporter.py`, `visualizer.py`

**Responsibilities**:

- `export_obj(model, path)` — vertices + line elements or thin quads per line.
- `export_glb(model, path)` — if `trimesh` added; otherwise defer.
- `draw_reprojection_overlay(scene_image, model, pose) -> np.ndarray`:
  - Project every 3D line endpoint; draw in green/red by error threshold.
- `side_by_side_panel(scene, overlay, reference_warp)` — match eval homography grid style.

---

### 6.8 Step 7 — Evaluation

**File**: `scripts/eval_3d_reconstruction.py`

Metrics on `GT_scene_samples.csv` (`scene_type == full_court`):

| Metric | Target (v1) |
|---|---|
| Success rate (≥ 4 inliers, reproj < 8 px) | ≥ 70% (match or beat homography success) |
| Mean reprojection error (inliers) | < 6 px |
| Line angular error vs manual baseline | < 2° (subset manually annotated) |

Output: `data/evaluation/3d_reconstruction_eval.json` + grid of overlays (same pattern as homography eval).

---

### 6.9 Step 8 — CLI Entry Point

**File**: `scripts/reconstruct_court_3d.py`

```bash
python scripts/reconstruct_court_3d.py \
  --scene data/scene_samples/scene_25_frame_13090.jpg \
  --reference data/Court_dimension.png \
  --out-dir data/verification/3d \
  --show
```

Prints: inlier count, reproj error, exported paths.

Wire into `main.py` as optional Step 3 once stable.

---

## 7. Integration with Existing Pipelines

```
main.py / broadcast pipeline
│
├── Step 1: scene split          (camera_split_segment)
├── Step 2: camera assignment    (approach_1 / 2 / 3)
└── Step 3: 3D court model       (src/2D_3D)  ← NEW
         │
         └── Uses camera_id to group frames for intrinsic refinement (future)
```

Recommended call site after homography succeeds:

```python
from src.D2_3D.pipeline import reconstruct_from_frame  # import path TBD

if homography_result.success:
    recon = reconstruct_from_frame(
        scene_path,
        reference_path,
        homography_result=homography_result,  # optional: reuse matches
    )
```

Passing `HomographyResult` avoids re-running matching when already computed.

---

## 8. Known Limitations

1. **Single plane** — Court surface only; stands, net mesh, players are not reconstructed.
2. **Unknown intrinsics** — Absolute scale along the optical axis is coupled with focal length; court dimensions fix the scale along the court plane, but foreshortening errors remain if FOV guess is wrong.
3. **Line detection failures** — Same failure modes as homography (occlusion, clay/grass tuning).
4. **No temporal fusion** — Each frame independent in v1; could average pose over a scene later.
5. **Distortion** — Wide-angle lenses at court level may need `k1` refinement in v2.

---

## 9. Milestones and Acceptance Criteria

| Phase | Scope | Done when |
|---|---|---|
| **M0** | `world_model.py` + tests | All ITF line endpoints generated; diagram corners map correctly |
| **M1** | Correspondences + PnP on one frame | Overlay lines align with white lines on `scene_25_frame_13090.jpg` |
| **M2** | Export OBJ + eval script | Batch eval JSON written; success rate measured |
| **M3** | Pipeline integration | Callable from `main.py`; docs in this plan updated with actual API |
| **M4** | Texture + net posts (optional) | Textured court plane; net line at 0.914 m |

---

## 10. Future Extensions

- **Multi-view bundle adjustment**: Multiple frames from same `camera_id` → refine `K` and shared 3D model.
- **Net and posts**: Add 3D segments at regulation height (0.914 m center).
- **Player height priors**: Known camera pose → rough metric player position on court plane via foot contact assumption.
- **Neural depth fusion**: Use monocular depth only for non-court geometry (crowd, backdrop), not for court lines.
- **Live viewer**: Open3D or web (Three.js) with exported GLB and camera frustum.

---

## Dependencies

| Package | Purpose | Already in project? |
|---|---|---|
| `opencv-python` | PnP, projection, drawing | Yes |
| `numpy` | Arrays | Yes |
| `trimesh` | GLB export (optional) | Add if needed |
| `open3d` | Interactive 3D viewer (optional dev) | Add if needed |

No GPU or deep-learning models required for v1.

---

## Suggested First PR (M0 + M1)

1. Add `src/2D_3D/world_model.py`, `config.py`, `camera_pose.py`.
2. Add `tests/2D_3D/test_world_model.py` and `test_camera_pose.py` (synthetic round-trip).
3. Add `scripts/reconstruct_court_3d.py` producing one overlay image.
4. Run on 3–5 frames from `GT_scene_samples.csv` and compare visually to homography warp.

This validates the core geometry before investing in export formats and batch eval.
