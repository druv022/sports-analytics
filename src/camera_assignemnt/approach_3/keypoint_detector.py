from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .config import HomographyConfig
from .types import KeypointDetection

ROOT = Path(__file__).resolve().parents[3]
NO_REFINE_INDICES = {8, 9, 12}


@lru_cache(maxsize=1)
def _load_model(model_path: str, device: str):
    import torch
    from third_party.tennis_court_detector.tracknet import BallTrackerNet

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"TennisCourtDetector weights not found: {path}. "
            "Run: python scripts/download_tennis_court_model.py"
        )
    model = BallTrackerNet(out_channels=15)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def detect_court_keypoints(
    scene: np.ndarray,
    config: HomographyConfig | None = None,
) -> KeypointDetection:
    """Detect 14 tennis court keypoints in scene pixel coordinates."""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for court keypoint detection. "
            "Install with: pip install -e '.[court]'"
        ) from exc

    from third_party.tennis_court_detector.homography import get_trans_matrix, refer_kps
    from third_party.tennis_court_detector.postprocess import postprocess, refine_kps

    config = config or HomographyConfig()
    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")

    orig_h, orig_w = scene.shape[:2]
    out_w, out_h = config.input_size
    # postprocess scale=2 maps heatmap coords to (out_w*2, out_h*2) network space
    scale_x = orig_w / (out_w * 2)
    scale_y = orig_h / (out_h * 2)

    resized = cv2.resize(scene, (out_w, out_h))

    inp = (resized.astype(np.float32) / 255.0)
    inp = torch.tensor(np.rollaxis(inp, 2, 0)).unsqueeze(0)

    model = _load_model(str(config.resolved_model_path()), device)
    with torch.no_grad():
        out = model(inp.float().to(device))[0]
    pred = F.sigmoid(out).detach().cpu().numpy()

    raw_points: list[tuple[float | None, float | None]] = []
    for kps_num in range(14):
        heatmap = (pred[kps_num] * 255).astype(np.uint8)
        x_pred, y_pred = postprocess(
            heatmap,
            scale=2,
            low_thresh=config.heatmap_thresh,
            max_radius=config.heatmap_max_radius,
        )
        if x_pred is not None and y_pred is not None:
            x_pred = float(x_pred) * scale_x
            y_pred = float(y_pred) * scale_y
        if (
            config.use_refine_kps
            and kps_num not in NO_REFINE_INDICES
            and x_pred is not None
            and y_pred is not None
        ):
            # refine_kps uses row/col indexing; swap back to OpenCV (x, y).
            y_pred, x_pred = refine_kps(scene, int(y_pred), int(x_pred))
            x_pred, y_pred = y_pred, x_pred
        raw_points.append((x_pred, y_pred))

    points_list: list[list[float | None]] = [
        [p[0], p[1]] if p[0] is not None and p[1] is not None else [None, None]
        for p in raw_points
    ]

    if config.use_tcd_homography_repair:
        matrix_trans = get_trans_matrix(points_list)
        if matrix_trans is not None:
            repaired = cv2.perspectiveTransform(refer_kps, matrix_trans)
            points_list = [[float(x), float(y)] for x, y in np.squeeze(repaired)]

    points = np.full((14, 2), np.nan, dtype=np.float32)
    valid = np.zeros(14, dtype=bool)
    for i, (x, y) in enumerate(points_list):
        if x is None or y is None:
            continue
        points[i, 0] = float(x)
        points[i, 1] = float(y)
        valid[i] = True

    raw_arr = np.full((14, 2), np.nan, dtype=np.float32)
    for i, (x, y) in enumerate(raw_points):
        if x is not None and y is not None:
            raw_arr[i, 0] = float(x)
            raw_arr[i, 1] = float(y)

    return KeypointDetection(points=points, valid=valid, raw_points=raw_arr)


def draw_detected_keypoints(
    scene: np.ndarray,
    detection: KeypointDetection,
    labels: list[str] | None = None,
) -> np.ndarray:
    vis = scene.copy()
    for i in range(14):
        if not detection.valid[i]:
            continue
        x, y = detection.points[i].astype(int)
        cv2.circle(vis, (x, y), 6, (0, 0, 255), -1)
        label = labels[i] if labels else str(i)
        cv2.putText(
            vis,
            label,
            (x + 8, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return vis
