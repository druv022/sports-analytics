"""YOLO11-seg ONNX person instance segmentation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from src.person_appearance.config import AppearanceConfig, COCO_PERSON_CLASS_ID

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - optional dependency
    ort = None  # type: ignore[assignment]


def _letterbox(
    image: NDArray[np.uint8],
    new_shape: int,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[NDArray[np.uint8], float, tuple[float, float]]:
    h, w = image.shape[:2]
    scale = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * scale)), int(round(h * scale)))
    resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    dw = new_shape - new_unpad[0]
    dh = new_shape - new_unpad[1]
    dw /= 2
    dh /= 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded, scale, (dw, dh)


def _xywh2xyxy(boxes: NDArray[np.float32]) -> NDArray[np.float32]:
    out = boxes.copy()
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2]
    out[:, 3] = boxes[:, 1] + boxes[:, 3]
    return out


def _nms(
    boxes: NDArray[np.float32],
    scores: NDArray[np.float32],
    iou_threshold: float,
) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[iou <= iou_threshold]
    return keep


def _scale_boxes_to_original(
    boxes: NDArray[np.float32],
    scale: float,
    pad: tuple[float, float],
    orig_shape: tuple[int, int],
) -> NDArray[np.float32]:
    boxes = boxes.copy()
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= scale
    h, w = orig_shape
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h)
    return boxes


def _decode_masks(
    mask_coeffs: NDArray[np.float32],
    protos: NDArray[np.float32],
    boxes: NDArray[np.float32],
    orig_shape: tuple[int, int],
    scale: float,
    pad: tuple[float, float],
    imgsz: int,
) -> list[NDArray[np.uint8]]:
    if len(mask_coeffs) == 0:
        return []
    # protos: (32, mh, mw)
    coeffs = mask_coeffs.astype(np.float32)
    proto = protos.reshape(protos.shape[0], -1)
    masks = 1 / (1 + np.exp(-(coeffs @ proto)))
    masks = masks.reshape(len(coeffs), protos.shape[1], protos.shape[2])

    h, w = orig_shape
    full_masks: list[NDArray[np.uint8]] = []
    for idx, mask in enumerate(masks):
        mask_resized = cv2.resize(mask, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        # Remove padding
        new_unpad = (int(round(w * scale)), int(round(h * scale)))
        top = int(round(pad[1] - 0.1))
        left = int(round(pad[0] - 0.1))
        cropped = mask_resized[top : top + new_unpad[1], left : left + new_unpad[0]]
        final = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        binary = (final > 0.5).astype(np.uint8) * 255
        x1, y1, x2, y2 = boxes[idx].astype(int)
        box_mask = np.zeros((h, w), dtype=np.uint8)
        box_mask[y1:y2, x1:x2] = binary[y1:y2, x1:x2]
        full_masks.append(box_mask)
    return full_masks


class PersonSegmenter(ABC):
    @abstractmethod
    def detect(self, image: NDArray[np.uint8]) -> list[tuple[tuple[int, int, int, int], float, NDArray[np.uint8] | None]]:
        """Return list of (bbox_xyxy, confidence, mask)."""


class MockPersonSegmenter(PersonSegmenter):
    """Deterministic segmenter for unit tests."""

    def __init__(self, detections: list[tuple[tuple[int, int, int, int], float]] | None = None) -> None:
        self._detections = detections or []

    def detect(self, image: NDArray[np.uint8]) -> list[tuple[tuple[int, int, int, int], float, NDArray[np.uint8] | None]]:
        h, w = image.shape[:2]
        results = []
        for bbox, conf in self._detections:
            x1, y1, x2, y2 = bbox
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
            results.append((bbox, conf, mask))
        return results


class Yolo11SegOnnxSegmenter(PersonSegmenter):
    """Run YOLO11-seg exported to ONNX via onnxruntime."""

    def __init__(self, config: AppearanceConfig, providers: list[str] | None = None) -> None:
        if ort is None:
            raise ImportError(
                "onnxruntime is required for person segmentation. "
                "Install with: pip install -e '.[appearance]'"
            )
        model_path = config.resolved_model_path()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Person segmentation model not found: {model_path}. "
                "Run: python scripts/download_person_seg_model.py"
            )
        self._config = config
        if providers is None:
            providers = _default_providers()
        self._session = ort.InferenceSession(str(model_path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]

    def detect(self, image: NDArray[np.uint8]) -> list[tuple[tuple[int, int, int, int], float, NDArray[np.uint8] | None]]:
        h, w = image.shape[:2]
        padded, scale, pad = _letterbox(image, self._config.imgsz)
        blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)

        outputs = self._session.run(self._output_names, {self._input_name: blob})
        if len(outputs) < 2:
            return []

        preds = outputs[0]
        protos = outputs[1]
        if preds.ndim == 3:
            preds = preds[0]
        if protos.ndim == 4:
            protos = protos[0]

        # (features, anchors) -> (anchors, features)
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T

        num_features = preds.shape[1]
        num_classes = num_features - 4 - 32
        if num_classes < 1:
            return []

        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4 : 4 + num_classes]
        mask_coeffs = preds[:, 4 + num_classes :]

        person_scores = class_scores[:, self._config.person_class_id]
        keep = person_scores >= self._config.min_confidence
        if not keep.any():
            return []

        boxes_xywh = boxes_xywh[keep]
        person_scores = person_scores[keep]
        mask_coeffs = mask_coeffs[keep]

        boxes_xyxy = _xywh2xyxy(boxes_xywh)
        nms_keep = _nms(boxes_xyxy, person_scores, self._config.nms_iou)
        if not nms_keep:
            return []

        boxes_xyxy = boxes_xyxy[nms_keep]
        person_scores = person_scores[nms_keep]
        mask_coeffs = mask_coeffs[nms_keep]
        boxes_orig = _scale_boxes_to_original(boxes_xyxy, scale, pad, (h, w))
        masks = _decode_masks(
            mask_coeffs,
            protos,
            boxes_orig,
            (h, w),
            scale,
            pad,
            self._config.imgsz,
        )

        detections: list[tuple[tuple[int, int, int, int], float, NDArray[np.uint8] | None]] = []
        for box, score, mask in zip(boxes_orig, person_scores, masks, strict=True):
            x1, y1, x2, y2 = [int(v) for v in box]
            detections.append(((x1, y1, x2, y2), float(score), mask))

        detections.sort(key=lambda item: (item[0][0] + item[0][2]) / 2)
        return detections


def _default_providers() -> list[str]:
    if ort is None:
        return ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CoreMLExecutionProvider" in available:
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def create_segmenter(config: AppearanceConfig) -> PersonSegmenter:
    return Yolo11SegOnnxSegmenter(config)
