"""Axis-aligned box geometry used by association and lifecycle code."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _boxes(value: Iterable[Iterable[float]] | np.ndarray) -> np.ndarray:
    if hasattr(value, "bbox_xyxy"):
        value = [getattr(value, "bbox_xyxy")]
    else:
        try:
            items = list(value)
        except TypeError:
            items = []
        if items and hasattr(items[0], "bbox_xyxy"):
            value = [getattr(item, "bbox_xyxy") for item in items]
        elif items and hasattr(items[0], "xyxy"):
            value = [getattr(item, "xyxy") for item in items]
        elif not items:
            value = items
    result = np.asarray(value, dtype=np.float64)
    if result.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    if result.ndim == 1:
        if result.shape[0] != 4:
            raise ValueError("boxes must have shape (N, 4)")
        result = result.reshape(1, 4)
    if result.ndim != 2 or result.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4)")
    if not np.all(np.isfinite(result)):
        raise ValueError("boxes must be finite")
    if np.any(result[:, 2] <= result[:, 0]) or np.any(result[:, 3] <= result[:, 1]):
        raise ValueError("boxes must have positive width and height")
    return result


def xyxy_to_xywh(box: Iterable[float]) -> np.ndarray:
    x1, y1, x2, y2 = _boxes(box)[0]
    return np.asarray(((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1), dtype=np.float64)


def xywh_to_xyxy(box: Iterable[float]) -> np.ndarray:
    value = np.asarray(tuple(float(item) for item in box), dtype=np.float64)
    if value.shape != (4,) or not np.all(np.isfinite(value)) or value[2] <= 0 or value[3] <= 0:
        raise ValueError("xywh must be finite (cx, cy, width, height) with positive size")
    cx, cy, width, height = value
    return np.asarray((cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2), dtype=np.float64)


def iou_matrix(boxes_a: Iterable[Iterable[float]] | np.ndarray, boxes_b: Iterable[Iterable[float]] | np.ndarray) -> np.ndarray:
    """Return pairwise IoU for two ``(x1, y1, x2, y2)`` collections."""

    a = _boxes(boxes_a)
    b = _boxes(boxes_b)
    result = np.zeros((len(a), len(b)), dtype=np.float64)
    if not len(a) or not len(b):
        return result
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection_size = np.maximum(0.0, bottom_right - top_left)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection
    np.divide(intersection, union, out=result, where=union > 0)
    return result


def iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    return float(iou_matrix(box_a, box_b)[0, 0])


def iou_distance(boxes_a: Iterable[Iterable[float]] | np.ndarray, boxes_b: Iterable[Iterable[float]] | np.ndarray) -> np.ndarray:
    return 1.0 - iou_matrix(boxes_a, boxes_b)


def fuse_score(cost_matrix: np.ndarray, scores: Iterable[float]) -> np.ndarray:
    """Fuse detection confidence into an IoU cost matrix."""

    cost = np.asarray(cost_matrix, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError("cost_matrix must be two-dimensional")
    if cost.size == 0:
        return cost.copy()
    values = np.asarray(
        tuple(float(getattr(score, "score", score)) for score in scores),
        dtype=np.float64,
    )
    if values.shape != (cost.shape[1],):
        raise ValueError("scores must contain one value per detection column")
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("scores must be finite and in [0, 1]")
    return 1.0 - (1.0 - cost) * values[None, :]


def warp_xywh_state(mean: np.ndarray, covariance: np.ndarray, warp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply a 2x3 affine camera warp to an XYWH Kalman state."""

    mean_value = np.asarray(mean, dtype=np.float64)
    covariance_value = np.asarray(covariance, dtype=np.float64)
    matrix = np.asarray(warp, dtype=np.float64)
    if mean_value.shape != (8,) or covariance_value.shape != (8, 8) or matrix.shape != (2, 3):
        raise ValueError("invalid Kalman/GMC shapes")
    rotation = matrix[:, :2]
    transform = np.kron(np.eye(4, dtype=np.float64), rotation)
    transformed_mean = transform @ mean_value
    transformed_mean[:2] += matrix[:, 2]
    transformed_covariance = transform @ covariance_value @ transform.T
    return transformed_mean, transformed_covariance


# Names used by common tracker implementations and convenient for callers.
box_iou = iou_matrix
