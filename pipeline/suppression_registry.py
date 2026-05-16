from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from suppression.cluster_diou_nms import cluster_diou_nms
from suppression.nms import nms as suppression_nms
from suppression.nms_ioa import nms_ioa as suppression_nms_ioa

from .types import SuppressionParams

SuppressionAdapter = Callable[
    [np.ndarray, np.ndarray, SuppressionParams, Mapping[str, float]],
    tuple[np.ndarray, np.ndarray],
]


def _apply_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    params: SuppressionParams,
    extra: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    iou_thresh = float(extra.get("iou_threshold", params.iou_threshold))
    return suppression_nms(boxes, scores, iou_thresh=iou_thresh)


def _apply_nms_ioa(
    boxes: np.ndarray,
    scores: np.ndarray,
    params: SuppressionParams,
    extra: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    return suppression_nms_ioa(
        boxes,
        scores,
        ioa_thresh=float(extra.get("ioa_threshold", params.iou_threshold)),
        conf_threshold=float(extra.get("conf_threshold", 0.4)),
        sigma=float(extra.get("sigma", 0.5)),
    )


def _apply_cluster_diou_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    params: SuppressionParams,
    extra: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    diou_thresh = float(extra.get("diou_threshold", params.diou_threshold))
    return cluster_diou_nms(boxes, scores, diou_thresh=diou_thresh)


SUPPRESSION_REGISTRY: Mapping[str, SuppressionAdapter] = {
    "nms": _apply_nms,
    "nms_ioa": _apply_nms_ioa,
    "cluster_diou_nms": _apply_cluster_diou_nms,
    "cluster_nms": _apply_cluster_diou_nms,
}
