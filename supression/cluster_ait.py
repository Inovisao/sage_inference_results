import numpy as np

from .cluster_diou_bws import compute_center_distance
from .cluster_diou_nms import compute_diou
from .nms import _prepare_inputs, compute_iou


def cluster_ait(
    boxes: np.ndarray | list[list[float]],
    scores: np.ndarray | list[float],
    T0: float = 0.5,
    alpha: float = 0.2,
    k: int = 5,
    lambda_weight: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Threshold adaptativo com agrupamento (Cluster-AIT)."""
    boxes_arr, scores_arr = _prepare_inputs(boxes, scores)
    if boxes_arr.size == 0:
        return boxes_arr, scores_arr

    order = np.argsort(scores_arr)[::-1]
    used = np.zeros_like(scores_arr, dtype=bool)
    keep: list[int] = []

    for anchor in order:
        if used[anchor]:
            continue

        cluster = [anchor]
        used[anchor] = True

        remaining = order[~used[order]]
        if remaining.size == 0:
            best_idx = anchor
            keep.append(best_idx)
            continue

        anchor_box = boxes_arr[anchor]
        dious = np.array(
            [compute_diou(anchor_box, boxes_arr[idx]) for idx in remaining],
            dtype=np.float32,
        )

        valid_dious = dious[dious > 0]
        if valid_dious.size:
            topk = np.sort(valid_dious)[-min(k, valid_dious.size):]
            density = float(np.mean(topk))
        else:
            density = 0.0

        adaptive_thresh = float(min(0.9, T0 + alpha * density))

        center_scores = np.array(
            [compute_center_distance(anchor_box, boxes_arr[idx]) for idx in remaining],
            dtype=np.float32,
        )
        iou_scores = np.array(
            [compute_iou(anchor_box, boxes_arr[idx]) for idx in remaining],
            dtype=np.float32,
        )
        affinity = lambda_weight * iou_scores + (1.0 - lambda_weight) * center_scores

        for candidate, diou_val, affinity_val in zip(remaining, dious, affinity):
            if diou_val <= 0:
                continue
            if affinity_val > adaptive_thresh:
                cluster.append(candidate)
                used[candidate] = True

        best_idx = max(cluster, key=lambda idx_: scores_arr[idx_])
        keep.append(best_idx)

    keep_arr = np.array(keep, dtype=np.int32)
    keep_arr = keep_arr[np.argsort(scores_arr[keep_arr])[::-1]]
    return boxes_arr[keep_arr], scores_arr[keep_arr]
