import numpy as np

from .cluster_diou_nms import compute_diou
from .nms import _prepare_inputs


def adaptive_cluster_diou_nms(
    boxes: np.ndarray | list[list[float]],
    scores: np.ndarray | list[float],
    T0: float = 0.45,
    alpha: float = 0.15,
    k: int = 5,
    score_ratio_thresh: float = 0.85,
    diou_dup_thresh: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Threshold AIT local aplicado ao Cluster-DIoU."""
    boxes_arr, scores_arr = _prepare_inputs(boxes, scores)
    if boxes_arr.size == 0:
        return boxes_arr, scores_arr

    order = np.argsort(scores_arr)[::-1]
    used = np.zeros_like(scores_arr, dtype=bool)
    keep: list[int] = []

    for anchor in order:
        if used[anchor]:
            continue

        keep.append(anchor)
        used[anchor] = True

        remaining = order[~used[order]]
        if remaining.size == 0:
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

        adaptive_thresh = float(min(0.05, T0 + alpha * density))

        anchor_score = float(scores_arr[anchor])
        other_scores = scores_arr[remaining]
        if anchor_score > 0:
            score_ratios = other_scores / anchor_score
        else:
            score_ratios = np.zeros_like(other_scores)

        dup_thresh = max(adaptive_thresh, diou_dup_thresh)
        duplicate_mask = (dious > dup_thresh) & (score_ratios >= score_ratio_thresh)
        suppression_mask = dious > adaptive_thresh
        to_remove = suppression_mask | duplicate_mask

        if np.any(to_remove):
            used[remaining[to_remove]] = True

    keep_arr = np.array(keep, dtype=np.int32)
    keep_arr = keep_arr[np.argsort(scores_arr[keep_arr])[::-1]]
    return boxes_arr[keep_arr], scores_arr[keep_arr]
