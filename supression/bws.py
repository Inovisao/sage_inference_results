import numpy as np

from .nms import _prepare_inputs, compute_iou


def bws(
    boxes: np.ndarray | list[list[float]],
    scores: np.ndarray | list[float],
    iou_thresh: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Média ponderada por score (BWS)."""
    boxes_arr, scores_arr = _prepare_inputs(boxes, scores)
    if boxes_arr.size == 0:
        return boxes_arr, scores_arr

    order = np.argsort(scores_arr)[::-1]
    used = np.zeros_like(scores_arr, dtype=bool)

    merged_boxes: list[np.ndarray] = []
    merged_scores: list[float] = []

    for anchor in order:
        if used[anchor]:
            continue

        cluster = [anchor]
        used[anchor] = True

        for candidate in order:
            if used[candidate]:
                continue
            if compute_iou(boxes_arr[anchor], boxes_arr[candidate]) > iou_thresh:
                cluster.append(candidate)
                used[candidate] = True

        cluster_boxes = boxes_arr[cluster]
        cluster_scores = scores_arr[cluster]
        total = float(cluster_scores.sum())

        if total > 0:
            weights = cluster_scores / total
        else:
            weights = np.full_like(cluster_scores, 1.0 / len(cluster), dtype=np.float32)

        merged_box = np.average(cluster_boxes, axis=0, weights=weights).astype(np.float32)
        merged_score = float(cluster_scores.max())

        merged_boxes.append(merged_box)
        merged_scores.append(merged_score)

    merged_order = np.argsort(merged_scores)[::-1]
    final_boxes = np.asarray(merged_boxes, dtype=np.float32)[merged_order]
    final_scores = np.asarray(merged_scores, dtype=np.float32)[merged_order]
    return final_boxes, final_scores
