import numpy as np

from .nms import _prepare_inputs, compute_iou


def compute_center_distance(box1: np.ndarray, box2: np.ndarray) -> float:
    """Return a normalised proximity score based on box centre distance."""
    box1 = np.asarray(box1, dtype=np.float32)
    box2 = np.asarray(box2, dtype=np.float32)

    cx1 = (box1[0] + box1[2]) * 0.5
    cy1 = (box1[1] + box1[3]) * 0.5
    cx2 = (box2[0] + box2[2]) * 0.5
    cy2 = (box2[1] + box2[3]) * 0.5

    dist = np.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)

    enc_x1 = min(box1[0], box2[0])
    enc_y1 = min(box1[1], box2[1])
    enc_x2 = max(box1[2], box2[2])
    enc_y2 = max(box1[3], box2[3])
    diag = np.sqrt((enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2)

    if diag <= 0.0:
        return 1.0
    return float(1.0 - min(1.0, dist / (diag + 1e-6)))


def cluster_diou_bws(
    boxes: np.ndarray | list[list[float]],
    scores: np.ndarray | list[float],
    affinity_thresh: float = 0.5,
    lambda_weight: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Fusão ponderada por afinidade (Cluster-DIoU-BWS)."""
    boxes_arr, scores_arr = _prepare_inputs(boxes, scores)
    if boxes_arr.size == 0:
        return boxes_arr, scores_arr

    order = np.argsort(scores_arr)[::-1]
    used = np.zeros_like(scores_arr, dtype=bool)
    clusters: list[tuple[np.ndarray, float]] = []

    for anchor in order:
        if used[anchor]:
            continue

        cluster = [anchor]
        used[anchor] = True

        for candidate in order:
            if used[candidate]:
                continue

            iou_val = compute_iou(boxes_arr[anchor], boxes_arr[candidate])
            center_aff = compute_center_distance(boxes_arr[anchor], boxes_arr[candidate])
            affinity = lambda_weight * iou_val + (1.0 - lambda_weight) * center_aff

            if affinity > affinity_thresh:
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

        clusters.append((merged_box, merged_score))

    if not clusters:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    merged_boxes, merged_scores = zip(*clusters)
    merged_scores_arr = np.asarray(merged_scores, dtype=np.float32)
    sort_idx = np.argsort(merged_scores_arr)[::-1]
    final_boxes = np.asarray(merged_boxes, dtype=np.float32)[sort_idx]
    final_scores = merged_scores_arr[sort_idx]
    return final_boxes, final_scores
