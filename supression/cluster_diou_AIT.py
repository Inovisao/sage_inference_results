import numpy as np


def bbox_center(box):
    """Return box center as (x, y)."""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_area(box):
    """Return box area."""
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def diou(box1, box2):
    """Compute Distance-IoU between two [x1, y1, x2, y2] boxes."""
    x1, y1, x2, y2 = box1
    xb1, yb1, xb2, yb2 = box2

    inter_x1 = max(x1, xb1)
    inter_y1 = max(y1, yb1)
    inter_x2 = min(x2, xb2)
    inter_y2 = min(y2, yb2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)

    area1 = bbox_area(box1)
    area2 = bbox_area(box2)
    union = area1 + area2 - inter_area
    iou = inter_area / union if union > 0 else 0.0

    cx1, cy1 = bbox_center(box1)
    cx2, cy2 = bbox_center(box2)
    center_dist_sq = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2

    c_x1 = min(x1, xb1)
    c_y1 = min(y1, yb1)
    c_x2 = max(x2, xb2)
    c_y2 = max(y2, yb2)
    diag_sq = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2

    return iou - center_dist_sq / diag_sq if diag_sq > 0 else iou


def adaptive_cluster_diou_nms(
    boxes,
    scores,
    T0=0.45,
    alpha=0.15,
    k=5,
    score_ratio_thresh=0.85,
    diou_dup_thresh=0.5,
):
    """
    Cluster-DIoU + Adaptive thresholding.

    The local density around the current highest-score detection is used to
    reduce the threshold (making suppression stricter) whenever many boxes
    overlap with it.
    """
    if len(boxes) == 0:
        return []

    sorted_indices = np.argsort(scores)[::-1]
    boxes_sorted = boxes[sorted_indices]
    scores_sorted = scores[sorted_indices]
    keep = []

    while len(boxes_sorted) > 0:
        keep.append(sorted_indices[0])
        current = boxes_sorted[0]

        if len(boxes_sorted) == 1:
            break

        others = boxes_sorted[1:]
        dious = np.array([diou(current, b) for b in others], dtype=np.float32)
        other_scores = scores_sorted[1:]
        anchor_score = float(scores_sorted[0])

        topk_count = min(k, len(dious))
        if topk_count > 0:
            topk = np.sort(dious)[-topk_count:]
            density = max(0.0, float(np.mean(topk)))
        else:
            density = 0.0
        adaptive_thresh = max(0.05, T0 - alpha * density)

        if anchor_score > 0:
            score_ratios = other_scores / anchor_score
        else:
            score_ratios = np.zeros_like(other_scores)
        duplicate_mask = (dious > diou_dup_thresh) & (score_ratios >= score_ratio_thresh)

        keep_mask = (dious < adaptive_thresh) & (~duplicate_mask)

        boxes_sorted = others[keep_mask]
        scores_sorted = other_scores[keep_mask]
        sorted_indices = sorted_indices[1:][keep_mask]

    return keep
