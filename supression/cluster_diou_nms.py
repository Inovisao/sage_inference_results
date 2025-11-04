import numpy as np

from .nms import _prepare_inputs


def compute_diou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute Distance-IoU between two boxes."""
    box1 = np.asarray(box1, dtype=np.float32)
    box2 = np.asarray(box2, dtype=np.float32)

    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    width1 = max(0.0, box1[2] - box1[0])
    height1 = max(0.0, box1[3] - box1[1])
    width2 = max(0.0, box2[2] - box2[0])
    height2 = max(0.0, box2[3] - box2[1])

    area1 = width1 * height1
    area2 = width2 * height2
    union = max(area1 + area2 - inter_area, 1e-6)
    iou = inter_area / union

    cx1 = (box1[0] + box1[2]) * 0.5
    cy1 = (box1[1] + box1[3]) * 0.5
    cx2 = (box2[0] + box2[2]) * 0.5
    cy2 = (box2[1] + box2[3]) * 0.5
    center_dist = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2

    enc_x1 = min(box1[0], box2[0])
    enc_y1 = min(box1[1], box2[1])
    enc_x2 = max(box1[2], box2[2])
    enc_y2 = max(box1[3], box2[3])
    diag = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2

    if diag <= 0.0:
        return float(iou)
    return float(iou - center_dist / (diag + 1e-7))


def cluster_diou_nms(
    boxes: np.ndarray | list[list[float]],
    scores: np.ndarray | list[float],
    diou_thresh: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Agrupamento por DIoU alto (Cluster-DIoU)."""
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
        for candidate in remaining:
            if used[candidate]:
                continue
            if compute_diou(boxes_arr[anchor], boxes_arr[candidate]) > diou_thresh:
                used[candidate] = True

    keep_arr = np.array(keep, dtype=np.int32)
    keep_arr = keep_arr[np.argsort(scores_arr[keep_arr])[::-1]]
    return boxes_arr[keep_arr], scores_arr[keep_arr]
