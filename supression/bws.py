import numpy as np
from .nms import compute_iou

def bws(boxes, scores, iou_thresh=0.5):
    """
    Box Weighted Suppression (BWS)
    Faz média ponderada das caixas com IoU > iou_thresh.
    """
    indices = np.argsort(scores)[::-1]
    boxes_final = []
    scores_final = []

    while len(indices) > 0:
        i = indices[0]
        overlaps = [i]
        rest = indices[1:]

        for j in rest:
            iou = compute_iou(boxes[i], boxes[j])
            if iou > iou_thresh:
                overlaps.append(j)

        cluster_boxes = boxes[overlaps]
        cluster_scores = scores[overlaps]

        total = cluster_scores.sum()
        if total <= 0:
            weights = np.full_like(cluster_scores, 1.0 / len(cluster_scores))
        else:
            weights = cluster_scores / total
        merged_box = np.sum(cluster_boxes * weights[:, None], axis=0)

        boxes_final.append(merged_box)
        scores_final.append(cluster_scores.max())

        indices = np.array([idx for idx in rest if idx not in overlaps])

    return np.array(boxes_final), np.array(scores_final)
