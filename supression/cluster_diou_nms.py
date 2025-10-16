import numpy as np

def compute_diou(box1, box2):
    """Distance-IoU entre duas caixas."""
    x1, y1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    x2, y2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2

    # Área de interseção
    xi1, yi1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    xi2, yi2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    iou = inter / union if union > 0 else 0

    # DIoU
    center_dist = (x1 - x2)**2 + (y1 - y2)**2
    c_x1, c_y1 = min(box1[0], box2[0]), min(box1[1], box2[1])
    c_x2, c_y2 = max(box1[2], box2[2]), max(box1[3], box2[3])
    diag = (c_x2 - c_x1)**2 + (c_y2 - c_y1)**2

    return iou - center_dist / (diag + 1e-6)


def cluster_diou_nms(boxes, scores, diou_thresh=0.5):
    """
    Agrupa caixas com DIoU alto e mantém a de maior score.
    """
    indices = np.argsort(scores)[::-1]
    keep = []
    used = set()

    for i in indices:
        if i in used:
            continue
        cluster = [i]
        used.add(i)
        for j in indices:
            if j in used:
                continue
            if compute_diou(boxes[i], boxes[j]) > diou_thresh:
                cluster.append(j)
                used.add(j)
        # Mantém apenas a de maior confiança
        best = max(cluster, key=lambda k: scores[k])
        keep.append(best)

    return boxes[keep], scores[keep]
