import numpy as np

def compute_iou(box1, box2):
    """Calcula IoU entre duas caixas [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area


def nms(boxes, scores, iou_thresh=0.5):
    """Supressão padrão por IoU."""
    indices = np.argsort(scores)[::-1]
    keep = []

    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        rest = indices[1:]
        ious = np.array([compute_iou(boxes[current], boxes[i]) for i in rest])
        indices = rest[ious < iou_thresh]

    return boxes[keep], scores[keep]
