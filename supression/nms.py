import numpy as np


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Non-Maximum Suppression (NMS) using vectorised numpy operations.

    Parameters
    ----------
    boxes : np.ndarray
        Array of shape (N, 4) with boxes in [x1, y1, x2, y2] format.
    scores : np.ndarray
        Confidence scores for each box.
    iou_thresh : float, optional
        IoU threshold for suppression, by default 0.5.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Filtered boxes and their respective scores.
    """
    if boxes.size == 0:
        return boxes, scores

    boxes = boxes.astype(np.float32, copy=False)
    scores = scores.astype(np.float32, copy=False)

    order = scores.argsort()[::-1]
    keep: list[int] = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]

        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter_area = inter_w * inter_h

        union = areas[i] + areas[rest] - inter_area
        iou = inter_area / np.maximum(union, 1e-9)

        remaining = np.where(iou <= iou_thresh)[0]
        order = rest[remaining]

    keep = np.array(keep, dtype=np.int32)
    return boxes[keep], scores[keep]
