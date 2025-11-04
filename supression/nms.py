import numpy as np


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Return IoU between two boxes in [x1, y1, x2, y2] format."""
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
    return float(inter_area / union)


def _prepare_inputs(
    boxes: np.ndarray | list[list[float]] | list[float],
    scores: np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert inputs to contiguous float32 arrays with consistent lengths."""
    boxes_arr = np.asarray(boxes, dtype=np.float32)
    scores_arr = np.asarray(scores, dtype=np.float32).reshape(-1)

    if boxes_arr.size == 0 or scores_arr.size == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    if boxes_arr.ndim == 1:
        if boxes_arr.size % 4 != 0:
            raise ValueError("Boxes array must contain coordinates in multiples of 4.")
        boxes_arr = boxes_arr.reshape(-1, 4)
    elif boxes_arr.shape[-1] != 4:
        boxes_arr = boxes_arr.reshape(-1, 4)

    count = min(boxes_arr.shape[0], scores_arr.shape[0])
    boxes_arr = boxes_arr[:count]
    scores_arr = scores_arr[:count]

    return np.ascontiguousarray(boxes_arr, dtype=np.float32), np.ascontiguousarray(
        scores_arr, dtype=np.float32
    )


def nms(
    boxes: np.ndarray | list[list[float]],
    scores: np.ndarray | list[float],
    iou_thresh: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Supressão padrão por IoU."""
    boxes_arr, scores_arr = _prepare_inputs(boxes, scores)
    if boxes_arr.size == 0:
        return boxes_arr, scores_arr

    order = np.argsort(scores_arr)[::-1]
    keep: list[int] = []

    x1 = boxes_arr[:, 0]
    y1 = boxes_arr[:, 1]
    x2 = boxes_arr[:, 2]
    y2 = boxes_arr[:, 3]

    widths = np.maximum(0.0, x2 - x1)
    heights = np.maximum(0.0, y2 - y1)
    areas = widths * heights

    while order.size > 0:
        idx = int(order[0])
        keep.append(idx)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[idx], x1[rest])
        yy1 = np.maximum(y1[idx], y1[rest])
        xx2 = np.minimum(x2[idx], x2[rest])
        yy2 = np.minimum(y2[idx], y2[rest])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter_area = inter_w * inter_h

        union = areas[idx] + areas[rest] - inter_area
        ious = inter_area / np.maximum(union, 1e-7)
        order = rest[ious <= iou_thresh]

    keep_arr = np.array(keep, dtype=np.int32)
    keep_arr = keep_arr[np.argsort(scores_arr[keep_arr])[::-1]]
    return boxes_arr[keep_arr], scores_arr[keep_arr]
