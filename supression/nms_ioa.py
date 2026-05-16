import torch


def calculate_ioa(better_box, others_box):
    """
    Calculate Intersection over Area.

    better_box: [4] in x1, y1, x2, y2 format.
    others_box: [N, 4] in x1, y1, x2, y2 format.
    """

    # 1 - Find the intersection coordinates
    intersection_left_top = torch.max(better_box[:2], others_box[:, :2])
    intersection_right_bottom = torch.min(better_box[2:], others_box[:, 2:])
    intersection_width_hight = (intersection_right_bottom - intersection_left_top).clamp(min=0)
    intersection_area = intersection_width_hight[:, 0] * intersection_width_hight[:, 1]

    # 2 - Calculate the area of the boxes being evaluated.
    area_other_boxes = (others_box[:, 2] - others_box[:, 0]).clamp(min=0) * (
        others_box[:, 3] - others_box[:, 1]
    ).clamp(min=0)

    # 3 - Return the ratio, using a small epsilon to avoid division by zero.
    ioa = intersection_area / area_other_boxes.clamp(min=1e-6)
    return ioa


def ioa_soft_nms(boxes, scores, sigma=0.5, ioa_threshold=0.75, conf_threshold=0.4):
    """Return indices kept by Soft-NMS using IoA as the overlap criterion."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    keep = []
    # Track the original indices so they can be returned at the end.
    indices = torch.arange(boxes.size(0), device=boxes.device)

    while boxes.numel() > 0:
        # Find the current highest-confidence box.
        max_idx = torch.argmax(scores)
        keep.append(indices[max_idx].item())

        if boxes.size(0) == 1:
            break

        # Split the highest-score box (box_m) from the remaining boxes.
        box_m = boxes[max_idx]

        # Remove the selected box from the iterable tensors.
        boxes = torch.cat((boxes[:max_idx], boxes[max_idx+1:]))
        scores = torch.cat((scores[:max_idx], scores[max_idx+1:]))
        indices = torch.cat((indices[:max_idx], indices[max_idx+1:]))

        # Step 1: Calculate IoA between the confirmed box and the remaining overlapping boxes.
        ioa = calculate_ioa(box_m, boxes)

        # Step 2: If IoA exceeds the threshold, apply Soft-NMS confidence decay.
        # Create the mask for boxes above the IoA threshold.
        penalty_mask = ioa > ioa_threshold

        # Calculate the Gaussian weight: e^(-IoA^2 / sigma).
        weight = torch.exp(-(ioa ** 2) / sigma)

        # Reduce confidence only for boxes above the IoA threshold.
        scores[penalty_mask] = scores[penalty_mask] * weight[penalty_mask]

        # Step 3: Drop boxes whose updated confidence fell below conf_threshold.
        keep_mask = scores >= conf_threshold

        boxes = boxes[keep_mask]
        scores = scores[keep_mask]
        indices = indices[keep_mask]

    # Return the original indices of the boxes that survived.
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def nms_ioa(boxes, scores, ioa_thresh=0.75, conf_threshold=0.4, sigma=0.5):
    """
    Apply IoA-based Soft-NMS and return filtered boxes and scores.

    The public API matches the other suppression modules:
    numpy boxes/scores in, numpy boxes/scores out.
    """
    if len(boxes) == 0:
        return boxes, scores

    boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.as_tensor(scores, dtype=torch.float32)
    keep = ioa_soft_nms(
        boxes_tensor,
        scores_tensor.clone(),
        sigma=float(sigma),
        ioa_threshold=float(ioa_thresh),
        conf_threshold=float(conf_threshold),
    )

    kept_boxes = boxes_tensor[keep].cpu().numpy()
    kept_scores = scores_tensor[keep].cpu().numpy()
    return kept_boxes, kept_scores
