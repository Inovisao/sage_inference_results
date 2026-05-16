import torch


def compute_diou_matrix(boxes):
    """
    Compute the vectorized Distance-IoU matrix for all boxes at once.

    boxes: Tensor with shape [N, 4] in x1, y1, x2, y2 format.
    """
    # 1 - Calculate intersection area and IoU.
    lt = torch.max(boxes[:, None, :2], boxes[None, :, :2])  # [N, N, 2]
    rb = torch.min(boxes[:, None, 2:], boxes[None, :, 2:])  # [N, N, 2]
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area[:, None] + area[None, :] - inter
    iou = inter / union.clamp(min=1e-6)

    # 2 - Calculate the DIoU penalty from center distance and enclosing diagonal.
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    center_dist = ((centers[:, None, :] - centers[None, :, :]) ** 2).sum(dim=-1)

    enc_lt = torch.min(boxes[:, None, :2], boxes[None, :, :2])
    enc_rb = torch.max(boxes[:, None, 2:], boxes[None, :, 2:])
    enc_wh = (enc_rb - enc_lt).clamp(min=0)
    diag_dist = (enc_wh[:, :, 0] ** 2 + enc_wh[:, :, 1] ** 2).clamp(min=1e-6)

    # DIoU = IoU - (squared center distance / squared enclosing diagonal).
    diou = iou - (center_dist / diag_dist)
    
    return diou


def compute_diou(box1, box2):
    """Compute Distance-IoU between two boxes."""
    box1_tensor = torch.as_tensor(box1, dtype=torch.float32).reshape(1, 4)
    box2_tensor = torch.as_tensor(box2, dtype=torch.float32).reshape(1, 4)
    boxes = torch.cat((box1_tensor, box2_tensor), dim=0)
    return float(compute_diou_matrix(boxes)[0, 1].item())


def cluster_diou_nms(boxes, scores, diou_thresh=0.5):
    """
    Matrix-based Cluster-DIoU-NMS.

    The public API matches the other suppression modules:
    numpy boxes/scores in, numpy boxes/scores out.
    """
    if len(boxes) == 0:
        return boxes, scores

    boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.as_tensor(scores, dtype=torch.float32)

    # 1 - Sort by descending confidence score.
    order = scores_tensor.argsort(descending=True)
    boxes_sorted = boxes_tensor[order]
    
    # 2 - Calculate the full N x N DIoU matrix.
    diou_matrix = compute_diou_matrix(boxes_sorted)
    
    # 3 - Convert to an upper triangular matrix.
    # The main diagonal is zeroed so a box cannot suppress itself.
    X = torch.triu(diou_matrix, diagonal=1)
    
    # 4 - Initialize the binary vector with all boxes active.
    b = torch.ones(boxes_sorted.size(0), device=boxes_sorted.device)
    
    # 5 - Run the iterative Cluster-NMS loop.
    while True:
        b_old = b.clone()
        
        # X_cluster = b x X. Rows for already suppressed boxes are zeroed.
        X_cluster = X * b.unsqueeze(1) 
        
        # Find the maximum penalty each box receives.
        col_max, _ = X_cluster.max(dim=0)
        
        # Suppress boxes whose received overlap is above the threshold.
        b = (col_max <= diou_thresh).float()
        
        # Stop once the binary vector stabilizes.
        if torch.equal(b, b_old):
            break
            
    # Map surviving sorted positions back to the original input order.
    keep = order[b.bool()]
    
    return boxes_tensor[keep].cpu().numpy(), scores_tensor[keep].cpu().numpy()
