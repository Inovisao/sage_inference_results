import torch


def calculate_diou(boxes1, boxes2):
    """
    Calcula o DIoU entre dois conjuntos de caixas.
    boxes: [N, 4] (x1, y1, x2, y2)
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.empty((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device, dtype=boxes1.dtype)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))[:, None]
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))[None, :]
    union = area1 + area2 - inter
    iou = torch.where(union > 0, inter / union, torch.zeros_like(inter))

    centers1 = (boxes1[:, :2] + boxes1[:, 2:]) / 2.0
    centers2 = (boxes2[:, :2] + boxes2[:, 2:]) / 2.0
    center_dist = ((centers1[:, None, :] - centers2[None, :, :]) ** 2).sum(dim=-1)

    enc_lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    enc_rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enc_wh = (enc_rb - enc_lt).clamp(min=0)
    enc_diag = (enc_wh[..., 0] ** 2 + enc_wh[..., 1] ** 2).clamp(min=1e-6)

    return iou - center_dist / enc_diag


def adaptive_diou_nms(boxes, scores, k=5, tau_0=0.5, alpha=0.1, tau_min=0.3, tau_dup=0.7, gamma=0.5):
    """
    boxes: Tensor [N, 4]
    scores: Tensor [N]
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0] 
        keep.append(i)
        
        if order.numel() == 1: break
        
     
        remaining_indices = order[1:]
        b_star = boxes[i].unsqueeze(0)
        b_others = boxes[remaining_indices]
        
        diou_similarities = calculate_diou(b_star, b_others).squeeze(0)
        
        num_k = min(k, diou_similarities.size(0))
        topk_diou, _ = torch.topk(diou_similarities, num_k)
        rho = topk_diou.mean()
        
        tau_adapt = max(tau_min, tau_0 - alpha * rho)
        
        # s_j / s* >= gamma
        score_ratio = scores[remaining_indices] / scores[i]
        
        suppress_mask = (diou_similarities > tau_adapt) | \
                        ((diou_similarities > tau_dup) & (score_ratio >= gamma))
        
        keep_mask = ~suppress_mask
        order = remaining_indices[keep_mask]
        
    return torch.stack(keep) if keep else torch.empty((0,), dtype=torch.long, device=boxes.device)


def nms_ioa(
    boxes,
    scores,
    *,
    k=5,
    tau_0=0.5,
    alpha=0.1,
    tau_min=0.3,
    tau_dup=0.7,
    gamma=0.5,
):
    """NMS variant based on adaptive DIoU thresholds."""
    boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.as_tensor(scores, dtype=torch.float32)
    keep = adaptive_diou_nms(
        boxes_tensor,
        scores_tensor,
        k=k,
        tau_0=tau_0,
        alpha=alpha,
        tau_min=tau_min,
        tau_dup=tau_dup,
        gamma=gamma,
    )
    if keep.numel() == 0:
        return boxes_tensor[:0].cpu().numpy(), scores_tensor[:0].cpu().numpy()
    return boxes_tensor[keep].cpu().numpy(), scores_tensor[keep].cpu().numpy()
