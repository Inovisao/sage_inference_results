import numpy as np
from itertools import combinations

def bbox_center(box):
    """Calcula o centro (x, y) da bounding box."""
    x1, y1, x2, y2 = box
    return ( (x1 + x2) / 2, (y1 + y2) / 2 )

def bbox_area(box):
    """Calcula a área da bounding box."""
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

def diou(box1, box2):
    """Calcula o Distance-IoU entre duas bounding boxes."""
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2

    # Interseção
    inter_x1 = max(x1, x1b)
    inter_y1 = max(y1, y1b)
    inter_x2 = min(x2, x2b)
    inter_y2 = min(y2, y2b)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # União
    area1 = bbox_area(box1)
    area2 = bbox_area(box2)
    union = area1 + area2 - inter_area
    iou = inter_area / union if union > 0 else 0.0

    # Distância entre centros
    cx1, cy1 = bbox_center(box1)
    cx2, cy2 = bbox_center(box2)
    rho2 = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2

    # Diagonal mínima que cobre ambas as caixas
    c_x1 = min(x1, x1b)
    c_y1 = min(y1, y1b)
    c_x2 = max(x2, x2b)
    c_y2 = max(y2, y2b)
    c2 = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2

    diou_value = iou - rho2 / c2 if c2 > 0 else iou
    return diou_value

def adaptive_cluster_diou_nms(boxes, scores, T0=0.45, alpha=0.15, k=5):
    """
    Cluster-DIoU + Adaptive IoU Threshold (AIT).
    Ajusta o threshold localmente com base na densidade de detecções.
    
    Args:
        boxes (np.ndarray): (N,4) -> [x1,y1,x2,y2]
        scores (np.ndarray): (N,)
        T0 (float): threshold base
        alpha (float): peso de ajuste local
        k (int): número de vizinhos próximos usados para densidade local
    
    Retorna:
        keep (list): índices das caixas mantidas
    """
    if len(boxes) == 0:
        return []

    # Ordena por score
    indices = np.argsort(scores)[::-1]
    boxes = boxes[indices]
    scores = scores[indices]
    keep = []

    while len(boxes) > 0:
        # Seleciona a caixa de maior score
        current = boxes[0]
        keep.append(indices[0])

        if len(boxes) == 1:
            break

        # Calcula DIoU de todas as demais com a atual
        dious = np.array([diou(current, b) for b in boxes[1:]])

        # Calcula densidade local (média dos k maiores valores de DIoU)
        topk = np.sort(dious)[-min(k, len(dious)):]
        mean_local = np.mean(topk)
        T_i = T0 + alpha * mean_local

        # Mantém apenas as caixas com DIoU abaixo do threshold adaptado
        keep_mask = dious < T_i
        boxes = boxes[1:][keep_mask]
        scores = scores[1:][keep_mask]
        indices = indices[1:][keep_mask]

    return keep
