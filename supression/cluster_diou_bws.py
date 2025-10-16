import numpy as np
from .nms import compute_iou
from .cluster_diou_nms import compute_diou

def cluster_diou_bws(boxes, scores, affinity_thresh=0.5, lambda_weight=0.6):
    """
    Cluster-DIoU-BWS híbrido:
    Agrupa caixas pela afinidade (IoU + distância entre centros)
    e realiza média ponderada das caixas em cada cluster.
    """
    n = len(boxes)
    used = set()
    clusters = []

    for i in range(n):
        if i in used:
            continue
        cluster = [i]
        used.add(i)

        for j in range(i + 1, n):
            if j in used:
                continue

            iou = compute_iou(boxes[i], boxes[j])
            diou = compute_diou(boxes[i], boxes[j])
            affinity = lambda_weight * iou + (1 - lambda_weight) * (1 - diou)

            if affinity > affinity_thresh:
                cluster.append(j)
                used.add(j)

        # Combina via média ponderada (BWS)
        cluster_boxes = boxes[cluster]
        cluster_scores = scores[cluster]
        weights = cluster_scores / cluster_scores.sum()
        merged_box = np.sum(cluster_boxes * weights[:, None], axis=0)
        merged_score = cluster_scores.max()

        clusters.append((merged_box, merged_score))

    final_boxes, final_scores = zip(*clusters)
    return np.array(final_boxes), np.array(final_scores)
