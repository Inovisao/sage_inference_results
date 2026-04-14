import numpy as np
from .nms import compute_iou
from .cluster_diou_nms import compute_diou


def cluster_diou_bws(boxes, scores, affinity_thresh=0.5, lambda_weight=0.6):
    """
    Cluster-DIoU-BWS híbrido:
    Agrupa caixas por afinidade geométrica local e realiza média ponderada.
    e média ponderada das caixas em cada cluster.
    """
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    order = np.argsort(scores)[::-1]
    used = set()
    clusters = []

    for pos, i in enumerate(order):
        if i in used:
            continue
        cluster = [i]
        used.add(i)

        for j in order[pos + 1:]:
            if j in used:
                continue

            iou = compute_iou(boxes[i], boxes[j])
            diou = max(0.0, float(compute_diou(boxes[i], boxes[j])))
            affinity = lambda_weight * float(iou) + (1 - lambda_weight) * diou

            if affinity > affinity_thresh:
                cluster.append(j)
                used.add(j)

        # Combina via média ponderada (BWS)
        cluster_boxes = boxes[cluster]
        cluster_scores = scores[cluster]
        score_sum = float(cluster_scores.sum())
        if score_sum <= 0:
            weights = np.full(len(cluster_scores), 1.0 / len(cluster_scores), dtype=np.float32)
        else:
            weights = cluster_scores / score_sum
        merged_box = np.sum(cluster_boxes * weights[:, None], axis=0)
        merged_score = cluster_scores.max()

        clusters.append((merged_box, merged_score))

    final_boxes, final_scores = zip(*clusters)
    return np.array(final_boxes), np.array(final_scores)
