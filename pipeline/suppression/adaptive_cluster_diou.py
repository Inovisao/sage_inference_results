"""
Thin wrapper module that exposes the Adaptive Cluster-DIoU suppression routine
under the pipeline.suppression namespace.
"""

from supression.cluster_diou_AIT import adaptive_cluster_diou_nms

__all__ = ["adaptive_cluster_diou_nms"]
