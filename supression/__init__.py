"""Unified interface for suppression strategies."""

from .bws import bws
from .cluster_ait import cluster_ait
from .cluster_diou_AIT import adaptive_cluster_diou_nms
from .cluster_diou_bws import cluster_diou_bws
from .cluster_diou_nms import cluster_diou_nms
from .nms import nms

__all__ = [
    "nms",
    "bws",
    "cluster_diou_nms",
    "adaptive_cluster_diou_nms",
    "cluster_ait",
    "cluster_diou_bws",
]
