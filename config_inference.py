"""Legacy modular pipeline configuration.

The canonical SAHI + Ultralytics entrypoint is `run_inference.py`. This module
is retained for scripts that still build `PipelineSettings` for the legacy
`SageInferencePipeline` flow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from pipeline import PipelineSettings
from pipeline.types import SuppressionParams


PROJECT_ROOT = Path(__file__).resolve().parent

# Models to execute in order. Leave empty/None to execute every model found in `pesos/`.
ENABLED_MODELS: Sequence[str] = ()

# Toggle generation of reconstructed mosaics for each original image.
CREATE_MOSAICS = True

# Map model folder names to detector registry keys when they differ.
DETECTOR_ALIASES = {
    "faster_rcnn": "faster",
    "tph_yolov5": "yolov5_tph",
}

# Override per-model detection thresholds (optional).
DETECTION_THRESHOLDS = {
    "yolov8": 0.25,
    "yolov11": 0.25,
    "yolov5_tph": 0.25,
    "faster": 0.5,
    "retinanet": 0.3,
}

# Add class offsets when the detector output class indices need shifting.
MODEL_CLASS_OFFSETS = {
    "yolov8": 1,
}

# Override the number of classes expected by each detector (optional).
MODEL_NUM_CLASSES = {
    "faster_rcnn": 2,
    "faster": 2,
    # Add entries like "yolov8": 3 if your weights were trained with a different class count.
}

# Supported suppression method names:
#   - "cluster_diou_nms" / "cluster_nms"
#   - "nms"
#   - "nms_ioa"
SUPPRESSION_METHOD = "cluster_diou_nms"
SUPPRESSION_EXTRA = {
    # Example for nms_ioa: "ioa_threshold": 0.75, "conf_threshold": 0.4
}
SUPPRESSION_PARAMS = SuppressionParams(
    method=SUPPRESSION_METHOD,
    affinity_threshold=0.5,
    lambda_weight=0.6,
    score_ratio_threshold=0.85,
    duplicate_iou_threshold=0.5,
    iou_threshold=0.5,
    diou_threshold=0.5,
    extra=SUPPRESSION_EXTRA,
)


def load_config(project_root: Path | None = None) -> PipelineSettings:
    root = project_root or PROJECT_ROOT
    return PipelineSettings(
        dataset_root=root / "dataset",
        models_root=root / "pesos",
        results_root=root / "results",
        originals_root=root / "dataset" / "imagens_originais",
        source_images_root=root / "dataset" / "all",
        create_mosaics=CREATE_MOSAICS,
        suppression=SUPPRESSION_PARAMS,
        detection_thresholds=DETECTION_THRESHOLDS,
        model_class_offsets=MODEL_CLASS_OFFSETS,
        enabled_models=ENABLED_MODELS or None,
        detector_name_aliases=DETECTOR_ALIASES,
        model_num_classes=MODEL_NUM_CLASSES,
        dataset_name="sage",
    )


__all__ = ["load_config", "PROJECT_ROOT"]
