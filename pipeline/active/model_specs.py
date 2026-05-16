from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence


MODEL_SPECS: Dict[str, Dict[str, object]] = {
    "YOLOV8": {
        "model_type": "ultralytics",
        "weight_relpath": Path("YOLOV8") / "train" / "weights" / "best.pt",
        "confidence": 0.25,
        "class_offset": 1,
        "slice_height": 640,
        "slice_width": 640,
        "overlap_height_ratio": 0.1,
        "overlap_width_ratio": 0.1,
    },
    "YOLOV11": {
        "model_type": "ultralytics",
        "weight_relpath": Path("YOLOV11") / "train" / "weights" / "best.pt",
        "confidence": 0.25,
        "class_offset": 0,
        "slice_height": 640,
        "slice_width": 640,
        "overlap_height_ratio": 0.1,
        "overlap_width_ratio": 0.1,
    },
}

SUPPRESSIONS: Sequence[str] = (
    "cluster_diou_nms",
    "nms",
    "nms_ioa",
)

MIN_ULTRALYTICS_VERSION = {
    "YOLOV8": (8, 0, 0),
    "YOLOV11": (8, 3, 161),
}

