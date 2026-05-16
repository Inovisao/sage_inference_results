from __future__ import annotations

from pathlib import Path
import time
from typing import Dict, List, Mapping, Sequence

import ultralytics
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from pipeline.types import DetectionRecord

def parse_version_tuple(version: str) -> tuple[int, int, int]:
    parts = []
    for chunk in version.split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
        if len(parts) == 3:
            break
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def ensure_ultralytics_compatibility(
    model_name: str,
    minimum_versions: Mapping[str, tuple[int, int, int]],
) -> None:
    current = parse_version_tuple(ultralytics.__version__)
    minimum = minimum_versions.get(model_name, (0, 0, 0))
    if current >= minimum:
        return

    minimum_str = ".".join(str(part) for part in minimum)
    current_str = ultralytics.__version__
    raise RuntimeError(
        f"{model_name} requires ultralytics>={minimum_str}, but the current environment has {current_str}. "
        "Update the package before running inference for this model."
    )


def build_detection_model(
    model_name: str,
    weight_path: Path,
    device: str,
    *,
    model_specs: Mapping[str, Mapping[str, object]],
    minimum_versions: Mapping[str, tuple[int, int, int]],
):
    ensure_ultralytics_compatibility(model_name, minimum_versions)
    spec = model_specs[model_name]
    return AutoDetectionModel.from_pretrained(
        model_type=str(spec["model_type"]),
        model_path=str(weight_path),
        confidence_threshold=float(spec["confidence"]),
        device=device,
    )


def prediction_to_detection(prediction: object, *, class_offset: int) -> DetectionRecord:
    bbox = prediction.bbox  # type: ignore[attr-defined]
    score = prediction.score.value if hasattr(prediction.score, "value") else float(prediction.score)  # type: ignore[attr-defined]
    category_id = int(prediction.category.id) + class_offset  # type: ignore[attr-defined]
    return DetectionRecord(
        x=float(bbox.minx),
        y=float(bbox.miny),
        width=float(bbox.maxx - bbox.minx),
        height=float(bbox.maxy - bbox.miny),
        score=float(score),
        category_id=category_id,
    )


def run_sahi_on_image(
    image_path: Path,
    detection_model: object,
    model_name: str,
    *,
    model_specs: Mapping[str, Mapping[str, object]],
) -> List[DetectionRecord]:
    spec = model_specs[model_name]
    result = get_sliced_prediction(
        str(image_path),
        detection_model,
        slice_height=int(spec["slice_height"]),
        slice_width=int(spec["slice_width"]),
        overlap_height_ratio=float(spec["overlap_height_ratio"]),
        overlap_width_ratio=float(spec["overlap_width_ratio"]),
    )
    return [
        prediction_to_detection(prediction, class_offset=int(spec["class_offset"]))
        for prediction in result.object_prediction_list
    ]


def run_model_inference(
    *,
    model_name: str,
    weight_path: Path,
    image_names: Sequence[str],
    source_images_root: Path,
    device: str,
    model_specs: Mapping[str, Mapping[str, object]],
    minimum_versions: Mapping[str, tuple[int, int, int]],
) -> tuple[Dict[str, List[DetectionRecord]], Dict[str, float]]:
    timings = {
        "model_load_time_s": 0.0,
        "tile_inference_time_s": 0.0,
    }

    load_start = time.perf_counter()
    detection_model = build_detection_model(
        model_name,
        weight_path,
        device,
        model_specs=model_specs,
        minimum_versions=minimum_versions,
    )
    timings["model_load_time_s"] = time.perf_counter() - load_start

    detections_by_image: Dict[str, List[DetectionRecord]] = {}
    inference_start = time.perf_counter()
    total_images = len(image_names)
    for index, image_name in enumerate(image_names, start=1):
        image_path = source_images_root / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Original image '{image_path}' not found.")
        print(f"[INFO]     [{index}/{total_images}] SAHI inference on {image_name}")
        image_start = time.perf_counter()
        detections = run_sahi_on_image(image_path, detection_model, model_name, model_specs=model_specs)
        elapsed = time.perf_counter() - image_start
        detections_by_image[image_name] = detections
        print(f"[INFO]     [{index}/{total_images}] completed in {elapsed:.2f}s with {len(detections)} detections")
    timings["tile_inference_time_s"] = time.perf_counter() - inference_start

    return detections_by_image, timings
