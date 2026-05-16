from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

import cv2
import ultralytics
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from calcula_estatisticas.evaluate_reconstructed import evaluate_fold
from pipeline.coco_utils import build_image_lookup_by_stem, extract_original_images, load_coco_json, save_coco_json
from pipeline.data_prep import build_tile_index, discover_fold_directories, prepare_original_test_split
from pipeline.reconstruction import apply_suppression
from pipeline.reporting import (
    write_fold_result,
    write_image_results,
    write_per_image_metrics_csv,
    write_run_metadata,
    write_summary_reports,
)
from pipeline.types import DetectionRecord, SuppressionParams

PROJECT_ROOT = Path(__file__).resolve().parent

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
    "cluster_diou_bws",
    "nms",
    "nms_ioa",
)

MIN_ULTRALYTICS_VERSION = {
    "YOLOV8": (8, 0, 0),
    "YOLOV11": (8, 3, 161),
}


def _parse_requested(items: Sequence[str] | None, available: Iterable[str]) -> List[str]:
    available_list = list(available)
    available_by_lower = {item.lower(): item for item in available_list}
    if not items:
        return available_list

    resolved: List[str] = []
    for item in items:
        match = available_by_lower.get(item.lower())
        if match is None:
            raise KeyError(f"Unknown item '{item}'. Available: {', '.join(sorted(available_by_lower.values()))}")
        resolved.append(match)
    return resolved


def _parse_folds(items: Sequence[str] | None) -> Sequence[int] | None:
    if not items:
        return None
    resolved = []
    for item in items:
        normalized = item.lower().replace("_", "").replace("-", "")
        if normalized.startswith("fold"):
            normalized = normalized[4:]
        resolved.append(int(normalized))
    return resolved


def _parse_version_tuple(version: str) -> tuple[int, int, int]:
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


def _ensure_ultralytics_compatibility(model_name: str) -> None:
    current = _parse_version_tuple(ultralytics.__version__)
    minimum = MIN_ULTRALYTICS_VERSION.get(model_name, (0, 0, 0))
    if current >= minimum:
        return

    minimum_str = ".".join(str(part) for part in minimum)
    current_str = ultralytics.__version__
    raise RuntimeError(
        f"{model_name} requires ultralytics>={minimum_str}, but the current environment has {current_str}. "
        "Update the package before running inference for this model."
    )


def _load_base_dataset(
    source_images_root: Path,
) -> tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
    base_coco_path = source_images_root / "_annotations.coco.json"
    if not base_coco_path.exists():
        raise FileNotFoundError(f"Base COCO annotations not found at {base_coco_path}")

    base_coco = load_coco_json(base_coco_path)
    original_images = extract_original_images(base_coco)
    original_lookup = build_image_lookup_by_stem(original_images)
    return base_coco, original_images, original_lookup


def _resolve_annotations_path(split_dir: Path) -> Path:
    for name in ("_annotations.coco.json", "annotations.coco.json"):
        candidate = split_dir / name
        if candidate.exists():
            return candidate
    return split_dir / "_annotations.coco.json"


def _resolve_weight_path(models_root: Path, model_name: str, fold_idx: int) -> Path:
    weight_relpath = MODEL_SPECS[model_name]["weight_relpath"]
    weight_path = models_root / f"fold_{fold_idx}" / weight_relpath
    if not weight_path.exists():
        raise FileNotFoundError(f"Weight for {model_name} fold_{fold_idx} not found at {weight_path}")
    return weight_path


def _build_detection_model(model_name: str, weight_path: Path, device: str):
    _ensure_ultralytics_compatibility(model_name)
    spec = MODEL_SPECS[model_name]
    return AutoDetectionModel.from_pretrained(
        model_type=str(spec["model_type"]),
        model_path=str(weight_path),
        confidence_threshold=float(spec["confidence"]),
        device=device,
    )


def _prediction_to_detection(prediction: object, *, class_offset: int) -> DetectionRecord:
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


def _run_sahi_on_image(
    image_path: Path,
    detection_model: object,
    model_name: str,
) -> List[DetectionRecord]:
    spec = MODEL_SPECS[model_name]
    result = get_sliced_prediction(
        str(image_path),
        detection_model,
        slice_height=int(spec["slice_height"]),
        slice_width=int(spec["slice_width"]),
        overlap_height_ratio=float(spec["overlap_height_ratio"]),
        overlap_width_ratio=float(spec["overlap_width_ratio"]),
    )
    return [
        _prediction_to_detection(prediction, class_offset=int(spec["class_offset"]))
        for prediction in result.object_prediction_list
    ]


def _run_model_inference(
    *,
    model_name: str,
    weight_path: Path,
    image_names: Sequence[str],
    source_images_root: Path,
    device: str,
) -> tuple[Dict[str, List[DetectionRecord]], Dict[str, float]]:
    timings = {
        "model_load_time_s": 0.0,
        "tile_inference_time_s": 0.0,
    }

    load_start = time.perf_counter()
    detection_model = _build_detection_model(model_name, weight_path, device)
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
        detections = _run_sahi_on_image(image_path, detection_model, model_name)
        elapsed = time.perf_counter() - image_start
        detections_by_image[image_name] = detections
        print(f"[INFO]     [{index}/{total_images}] completed in {elapsed:.2f}s with {len(detections)} detections")
    timings["tile_inference_time_s"] = time.perf_counter() - inference_start

    return detections_by_image, timings


def _build_prediction_dataset(
    *,
    filtered_coco: Mapping[str, object],
    detections_by_image: Mapping[str, Sequence[DetectionRecord]],
) -> Mapping[str, object]:
    annotations: List[MutableMapping[str, object]] = []
    ann_id = 0

    for image_entry in filtered_coco.get("images", []):
        image_id = int(image_entry["id"])
        file_name = str(image_entry["file_name"])
        for det in detections_by_image.get(file_name, []):
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": det.category_id,
                    "bbox": det.to_bbox(),
                    "area": det.width * det.height,
                    "score": det.score,
                }
            )
            ann_id += 1

    return {
        "info": filtered_coco.get("info", {}),
        "licenses": filtered_coco.get("licenses", []),
        "images": filtered_coco.get("images", []),
        "annotations": annotations,
        "categories": filtered_coco.get("categories", []),
    }


def _apply_suppression_to_images(
    *,
    filtered_coco: Mapping[str, object],
    raw_detections_by_image: Mapping[str, Sequence[DetectionRecord]],
    params: SuppressionParams,
) -> tuple[Dict[str, List[DetectionRecord]], float]:
    dimensions = {
        str(image["file_name"]): (int(image.get("width", 0)), int(image.get("height", 0)))
        for image in filtered_coco.get("images", [])
    }
    suppressed: Dict[str, List[DetectionRecord]] = {}
    start = time.perf_counter()
    for image_name, detections in raw_detections_by_image.items():
        width, height = dimensions.get(image_name, (0, 0))
        suppressed[image_name] = apply_suppression(
            detections,
            image_width=width,
            image_height=height,
            params=params,
        )
    return suppressed, time.perf_counter() - start


def _draw_boxes(
    image_path: Path,
    gt_boxes: Sequence[Sequence[float]],
    pred_boxes: Sequence[DetectionRecord],
    output_path: Path,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image at {image_path}")

    for bbox in gt_boxes:
        x, y, w, h = [float(v) for v in bbox]
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2)

    for det in pred_boxes:
        x1 = int(round(det.x))
        y1 = int(round(det.y))
        x2 = int(round(det.x + det.width))
        y2 = int(round(det.y + det.height))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{det.category_id}:{det.score:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def _render_fold_visualizations(
    *,
    filtered_coco: Mapping[str, object],
    detections_by_image: Mapping[str, Sequence[DetectionRecord]],
    source_images_root: Path,
    output_images_dir: Path,
) -> None:
    gt_by_image: Dict[str, List[Sequence[float]]] = {}
    image_id_to_name = {int(image["id"]): str(image["file_name"]) for image in filtered_coco.get("images", [])}
    for ann in filtered_coco.get("annotations", []):
        image_name = image_id_to_name.get(int(ann["image_id"]))
        if image_name is None:
            continue
        gt_by_image.setdefault(image_name, []).append(ann["bbox"])

    for image_name in image_id_to_name.values():
        _draw_boxes(
            source_images_root / image_name,
            gt_by_image.get(image_name, []),
            detections_by_image.get(image_name, []),
            output_images_dir / image_name,
        )


def _build_per_image_rows(
    *,
    dataset_name: str,
    suppression_name: str,
    model_name: str,
    fold_name: str,
    per_image: Sequence[object],
) -> List[Mapping[str, object]]:
    rows = []
    for metric in per_image:
        if metric.image_name == "__summary__":
            continue
        rows.append(
            {
                "dataset": dataset_name,
                "suppression": suppression_name,
                "model": model_name,
                "fold": fold_name,
                "image_name": metric.image_name,
                "precision": f"{metric.precision:.6f}",
                "recall": f"{metric.recall:.6f}",
                "f1": f"{metric.f1:.6f}",
                "mAP50": f"{metric.map50:.6f}",
                "mAP75": f"{metric.map75:.6f}",
                "mAP": f"{metric.map_all:.6f}",
                "MAE": f"{metric.mae:.6f}",
                "RMSE": f"{metric.rmse:.6f}",
                "pred_count": metric.pred_count,
                "gt_count": metric.gt_count,
                "avg_iou": f"{metric.avg_iou:.6f}",
            }
        )
    return rows


def _write_outputs_for_suppression(
    *,
    dataset_name: str,
    suppression_name: str,
    model_name: str,
    fold_idx: int,
    weight_path: Path,
    filtered_coco: Mapping[str, object],
    filtered_coco_path: Path,
    raw_timings: Mapping[str, float],
    detections_by_image: Mapping[str, Sequence[DetectionRecord]],
    source_images_root: Path,
    results_root: Path,
    reports_root: Path,
    total_tiles: int,
    train_annotations: Path,
    val_annotations: Path,
    test_annotations: Path,
) -> None:
    reconstructed_dir = results_root / "reconstructed" / suppression_name / model_name / f"fold{fold_idx}"
    annotations_output = reconstructed_dir / "_annotations.coco.json"
    images_output_dir = reconstructed_dir / "images"
    metadata_output = reconstructed_dir / "run_metadata.json"
    per_image_metrics_output = reconstructed_dir / "per_image_metrics.csv"

    suppression_params = SuppressionParams(method=suppression_name)
    suppressed_detections, suppression_time_s = _apply_suppression_to_images(
        filtered_coco=filtered_coco,
        raw_detections_by_image=detections_by_image,
        params=suppression_params,
    )
    reconstruction_start = time.perf_counter()
    prediction_dataset = _build_prediction_dataset(
        filtered_coco=filtered_coco,
        detections_by_image=suppressed_detections,
    )
    save_coco_json(prediction_dataset, annotations_output)
    _render_fold_visualizations(
        filtered_coco=filtered_coco,
        detections_by_image=suppressed_detections,
        source_images_root=source_images_root,
        output_images_dir=images_output_dir,
    )
    reconstruction_time_s = time.perf_counter() - reconstruction_start

    evaluation_start = time.perf_counter()
    per_image, summary = evaluate_fold(annotations_output, filtered_coco_path)
    evaluation_time_s = time.perf_counter() - evaluation_start

    created_at = datetime.now().isoformat(timespec="seconds")
    fold_name = f"fold_{fold_idx}"
    per_image_rows = _build_per_image_rows(
        dataset_name=dataset_name,
        suppression_name=suppression_name,
        model_name=model_name,
        fold_name=fold_name,
        per_image=per_image,
    )
    write_image_results(reports_root, per_image_rows)
    write_fold_result(
        reports_root,
        {
            "dataset": dataset_name,
            "suppression": suppression_name,
            "model": model_name,
            "fold": fold_name,
            "split": "test",
            "weight_path": str(weight_path),
            "train_annotations": str(train_annotations),
            "val_annotations": str(val_annotations),
            "test_annotations": str(test_annotations),
            "images": len(per_image_rows),
            "tiles": total_tiles,
            "precision": f"{summary.precision:.6f}",
            "recall": f"{summary.recall:.6f}",
            "f1": f"{summary.f1:.6f}",
            "mAP": f"{summary.map_all:.6f}",
            "mAP50": f"{summary.map50:.6f}",
            "mAP75": f"{summary.map75:.6f}",
            "MAE": f"{summary.mae:.6f}",
            "RMSE": f"{summary.rmse:.6f}",
            "model_load_time_s": f"{raw_timings['model_load_time_s']:.6f}",
            "tile_inference_time_s": f"{raw_timings['tile_inference_time_s']:.6f}",
            "reconstruction_time_s": f"{reconstruction_time_s:.6f}",
            "suppression_time_s": f"{suppression_time_s:.6f}",
            "evaluation_time_s": f"{evaluation_time_s:.6f}",
            "total_time_s": f"{(raw_timings['model_load_time_s'] + raw_timings['tile_inference_time_s'] + reconstruction_time_s + suppression_time_s + evaluation_time_s):.6f}",
            "created_at": created_at,
        },
    )
    write_per_image_metrics_csv(
        per_image_metrics_output,
        [
            {
                "image_name": row["image_name"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "mAP50": row["mAP50"],
                "mAP75": row["mAP75"],
                "mAP": row["mAP"],
                "MAE": row["MAE"],
                "RMSE": row["RMSE"],
                "pred_count": row["pred_count"],
                "gt_count": row["gt_count"],
                "avg_iou": row["avg_iou"],
            }
            for row in per_image_rows
        ],
    )
    write_run_metadata(
        metadata_output,
        {
            "dataset": dataset_name,
            "suppression": suppression_name,
            "model": model_name,
            "fold": fold_name,
            "split": "test",
            "weight_path": str(weight_path),
            "images": len(per_image_rows),
            "tiles": total_tiles,
            "timings": {
                **raw_timings,
                "reconstruction_time_s": reconstruction_time_s,
                "suppression_time_s": suppression_time_s,
                "evaluation_time_s": evaluation_time_s,
                "total_time_s": raw_timings["model_load_time_s"] + raw_timings["tile_inference_time_s"] + reconstruction_time_s + suppression_time_s + evaluation_time_s,
            },
            "metrics": {
                "precision": round(summary.precision, 6),
                "recall": round(summary.recall, 6),
                "f1": round(summary.f1, 6),
                "mAP": round(summary.map_all, 6),
                "mAP50": round(summary.map50, 6),
                "mAP75": round(summary.map75, 6),
                "MAE": round(summary.mae, 6),
                "RMSE": round(summary.rmse, 6),
            },
            "created_at": created_at,
        },
    )


def _prepare_fold_context(
    *,
    fold_dir: Path,
    base_coco: Mapping[str, object],
    original_lookup: Mapping[str, object],
    source_images_root: Path,
    originals_root: Path,
) -> tuple[Mapping[str, object], Dict[str, List[object]], Path, int]:
    test_dir = fold_dir / "test"
    _, original_to_tiles = build_tile_index(test_dir, original_lookup)
    filtered_coco_path = prepare_original_test_split(
        base_coco,
        original_to_tiles,
        output_dir=originals_root / fold_dir.name,
        source_images_dir=source_images_root,
    )
    filtered_coco = load_coco_json(filtered_coco_path)
    total_tiles = sum(len(tiles) for tiles in original_to_tiles.values())
    return filtered_coco, original_to_tiles, filtered_coco_path, total_tiles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Principal SAHI-based inference pipeline for YOLOV8 and YOLOV11. "
            "Select the models and suppression methods you want to evaluate."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--source-images-root", type=Path, default=PROJECT_ROOT / "dataset" / "all")
    parser.add_argument("--models-root", type=Path, default=PROJECT_ROOT / "pesos")
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--reports-root", type=Path, default=None)
    parser.add_argument("--originals-root", type=Path, default=None)
    parser.add_argument("--dataset-name", default="sage")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--models",
        nargs="*",
        help="Subset of models to execute. Defaults to all supported models.",
    )
    parser.add_argument(
        "--suppressions",
        nargs="*",
        help="Subset of suppression methods to execute. Defaults to all supported methods.",
    )
    parser.add_argument("--folds", nargs="*", help="Optional list of folds, e.g. fold_1 fold_3 or 1 3.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing outputs and rebuild everything.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    reports_root = args.reports_root or (args.results_root / "reports")
    originals_root = args.originals_root or (args.dataset_root / "imagens_originais")
    requested_models = _parse_requested(args.models, MODEL_SPECS.keys())
    requested_suppressions = _parse_requested(args.suppressions, SUPPRESSIONS)
    allowed_folds = set(_parse_folds(args.folds) or [])

    base_coco, _, original_lookup = _load_base_dataset(args.source_images_root)
    fold_dirs = discover_fold_directories(args.dataset_root / "tiles")
    if not fold_dirs:
        raise RuntimeError(f"No folds found under {args.dataset_root / 'tiles'}")

    for fold_dir in fold_dirs:
        fold_idx = int(fold_dir.name.split("_")[-1])
        if allowed_folds and fold_idx not in allowed_folds:
            print(f"[INFO] Skipping {fold_dir.name} because it was filtered out during preflight validation.")
            continue

        print(f"\n[INFO] Processing {fold_dir.name} (fold {fold_idx})")
        filtered_coco, original_to_tiles, filtered_coco_path, total_tiles = _prepare_fold_context(
            fold_dir=fold_dir,
            base_coco=base_coco,
            original_lookup=original_lookup,
            source_images_root=args.source_images_root,
            originals_root=originals_root,
        )
        image_names = sorted(original_to_tiles.keys())
        train_annotations = _resolve_annotations_path(fold_dir / "train")
        val_annotations = _resolve_annotations_path(fold_dir / "val")
        test_annotations = _resolve_annotations_path(fold_dir / "test")

        for model_name in requested_models:
            weight_path = _resolve_weight_path(args.models_root, model_name, fold_idx)
            suppression_outputs = {
                suppression: args.results_root / "reconstructed" / suppression / model_name / f"fold{fold_idx}" / "_annotations.coco.json"
                for suppression in requested_suppressions
            }
            all_outputs_exist = all(path.exists() for path in suppression_outputs.values())

            if args.no_resume or not all_outputs_exist:
                print(f"[INFO]  +- Running SAHI inference for model '{model_name}' on {fold_dir.name}")
                raw_detections_by_image, raw_timings = _run_model_inference(
                    model_name=model_name,
                    weight_path=weight_path,
                    image_names=image_names,
                    source_images_root=args.source_images_root,
                    device=args.device,
                )
            else:
                print(f"[INFO]  +- Reusing existing outputs for model '{model_name}' on {fold_dir.name}")
                raw_detections_by_image = {}
                raw_timings = {"model_load_time_s": 0.0, "tile_inference_time_s": 0.0}

            for suppression_name in requested_suppressions:
                annotations_output = suppression_outputs[suppression_name]
                if not args.no_resume and annotations_output.exists() and not raw_detections_by_image:
                    prediction_dataset = load_coco_json(annotations_output)
                    image_id_to_name = {int(image["id"]): str(image["file_name"]) for image in prediction_dataset.get("images", [])}
                    suppressed_detections: Dict[str, List[DetectionRecord]] = {}
                    for ann in prediction_dataset.get("annotations", []):
                        image_name = image_id_to_name.get(int(ann["image_id"]))
                        if image_name is None:
                            continue
                        bbox = ann["bbox"]
                        suppressed_detections.setdefault(image_name, []).append(
                            DetectionRecord(
                                x=float(bbox[0]),
                                y=float(bbox[1]),
                                width=float(bbox[2]),
                                height=float(bbox[3]),
                                score=float(ann.get("score", 0.0)),
                                category_id=int(ann["category_id"]),
                            )
                        )
                    _render_fold_visualizations(
                        filtered_coco=filtered_coco,
                        detections_by_image=suppressed_detections,
                        source_images_root=args.source_images_root,
                        output_images_dir=annotations_output.parent / "images",
                    )
                    per_image, summary = evaluate_fold(annotations_output, filtered_coco_path)
                    per_image_rows = _build_per_image_rows(
                        dataset_name=args.dataset_name,
                        suppression_name=suppression_name,
                        model_name=model_name,
                        fold_name=f"fold_{fold_idx}",
                        per_image=per_image,
                    )
                    write_image_results(reports_root, per_image_rows)
                    write_fold_result(
                        reports_root,
                        {
                            "dataset": args.dataset_name,
                            "suppression": suppression_name,
                            "model": model_name,
                            "fold": f"fold_{fold_idx}",
                            "split": "test",
                            "weight_path": str(weight_path),
                            "train_annotations": str(train_annotations),
                            "val_annotations": str(val_annotations),
                            "test_annotations": str(test_annotations),
                            "images": len(per_image_rows),
                            "tiles": total_tiles,
                            "precision": f"{summary.precision:.6f}",
                            "recall": f"{summary.recall:.6f}",
                            "f1": f"{summary.f1:.6f}",
                            "mAP": f"{summary.map_all:.6f}",
                            "mAP50": f"{summary.map50:.6f}",
                            "mAP75": f"{summary.map75:.6f}",
                            "MAE": f"{summary.mae:.6f}",
                            "RMSE": f"{summary.rmse:.6f}",
                            "model_load_time_s": "0.000000",
                            "tile_inference_time_s": "0.000000",
                            "reconstruction_time_s": "0.000000",
                            "suppression_time_s": "0.000000",
                            "evaluation_time_s": "0.000000",
                            "total_time_s": "0.000000",
                            "created_at": datetime.now().isoformat(timespec="seconds"),
                        },
                    )
                    continue

                if not raw_detections_by_image:
                    raise RuntimeError(
                        f"Missing raw detections for {model_name}/{fold_dir.name}. "
                        "Use --no-resume or remove stale outputs."
                    )

                _write_outputs_for_suppression(
                    dataset_name=args.dataset_name,
                    suppression_name=suppression_name,
                    model_name=model_name,
                    fold_idx=fold_idx,
                    weight_path=weight_path,
                    filtered_coco=filtered_coco,
                    filtered_coco_path=filtered_coco_path,
                    raw_timings=raw_timings,
                    detections_by_image=raw_detections_by_image,
                    source_images_root=args.source_images_root,
                    results_root=args.results_root,
                    reports_root=reports_root,
                    total_tiles=total_tiles,
                    train_annotations=train_annotations,
                    val_annotations=val_annotations,
                    test_annotations=test_annotations,
                )

    summary_paths = write_summary_reports(reports_root)
    for summary_path in summary_paths:
        print(f"[INFO] Summary report updated at {summary_path}")
    print("\n[DONE] SAHI pipeline execution completed.")


if __name__ == "__main__":
    main()
