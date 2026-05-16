from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time
from typing import Dict, List, Mapping, MutableMapping, Sequence

import cv2

from calcula_estatisticas.evaluate_reconstructed import evaluate_fold
from pipeline.coco_utils import save_coco_json
from pipeline.reconstruction import apply_suppression
from pipeline.reporting import (
    write_fold_result,
    write_image_results,
    write_per_image_metrics_csv,
    write_run_metadata,
)
from pipeline.types import DetectionRecord, SuppressionParams


def build_prediction_dataset(
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


def apply_suppression_to_images(
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


def draw_boxes(
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


def render_fold_visualizations(
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
        draw_boxes(
            source_images_root / image_name,
            gt_by_image.get(image_name, []),
            detections_by_image.get(image_name, []),
            output_images_dir / image_name,
        )


def build_per_image_rows(
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


def write_outputs_for_suppression(
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
    suppression_params: SuppressionParams | None = None,
) -> None:
    reconstructed_dir = results_root / "reconstructed" / suppression_name / model_name / f"fold{fold_idx}"
    annotations_output = reconstructed_dir / "_annotations.coco.json"
    images_output_dir = reconstructed_dir / "images"
    metadata_output = reconstructed_dir / "run_metadata.json"
    per_image_metrics_output = reconstructed_dir / "per_image_metrics.csv"

    suppression_params = suppression_params or SuppressionParams(method=suppression_name)
    suppressed_detections, suppression_time_s = apply_suppression_to_images(
        filtered_coco=filtered_coco,
        raw_detections_by_image=detections_by_image,
        params=suppression_params,
    )
    reconstruction_start = time.perf_counter()
    prediction_dataset = build_prediction_dataset(
        filtered_coco=filtered_coco,
        detections_by_image=suppressed_detections,
    )
    save_coco_json(prediction_dataset, annotations_output)
    render_fold_visualizations(
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
    per_image_rows = build_per_image_rows(
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
