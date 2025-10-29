from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from calcula_estatisticas.evaluate_reconstructed import (
    evaluate_fold,
    fold_to_gt_path,
    write_details_csv,
    write_results_csv,
)
from pipeline.reconstruction import build_dataset_from_detections, apply_suppression_to_detections
from pipeline.types import DetectionRecord, OriginalImage, SuppressionParams
from pipeline.coco_utils import save_coco_json


AVAILABLE_METHODS = ["cluster_diou_ait", "cluster_diou_nms", "cluster_diou_bws", "nms", "bws"]


def discover_models(raw_root: Path) -> Sequence[Path]:
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw detections root '{raw_root}' not found.")
    return [p for p in raw_root.iterdir() if p.is_dir()]


def discover_folds(model_dir: Path) -> Sequence[Path]:
    folds = [p for p in model_dir.iterdir() if p.is_dir() and p.name.lower().startswith("fold")]
    folds.sort(key=lambda path: path.name)
    return folds


def load_raw_detections(raw_path: Path) -> Tuple[Mapping[str, object], Dict[int, List[DetectionRecord]], Dict[int, OriginalImage]]:
    with raw_path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    image_meta: Dict[int, OriginalImage] = {}
    for image_entry in coco.get("images", []):
        image_id = int(image_entry["id"])
        file_name = str(image_entry["file_name"])
        width = int(image_entry.get("width", 0))
        height = int(image_entry.get("height", 0))
        image_meta[image_id] = OriginalImage(
            id=image_id,
            file_name=file_name,
            width=width,
            height=height,
            stem=Path(file_name).stem,
        )

    detections: Dict[int, List[DetectionRecord]] = {image_id: [] for image_id in image_meta}
    for ann in coco.get("annotations", []):
        image_id = int(ann["image_id"])
        record = DetectionRecord(
            x=float(ann["bbox"][0]),
            y=float(ann["bbox"][1]),
            width=float(ann["bbox"][2]),
            height=float(ann["bbox"][3]),
            score=float(ann.get("score", 0.0)),
            category_id=int(ann["category_id"]),
        )
        detections.setdefault(image_id, []).append(record)

    return coco, detections, image_meta


def default_params_for_method(method: str) -> SuppressionParams:
    method = method.lower()
    if method == "cluster_diou_ait":
        extra = {
            "T0": 0.45,
            "alpha": 0.15,
            "k": 5,
            "score_ratio_threshold": 0.85,
            "duplicate_iou_threshold": 0.5,
        }
        return SuppressionParams(method=method, extra=extra)
    if method == "cluster_diou_nms":
        return SuppressionParams(method=method, diou_threshold=0.5)
    if method == "cluster_diou_bws":
        return SuppressionParams(method=method, affinity_threshold=0.4, lambda_weight=0.3)
    if method in {"nms", "bws"}:
        return SuppressionParams(method=method, iou_threshold=0.5)
    raise ValueError(f"Unsupported suppression method '{method}'.")


def suppress_raw_annotations(
    *,
    raw_path: Path,
    params: SuppressionParams,
) -> Mapping[str, object]:
    base_coco, detections_by_image, image_meta = load_raw_detections(raw_path)
    suppressed = apply_suppression_to_detections(
        detections_by_image=detections_by_image,
        image_meta_by_id=image_meta,
        params=params,
    )
    dataset = build_dataset_from_detections(
        base_coco=base_coco,
        detections_by_image=suppressed,
    )
    return dataset


def save_suppressed_dataset(
    *,
    dataset: Mapping[str, object],
    destination: Path,
) -> None:
    save_coco_json(dataset, destination)


def evaluate_method_predictions(
    *,
    method_root: Path,
    dataset_root: Path,
) -> None:
    reconstructed_root = method_root / "reconstructed"
    if not reconstructed_root.exists():
        print(f"[WARN] No reconstructed predictions found under {reconstructed_root}. Skipping evaluation.")
        return

    aggregate_rows: List[List[str]] = []

    for model_dir in discover_models(reconstructed_root):
        model_name = model_dir.name
        for fold_dir in discover_folds(model_dir):
            pred_path = fold_dir / "_annotations.coco.json"
            if not pred_path.exists():
                print(f"[WARN] Predictions not found at {pred_path}; skipping.")
                continue
            try:
                gt_path = fold_to_gt_path(dataset_root, fold_dir.name)
            except FileNotFoundError as exc:
                print(f"[WARN] {exc}")
                continue

            print(f"[INFO] Evaluating {model_name} {fold_dir.name} for method '{method_root.name}'")
            per_image, summary = evaluate_fold(pred_path, gt_path)
            write_details_csv(method_root, model_name, fold_dir.name, per_image)

            aggregate_rows.append(
                [
                    model_name,
                    fold_dir.name,
                    str(len(per_image)),
                    f"{summary.precision:.6f}",
                    f"{summary.recall:.6f}",
                    f"{summary.f1:.6f}",
                    f"{summary.map_all:.6f}",
                    f"{summary.map50:.6f}",
                    f"{summary.map75:.6f}",
                    f"{summary.mae:.6f}",
                    f"{summary.rmse:.6f}",
                ]
            )

    if aggregate_rows:
        write_results_csv(method_root, aggregate_rows)
    else:
        print(f"[WARN] No evaluation rows generated for method '{method_root.name}'.")


def aggregate_method_results(results_paths: Mapping[str, Path], output_csv: Path) -> None:
    metric_columns = ["precision", "recall", "f1", "mAP", "mAP50", "mAP75", "MAE", "RMSE"]
    header = ["method", "model"] + metric_columns
    rows: List[List[str]] = []

    for method, csv_path in sorted(results_paths.items()):
        if not csv_path.exists():
            print(f"[WARN] Results CSV not found for method '{method}' at {csv_path}. Skipping.")
            continue

        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            metrics_map: Dict[str, Dict[str, List[float]]] = {}
            for row in reader:
                model = row.get("model")
                if not model:
                    continue
                metrics_map.setdefault(model, {})
                for metric in metric_columns:
                    value = row.get(metric)
                    if value is None:
                        continue
                    try:
                        parsed = float(value)
                    except ValueError:
                        continue
                    metrics_map[model].setdefault(metric, []).append(parsed)

        for model, metrics in metrics_map.items():
            row = [method, model]
            for metric in metric_columns:
                values = metrics.get(metric, [])
                if not values:
                    row.append("")
                    continue
                mean_val = statistics.mean(values)
                row.append(f"{mean_val:.6f}")
            rows.append(row)

    if not rows:
        print("[WARN] No aggregated statistics to write.")
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"[INFO] Final aggregated statistics written to {output_csv}")


def run_method(
    *,
    method: str,
    raw_root: Path,
    output_root: Path,
) -> None:
    params = default_params_for_method(method)
    reconstructed_root = raw_root
    method_root = output_root / method

    for model_dir in discover_models(reconstructed_root):
        model_name = model_dir.name
        for fold_dir in discover_folds(model_dir):
            raw_path = fold_dir / "_detections.coco.json"
            if not raw_path.exists():
                fallback = fold_dir / "_annotations.coco.json"
                if fallback.exists():
                    print(
                        f"[WARN] Expected raw detections at {raw_path} not found; "
                        f"using fallback {fallback} (may already be suppressed)."
                    )
                    raw_path = fallback
                else:
                    print(
                        f"[WARN] Raw detections not found at {raw_path} "
                        f"or fallback {fallback}; skipping."
                    )
                    continue

            dataset = suppress_raw_annotations(raw_path=raw_path, params=params)

            destination = method_root / "reconstructed" / model_name / fold_dir.name / "_annotations.coco.json"
            save_suppressed_dataset(dataset=dataset, destination=destination)
            print(f"[INFO] Saved {method} suppression for {model_name}/{fold_dir.name} to {destination}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply every suppression method to raw detections and evaluate the outcomes."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("results") / "reconstructed",
        help="Directory containing model/fold raw detections (_detections.coco.json).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results") / "suppression_comparison",
        help="Directory where method-specific suppression outputs will be saved.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
        help="Dataset root used to locate ground-truth annotations.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        choices=AVAILABLE_METHODS,
        help="Optional subset of suppression methods to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = args.methods or AVAILABLE_METHODS

    results_csv_map: Dict[str, Path] = {}

    for method in methods:
        print(f"\n[INFO] Running suppression method '{method}'")
        run_method(method=method, raw_root=args.raw_root, output_root=args.output_root)

        method_root = args.output_root / method
        evaluate_method_predictions(method_root=method_root, dataset_root=args.dataset_root)
        results_csv_map[method] = method_root / "results.csv"

    final_results_path = args.output_root / "final_results.csv"
    aggregate_method_results(results_csv_map, final_results_path)


if __name__ == "__main__":
    main()
