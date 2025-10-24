from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

EPS = 1e-6
IOU_DEFAULT = 0.5
MAP_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]


@dataclass
class ImageMetrics:
    image_name: str
    precision: float
    recall: float
    f1: float
    map50: float
    map75: float
    map_all: float
    mae: float
    rmse: float
    avg_iou: float
    pred_count: int
    gt_count: int


def load_coco_boxes(path: Path, with_scores: bool = False) -> Dict[str, List[Tuple[List[float], float]]]:
    with path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    id_to_file: Dict[int, str] = {int(img["id"]): img["file_name"] for img in coco.get("images", [])}
    boxes: Dict[str, List[Tuple[List[float], float]]] = {file: [] for file in id_to_file.values()}

    for ann in coco.get("annotations", []):
        image_id = int(ann["image_id"])
        bbox = [float(v) for v in ann["bbox"]]
        score = float(ann.get("score", 1.0)) if with_scores else 1.0
        file_name = id_to_file.get(image_id)
        if file_name is None:
            continue
        boxes.setdefault(file_name, []).append((bbox, score))

    return boxes


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


@dataclass
class MatchResult:
    tp: int
    fp: int
    fn: int
    matched_ious: List[float]


def match_predictions(gt_boxes: Sequence[Sequence[float]], pred_boxes: Sequence[Sequence[float]], threshold: float) -> MatchResult:
    matched_gt: set[int] = set()
    matched_ious: List[float] = []
    tp = fp = 0

    for pred in pred_boxes:
        best_iou = 0.0
        best_idx = None
        for idx, gt in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            value = iou(pred, gt)
            if value > best_iou:
                best_iou = value
                best_idx = idx
        if best_iou >= threshold and best_idx is not None:
            tp += 1
            matched_gt.add(best_idx)
            matched_ious.append(best_iou)
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    return MatchResult(tp=tp, fp=fp, fn=fn, matched_ious=matched_ious)


def evaluate_image(gt: Sequence[Sequence[float]], preds: List[Tuple[List[float], float]]) -> ImageMetrics:
    sorted_preds = sorted(preds, key=lambda item: item[1], reverse=True)
    pred_boxes = [p[0] for p in sorted_preds]

    match = match_predictions(gt, pred_boxes, IOU_DEFAULT)
    precision = match.tp / (match.tp + match.fp + EPS)
    recall = match.tp / (match.tp + match.fn + EPS)
    f1 = 2 * precision * recall / (precision + recall + EPS)

    match_50 = match_predictions(gt, pred_boxes, 0.5)
    map50 = match_50.tp / (match_50.tp + match_50.fp + EPS)
    match_75 = match_predictions(gt, pred_boxes, 0.75)
    map75 = match_75.tp / (match_75.tp + match_75.fp + EPS)

    precisions = []
    for thr in MAP_THRESHOLDS:
        result = match_predictions(gt, pred_boxes, thr)
        precision_thr = result.tp / (result.tp + result.fp + EPS)
        precisions.append(precision_thr)
    map_all = float(sum(precisions) / len(precisions)) if precisions else 0.0

    mae = abs(len(pred_boxes) - len(gt))
    rmse = float(np.sqrt((len(pred_boxes) - len(gt)) ** 2))
    avg_iou = float(sum(match.matched_ious) / len(match.matched_ious)) if match.matched_ious else 0.0

    return ImageMetrics(
        image_name="",
        precision=precision,
        recall=recall,
        f1=f1,
        map50=map50,
        map75=map75,
        map_all=map_all,
        mae=float(mae),
        rmse=rmse,
        avg_iou=avg_iou,
        pred_count=len(pred_boxes),
        gt_count=len(gt),
    )


def evaluate_fold(pred_path: Path, gt_path: Path) -> Tuple[List[ImageMetrics], ImageMetrics]:
    pred_map = load_coco_boxes(pred_path, with_scores=True)
    gt_map = load_coco_boxes(gt_path, with_scores=False)

    image_names = sorted(set(gt_map.keys()) | set(pred_map.keys()))
    results: List[ImageMetrics] = []

    precisions = []
    recalls = []
    f1s = []
    map50_list = []
    map75_list = []
    map_all_list = []
    mae_list = []
    rmse_list = []
    avg_ious = []

    for name in image_names:
        gt_boxes = [bbox for bbox, _ in gt_map.get(name, [])]
        preds = pred_map.get(name, [])
        metrics = evaluate_image(gt_boxes, preds)
        metrics.image_name = name
        results.append(metrics)

        precisions.append(metrics.precision)
        recalls.append(metrics.recall)
        f1s.append(metrics.f1)
        map50_list.append(metrics.map50)
        map75_list.append(metrics.map75)
        map_all_list.append(metrics.map_all)
        mae_list.append(metrics.mae)
        rmse_list.append(metrics.rmse)
        avg_ious.append(metrics.avg_iou)

    summary = ImageMetrics(
        image_name="__summary__",
        precision=float(sum(precisions) / len(precisions)) if precisions else 0.0,
        recall=float(sum(recalls) / len(recalls)) if recalls else 0.0,
        f1=float(sum(f1s) / len(f1s)) if f1s else 0.0,
        map50=float(sum(map50_list) / len(map50_list)) if map50_list else 0.0,
        map75=float(sum(map75_list) / len(map75_list)) if map75_list else 0.0,
        map_all=float(sum(map_all_list) / len(map_all_list)) if map_all_list else 0.0,
        mae=float(sum(mae_list) / len(mae_list)) if mae_list else 0.0,
        rmse=float(sum(rmse_list) / len(rmse_list)) if rmse_list else 0.0,
        avg_iou=float(sum(avg_ious) / len(avg_ious)) if avg_ious else 0.0,
        pred_count=sum(m.pred_count for m in results),
        gt_count=sum(m.gt_count for m in results),
    )
    return results, summary


def write_results_csv(results_root: Path, rows: List[List[str]]) -> None:
    csv_path = results_root / "results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["model", "fold", "images", "precision", "recall", "f1", "mAP", "mAP50", "mAP75", "MAE", "RMSE"]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(header)
        for row in rows:
            writer.writerow(row)
    print(f"[INFO] Metrics written to {csv_path}")


def write_details_csv(results_root: Path, model: str, fold: str, metrics: Sequence[ImageMetrics]) -> None:
    details_path = results_root / f"details_{model}_{fold}.csv"
    with details_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "image_name",
            "precision",
            "recall",
            "f1",
            "mAP",
            "mAP50",
            "mAP75",
            "MAE",
            "RMSE",
            "avg_iou",
            "pred_count",
            "gt_count",
        ])
        for metric in metrics:
            if metric.image_name == "__summary__":
                continue
            writer.writerow([
                metric.image_name,
                f"{metric.precision:.6f}",
                f"{metric.recall:.6f}",
                f"{metric.f1:.6f}",
                f"{metric.map_all:.6f}",
                f"{metric.map50:.6f}",
                f"{metric.map75:.6f}",
                f"{metric.mae:.6f}",
                f"{metric.rmse:.6f}",
                f"{metric.avg_iou:.6f}",
                metric.pred_count,
                metric.gt_count,
            ])
    print(f"[INFO] Detailed metrics written to {details_path}")


def discover_models(results_root: Path, models: Sequence[str] | None) -> List[Path]:
    reconstructed_root = results_root / "reconstructed"
    if not reconstructed_root.exists():
        raise FileNotFoundError(f"Reconstructed directory not found at {reconstructed_root}")

    if models:
        paths = []
        for name in models:
            candidate = reconstructed_root / name
            if not candidate.exists():
                raise FileNotFoundError(f"Model directory '{candidate}' not found.")
            paths.append(candidate)
        return paths

    return [p for p in reconstructed_root.iterdir() if p.is_dir()]


def discover_folds(model_dir: Path) -> List[Path]:
    folds = [p for p in model_dir.iterdir() if p.is_dir() and p.name.lower().startswith("fold")]
    folds.sort(key=lambda path: path.name)
    return folds


def fold_to_gt_path(dataset_root: Path, fold_name: str) -> Path:
    fold_clean = fold_name.replace("_", "")
    candidates = [
        dataset_root / "imagens_originais" / fold_name / "_annotations.coco.json",
        dataset_root / "imagens_originais" / fold_clean / "_annotations.coco.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Ground-truth annotations not found for fold '{fold_name}'. Checked: {candidates}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate reconstructed predictions against ground truth.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--models", nargs="*", help="Optional list of model names to evaluate.")
    args = parser.parse_args()

    model_dirs = discover_models(args.results_root, args.models)
    aggregate_rows: List[List[str]] = []

    for model_dir in model_dirs:
        model_name = model_dir.name
        fold_dirs = discover_folds(model_dir)
        if not fold_dirs:
            print(f"[WARN] No folds found for model '{model_name}'. Skipping.")
            continue

        for fold_dir in fold_dirs:
            fold_name = fold_dir.name
            pred_path = fold_dir / "_annotations.coco.json"
            if not pred_path.exists():
                print(f"[WARN] Predictions not found at {pred_path}. Skipping.")
                continue

            gt_path = fold_to_gt_path(args.dataset_root, fold_name)
            print(f"\n[INFO] Evaluating model '{model_name}' on {fold_name}")
            print(f"       Predictions: {pred_path}")
            print(f"       Ground truth: {gt_path}")

            per_image, summary = evaluate_fold(pred_path, gt_path)
            write_details_csv(args.results_root, model_name, fold_name, per_image)

            aggregate_rows.append([
                model_name,
                fold_name,
                str(len(per_image)),
                f"{summary.precision:.6f}",
                f"{summary.recall:.6f}",
                f"{summary.f1:.6f}",
                f"{summary.map_all:.6f}",
                f"{summary.map50:.6f}",
                f"{summary.map75:.6f}",
                f"{summary.mae:.6f}",
                f"{summary.rmse:.6f}",
            ])

    if aggregate_rows:
        write_results_csv(args.results_root, aggregate_rows)
    else:
        print("[WARN] No evaluation rows generated.")


if __name__ == "__main__":
    main()
