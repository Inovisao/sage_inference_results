from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from utils.csv_utils import save_csv

FOLD_RESULTS_HEADER = [
    "dataset",
    "suppression",
    "model",
    "fold",
    "split",
    "weight_path",
    "train_annotations",
    "val_annotations",
    "test_annotations",
    "images",
    "tiles",
    "precision",
    "recall",
    "f1",
    "mAP",
    "mAP50",
    "mAP75",
    "MAE",
    "RMSE",
    "model_load_time_s",
    "tile_inference_time_s",
    "reconstruction_time_s",
    "suppression_time_s",
    "evaluation_time_s",
    "total_time_s",
    "created_at",
]

IMAGE_RESULTS_HEADER = [
    "dataset",
    "suppression",
    "model",
    "fold",
    "image_name",
    "precision",
    "recall",
    "f1",
    "mAP50",
    "mAP75",
    "mAP",
    "MAE",
    "RMSE",
    "pred_count",
    "gt_count",
    "avg_iou",
]

PER_IMAGE_HEADER = [
    "image_name",
    "precision",
    "recall",
    "f1",
    "mAP50",
    "mAP75",
    "mAP",
    "MAE",
    "RMSE",
    "pred_count",
    "gt_count",
    "avg_iou",
]

SUMMARY_METRICS = [
    "precision",
    "recall",
    "f1",
    "mAP",
    "mAP50",
    "mAP75",
    "MAE",
    "RMSE",
    "model_load_time_s",
    "tile_inference_time_s",
    "reconstruction_time_s",
    "suppression_time_s",
    "evaluation_time_s",
    "total_time_s",
]

RESULTS_CSV_HEADER = [
    "dataset",
    "suppression",
    "model",
    "fold",
    "images",
    "tiles",
    "precision",
    "recall",
    "f1",
    "mAP",
    "mAP50",
    "mAP75",
    "MAE",
    "RMSE",
    "model_load_time_s",
    "tile_inference_time_s",
    "reconstruction_time_s",
    "suppression_time_s",
    "evaluation_time_s",
    "total_time_s",
    "created_at",
]


def _safe_float(value: str | object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_rows(path: Path, header: Sequence[str]) -> List[Dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []
        missing = [column for column in header if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing expected columns in {path}: {', '.join(missing)}")
        return [{column: str(row.get(column, "")) for column in header} for row in reader]


def _upsert_rows(
    path: Path,
    header: Sequence[str],
    rows: Iterable[Mapping[str, object]],
    *,
    key_columns: Sequence[str],
) -> None:
    rows_by_key: Dict[Tuple[str, ...], List[str]] = {}
    for existing in _load_rows(path, header):
        key = tuple(existing[column] for column in key_columns)
        rows_by_key[key] = [existing.get(column, "") for column in header]

    for row in rows:
        normalized = {column: str(row.get(column, "")) for column in header}
        key = tuple(normalized[column] for column in key_columns)
        rows_by_key[key] = [normalized[column] for column in header]

    save_csv(path, header, rows_by_key.values())


def write_fold_result(reports_root: Path, row: Mapping[str, object]) -> Path:
    csv_path = reports_root / "fold_results.csv"
    _upsert_rows(
        csv_path,
        FOLD_RESULTS_HEADER,
        [row],
        key_columns=("dataset", "suppression", "model", "fold"),
    )
    return csv_path


def write_image_results(reports_root: Path, rows: Iterable[Mapping[str, object]]) -> Path:
    csv_path = reports_root / "image_results.csv"
    _upsert_rows(
        csv_path,
        IMAGE_RESULTS_HEADER,
        rows,
        key_columns=("dataset", "suppression", "model", "fold", "image_name"),
    )
    return csv_path


def write_per_image_metrics_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> Path:
    normalized_rows = []
    for row in rows:
        normalized_rows.append([str(row.get(column, "")) for column in PER_IMAGE_HEADER])
    save_csv(path, PER_IMAGE_HEADER, normalized_rows)
    return path


def write_run_metadata(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _compute_statistics(values: List[float]) -> Tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    mean_val = float(statistics.mean(values))
    median_val = float(statistics.median(values))
    std_val = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    return mean_val, median_val, std_val


def _summarize_by_groups(
    rows: List[Dict[str, str]],
    group_columns: Sequence[str],
) -> List[Sequence[object]]:
    grouped: Dict[Tuple[str, ...], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        key = tuple(row.get(column, "") for column in group_columns)
        for metric in SUMMARY_METRICS:
            value = _safe_float(row.get(metric, ""))
            if value is not None:
                grouped[key][metric].append(value)

    summary_rows: List[Sequence[object]] = []
    for key, metrics in sorted(grouped.items()):
        for metric_name, values in sorted(metrics.items()):
            mean_val, median_val, std_val = _compute_statistics(values)
            summary_rows.append((*key, metric_name, f"{mean_val:.6f}", f"{median_val:.6f}", f"{std_val:.6f}"))
    return summary_rows


def write_summary_reports(reports_root: Path) -> List[Path]:
    reports_root.mkdir(parents=True, exist_ok=True)
    fold_rows = _load_rows(reports_root / "fold_results.csv", FOLD_RESULTS_HEADER)
    if not fold_rows:
        return []

    outputs: List[Path] = []
    ordered_results_rows = [
        [row.get(column, "") for column in RESULTS_CSV_HEADER]
        for row in sorted(
            fold_rows,
            key=lambda row: (
                row.get("dataset", ""),
                row.get("suppression", ""),
                row.get("model", ""),
                row.get("fold", ""),
            ),
        )
    ]
    results_csv = reports_root / "results.csv"
    save_csv(results_csv, RESULTS_CSV_HEADER, ordered_results_rows)
    outputs.append(results_csv)

    parent_results_csv = reports_root.parent / "results.csv"
    save_csv(parent_results_csv, RESULTS_CSV_HEADER, ordered_results_rows)
    outputs.append(parent_results_csv)

    by_model = reports_root / "summary_by_model.csv"
    save_csv(
        by_model,
        ["model", "metric", "mean", "median", "std"],
        _summarize_by_groups(fold_rows, ("model",)),
    )
    outputs.append(by_model)

    by_suppression = reports_root / "summary_by_suppression.csv"
    save_csv(
        by_suppression,
        ["suppression", "metric", "mean", "median", "std"],
        _summarize_by_groups(fold_rows, ("suppression",)),
    )
    outputs.append(by_suppression)

    by_model_suppression = reports_root / "summary_by_model_suppression.csv"
    save_csv(
        by_model_suppression,
        ["model", "suppression", "metric", "mean", "median", "std"],
        _summarize_by_groups(fold_rows, ("model", "suppression")),
    )
    outputs.append(by_model_suppression)

    return outputs
