from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from utils.csv_utils import save_csv

RESULTS_CSV = Path("results") / "results.csv"
OUTPUT_CSV = Path("results") / "fold_statistics.csv"

# Metrics to aggregate across folds. Extend the list if results.csv gains more columns.
METRIC_COLUMNS = [
    "precision",
    "recall",
    "f1",
    "mAP",
    "mAP50",
    "mAP75",
    "MAE",
    "RMSE",
]


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_results(path: Path) -> Dict[str, Dict[str, List[float]]]:
    """Load per-fold metrics grouped by model name."""
    if not path.exists():
        raise FileNotFoundError(f"Results CSV not found at {path}")

    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [col for col in METRIC_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing expected columns in results file: {', '.join(missing)}")

        for row in reader:
            model = row.get("model")
            if not model:
                continue
            for metric in METRIC_COLUMNS:
                value = _safe_float(row.get(metric, ""))
                if value is not None:
                    grouped[model][metric].append(value)

    return grouped


def compute_statistics(values: List[float]) -> Tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")

    mean_val = float(statistics.mean(values))
    median_val = float(statistics.median(values))
    std_val = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    return mean_val, median_val, std_val


def main() -> None:
    grouped = load_results(RESULTS_CSV)
    statistics_map: Dict[str, Dict[str, Tuple[float, float, float]]] = {}

    for model, metrics in grouped.items():
        statistics_map[model] = {}
        for metric_name, values in metrics.items():
            statistics_map[model][metric_name] = compute_statistics(values)

    if not statistics_map:
        raise RuntimeError("No metrics found to aggregate.")

    rows = []
    for model, metrics in sorted(statistics_map.items()):
        for metric_name, values in sorted(metrics.items()):
            mean_val, median_val, std_val = values
            rows.append(
                (
                    model,
                    metric_name,
                    f"{mean_val:.6f}",
                    f"{median_val:.6f}",
                    f"{std_val:.6f}",
                )
            )

    save_csv(OUTPUT_CSV, ["model", "metric", "mean", "median", "std"], rows)

    for model, metrics in sorted(statistics_map.items()):
        print(f"\nModel: {model}")
        for metric_name, (mean_val, median_val, std_val) in sorted(metrics.items()):
            print(
                f"  {metric_name:>8s} | mean={mean_val:.6f}  median={median_val:.6f}  std={std_val:.6f}"
            )
    print(f"\n[INFO] Fold statistics written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
