from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd


RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = RESULTS_DIR / "anova_all_results.txt"
OUTPUT_FILE.write_text("", encoding="utf-8")
ANOVA_CSV = RESULTS_DIR / "anova_summary.csv"
TUKEY_CSV = RESULTS_DIR / "anova_tukey.csv"


def plot_metric_boxplots(results_df: pd.DataFrame, metrics: Iterable[str]) -> None:
    ncols = 3
    metrics = list(metrics)
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        sns.boxplot(
            data=results_df,
            x="model",
            y=metric,
            hue="model",
            palette="Purples",
            ax=axes[idx],
        )
        axes[idx].set_title(f"Boxplot for {metric}")
        axes[idx].set_xlabel("Models")
        axes[idx].set_ylabel(metric)
        axes[idx].grid(True, linestyle="--", alpha=0.4)
        legend = axes[idx].get_legend()
        if legend is not None:
            legend.remove()

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "boxplot.png", dpi=300)
    plt.close(fig)


def safe_pearsonr(a: np.ndarray, b: np.ndarray) -> float:
    """Return Pearson's r or NaN when undefined."""
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.allclose(a, a.mean()) or np.allclose(b, b.mean()):
        return float("nan")
    try:
        value, _ = pearsonr(a, b)
        return float(value)
    except Exception:
        return float("nan")


def evaluate_counting_models(counting_df: pd.DataFrame) -> None:
    rmse_records = []

    for model_name, subset in counting_df.groupby("ml"):
        gt = subset["groundtruth"].to_numpy(dtype=float)
        pred = subset["predicted"].to_numpy(dtype=float)

        rmse = float(np.sqrt(mean_squared_error(gt, pred)))
        mae = float(mean_absolute_error(gt, pred))
        mape = float(np.mean(np.abs((gt - pred) / np.clip(gt, 1e-8, None))))
        r = safe_pearsonr(gt, pred)

        title = f"{model_name} RMSE={rmse:.3f} MAE={mae:.3f} MAPE={mape:.3f} r={r:.3f}"
        rmse_records.append({"Model": model_name, "RMSE": rmse})

        plt.figure()
        sns.regplot(x=gt, y=pred, scatter_kws={"s": 15})
        plt.title(title)
        plt.xlabel("Manual Counting (Ground Truth)")
        plt.ylabel("Predicted Counting")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.savefig(RESULTS_DIR / f"{model_name}_counting.png", dpi=300)
        plt.close()

    pd.DataFrame(rmse_records).to_csv(RESULTS_DIR / "rmse_values.csv", index=False)


def run_anova(results_df: pd.DataFrame, metric: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Execute one-way ANOVA followed by Tukey HSD when significant."""
    tukey_df: Optional[pd.DataFrame] = None
    anova_table: Optional[pd.DataFrame] = None
    with OUTPUT_FILE.open("a", encoding="utf-8") as handle:
        handle.write("\n------------------------------------------------------------\n")
        handle.write(f"ANOVA for {metric}\n")

        try:
            model = ols(f"{metric} ~ C(model)", data=results_df).fit()
            anova_table = anova_lm(model, typ=2)
            handle.write(f"{anova_table}\n")

            p_value = float(anova_table.loc["C(model)", "PR(>F)"])
            if p_value < 0.05:
                handle.write(f"\nTukey HSD for {metric}\n")
                tukey = pairwise_tukeyhsd(results_df[metric], results_df["model"], alpha=0.05)
                summary = tukey.summary()
                handle.write(f"{summary}\n")
                tukey_df = pd.DataFrame(summary.data[1:], columns=summary.data[0])
        except Exception as exc:
            handle.write(f"Failed to run ANOVA for {metric}: {exc}\n")
            anova_table = None
            tukey_df = None

    return anova_table, tukey_df


def main() -> None:
    results_path = RESULTS_DIR / "results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found at {results_path}")
    results_df = pd.read_csv(results_path)

    counting_path = RESULTS_DIR / "counting.csv"
    counting_df = None
    if counting_path.exists():
        counting_df = pd.read_csv(counting_path)
    else:
        print(f"[WARN] Skipping counting analysis. File not found: {counting_path}")

    metrics = ["mAP50", "mAP75", "mAP", "precision", "recall", "f1", "MAE", "RMSE"]
    plot_metric_boxplots(results_df, metrics)
    if counting_df is not None:
        evaluate_counting_models(counting_df)

    anova_metrics = ["mAP", "mAP50", "mAP75", "MAE", "RMSE", "precision", "recall", "f1"]
    anova_tables: List[pd.DataFrame] = []
    tukey_tables: List[pd.DataFrame] = []
    for metric in anova_metrics:
        anova_table, tukey_df = run_anova(results_df, metric)
        if anova_table is not None:
            table = anova_table.reset_index().rename(columns={"index": "source"})
            table.insert(0, "metric", metric)
            anova_tables.append(table)
        if tukey_df is not None:
            tukey_df.insert(0, "metric", metric)
            tukey_tables.append(tukey_df)

    if anova_tables:
        pd.concat(anova_tables, ignore_index=True).to_csv(ANOVA_CSV, index=False)
    if tukey_tables:
        pd.concat(tukey_tables, ignore_index=True).to_csv(TUKEY_CSV, index=False)

    print(f"\nDone. Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
