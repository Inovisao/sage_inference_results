from __future__ import annotations

from pathlib import Path

from pipeline.reporting import write_summary_reports

REPORTS_ROOT = Path("results") / "reports"
FOLD_RESULTS_CSV = REPORTS_ROOT / "fold_results.csv"


def main() -> None:
    if not FOLD_RESULTS_CSV.exists():
        raise FileNotFoundError(f"Consolidated fold results not found at {FOLD_RESULTS_CSV}")

    summary_paths = write_summary_reports(REPORTS_ROOT)
    if not summary_paths:
        raise RuntimeError("No summary reports were generated.")

    print("[INFO] Summary reports regenerated from consolidated fold results:")
    for path in summary_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
