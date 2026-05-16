from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

from utils.csv_utils import save_csv

DEFAULT_DATASET_ROOT = Path("dataset") / "tiles"
DEFAULT_OUTPUT_ROOT = Path("results")
SUMMARY_CSV_NAME = "tile_split_summary.csv"
DETAILS_CSV_NAME = "tile_split_originals.csv"

TILE_SUFFIX_PATTERN = re.compile(r"_tile_\d+_\d+\.[^.]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export train/val/test split membership from COCO annotations stored under "
            "dataset/tiles/fold_*/<split>/_annotations.coco.json."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory containing fold_* subdirectories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where the CSV files will be written.",
    )
    return parser.parse_args()


def _load_original_names(coco_path: Path) -> Tuple[int, List[str]]:
    data = json.loads(coco_path.read_text(encoding="utf-8"))
    images = data.get("images", [])
    original_names = sorted(
        {
            TILE_SUFFIX_PATTERN.sub("", str(image.get("file_name", "")).strip())
            for image in images
            if image.get("file_name")
        }
    )
    return len(images), original_names


def _fold_sort_key(path: Path) -> Tuple[int, str]:
    match = re.search(r"(\d+)", path.name)
    return (int(match.group(1)), path.name) if match else (10**9, path.name)


def collect_fold_data(dataset_root: Path) -> Tuple[List[Sequence[object]], List[Sequence[object]]]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    summary_rows: List[Sequence[object]] = []
    details_rows: List[Sequence[object]] = []
    split_sets_by_fold: Dict[str, Dict[str, Set[str]]] = {}

    fold_dirs = sorted(
        [path for path in dataset_root.iterdir() if path.is_dir() and path.name.startswith("fold_")],
        key=_fold_sort_key,
    )
    if not fold_dirs:
        raise RuntimeError(f"No fold directories found under {dataset_root}")

    for fold_dir in fold_dirs:
        split_sets_by_fold[fold_dir.name] = {}
        for split in ("train", "val", "test"):
            coco_path = fold_dir / split / "_annotations.coco.json"
            if not coco_path.exists():
                raise FileNotFoundError(f"Missing annotations file: {coco_path}")

            tile_count, original_names = _load_original_names(coco_path)
            original_set = set(original_names)
            split_sets_by_fold[fold_dir.name][split] = original_set

            summary_rows.append((fold_dir.name, split, tile_count, len(original_names)))
            details_rows.extend((fold_dir.name, split, original_name) for original_name in original_names)

    for fold_name, split_sets in split_sets_by_fold.items():
        train_set = split_sets["train"]
        val_set = split_sets["val"]
        test_set = split_sets["test"]
        summary_rows.extend(
            [
                (fold_name, "overlap_train_val", "", len(train_set & val_set)),
                (fold_name, "overlap_train_test", "", len(train_set & test_set)),
                (fold_name, "overlap_val_test", "", len(val_set & test_set)),
            ]
        )

    return summary_rows, details_rows


def main() -> None:
    args = parse_args()

    summary_rows, details_rows = collect_fold_data(args.dataset_root)

    summary_path = args.output_root / SUMMARY_CSV_NAME
    details_path = args.output_root / DETAILS_CSV_NAME

    save_csv(
        summary_path,
        ["fold", "split", "tile_count", "original_image_count"],
        summary_rows,
    )
    save_csv(
        details_path,
        ["fold", "split", "original_image"],
        details_rows,
    )

    print(f"[INFO] Summary written to {summary_path}")
    print(f"[INFO] Original image listing written to {details_path}")


if __name__ == "__main__":
    main()
