from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple

from utils.csv_utils import save_csv

DEFAULT_MODELS_ROOT = Path("pesos")
DEFAULT_TILES_ROOT = Path("dataset") / "tiles"
DEFAULT_OUTPUT_ROOT = Path("results") / "model_split_exports"

WEIGHT_SUFFIXES = {".pt", ".pth", ".onnx"}
TILE_SUFFIX_PATTERN = re.compile(r"_tile_\d+_\d+\.[^.]+$")
FOLD_PATTERN = re.compile(r"fold[_\-]?(\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export train/val/test split membership for each model discovered under "
            "the weights directory."
        )
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=DEFAULT_MODELS_ROOT,
        help="Root directory containing fold_*/<model>/... weight files.",
    )
    parser.add_argument(
        "--tiles-root",
        type=Path,
        default=DEFAULT_TILES_ROOT,
        help="Root directory containing dataset/tiles/fold_*/<split>/_annotations.coco.json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where the per-model CSV files will be written.",
    )
    return parser.parse_args()


def _fold_sort_key(name: str) -> Tuple[int, str]:
    match = FOLD_PATTERN.search(name)
    return (int(match.group(1)), name) if match else (10**9, name)


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


def _collect_split_index(tiles_root: Path) -> Dict[str, Dict[str, Dict[str, object]]]:
    if not tiles_root.exists():
        raise FileNotFoundError(f"Tiles root not found: {tiles_root}")

    split_index: Dict[str, Dict[str, Dict[str, object]]] = {}
    fold_dirs = sorted(
        [path for path in tiles_root.iterdir() if path.is_dir() and FOLD_PATTERN.match(path.name)],
        key=lambda path: _fold_sort_key(path.name),
    )
    if not fold_dirs:
        raise RuntimeError(f"No fold directories found under {tiles_root}")

    for fold_dir in fold_dirs:
        split_index[fold_dir.name] = {}
        for split in ("train", "val", "test"):
            annotations_path = fold_dir / split / "_annotations.coco.json"
            if not annotations_path.exists():
                raise FileNotFoundError(f"Missing annotations file: {annotations_path}")
            tile_count, original_names = _load_original_names(annotations_path)
            split_index[fold_dir.name][split] = {
                "annotations_path": annotations_path,
                "tile_count": tile_count,
                "original_names": original_names,
                "original_set": set(original_names),
            }
    return split_index


def _preferred_weight(weights: Sequence[Path]) -> Path:
    for candidate in weights:
        if candidate.stem.lower() == "best":
            return candidate
    for candidate in weights:
        if "best" in candidate.stem.lower():
            return candidate
    return weights[0]


def _discover_model_weights(models_root: Path) -> Dict[str, Dict[str, List[Path]]]:
    if not models_root.exists():
        raise FileNotFoundError(f"Models root not found: {models_root}")

    discovered: DefaultDict[str, DefaultDict[str, List[Path]]] = defaultdict(lambda: defaultdict(list))

    for fold_dir in sorted(
        [path for path in models_root.iterdir() if path.is_dir() and FOLD_PATTERN.match(path.name)],
        key=lambda path: _fold_sort_key(path.name),
    ):
        for model_dir in sorted([path for path in fold_dir.iterdir() if path.is_dir()], key=lambda path: path.name.lower()):
            weights = sorted(
                [
                    path
                    for path in model_dir.rglob("*")
                    if path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES
                ]
            )
            if weights:
                discovered[model_dir.name][fold_dir.name].extend(weights)

    return {model: dict(folds) for model, folds in discovered.items()}


def _sanitize_model_name(model_name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name.strip())
    return sanitized or "model"


def _summary_rows_for_model(
    model_name: str,
    fold_to_weights: Dict[str, List[Path]],
    split_index: Dict[str, Dict[str, Dict[str, object]]],
) -> List[Sequence[object]]:
    rows: List[Sequence[object]] = []
    for fold_name in sorted(fold_to_weights.keys(), key=_fold_sort_key):
        if fold_name not in split_index:
            continue
        weights = sorted(fold_to_weights[fold_name])
        preferred = _preferred_weight(weights)
        split_data = split_index[fold_name]
        train_set = split_data["train"]["original_set"]
        val_set = split_data["val"]["original_set"]
        test_set = split_data["test"]["original_set"]
        rows.append(
            (
                model_name,
                fold_name,
                str(preferred),
                len(weights),
                "test",
                split_data["train"]["tile_count"],
                len(split_data["train"]["original_names"]),
                split_data["val"]["tile_count"],
                len(split_data["val"]["original_names"]),
                split_data["test"]["tile_count"],
                len(split_data["test"]["original_names"]),
                len(train_set & val_set),
                len(train_set & test_set),
                len(val_set & test_set),
                str(split_data["train"]["annotations_path"]),
                str(split_data["val"]["annotations_path"]),
                str(split_data["test"]["annotations_path"]),
                " | ".join(str(path) for path in weights),
            )
        )
    return rows


def _detail_rows_for_model(
    model_name: str,
    fold_to_weights: Dict[str, List[Path]],
    split_index: Dict[str, Dict[str, Dict[str, object]]],
) -> List[Sequence[object]]:
    rows: List[Sequence[object]] = []
    for fold_name in sorted(fold_to_weights.keys(), key=_fold_sort_key):
        if fold_name not in split_index:
            continue
        preferred = _preferred_weight(sorted(fold_to_weights[fold_name]))
        for split in ("train", "val", "test"):
            for original_name in split_index[fold_name][split]["original_names"]:
                rows.append((model_name, fold_name, str(preferred), split, original_name))
    return rows


def _write_model_exports(
    output_root: Path,
    model_name: str,
    summary_rows: Iterable[Sequence[object]],
    detail_rows: Iterable[Sequence[object]],
) -> Tuple[Path, Path]:
    model_dir = output_root / _sanitize_model_name(model_name)
    summary_path = model_dir / "summary.csv"
    details_path = model_dir / "originals.csv"

    save_csv(
        summary_path,
        [
            "model",
            "fold",
            "preferred_weight",
            "weight_file_count",
            "recommended_inference_split",
            "train_tile_count",
            "train_original_count",
            "val_tile_count",
            "val_original_count",
            "test_tile_count",
            "test_original_count",
            "overlap_train_val",
            "overlap_train_test",
            "overlap_val_test",
            "train_annotations",
            "val_annotations",
            "test_annotations",
            "available_weights",
        ],
        summary_rows,
    )
    save_csv(
        details_path,
        ["model", "fold", "preferred_weight", "split", "original_image"],
        detail_rows,
    )
    return summary_path, details_path


def main() -> None:
    args = parse_args()
    split_index = _collect_split_index(args.tiles_root)
    model_weights = _discover_model_weights(args.models_root)

    if not model_weights:
        raise RuntimeError(f"No weight files were found under {args.models_root}")

    combined_summary_rows: List[Sequence[object]] = []
    combined_detail_rows: List[Sequence[object]] = []

    for model_name in sorted(model_weights.keys(), key=str.lower):
        summary_rows = _summary_rows_for_model(model_name, model_weights[model_name], split_index)
        detail_rows = _detail_rows_for_model(model_name, model_weights[model_name], split_index)
        summary_path, details_path = _write_model_exports(args.output_root, model_name, summary_rows, detail_rows)
        combined_summary_rows.extend(summary_rows)
        combined_detail_rows.extend(detail_rows)
        print(f"[INFO] Wrote {summary_path}")
        print(f"[INFO] Wrote {details_path}")

    save_csv(
        args.output_root / "all_models_summary.csv",
        [
            "model",
            "fold",
            "preferred_weight",
            "weight_file_count",
            "recommended_inference_split",
            "train_tile_count",
            "train_original_count",
            "val_tile_count",
            "val_original_count",
            "test_tile_count",
            "test_original_count",
            "overlap_train_val",
            "overlap_train_test",
            "overlap_val_test",
            "train_annotations",
            "val_annotations",
            "test_annotations",
            "available_weights",
        ],
        combined_summary_rows,
    )
    save_csv(
        args.output_root / "all_models_originals.csv",
        ["model", "fold", "preferred_weight", "split", "original_image"],
        combined_detail_rows,
    )
    print(f"[INFO] Wrote {args.output_root / 'all_models_summary.csv'}")
    print(f"[INFO] Wrote {args.output_root / 'all_models_originals.csv'}")


if __name__ == "__main__":
    main()
