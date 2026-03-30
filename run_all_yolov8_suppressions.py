from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cv2

from calcula_estatisticas.evaluate_reconstructed import evaluate_results_root
from pipeline import PipelineSettings, SageInferencePipeline
from pipeline.coco_utils import build_image_lookup_by_stem, extract_original_images, load_coco_json
from pipeline.data_prep import build_tile_index, parse_tile_filename
from pipeline.types import SuppressionParams
from utils.csv_utils import save_csv


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "yolov8_all_suppressions"
COMPLETE_CONFIGS_CSV = "configuracoes_completas.csv"
COMPLETE_CONFIGS_HEADER = [
    "slicing",
    "method",
    "model",
    "fold",
    "images",
    "precision",
    "recall",
    "f1",
    "fscore",
    "mAP",
    "mAP50",
    "mAP75",
    "MAE",
    "RMSE",
    "r",
]
ALL_SUPPRESSIONS: Sequence[str] = (
    "cluster_diou_ait",
    "nms",
    "nms_ioa",
    "bws",
    "cluster_diou_nms",
    "cluster_diou_bws",
)

DATASET_CONFIGS: Dict[str, Dict[str, Path]] = {
    "asahi": {
        "train_root": PROJECT_ROOT / "datasets" / "all",
        "tiles_root": PROJECT_ROOT / "datasets" / "methods" / "asahi",
        "models_root": PROJECT_ROOT / "pesos" / "asahi" / "experimento_p_adaptativo" / "model_checkpoints",
    },
    "sage": {
        "train_root": PROJECT_ROOT / "datasets" / "all",
        "tiles_root": PROJECT_ROOT / "datasets" / "methods" / "sage",
        "models_root": PROJECT_ROOT / "pesos" / "pesos_teste_sage" / "model_checkpoints",
    },
    "sahi": {
        "train_root": PROJECT_ROOT / "datasets" / "all",
        "tiles_root": PROJECT_ROOT / "datasets" / "methods" / "sahi",
        "models_root": PROJECT_ROOT / "pesos" / "sahi" / "model_checkpoints",
    },
    "slicing_common": {
        "train_root": PROJECT_ROOT / "datasets" / "all",
        "tiles_root": PROJECT_ROOT / "datasets" / "methods" / "slicing_common",
        "models_root": PROJECT_ROOT / "pesos" / "slicing_common",
    },
}

ALIGNMENT_DIFF_THRESHOLD = 5.0
MIN_ALIGNMENT_SAMPLES = 20
MIN_ALIGNMENT_RATIO = 0.9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOV8 across all configured datasets and suppression methods.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=sorted(DATASET_CONFIGS.keys()),
        default=list(DATASET_CONFIGS.keys()),
        help="Datasets to process. Defaults to all configured datasets.",
    )
    parser.add_argument(
        "--suppressions",
        nargs="*",
        default=list(ALL_SUPPRESSIONS),
        help="Suppression methods to test. Defaults to all supported methods.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where per-dataset and per-suppression outputs will be written.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confidence threshold used by YOLOV8.",
    )
    parser.add_argument(
        "--class-offset",
        type=int,
        default=1,
        help="Class index offset applied to YOLOV8 detections.",
    )
    parser.add_argument(
        "--create-mosaics",
        action="store_true",
        help="Generate reconstructed mosaic images.",
    )
    parser.add_argument("--tau-0", type=float, default=0.5, help="Base adaptive DIoU threshold.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Adaptive threshold sensitivity.")
    parser.add_argument("--tau-min", type=float, default=0.3, help="Minimum adaptive DIoU threshold.")
    parser.add_argument("--tau-dup", type=float, default=0.7, help="Duplicate DIoU threshold.")
    parser.add_argument("--gamma", type=float, default=0.5, help="Minimum score ratio for duplicates.")
    parser.add_argument("--topk", type=int, default=5, help="Number of nearest neighbors for local density.")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable checkpoint reuse and force regeneration of reconstructed predictions.",
    )
    parser.add_argument(
        "--append-complete-csv",
        action="store_true",
        help=(
            "Preserve the current row order in configuracoes_completas.csv and append any new "
            "dataset/suppression/fold rows at the end instead of sorting the whole file."
        ),
    )
    return parser.parse_args()


def ensure_symlink(link_path: Path, target_path: Path) -> None:
    target_path = target_path.resolve()
    link_path.parent.mkdir(parents=True, exist_ok=True)

    if link_path.is_symlink():
        if link_path.resolve() == target_path:
            return
        link_path.unlink()
    elif link_path.exists():
        raise FileExistsError(f"Path '{link_path}' already exists and is not a symlink.")

    link_path.symlink_to(target_path, target_is_directory=True)


def prepare_dataset_root(dataset_name: str, dataset_output_root: Path) -> Path:
    config = DATASET_CONFIGS[dataset_name]
    dataset_root = dataset_output_root / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)

    ensure_symlink(dataset_root / "train", config["train_root"])
    ensure_symlink(dataset_root / "tiles", config["tiles_root"])
    (dataset_root / "imagens_originais").mkdir(parents=True, exist_ok=True)
    return dataset_root


def run_pipeline(
    dataset_name: str,
    suppression_method: str,
    output_root: Path,
    confidence: float,
    class_offset: int,
    create_mosaics: bool,
    resume: bool,
    allowed_folds: Sequence[int] | None,
    suppression_extra: Dict[str, float],
) -> tuple[Path, Path]:
    config = DATASET_CONFIGS[dataset_name]
    run_root = output_root / dataset_name / suppression_method
    dataset_root = prepare_dataset_root(dataset_name, run_root)
    results_root = run_root / "results"
    originals_root = dataset_root / "imagens_originais"

    settings = PipelineSettings(
        dataset_root=dataset_root,
        models_root=config["models_root"],
        results_root=results_root,
        originals_root=originals_root,
        create_mosaics=create_mosaics,
        suppression=SuppressionParams(method=suppression_method, extra=suppression_extra),
        detection_thresholds={"yolov8": confidence},
        model_class_offsets={"yolov8": class_offset},
        enabled_models=("yolov8",),
        skip_existing_predictions=resume,
        allowed_folds=allowed_folds,
    )
    SageInferencePipeline(settings).run()
    return dataset_root, results_root


def write_complete_configurations_csv(output_root: Path, rows: Sequence[Sequence[str]]) -> None:
    csv_path = output_root / COMPLETE_CONFIGS_CSV
    save_csv(
        csv_path,
        COMPLETE_CONFIGS_HEADER,
        rows,
    )
    print(f"[INFO] Consolidated metrics written to {csv_path}")


def build_complete_configuration_row(
    dataset_name: str,
    suppression_method: str,
    row: Sequence[str],
) -> tuple[tuple[str, str, str, str], List[str]]:
    model_name = str(row[0]).strip().upper()
    fold_name = str(row[1]).strip()
    key = (dataset_name, suppression_method, model_name, fold_name)
    complete_row = [
        dataset_name,
        suppression_method,
        model_name,
        fold_name,
        str(row[2]),
        str(row[3]),
        str(row[4]),
        str(row[5]),
        str(row[5]),
        str(row[6]),
        str(row[7]),
        str(row[8]),
        str(row[9]),
        str(row[10]),
        "",
    ]
    return key, complete_row


def load_existing_complete_rows(output_root: Path) -> Dict[tuple[str, str, str, str], List[str]]:
    csv_path = output_root / COMPLETE_CONFIGS_CSV
    if not csv_path.exists():
        return {}

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return {}

        rows_by_key: Dict[tuple[str, str, str, str], List[str]] = {}
        for row in reader:
            dataset = (row.get("slicing") or "").strip()
            suppression = (row.get("method") or "").strip()
            model = (row.get("model") or "").strip()
            fold = (row.get("fold") or "").strip()
            if not dataset or not suppression or not model or not fold:
                continue
            rows_by_key[(dataset, suppression, model, fold)] = [
                *[row.get(column, "") for column in COMPLETE_CONFIGS_HEADER],
            ]
    return rows_by_key


def _fold_alignment_score(
    dataset_name: str,
    fold_dir: Path,
    original_lookup: Dict[str, object],
) -> tuple[int, int, float | None]:
    if dataset_name != "slicing_common":
        return 0, 0, None

    annotations_path = fold_dir / "test" / "_annotations.coco.json"
    if not annotations_path.exists():
        return 0, 0, None

    coco = load_coco_json(annotations_path)
    top_left_tiles: Dict[str, str] = {}
    for image_entry in coco.get("images", []):
        file_name = str(image_entry["file_name"])
        stem, offset_x, offset_y = parse_tile_filename(file_name)
        if offset_x == 0 and offset_y == 0 and stem not in top_left_tiles:
            top_left_tiles[stem] = file_name

    if not top_left_tiles:
        return 0, 0, None

    matched = 0
    checked = 0
    for stem, tile_name in top_left_tiles.items():
        original = original_lookup.get(stem)
        if original is None:
            continue

        tile_path = fold_dir / "test" / tile_name
        original_path = DATASET_CONFIGS[dataset_name]["train_root"] / original.file_name
        tile = cv2.imread(str(tile_path), cv2.IMREAD_GRAYSCALE)
        original_image = cv2.imread(str(original_path), cv2.IMREAD_GRAYSCALE)
        if tile is None or original_image is None:
            continue

        tile_small = cv2.resize(tile, (64, 64), interpolation=cv2.INTER_AREA)
        patch = original_image[: tile.shape[0], : tile.shape[1]]
        if patch.shape != tile.shape:
            continue
        patch_small = cv2.resize(patch, (64, 64), interpolation=cv2.INTER_AREA)
        diff = float(cv2.absdiff(tile_small, patch_small).mean())
        checked += 1
        if diff <= ALIGNMENT_DIFF_THRESHOLD:
            matched += 1

    ratio = (matched / checked) if checked else None
    return checked, matched, ratio


def filter_rows_for_dataset(
    rows_by_key: Dict[tuple[str, str, str, str], List[str]],
    dataset_name: str,
    valid_folds: Sequence[int],
) -> Dict[tuple[str, str, str, str], List[str]]:
    valid_fold_names = {f"fold{fold}" for fold in valid_folds}
    filtered: Dict[tuple[str, str, str, str], List[str]] = {}
    removed = 0
    for key, row in rows_by_key.items():
        row_dataset = key[0]
        row_fold = key[3]
        if row_dataset == dataset_name and row_fold not in valid_fold_names:
            removed += 1
            continue
        filtered[key] = row

    if removed:
        print(
            f"[INFO] Removed {removed} stale row(s) for dataset '{dataset_name}' "
            f"from {COMPLETE_CONFIGS_CSV} because the fold is no longer considered valid."
        )
    return filtered


def validate_paths(dataset_names: Iterable[str]) -> None:
    missing_paths: List[Path] = []
    for dataset_name in dataset_names:
        config = DATASET_CONFIGS[dataset_name]
        missing_paths.extend(
            path for path in (config["train_root"], config["tiles_root"], config["models_root"]) if not path.exists()
        )

    if missing_paths:
        missing = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(f"Required paths were not found:\n{missing}")


def discover_valid_folds(dataset_name: str) -> List[int]:
    config = DATASET_CONFIGS[dataset_name]
    train_coco = load_coco_json(config["train_root"] / "_annotations.coco.json")
    original_lookup = build_image_lookup_by_stem(extract_original_images(train_coco))
    valid_folds: List[int] = []

    for fold_dir in sorted(p for p in config["tiles_root"].iterdir() if p.is_dir() and p.name.lower().startswith("fold")):
        test_dir = fold_dir / "test"
        try:
            build_tile_index(test_dir, original_lookup)
        except Exception as exc:
            print(f"[WARN] Skipping {dataset_name}/{fold_dir.name} during preflight validation: {exc}")
            continue

        checked, matched, ratio = _fold_alignment_score(dataset_name, fold_dir, original_lookup)
        if ratio is not None and checked >= MIN_ALIGNMENT_SAMPLES and ratio < MIN_ALIGNMENT_RATIO:
            print(
                f"[WARN] Skipping {dataset_name}/{fold_dir.name} during preflight validation: "
                f"only {matched}/{checked} top-left tiles matched the claimed original image "
                f"(ratio={ratio:.3f}). This fold appears to have mislabeled tiles."
            )
            continue

        digits = "".join(ch for ch in fold_dir.name if ch.isdigit())
        if digits:
            valid_folds.append(int(digits))

    return valid_folds


def build_suppression_extra(args: argparse.Namespace) -> Dict[str, float]:
    return {
        "tau_0": float(args.tau_0),
        "alpha": float(args.alpha),
        "tau_min": float(args.tau_min),
        "tau_dup": float(args.tau_dup),
        "gamma": float(args.gamma),
        "k": float(args.topk),
    }


def main() -> None:
    args = parse_args()
    validate_paths(args.datasets)
    complete_rows_by_key = load_existing_complete_rows(args.output_root)
    suppression_extra = build_suppression_extra(args)

    for dataset_name in args.datasets:
        valid_folds = discover_valid_folds(dataset_name)
        complete_rows_by_key = filter_rows_for_dataset(complete_rows_by_key, dataset_name, valid_folds)
        if not valid_folds:
            print(f"[WARN] No valid folds found for dataset '{dataset_name}'. Skipping dataset.")
            if args.append_complete_csv:
                ordered_rows = list(complete_rows_by_key.values())
            else:
                ordered_rows = [row for _, row in sorted(complete_rows_by_key.items())]
            write_complete_configurations_csv(args.output_root, ordered_rows)
            continue
        for suppression_method in args.suppressions:
            print(f"\n[INFO] Dataset='{dataset_name}' Suppression='{suppression_method}'")
            dataset_root, results_root = run_pipeline(
                dataset_name=dataset_name,
                suppression_method=suppression_method,
                output_root=args.output_root,
                confidence=args.confidence,
                class_offset=args.class_offset,
                create_mosaics=args.create_mosaics,
                resume=not args.no_resume,
                allowed_folds=valid_folds,
                suppression_extra=suppression_extra,
            )
            aggregate_rows = evaluate_results_root(dataset_root, results_root, ("yolov8",))
            for row in aggregate_rows:
                key, complete_row = build_complete_configuration_row(
                    dataset_name,
                    suppression_method,
                    row,
                )
                if args.append_complete_csv and key in complete_rows_by_key:
                    del complete_rows_by_key[key]
                complete_rows_by_key[key] = complete_row
            if args.append_complete_csv:
                ordered_rows = list(complete_rows_by_key.values())
            else:
                ordered_rows = [row for _, row in sorted(complete_rows_by_key.items())]
            write_complete_configurations_csv(args.output_root, ordered_rows)
            print(f"[DONE] Results available at {results_root}")

    if not complete_rows_by_key:
        print(f"[WARN] No rows available to write '{COMPLETE_CONFIGS_CSV}'.")


if __name__ == "__main__":
    main()
