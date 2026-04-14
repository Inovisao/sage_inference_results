from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Dict, List

import cv2

from calcula_estatisticas.evaluate_reconstructed import evaluate_results_root
from pipeline import PipelineSettings, SageInferencePipeline
from pipeline.coco_utils import load_coco_json, save_coco_json
from pipeline.data_prep import build_tile_index, prepare_original_test_split
from pipeline.reconstruction import build_prediction_dataset
from pipeline.types import DetectionRecord, SuppressionParams
from run_all_yolov8_suppressions import (
    DATASET_CONFIGS,
    build_complete_configuration_row,
    load_existing_complete_rows,
    prepare_dataset_root,
    write_complete_configurations_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "yolov8_random_single"
_FOLD_REGEX = re.compile(r"fold[_\-]?(\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOV8 on the tiles of a single random image from one fold.",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_CONFIGS.keys()),
        default="asahi",
        help="Dataset configuration to use.",
    )
    parser.add_argument(
        "--fold",
        default="fold_1",
        help="Fold identifier (for example: fold_1 or fold1).",
    )
    parser.add_argument(
        "--suppression",
        default="cluster_diou_ait",
        help="Suppression method applied after reprojection.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Base directory for the single-image run outputs.",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for image selection.",
    )
    parser.add_argument(
        "--create-mosaics",
        action="store_true",
        help="Generate the reconstructed mosaic image for the selected original.",
    )
    return parser.parse_args()


def _normalize_fold_name(name: str) -> str:
    return name.lower().replace("_", "").replace("-", "")


def _extract_fold_index(name: str) -> int:
    match = _FOLD_REGEX.search(name)
    if not match:
        raise ValueError(f"Unable to extract fold index from '{name}'.")
    return int(match.group(1))


def _select_fold_dir(tiles_root: Path, requested_fold: str) -> Path:
    normalized = _normalize_fold_name(requested_fold)
    for candidate in sorted(p for p in tiles_root.iterdir() if p.is_dir()):
        if _normalize_fold_name(candidate.name) == normalized:
            return candidate
    raise FileNotFoundError(f"Fold '{requested_fold}' not found under '{tiles_root}'.")


def _select_yolov8_weight(pipeline: SageInferencePipeline, fold_idx: int) -> tuple[str, Path]:
    for spec in pipeline._discover_model_weights():
        if spec.name.lower() != "yolov8":
            continue
        weight_path = spec.get(fold_idx)
        if weight_path is None:
            break
        return spec.name, weight_path
    raise FileNotFoundError(f"YOLOV8 weights for fold {fold_idx} were not found.")


def main() -> None:
    args = parse_args()
    dataset_config = DATASET_CONFIGS[args.dataset]
    fold_idx = _extract_fold_index(args.fold)
    run_root = args.output_root / args.dataset / args.suppression / f"fold{fold_idx}"
    dataset_root = prepare_dataset_root(args.dataset, run_root)
    results_root = run_root / "results"
    originals_root = dataset_root / "imagens_originais"

    settings = PipelineSettings(
        dataset_root=dataset_root,
        models_root=dataset_config["models_root"],
        results_root=results_root,
        originals_root=originals_root,
        create_mosaics=args.create_mosaics,
        suppression=SuppressionParams(method=args.suppression),
        detection_thresholds={"yolov8": args.confidence},
        model_class_offsets={"yolov8": args.class_offset},
        enabled_models=("yolov8",),
    )
    pipeline = SageInferencePipeline(settings)

    fold_dir = _select_fold_dir(pipeline.tiles_root, args.fold)
    test_dir = fold_dir / "test"
    tile_index, original_to_tiles = build_tile_index(test_dir, pipeline.original_images_by_stem)
    if not original_to_tiles:
        raise RuntimeError(f"No original images found in '{test_dir}'.")

    selected_image = random.Random(args.seed).choice(sorted(original_to_tiles.keys()))
    selected_tiles = original_to_tiles[selected_image]
    selected_original_to_tiles = {selected_image: selected_tiles}
    print(f"[INFO] Selected image '{selected_image}' with {len(selected_tiles)} tiles.")

    originals_output_dir = originals_root / f"fold{fold_idx}"
    annotations_path = prepare_original_test_split(
        pipeline.train_coco,
        selected_original_to_tiles,
        output_dir=originals_output_dir,
        source_images_dir=pipeline.train_images_dir,
    )
    filtered_coco = load_coco_json(annotations_path)
    orientation_by_image = pipeline._detect_tile_orientations(selected_original_to_tiles, originals_output_dir)
    model_name, weight_path = _select_yolov8_weight(pipeline, fold_idx)

    tile_predictions: Dict[str, List[DetectionRecord]] = {}
    detector = pipeline._instantiate_detector(model_name, weight_path)
    try:
        with detector:
            for tile_idx, tile in enumerate(selected_tiles, 1):
                image = cv2.imread(str(tile.path))
                if image is None:
                    raise FileNotFoundError(f"Unable to read tile image '{tile.path}'.")
                detections = detector.predict(image, args.confidence)
                tile_predictions[tile.file_name] = list(detections)
                print(
                    f"[INFO] Tile {tile_idx}/{len(selected_tiles)} '{tile.file_name}' -> "
                    f"{len(detections)} detections"
                )
    finally:
        detector.close()

    reconstructed_dir = results_root / "reconstructed" / model_name / f"fold{fold_idx}"
    dataset = build_prediction_dataset(
        fold_original_to_tiles=selected_original_to_tiles,
        tile_predictions=tile_predictions,
        suppression=settings.suppression,
        original_images=pipeline.original_images,
        base_coco=filtered_coco,
        output_images_dir=reconstructed_dir / "images",
        create_mosaics=args.create_mosaics,
        orientation_by_image=orientation_by_image,
    )
    annotations_output = reconstructed_dir / "_annotations.coco.json"
    save_coco_json(dataset, annotations_output)
    print(f"[INFO] Single-image reconstruction saved to {annotations_output}")

    aggregate_rows = evaluate_results_root(dataset_root, results_root, ("yolov8",))
    complete_rows_by_key = load_existing_complete_rows(args.output_root)
    for row in aggregate_rows:
        key, complete_row = build_complete_configuration_row(args.dataset, args.suppression, row)
        complete_rows_by_key[key] = complete_row
    if complete_rows_by_key:
        ordered_rows = [row for _, row in sorted(complete_rows_by_key.items())]
        write_complete_configurations_csv(args.output_root, ordered_rows)

    print(f"[DONE] Single-image test finished for '{selected_image}'.")


if __name__ == "__main__":
    main()
