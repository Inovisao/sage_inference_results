#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2

from debug_single_image import (
    DEFAULT_SUPPRESSION,
    _discover_tiles_for_image,
    _draw_detections,
    _load_ground_truth_boxes,
    _project_detections,
)
from pipeline.coco_utils import (
    build_image_lookup_by_stem,
    extract_original_images,
    load_coco_json,
)
from pipeline.data_prep import build_tile_index
from pipeline.detectors import resolve_detector
from pipeline.reconstruction import (
    apply_suppression,
    detect_tile_orientation,
    remap_detections_by_rotation,
)
from pipeline.types import DetectionRecord


def _select_samples(
    names: Sequence[str],
    *,
    limit: int,
    rng: random.Random,
) -> List[str]:
    if len(names) <= limit:
        return list(names)
    return rng.sample(list(names), limit)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate debug visualisations (detections + ground-truth boxes) "
        "for a subset of images per fold."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--tiles-root", type=Path, default=None,
                        help="Optional override for tiles root directory.")
    parser.add_argument("--folds", nargs="*", help="Specific folds to process (e.g. fold_1 fold_2).")
    parser.add_argument("--images-per-fold", type=int, default=5)
    parser.add_argument("--model", type=str, default="yolov8",
                        help="Detector name (yolov8, yolov11, yolov5_tph, retinanet, faster, ...).")
    parser.add_argument("--weight-template", type=str,
                        default="model_checkpoints/{fold}/YOLOV8/best.pt",
                        help="Template for locating weights (use {fold} placeholder).")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--output-root", type=Path, default=Path("results/batch_debug"))
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for sampling images per fold.")
    parser.add_argument("--class-id-offset", type=int, default=0,
                        help="Optional category id offset to apply to detections.")
    args = parser.parse_args()

    dataset_root = args.dataset_root
    tiles_root = args.tiles_root or (dataset_root / "tiles")
    train_coco_path = dataset_root / "train" / "_annotations.coco.json"
    if not train_coco_path.exists():
        raise FileNotFoundError(f"Training COCO annotations not found at '{train_coco_path}'.")

    train_coco = load_coco_json(train_coco_path)
    original_images = extract_original_images(train_coco)
    originals_by_stem = build_image_lookup_by_stem(original_images)

    available_folds = []
    for path in sorted(tiles_root.iterdir()):
        if path.is_dir():
            available_folds.append(path.name)

    if args.folds:
        requested = set(args.folds)
        folds = [name for name in available_folds if name in requested]
        missing = requested - set(folds)
        for missing_fold in sorted(missing):
            print(f"[WARN] Requested fold '{missing_fold}' not found under '{tiles_root}'.")
    else:
        folds = available_folds

    if not folds:
        print("[WARN] No folds to process. Exiting.")
        return

    rng = random.Random(args.seed)

    categories = train_coco.get("categories", [])
    if categories:
        max_id = max(int(cat["id"]) for cat in categories)
        inferred_num_classes = max_id + 1
    else:
        inferred_num_classes = 2

    for fold_name in folds:
        fold_dir = tiles_root / fold_name
        test_dir = fold_dir / "test"
        if not test_dir.exists():
            print(f"[WARN] Skipping {fold_name}: test directory '{test_dir}' not found.")
            continue

        weight_path = Path(args.weight_template.format(fold=fold_name))
        if not weight_path.exists():
            print(f"[WARN] Skipping {fold_name}: weight file '{weight_path}' not found.")
            continue

        try:
            _, original_to_tiles = build_tile_index(test_dir, originals_by_stem)
        except Exception as exc:
            print(f"[WARN] Unable to build tile index for {fold_name}: {exc}")
            continue

        original_names = sorted(original_to_tiles.keys())
        if not original_names:
            print(f"[WARN] No original images mapped for {fold_name}; skipping.")
            continue

        samples = _select_samples(original_names, limit=args.images_per_fold, rng=rng)
        detector_cls = resolve_detector(args.model)
        extra_kwargs = {"class_id_offset": args.class_id_offset}

        if detector_cls.model_name in {"faster", "fasterrcnn"}:
            extra_kwargs["num_classes"] = inferred_num_classes

        print(f"[INFO] Processing {fold_name} with weights '{weight_path}'.")
        detector = detector_cls(weight_path, **extra_kwargs)
        try:
            for idx, image_name in enumerate(samples, 1):
                print(f"    [{idx}/{len(samples)}] Fold {fold_name}: {image_name}")
                try:
                    tiles = _discover_tiles_for_image(test_dir, image_name)
                except FileNotFoundError as exc:
                    print(f"        [WARN] {exc}")
                    continue

                if not tiles:
                    print(f"        [WARN] No tiles discovered for '{image_name}'. Skipping.")
                    continue

                aggregated: List[DetectionRecord] = []
                for tile_path, offset_x, offset_y in tiles:
                    tile_img = cv2.imread(str(tile_path))
                    if tile_img is None:
                        print(f"        [WARN] Unable to read tile '{tile_path}'. Skipping tile.")
                        continue
                    detections = detector.predict(tile_img, args.threshold)
                    aggregated.extend(_project_detections(detections, offset_x, offset_y))

                original_image_path = dataset_root / "train" / image_name
                original_image = cv2.imread(str(original_image_path))
                if original_image is None:
                    print(f"        [WARN] Unable to read original image '{original_image_path}'. Skipping image.")
                    continue

                angle = detect_tile_orientation(
                    original_image,
                    tile_path=tiles[0][0],
                    offset_x=tiles[0][1],
                    offset_y=tiles[0][2],
                )

                suppressed = apply_suppression(
                    aggregated,
                    image_width=original_image.shape[1],
                    image_height=original_image.shape[0],
                    params=DEFAULT_SUPPRESSION,
                )
                suppressed = remap_detections_by_rotation(
                    suppressed,
                    angle,
                    image_width=original_image.shape[1],
                    image_height=original_image.shape[0],
                )

                ground_truth_boxes = _load_ground_truth_boxes(dataset_root, image_name)

                output_dir = args.output_root / fold_name
                output_name = f"{Path(image_name).stem}_debug.png"
                output_path = output_dir / output_name
                _draw_detections(
                    original_image_path,
                    suppressed,
                    ground_truth_boxes,
                    output_path,
                    score_threshold=args.threshold,
                )
        finally:
            detector.close()


if __name__ == "__main__":
    main()
