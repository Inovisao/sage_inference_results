from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence

import cv2

from .coco_utils import build_image_lookup_by_stem, extract_original_images, load_coco_json, save_coco_json
from .data_prep import build_tile_index, discover_fold_directories, prepare_original_test_split
from .detectors import BaseDetector, resolve_detector
from .reconstruction import build_prediction_dataset, build_raw_detection_dataset
from .types import DetectionRecord, ModelWeights, SuppressionParams

_WEIGHT_SUFFIXES = {".pt", ".pth", ".onnx"}
_FOLD_REGEX = re.compile(r"fold[_\-]?(\d+)", re.IGNORECASE)


@dataclass
class PipelineSettings:
    dataset_root: Path
    models_root: Path
    results_root: Path
    originals_root: Path
    suppression: SuppressionParams = field(default_factory=SuppressionParams)
    create_mosaics: bool = False
    detection_thresholds: Mapping[str, float] = field(default_factory=dict)
    model_class_offsets: Mapping[str, int] = field(default_factory=dict)


class SageInferencePipeline:
    """Coordinate inference, reconstruction, and evaluation for each fold."""

    def __init__(self, settings: PipelineSettings):
        self.settings = settings
        self.dataset_root = settings.dataset_root
        self.models_root = settings.models_root
        self.results_root = settings.results_root
        self.originals_root = settings.originals_root
        self.suppression = settings.suppression
        self.create_mosaics = settings.create_mosaics
        self.detection_thresholds = {k.lower(): v for k, v in settings.detection_thresholds.items()}
        self.model_class_offsets = {k.lower(): v for k, v in settings.model_class_offsets.items()}

        # Defaults for known detectors
        self.detection_thresholds.setdefault("yolov8", 0.25)
        self.detection_thresholds.setdefault("yolov5_tph", 0.25)
        self.detection_thresholds.setdefault("faster", 0.5)
        self.detection_thresholds.setdefault("fasterrcnn", 0.5)
        self.detection_thresholds.setdefault("retinanet", 0.3)
        self.detection_thresholds.setdefault("yolov11", 0.25)

        self.model_class_offsets.setdefault("yolov8", 1)
        self.model_class_offsets.setdefault("faster", 0)
        self.model_class_offsets.setdefault("fasterrcnn", 0)
        self.model_class_offsets.setdefault("retinanet", 0)
        self.model_class_offsets.setdefault("yolov11", 0)
        self.model_class_offsets.setdefault("yolov5_tph", 0)

        self.train_coco_path = self.dataset_root / "train" / "_annotations.coco.json"
        self.train_images_dir = self.dataset_root / "train"
        self.tiles_root = self.dataset_root / "tiles"

        if not self.train_coco_path.exists():
            raise FileNotFoundError(f"Ground-truth COCO not found at {self.train_coco_path}")
        if not self.tiles_root.exists():
            raise FileNotFoundError(f"Tiles root not found at {self.tiles_root}")

        self.train_coco = load_coco_json(self.train_coco_path)
        self.original_images = extract_original_images(self.train_coco)
        self.original_images_by_stem = build_image_lookup_by_stem(self.original_images)

        self.categories = self.train_coco.get("categories", [])
        self.num_classes = self._infer_num_classes(self.categories)

    @staticmethod
    def _infer_num_classes(categories: Sequence[Mapping[str, object]]) -> int:
        if not categories:
            return 1
        ids = [int(cat["id"]) for cat in categories]
        return max(ids) + 1

    def _discover_model_weights(self) -> List[ModelWeights]:
        specs_by_name: Dict[str, ModelWeights] = {}
        if not self.models_root.exists():
            print(f"[WARN] Models directory '{self.models_root}' not found. Skipping inference.")
            return []

        for entry in sorted(self.models_root.iterdir()):
            if not entry.is_dir():
                continue

            fold_match = _FOLD_REGEX.match(entry.name)
            if fold_match:
                fold_idx = int(fold_match.group(1))
                for model_dir in sorted(entry.iterdir()):
                    if not model_dir.is_dir():
                        continue
                    spec = specs_by_name.setdefault(model_dir.name, ModelWeights(name=model_dir.name))
                    for weight_path in sorted(model_dir.rglob("*")):
                        if not weight_path.is_file() or weight_path.suffix.lower() not in _WEIGHT_SUFFIXES:
                            continue
                        if fold_idx in spec.fold_to_path:
                            existing = spec.fold_to_path[fold_idx]
                            print(
                                f"[WARN] Model '{model_dir.name}' already has weight for fold {fold_idx}. "
                                f"Keeping '{existing}' and skipping '{weight_path}'."
                            )
                            continue
                        spec.fold_to_path[fold_idx] = weight_path
                continue

            spec = specs_by_name.setdefault(entry.name, ModelWeights(name=entry.name))
            for weight_path in sorted(entry.rglob("*")):
                if not weight_path.is_file() or weight_path.suffix.lower() not in _WEIGHT_SUFFIXES:
                    continue
                fold_idx = self._extract_fold_index(weight_path, entry)
                if fold_idx is None:
                    continue
                if fold_idx in spec.fold_to_path:
                    existing = spec.fold_to_path[fold_idx]
                    print(
                        f"[WARN] Model '{entry.name}' already has weight for fold {fold_idx}. "
                        f"Keeping '{existing}' and skipping '{weight_path}'."
                    )
                    continue
                spec.fold_to_path[fold_idx] = weight_path

        specs = [spec for spec in specs_by_name.values() if spec.fold_to_path]
        specs.sort(key=lambda item: item.name.lower())
        return specs

    @staticmethod
    def _extract_fold_index(weight_path: Path, model_dir: Path) -> Optional[int]:
        match = _FOLD_REGEX.search(weight_path.stem)
        if match:
            return int(match.group(1))
        current = weight_path.parent
        while model_dir in current.parents:
            match = _FOLD_REGEX.search(current.name)
            if match:
                return int(match.group(1))
            current = current.parent
        return None

    def _instantiate_detector(self, model_name: str, weight_path: Path) -> BaseDetector:
        detector_cls = resolve_detector(model_name)
        threshold = self.detection_thresholds.get(model_name.lower(), detector_cls.default_threshold)
        class_offset = self.model_class_offsets.get(model_name.lower(), 0)
        extra_kwargs = {}
        if detector_cls.model_name in {"faster", "fasterrcnn"}:
            extra_kwargs["num_classes"] = self.num_classes
        detector = detector_cls(weight_path, class_id_offset=class_offset, **extra_kwargs)
        detector.threshold = threshold  # convenience attribute
        return detector

    def run(self) -> None:
        folds = discover_fold_directories(self.tiles_root)
        if not folds:
            print(f"[WARN] No folds discovered under '{self.tiles_root}'.")
            return

        model_specs = self._discover_model_weights()
        if not model_specs:
            print("[WARN] No model weights discovered. Run will only prepare original test splits.")

        for fold_dir in folds:
            match = _FOLD_REGEX.match(fold_dir.name)
            if not match:
                continue
            fold_idx = int(match.group(1))
            print(f"\n[INFO] Processing {fold_dir.name} (fold {fold_idx})")

            test_dir = fold_dir / "test"
            tile_index, original_to_tiles = build_tile_index(test_dir, self.original_images_by_stem)

            total_tiles = len(tile_index)
            originals_output_dir = self.originals_root / f"fold{fold_idx}"
            annotations_path = prepare_original_test_split(
                self.train_coco,
                original_to_tiles,
                output_dir=originals_output_dir,
                source_images_dir=self.train_images_dir,
            )
            filtered_coco = load_coco_json(annotations_path)

            if not model_specs:
                continue

            for spec in model_specs:
                weight_path = spec.get(fold_idx)
                if weight_path is None:
                    print(f"[WARN] Model '{spec.name}' has no weight for fold {fold_idx}. Skipping.")
                    continue

                print(f"[INFO]  +- Running model '{spec.name}' with weights '{weight_path.name}'")
                start_time = time.time()
                try:
                    detector = self._instantiate_detector(spec.name, weight_path)
                except Exception as exc:
                    print(f"[ERROR] Failed to instantiate detector '{spec.name}': {exc}")
                    continue

                tile_predictions: Optional[MutableMapping[str, Sequence[DetectionRecord]]] = {}
                try:
                    with detector:
                        for tile_idx, (tile_name, metadata) in enumerate(sorted(tile_index.items()), 1):
                            print(
                                f"        [fold {fold_idx}][{spec.name}] starting tile "
                                f"{tile_idx}/{total_tiles}: {tile_name}"
                            )
                            image = cv2.imread(str(metadata.path))
                            if image is None:
                                raise FileNotFoundError(f"Unable to read tile image '{metadata.path}'.")
                            threshold = self.detection_thresholds.get(spec.name.lower(), detector.threshold)
                            try:
                                detections = detector.predict(image, threshold)
                            except ImportError as exc:
                                print(
                                    f"[ERROR] Missing dependency while running '{spec.name}' on fold {fold_idx}: {exc}"
                                )
                                tile_predictions = None
                                print(
                                    f"        [fold {fold_idx}][{spec.name}] dependency missing; aborting model"
                                )
                                break
                            tile_predictions[tile_name] = detections
                            print(
                                f"        [fold {fold_idx}][{spec.name}] finished {tile_name} "
                                f"with {len(detections)} detections"
                            )
                            if tile_idx % 100 == 0 or tile_idx == total_tiles:
                                print(
                                    f"        [fold {fold_idx}][{spec.name}] "
                                    f"{tile_idx}/{total_tiles} tiles processed"
                                )
                finally:
                    detector.close()

                if tile_predictions is None:
                    print(
                        f"[WARN] Skipping model '{spec.name}' on fold {fold_idx} due to unresolved dependencies."
                    )
                    continue

                reconstructed_dir = self.results_root / "reconstructed" / spec.name / f"fold{fold_idx}"
                images_dir = reconstructed_dir / "images"

                raw_dataset, projected_detections, image_meta_by_id = build_raw_detection_dataset(
                    fold_original_to_tiles=original_to_tiles,
                    tile_predictions=tile_predictions,
                    original_images=self.original_images,
                    base_coco=filtered_coco,
                )
                raw_output = reconstructed_dir / "_detections.coco.json"
                save_coco_json(raw_dataset, raw_output)
                print(
                    f"[INFO]  +- Stored raw detections for '{spec.name}' fold {fold_idx} at {raw_output}"
                )

                dataset = build_prediction_dataset(
                    fold_original_to_tiles=original_to_tiles,
                    tile_predictions=tile_predictions,
                    suppression=self.suppression,
                    original_images=self.original_images,
                    base_coco=filtered_coco,
                    output_images_dir=images_dir,
                    create_mosaics=self.create_mosaics,
                    projected_detections=projected_detections,
                    image_meta_by_id=image_meta_by_id,
                )
                annotations_output = reconstructed_dir / "_annotations.coco.json"
                save_coco_json(dataset, annotations_output)
                elapsed = time.time() - start_time
                print(
                    f"[INFO]  +- Completed model '{spec.name}' on fold {fold_idx} "
                    f"in {elapsed:.1f}s. Saved to {annotations_output}"
                )

        print("\n[DONE] Pipeline execution completed.")
