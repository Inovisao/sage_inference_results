from __future__ import annotations

from datetime import datetime
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence

import cv2

from calcula_estatisticas.evaluate_reconstructed import evaluate_fold
from .coco_utils import build_image_lookup_by_stem, extract_original_images, load_coco_json, save_coco_json
from .data_prep import build_tile_index, discover_fold_directories, prepare_original_test_split
from .detectors import BaseDetector, resolve_detector
from .reconstruction import build_prediction_dataset, detect_tile_orientation, render_visualizations_from_dataset
from .reporting import write_fold_result, write_image_results, write_per_image_metrics_csv, write_run_metadata, write_summary_reports
from .types import DetectionRecord, ModelWeights, OriginalToTiles, SuppressionParams

_WEIGHT_SUFFIXES = {".pt", ".pth", ".onnx"}
_FOLD_REGEX = re.compile(r"fold[_\-]?(\d+)", re.IGNORECASE)


@dataclass
class PipelineSettings:
    dataset_root: Path
    models_root: Path
    results_root: Path
    originals_root: Path
    source_images_root: Optional[Path] = None
    reports_root: Optional[Path] = None
    suppression: SuppressionParams = field(default_factory=SuppressionParams)
    create_mosaics: bool = False
    detection_thresholds: Mapping[str, float] = field(default_factory=dict)
    model_class_offsets: Mapping[str, int] = field(default_factory=dict)
    enabled_models: Optional[Sequence[str]] = None
    detector_name_aliases: Mapping[str, str] = field(default_factory=dict)
    model_num_classes: Mapping[str, int] = field(default_factory=dict)
    skip_existing_predictions: bool = True
    allowed_folds: Optional[Sequence[int]] = None
    dataset_name: str = "dataset"


class SageInferencePipeline:
    """Coordinate inference, reconstruction, and evaluation for each fold."""

    def __init__(self, settings: PipelineSettings):
        self.settings = settings
        self.dataset_root = settings.dataset_root
        self.models_root = settings.models_root
        self.results_root = settings.results_root
        self.originals_root = settings.originals_root
        self.reports_root = settings.reports_root or (self.results_root / "reports")
        self.suppression = settings.suppression
        self.dataset_name = settings.dataset_name
        self.create_mosaics = settings.create_mosaics
        self.detection_thresholds = {k.lower(): v for k, v in settings.detection_thresholds.items()}
        self.model_class_offsets = {k.lower(): v for k, v in settings.model_class_offsets.items()}
        self.model_num_classes = {k.lower(): v for k, v in settings.model_num_classes.items()}
        self.skip_existing_predictions = settings.skip_existing_predictions
        self.allowed_folds = set(settings.allowed_folds) if settings.allowed_folds is not None else None
        self.enabled_models = None
        if settings.enabled_models:
            self.enabled_models = {name.lower() for name in settings.enabled_models}
        self.detector_aliases = {k.lower(): v.lower() for k, v in settings.detector_name_aliases.items()}

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

        self.train_images_dir = settings.source_images_root or (self.dataset_root / "all")
        self.train_coco_path = self.train_images_dir / "_annotations.coco.json"
        if not self.train_coco_path.exists():
            fallback_images_dir = self.dataset_root / "train"
            fallback_coco_path = fallback_images_dir / "_annotations.coco.json"
            if fallback_coco_path.exists():
                self.train_images_dir = fallback_images_dir
                self.train_coco_path = fallback_coco_path
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
    def _prefer_weight(existing: Path, candidate: Path) -> Path:
        existing_name = existing.name.lower()
        candidate_name = candidate.name.lower()
        if "best" in candidate_name and "best" not in existing_name:
            return candidate
        if candidate_name == existing_name:
            return min(existing, candidate, key=lambda path: str(path))
        return existing

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
                direct_weight_paths = [
                    path
                    for path in sorted(entry.iterdir())
                    if path.is_file() and path.suffix.lower() in _WEIGHT_SUFFIXES
                ]

                # Support layouts like models_root/fold_1/best.pt when there is a single
                # enabled model (for example YOLOv8-only runs).
                if direct_weight_paths and self.enabled_models and len(self.enabled_models) == 1:
                    model_name = next(iter(self.enabled_models))
                    spec = specs_by_name.setdefault(model_name, ModelWeights(name=model_name))
                    selected = sorted(direct_weight_paths, key=lambda path: ("best" not in path.name.lower(), str(path)))[0]
                    existing = spec.fold_to_path.get(fold_idx)
                    if existing is None:
                        spec.fold_to_path[fold_idx] = selected
                    else:
                        spec.fold_to_path[fold_idx] = self._prefer_weight(existing, selected)

                for model_dir in sorted(entry.iterdir()):
                    if not model_dir.is_dir():
                        continue
                    spec = specs_by_name.setdefault(model_dir.name, ModelWeights(name=model_dir.name))
                    for weight_path in sorted(model_dir.rglob("*")):
                        if not weight_path.is_file() or weight_path.suffix.lower() not in _WEIGHT_SUFFIXES:
                            continue
                        existing = spec.fold_to_path.get(fold_idx)
                        spec.fold_to_path[fold_idx] = (
                            weight_path if existing is None else self._prefer_weight(existing, weight_path)
                        )
                continue

            spec = specs_by_name.setdefault(entry.name, ModelWeights(name=entry.name))
            for weight_path in sorted(entry.rglob("*")):
                if not weight_path.is_file() or weight_path.suffix.lower() not in _WEIGHT_SUFFIXES:
                    continue
                fold_idx = self._extract_fold_index(weight_path, entry)
                if fold_idx is None:
                    continue
                existing = spec.fold_to_path.get(fold_idx)
                spec.fold_to_path[fold_idx] = (
                    weight_path if existing is None else self._prefer_weight(existing, weight_path)
                )

        specs = [spec for spec in specs_by_name.values() if spec.fold_to_path]
        specs.sort(key=lambda item: item.name.lower())

        if self.enabled_models is not None:
            filtered: List[ModelWeights] = []
            discovered = {spec.name.lower() for spec in specs}
            missing = sorted(self.enabled_models - discovered)
            for spec in specs:
                if spec.name.lower() in self.enabled_models:
                    filtered.append(spec)
            for name in missing:
                print(f"[WARN] Enabled model '{name}' not found under '{self.models_root}'.")
            specs = filtered

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
        lookup_name = self.detector_aliases.get(model_name.lower(), model_name)
        detector_cls = resolve_detector(lookup_name)
        lookup_key = lookup_name.lower()
        model_key = model_name.lower()
        threshold = (
            self.detection_thresholds.get(model_key)
            or self.detection_thresholds.get(lookup_key, detector_cls.default_threshold)
        )
        class_offset = self.model_class_offsets.get(model_key, self.model_class_offsets.get(lookup_key, 0))
        extra_kwargs = {}
        if detector_cls.model_name in {"faster", "fasterrcnn"}:
            num_classes = (
                self.model_num_classes.get(model_key)
                or self.model_num_classes.get(lookup_key)
                or self.num_classes
            )
            extra_kwargs["num_classes"] = num_classes
        detector = detector_cls(weight_path, class_id_offset=class_offset, **extra_kwargs)
        detector.threshold = threshold  # convenience attribute
        return detector

    def _detect_tile_orientations(
        self,
        original_to_tiles: OriginalToTiles,
        originals_output_dir: Path,
    ) -> Dict[str, int]:
        """
        Detect rotation mismatches between stored originals and their tiles.
        Returns a mapping of original image names to the required rotation (currently only 180°).
        """

        orientation: Dict[str, int] = {}
        for original_name, tiles in original_to_tiles.items():
            if not tiles:
                continue

            original_path = originals_output_dir / original_name
            original_image = cv2.imread(str(original_path))
            if original_image is None:
                fallback_path = self.train_images_dir / original_name
                original_image = cv2.imread(str(fallback_path))

            if original_image is None:
                print(
                    f"[WARN] Unable to read original image '{original_name}' for orientation detection "
                    f"(looked in '{original_path}' and dataset train folder)."
                )
                continue

            sample_tile = tiles[0]
            try:
                angle = detect_tile_orientation(
                    original_image,
                    tile_path=sample_tile.path,
                    offset_x=sample_tile.offset_x,
                    offset_y=sample_tile.offset_y,
                )
            except Exception as exc:
                print(f"[WARN] Orientation detection failed for '{original_name}': {exc}")
                continue

            if angle:
                orientation[original_name] = angle
                print(f"[INFO]     Detected {angle}° rotation gap for '{original_name}'.")

        return orientation

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
            if self.allowed_folds is not None and fold_idx not in self.allowed_folds:
                print(f"[INFO] Skipping {fold_dir.name} because it was filtered out during preflight validation.")
                continue
            print(f"\n[INFO] Processing {fold_dir.name} (fold {fold_idx})")
            try:
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
                train_annotations = self.tiles_root / fold_dir.name / "train" / "_annotations.coco.json"
                val_annotations = self.tiles_root / fold_dir.name / "val" / "_annotations.coco.json"
                test_annotations = self.tiles_root / fold_dir.name / "test" / "_annotations.coco.json"

                if not model_specs:
                    continue

                orientation_by_image = self._detect_tile_orientations(original_to_tiles, originals_output_dir)
                if orientation_by_image:
                    print(
                        f"[INFO]  +- Rotation compensation needed for {len(orientation_by_image)} original images."
                    )

                for spec in model_specs:
                    weight_path = spec.get(fold_idx)
                    if weight_path is None:
                        print(f"[WARN] Model '{spec.name}' has no weight for fold {fold_idx}. Skipping.")
                        continue

                    suppression_name = str(self.suppression.method).lower().replace("-", "_")
                    reconstructed_dir = self.results_root / "reconstructed" / suppression_name / spec.name / f"fold{fold_idx}"
                    annotations_output = reconstructed_dir / "_annotations.coco.json"
                    metadata_output = reconstructed_dir / "run_metadata.json"
                    metrics_output = reconstructed_dir / "per_image_metrics.csv"
                    created_at = datetime.now().isoformat(timespec="seconds")
                    timings = {
                        "model_load_time_s": 0.0,
                        "tile_inference_time_s": 0.0,
                        "reconstruction_time_s": 0.0,
                        "suppression_time_s": 0.0,
                        "evaluation_time_s": 0.0,
                        "total_time_s": 0.0,
                    }
                    if self.skip_existing_predictions and annotations_output.exists():
                        print(
                            f"[INFO]  +- Checkpoint found for model '{spec.name}' on fold {fold_idx}. "
                            f"Skipping inference and reusing {annotations_output}"
                        )
                        reused_dataset = load_coco_json(annotations_output)
                        render_visualizations_from_dataset(
                            prediction_dataset=reused_dataset,
                            fold_original_to_tiles=original_to_tiles,
                            original_images=self.original_images,
                            output_images_dir=reconstructed_dir / "images",
                            source_images_dir=self.train_images_dir,
                            create_mosaics=self.create_mosaics,
                        )
                        if metadata_output.exists():
                            try:
                                existing_metadata = json.loads(metadata_output.read_text(encoding="utf-8"))
                                existing_timings = existing_metadata.get("timings", {})
                                for key in timings:
                                    timings[key] = float(existing_timings.get(key, 0.0))
                                created_at = str(existing_metadata.get("created_at", created_at))
                            except (OSError, ValueError, TypeError):
                                pass
                    else:
                        print(f"[INFO]  +- Running model '{spec.name}' with weights '{weight_path.name}'")
                        total_start = time.perf_counter()
                        load_start = time.perf_counter()
                        try:
                            detector = self._instantiate_detector(spec.name, weight_path)
                        except Exception as exc:
                            print(f"[ERROR] Failed to instantiate detector '{spec.name}': {exc}")
                            continue
                        timings["model_load_time_s"] = time.perf_counter() - load_start

                        tile_predictions: Optional[MutableMapping[str, Sequence[DetectionRecord]]] = {}
                        inference_start = time.perf_counter()
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
                        timings["tile_inference_time_s"] = time.perf_counter() - inference_start

                        if tile_predictions is None:
                            print(
                                f"[WARN] Skipping model '{spec.name}' on fold {fold_idx} due to unresolved dependencies."
                            )
                            continue

                        images_dir = reconstructed_dir / "images"
                        dataset, reconstruction_stats = build_prediction_dataset(
                            fold_original_to_tiles=original_to_tiles,
                            tile_predictions=tile_predictions,
                            suppression=self.suppression,
                            original_images=self.original_images,
                            base_coco=filtered_coco,
                            output_images_dir=images_dir,
                            source_images_dir=self.train_images_dir,
                            create_mosaics=self.create_mosaics,
                            orientation_by_image=orientation_by_image,
                        )
                        save_coco_json(dataset, annotations_output)
                        timings["suppression_time_s"] = reconstruction_stats.suppression_time_s
                        timings["reconstruction_time_s"] = max(
                            0.0,
                            reconstruction_stats.total_time_s - reconstruction_stats.suppression_time_s,
                        )
                        timings["total_time_s"] = time.perf_counter() - total_start
                        print(
                            f"[INFO]  +- Completed model '{spec.name}' on fold {fold_idx} "
                            f"in {timings['total_time_s']:.1f}s. Saved to {annotations_output}"
                        )

                    eval_start = time.perf_counter()
                    per_image, summary = evaluate_fold(annotations_output, annotations_path)
                    timings["evaluation_time_s"] = time.perf_counter() - eval_start
                    if timings["total_time_s"] == 0.0:
                        timings["total_time_s"] = (
                            timings["model_load_time_s"]
                            + timings["tile_inference_time_s"]
                            + timings["reconstruction_time_s"]
                            + timings["suppression_time_s"]
                            + timings["evaluation_time_s"]
                        )

                    per_image_rows = []
                    for metric in per_image:
                        if metric.image_name == "__summary__":
                            continue
                        per_image_rows.append(
                            {
                                "dataset": self.dataset_name,
                                "suppression": suppression_name,
                                "model": spec.name,
                                "fold": f"fold_{fold_idx}",
                                "image_name": metric.image_name,
                                "precision": f"{metric.precision:.6f}",
                                "recall": f"{metric.recall:.6f}",
                                "f1": f"{metric.f1:.6f}",
                                "mAP50": f"{metric.map50:.6f}",
                                "mAP75": f"{metric.map75:.6f}",
                                "mAP": f"{metric.map_all:.6f}",
                                "MAE": f"{metric.mae:.6f}",
                                "RMSE": f"{metric.rmse:.6f}",
                                "pred_count": metric.pred_count,
                                "gt_count": metric.gt_count,
                                "avg_iou": f"{metric.avg_iou:.6f}",
                            }
                        )

                    reports_root = self.reports_root
                    reports_root.mkdir(parents=True, exist_ok=True)
                    write_fold_result(
                        reports_root,
                        {
                            "dataset": self.dataset_name,
                            "suppression": suppression_name,
                            "model": spec.name,
                            "fold": f"fold_{fold_idx}",
                            "split": "test",
                            "weight_path": str(weight_path),
                            "train_annotations": str(train_annotations),
                            "val_annotations": str(val_annotations),
                            "test_annotations": str(test_annotations),
                            "images": len(per_image_rows),
                            "tiles": total_tiles,
                            "precision": f"{summary.precision:.6f}",
                            "recall": f"{summary.recall:.6f}",
                            "f1": f"{summary.f1:.6f}",
                            "mAP": f"{summary.map_all:.6f}",
                            "mAP50": f"{summary.map50:.6f}",
                            "mAP75": f"{summary.map75:.6f}",
                            "MAE": f"{summary.mae:.6f}",
                            "RMSE": f"{summary.rmse:.6f}",
                            "model_load_time_s": f"{timings['model_load_time_s']:.6f}",
                            "tile_inference_time_s": f"{timings['tile_inference_time_s']:.6f}",
                            "reconstruction_time_s": f"{timings['reconstruction_time_s']:.6f}",
                            "suppression_time_s": f"{timings['suppression_time_s']:.6f}",
                            "evaluation_time_s": f"{timings['evaluation_time_s']:.6f}",
                            "total_time_s": f"{timings['total_time_s']:.6f}",
                            "created_at": created_at,
                        },
                    )
                    write_image_results(reports_root, per_image_rows)
                    write_per_image_metrics_csv(
                        metrics_output,
                        [
                            {
                                "image_name": row["image_name"],
                                "precision": row["precision"],
                                "recall": row["recall"],
                                "f1": row["f1"],
                                "mAP50": row["mAP50"],
                                "mAP75": row["mAP75"],
                                "mAP": row["mAP"],
                                "MAE": row["MAE"],
                                "RMSE": row["RMSE"],
                                "pred_count": row["pred_count"],
                                "gt_count": row["gt_count"],
                                "avg_iou": row["avg_iou"],
                            }
                            for row in per_image_rows
                        ],
                    )
                    write_run_metadata(
                        metadata_output,
                        {
                            "dataset": self.dataset_name,
                            "suppression": suppression_name,
                            "model": spec.name,
                            "fold": f"fold_{fold_idx}",
                            "split": "test",
                            "weight_path": str(weight_path),
                            "train_annotations": str(train_annotations),
                            "val_annotations": str(val_annotations),
                            "test_annotations": str(test_annotations),
                            "images": len(per_image_rows),
                            "tiles": total_tiles,
                            "create_mosaics": self.create_mosaics,
                            "timings": timings,
                            "metrics": {
                                "precision": round(summary.precision, 6),
                                "recall": round(summary.recall, 6),
                                "f1": round(summary.f1, 6),
                                "mAP": round(summary.map_all, 6),
                                "mAP50": round(summary.map50, 6),
                                "mAP75": round(summary.map75, 6),
                                "MAE": round(summary.mae, 6),
                                "RMSE": round(summary.rmse, 6),
                            },
                            "created_at": created_at,
                        },
                    )
            except Exception as exc:
                print(f"[ERROR] Failed while processing {fold_dir.name}: {exc}")
                continue

        summary_paths = write_summary_reports(self.reports_root)
        for summary_path in summary_paths:
            print(f"[INFO] Summary report updated at {summary_path}")
        print("\n[DONE] Pipeline execution completed.")
