from __future__ import annotations

from datetime import datetime
from typing import Dict, List

from calcula_estatisticas.evaluate_reconstructed import evaluate_fold
from pipeline.coco_utils import load_coco_json
from pipeline.data_prep import discover_fold_directories
from pipeline.reporting import write_fold_result, write_image_results, write_summary_reports
from pipeline.types import DetectionRecord

from .cli import parse_folds, parse_requested
from .config import load_active_config
from .dataset_context import (
    load_base_dataset,
    prepare_fold_context,
    resolve_annotations_path,
    resolve_weight_path,
)
from .output_writer import build_per_image_rows, render_fold_visualizations, write_outputs_for_suppression
from .sahi_inference import run_model_inference


def run(args) -> None:
    active_config = load_active_config(args.config)
    reports_root = args.reports_root or (args.results_root / "reports")
    originals_root = args.originals_root or (args.dataset_root / "imagens_originais")
    requested_models = parse_requested(args.models, active_config.model_specs.keys())
    requested_suppressions = parse_requested(args.suppressions, active_config.suppressions)
    allowed_folds = set(parse_folds(args.folds) or [])

    base_coco, _, original_lookup = load_base_dataset(args.source_images_root)
    fold_dirs = discover_fold_directories(args.dataset_root / "tiles")
    if not fold_dirs:
        raise RuntimeError(f"No folds found under {args.dataset_root / 'tiles'}")

    for fold_dir in fold_dirs:
        fold_idx = int(fold_dir.name.split("_")[-1])
        if allowed_folds and fold_idx not in allowed_folds:
            print(f"[INFO] Skipping {fold_dir.name} because it was filtered out during preflight validation.")
            continue

        print(f"\n[INFO] Processing {fold_dir.name} (fold {fold_idx})")
        filtered_coco, original_to_tiles, filtered_coco_path, total_tiles = prepare_fold_context(
            fold_dir=fold_dir,
            base_coco=base_coco,
            original_lookup=original_lookup,
            source_images_root=args.source_images_root,
            originals_root=originals_root,
        )
        image_names = sorted(original_to_tiles.keys())
        train_annotations = resolve_annotations_path(fold_dir / "train")
        val_annotations = resolve_annotations_path(fold_dir / "val")
        test_annotations = resolve_annotations_path(fold_dir / "test")

        for model_name in requested_models:
            weight_path = resolve_weight_path(args.models_root, model_name, fold_idx, active_config.model_specs)
            suppression_outputs = {
                suppression: args.results_root / "reconstructed" / suppression / model_name / f"fold{fold_idx}" / "_annotations.coco.json"
                for suppression in requested_suppressions
            }
            all_outputs_exist = all(path.exists() for path in suppression_outputs.values())

            if args.no_resume or not all_outputs_exist:
                print(f"[INFO]  +- Running SAHI inference for model '{model_name}' on {fold_dir.name}")
                raw_detections_by_image, raw_timings = run_model_inference(
                    model_name=model_name,
                    weight_path=weight_path,
                    image_names=image_names,
                    source_images_root=args.source_images_root,
                    device=args.device,
                    model_specs=active_config.model_specs,
                    minimum_versions=active_config.ultralytics_minimum_versions,
                )
            else:
                print(f"[INFO]  +- Reusing existing outputs for model '{model_name}' on {fold_dir.name}")
                raw_detections_by_image = {}
                raw_timings = {"model_load_time_s": 0.0, "tile_inference_time_s": 0.0}

            for suppression_name in requested_suppressions:
                annotations_output = suppression_outputs[suppression_name]
                if not args.no_resume and annotations_output.exists() and not raw_detections_by_image:
                    prediction_dataset = load_coco_json(annotations_output)
                    image_id_to_name = {int(image["id"]): str(image["file_name"]) for image in prediction_dataset.get("images", [])}
                    suppressed_detections: Dict[str, List[DetectionRecord]] = {}
                    for ann in prediction_dataset.get("annotations", []):
                        image_name = image_id_to_name.get(int(ann["image_id"]))
                        if image_name is None:
                            continue
                        bbox = ann["bbox"]
                        suppressed_detections.setdefault(image_name, []).append(
                            DetectionRecord(
                                x=float(bbox[0]),
                                y=float(bbox[1]),
                                width=float(bbox[2]),
                                height=float(bbox[3]),
                                score=float(ann.get("score", 0.0)),
                                category_id=int(ann["category_id"]),
                            )
                        )
                    render_fold_visualizations(
                        filtered_coco=filtered_coco,
                        detections_by_image=suppressed_detections,
                        source_images_root=args.source_images_root,
                        output_images_dir=annotations_output.parent / "images",
                    )
                    per_image, summary = evaluate_fold(annotations_output, filtered_coco_path)
                    per_image_rows = build_per_image_rows(
                        dataset_name=args.dataset_name,
                        suppression_name=suppression_name,
                        model_name=model_name,
                        fold_name=f"fold_{fold_idx}",
                        per_image=per_image,
                    )
                    write_image_results(reports_root, per_image_rows)
                    write_fold_result(
                        reports_root,
                        {
                            "dataset": args.dataset_name,
                            "suppression": suppression_name,
                            "model": model_name,
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
                            "model_load_time_s": "0.000000",
                            "tile_inference_time_s": "0.000000",
                            "reconstruction_time_s": "0.000000",
                            "suppression_time_s": "0.000000",
                            "evaluation_time_s": "0.000000",
                            "total_time_s": "0.000000",
                            "created_at": datetime.now().isoformat(timespec="seconds"),
                        },
                    )
                    continue

                if not raw_detections_by_image:
                    raise RuntimeError(
                        f"Missing raw detections for {model_name}/{fold_dir.name}. "
                        "Use --no-resume or remove stale outputs."
                    )

                write_outputs_for_suppression(
                    dataset_name=args.dataset_name,
                    suppression_name=suppression_name,
                    model_name=model_name,
                    fold_idx=fold_idx,
                    weight_path=weight_path,
                    filtered_coco=filtered_coco,
                    filtered_coco_path=filtered_coco_path,
                    raw_timings=raw_timings,
                    detections_by_image=raw_detections_by_image,
                    source_images_root=args.source_images_root,
                    results_root=args.results_root,
                    reports_root=reports_root,
                    total_tiles=total_tiles,
                    train_annotations=train_annotations,
                    val_annotations=val_annotations,
                    test_annotations=test_annotations,
                    suppression_params=active_config.suppression_params(suppression_name),
                )

    summary_paths = write_summary_reports(reports_root)
    for summary_path in summary_paths:
        print(f"[INFO] Summary report updated at {summary_path}")
    print("\n[DONE] SAHI pipeline execution completed.")
