from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image

from pipeline.reconstruction import apply_suppression
from pipeline.types import DetectionRecord, SuppressionParams



def _ensure_image_dimensions(image_entry: Dict[str, object], images_dir: Path) -> Tuple[int, int]:
    width = image_entry.get("width")
    height = image_entry.get("height")
    if isinstance(width, int) and isinstance(height, int):
        return width, height

    image_path = images_dir / image_entry["file_name"]
    with Image.open(image_path) as img:
        width, height = img.size

    image_entry["width"] = width
    image_entry["height"] = height
    return width, height


def _reapply_for_file(
    annotations_path: Path,
    *,
    images_dir: Path,
    params: SuppressionParams,
    create_backup: bool,
) -> Tuple[int, int, int]:
    raw_text = annotations_path.read_text(encoding="utf-8")
    dataset = json.loads(raw_text)

    if create_backup:
        backup_path = annotations_path.with_suffix(annotations_path.suffix + ".bak")
        if not backup_path.exists():
            backup_path.write_text(raw_text, encoding="utf-8")

    image_lookup = {img["id"]: img for img in dataset.get("images", [])}

    grouped: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for ann in dataset.get("annotations", []):
        grouped[int(ann["image_id"])].append(ann)

    new_annotations: List[Dict[str, object]] = []
    next_id = 0
    original_count = len(dataset.get("annotations", []))

    for image_id, anns in grouped.items():
        image_entry = image_lookup.get(image_id)
        if image_entry is None:
            continue

        width, height = _ensure_image_dimensions(image_entry, images_dir)

        detections = [
            DetectionRecord(
                x=float(ann["bbox"][0]),
                y=float(ann["bbox"][1]),
                width=float(ann["bbox"][2]),
                height=float(ann["bbox"][3]),
                score=float(ann.get("score", 0.0)),
                category_id=int(ann["category_id"]),
            )
            for ann in anns
        ]

        suppressed = apply_suppression(
            detections,
            image_width=width,
            image_height=height,
            params=params,
        )

        for det in suppressed:
            new_annotations.append(
                {
                    "id": next_id,
                    "image_id": image_id,
                    "category_id": det.category_id,
                    "bbox": [det.x, det.y, det.width, det.height],
                    "area": det.width * det.height,
                    "score": det.score,
                }
            )
            next_id += 1

    dataset["annotations"] = new_annotations
    annotations_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")

    return len(grouped), original_count, len(new_annotations)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reapply suppression to annotations produced in results/reconstructed.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/reconstructed"),
        help="Root directory that contains model/fold subdirectories.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="nms",
        help="Suppression method (cluster_diou_ait [auto thresholds], cluster_ait, nms, bws, cluster_diou_nms, cluster_diou_bws).",
    )
    parser.add_argument(
        "--affinity-threshold",
        type=float,
        default=0.4,
        help="Affinity threshold used by cluster_diou_* variants.",
    )
    parser.add_argument(
        "--lambda-weight",
        type=float,
        default=0.3,
        help="Lambda weight for adaptive/cluster suppression heuristics.",
    )
    parser.add_argument(
        "--score-ratio-threshold",
        type=float,
        default=0.85,
        help="Minimum score ratio to classify a box as a duplicate (cluster_diou_ait).",
    )
    parser.add_argument(
        "--duplicate-iou-threshold",
        type=float,
        default=0.5,
        help="DIoU threshold used to mark strong duplicates (cluster_diou_ait).",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for NMS/BWS methods.",
    )
    parser.add_argument(
        "--diou-threshold",
        type=float,
        default=0.5,
        help="DIoU threshold for cluster_diou_nms.",
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional suppression parameters (repeatable).",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak backups before rewriting JSON files.",
    )
    args = parser.parse_args()

    if args.method == "cluster_diou_ait":
        print("[INFO] Método AIT detectado: ignorando thresholds externos.")
        args.affinity_threshold = None
        args.lambda_weight = None
        args.iou_threshold = None
        args.diou_threshold = None

    return args


def main() -> None:
    args = _parse_args()
    root = args.root

    if not root.exists():
        raise FileNotFoundError(f"Results root '{root}' not found.")

    extra_params = {}
    for item in args.extra:
        if '=' not in item:
            raise ValueError(f"Invalid extra parameter format: {item!r}. Expected key=value.")
        key, value = item.split('=', 1)
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError(f"Could not parse extra parameter '{item}'.") from exc
        extra_params[key.strip()] = parsed

    params = SuppressionParams(
        method=args.method,
        affinity_threshold=args.affinity_threshold,
        lambda_weight=args.lambda_weight,
        score_ratio_threshold=args.score_ratio_threshold,
        duplicate_iou_threshold=args.duplicate_iou_threshold,
        iou_threshold=args.iou_threshold,
        diou_threshold=args.diou_threshold,
        extra=extra_params,
    )

    if params.method == "cluster_diou_ait":
        print("[INFO] Usando Adaptive IoU Threshold (AIT): thresholds externos desativados.")

    total_annotations_before = 0
    total_annotations_after = 0

    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue

        for fold_dir in sorted(model_dir.iterdir()):
            if not fold_dir.is_dir():
                continue

            annotations_path = fold_dir / "_annotations.coco.json"
            if not annotations_path.exists():
                continue

            images_dir = fold_dir / "images"
            num_images, original_annotations, new_annotations = _reapply_for_file(
                annotations_path,
                images_dir=images_dir,
                params=params,
                create_backup=not args.no_backup,
            )

            total_annotations_after += new_annotations
            total_annotations_before += original_annotations

            print(
                f"[INFO] Processed {model_dir.name}/{fold_dir.name}: "
                f"{num_images} images, {original_annotations} -> {new_annotations} annotations",
            )

    if total_annotations_before > 0:
        removed = total_annotations_before - total_annotations_after
        print(
            f"[DONE] Total annotations: {total_annotations_before} -> {total_annotations_after} "
            f"({removed} removed)",
        )
    else:
        print("[WARN] No annotations files were updated.")


if __name__ == "__main__":
    main()
