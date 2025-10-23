from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image

from pipeline.types import DetectionRecord, SuppressionParams
from supression.cluster_diou_AIT import adaptive_cluster_diou_nms


def _clip_detection(det: DetectionRecord, *, width: float, height: float) -> DetectionRecord | None:
    x1 = max(0.0, min(det.x, float(width)))
    y1 = max(0.0, min(det.y, float(height)))
    x2 = max(0.0, min(det.x + det.width, float(width)))
    y2 = max(0.0, min(det.y + det.height, float(height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return DetectionRecord(
        x=x1,
        y=y1,
        width=x2 - x1,
        height=y2 - y1,
        score=det.score,
        category_id=det.category_id,
    )


def _apply_suppression(
    detections: Iterable[DetectionRecord],
    *,
    image_width: float,
    image_height: float,
    params: SuppressionParams,
) -> List[DetectionRecord]:
    grouped: Dict[int, List[DetectionRecord]] = defaultdict(list)
    for det in detections:
        grouped[det.category_id].append(det)

    suppressed: List[DetectionRecord] = []
    for class_id, class_dets in grouped.items():
        if len(class_dets) == 1:
            clipped = _clip_detection(class_dets[0], width=image_width, height=image_height)
            if clipped:
                suppressed.append(clipped)
            continue

        boxes = np.array(
            [
                [
                    det.x,
                    det.y,
                    det.x + det.width,
                    det.y + det.height,
                ]
                for det in class_dets
            ],
            dtype=np.float32,
        )
        scores = np.array([det.score for det in class_dets], dtype=np.float32)

        keep_indices = adaptive_cluster_diou_nms(
            boxes,
            scores,
            T0=params.affinity_threshold,
            alpha=params.lambda_weight,
            score_ratio_thresh=params.score_ratio_threshold,
            diou_dup_thresh=params.duplicate_iou_threshold,
        )

        for idx in keep_indices:
            x1, y1, x2, y2 = boxes[idx].tolist()
            score = float(scores[idx])
            clipped = _clip_detection(
                DetectionRecord(
                    x=float(x1),
                    y=float(y1),
                    width=float(x2 - x1),
                    height=float(y2 - y1),
                    score=score,
                    category_id=class_id,
                ),
                width=image_width,
                height=image_height,
            )
            if clipped:
                suppressed.append(clipped)

    suppressed.sort(key=lambda det: det.score, reverse=True)
    return suppressed


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

        suppressed = _apply_suppression(
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
        description="Reapply adaptive suppression to annotations produced in results/reconstructed.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/reconstructed"),
        help="Root directory that contains model/fold subdirectories.",
    )
    parser.add_argument(
        "--affinity-threshold",
        type=float,
        default=0.4,
        help="Base DIoU threshold (T0).",
    )
    parser.add_argument(
        "--lambda-weight",
        type=float,
        default=0.3,
        help="Alpha value used to adapt the threshold.",
    )
    parser.add_argument(
        "--score-ratio-threshold",
        type=float,
        default=0.85,
        help="Minimum score ratio to classify a box as a duplicate.",
    )
    parser.add_argument(
        "--duplicate-iou-threshold",
        type=float,
        default=0.5,
        help="DIoU threshold used to mark strong duplicates.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak backups before rewriting JSON files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.root

    if not root.exists():
        raise FileNotFoundError(f"Results root '{root}' not found.")

    params = SuppressionParams(
        affinity_threshold=args.affinity_threshold,
        lambda_weight=args.lambda_weight,
        score_ratio_threshold=args.score_ratio_threshold,
        duplicate_iou_threshold=args.duplicate_iou_threshold,
    )

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
