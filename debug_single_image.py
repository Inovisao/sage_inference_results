from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from pipeline.data_prep import parse_tile_filename
from pipeline.detectors import resolve_detector
from pipeline.types import DetectionRecord, SuppressionParams
from supression.nms import nms


def _discover_tiles_for_image(test_dir: Path, target_stem: str) -> List[Tuple[Path, int, int]]:
    """Return a sorted list of (tile_path, offset_x, offset_y) for a given image stem."""
    candidate_files = []
    tiles_dir = test_dir
    images_dir = test_dir / "images"

    for directory in [tiles_dir, images_dir]:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.jpg")):
            stem, offset_x, offset_y = parse_tile_filename(path.name)
            if stem == target_stem:
                candidate_files.append((path, offset_x, offset_y))

    if not candidate_files:
        raise FileNotFoundError(
            f"No tiles found for image '{target_stem}' under '{test_dir}'. "
            "Ensure the fold/test split contains the target image."
        )

    candidate_files.sort(key=lambda item: (item[2], item[1]))  # sort by offset_y, then offset_x
    return candidate_files


def _project_detections(
    detections: Sequence[DetectionRecord],
    offset_x: int,
    offset_y: int,
) -> List[DetectionRecord]:
    """Shift detections from tile space into original image space."""
    projected: List[DetectionRecord] = []
    for det in detections:
        projected.append(
            DetectionRecord(
                x=det.x + offset_x,
                y=det.y + offset_y,
                width=det.width,
                height=det.height,
                score=det.score,
                category_id=det.category_id,
            )
        )
    return projected


def _apply_suppression(
    detections: Sequence[DetectionRecord],
    params: SuppressionParams,
) -> List[DetectionRecord]:
    if not detections:
        return []

    by_class: dict[int, List[DetectionRecord]] = {}
    for det in detections:
        by_class.setdefault(det.category_id, []).append(det)

    suppressed: List[DetectionRecord] = []
    for class_id, dets in by_class.items():
        if len(dets) == 1:
            suppressed.extend(dets)
            continue

        boxes = np.array(
            [[det.x, det.y, det.x + det.width, det.y + det.height] for det in dets],
            dtype=np.float32,
        )
        scores = np.array([det.score for det in dets], dtype=np.float32)

        keep_boxes, keep_scores = nms(
            boxes,
            scores,
            iou_thresh=params.affinity_threshold,
        )

        for (x1, y1, x2, y2), score in zip(keep_boxes, keep_scores):
            suppressed.append(
                DetectionRecord(
                    x=float(x1),
                    y=float(y1),
                    width=float(x2 - x1),
                    height=float(y2 - y1),
                    score=float(score),
                    category_id=class_id,
                )
            )

    return suppressed


def _draw_detections(
    image_path: Path,
    detections: Sequence[DetectionRecord],
    output_path: Path,
    score_threshold: float,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read original image at '{image_path}'.")

    for det in detections:
        if det.score < score_threshold:
            continue
        x1 = int(det.x)
        y1 = int(det.y)
        x2 = int(det.x + det.width)
        y2 = int(det.y + det.height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{det.score:.2f}",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    print(f"[INFO] Saved visualization to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference on all tiles of a single image and visualise the suppressed detections."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--fold", type=str, default="fold_3", help="Fold identifier (e.g., fold_1)")
    parser.add_argument("--image-name", type=str, default="206.jpg", help="Original image file name, e.g., 100.jpg")
    parser.add_argument("--model", type=str, default="yolov5_tph", help="Detector name (yolov8, faster, yolov5_tph)")
    parser.add_argument(
        "--weight",
        type=Path,
        default=Path("pesos/yolov5_tph/fold_02/weights/best.pt"),
        help="Path to the trained weight file (.pt/.pth)",
    )
    parser.add_argument("--threshold", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/debug_100.png"),
        help="Output path for the visualization PNG",
    )
    args = parser.parse_args()

    dataset_root: Path = args.dataset_root
    fold_dir = dataset_root / "tiles" / args.fold / "test"
    if not fold_dir.exists():
        raise FileNotFoundError(f"Fold directory '{fold_dir}' not found.")

    image_stem = Path(args.image_name).stem
    tiles = _discover_tiles_for_image(fold_dir, image_stem)
    print(f"[INFO] Located {len(tiles)} tiles for image '{args.image_name}'.")

    detector_cls = resolve_detector(args.model)
    
    # Handle Faster R-CNN specific requirements
    extra_kwargs = {}
    if detector_cls.model_name in {"faster", "fasterrcnn"}:
        # Infer num_classes from the dataset
        coco_path = dataset_root / "train" / "_annotations.coco.json"
        if coco_path.exists():
            import json
            with open(coco_path, 'r') as f:
                coco_data = json.load(f)
            categories = coco_data.get("categories", [])
            if categories:
                max_id = max(int(cat["id"]) for cat in categories)
                extra_kwargs["num_classes"] = max_id + 1
            else:
                extra_kwargs["num_classes"] = 2  # default
        else:
            extra_kwargs["num_classes"] = 2  # default
    
    detector = detector_cls(args.weight, **extra_kwargs)
    print(f"[INFO] Loaded detector '{args.model}' from '{args.weight}'.")

    aggregated: List[DetectionRecord] = []
    try:
        for idx, (tile_path, offset_x, offset_y) in enumerate(tiles, 1):
            tile_img = cv2.imread(str(tile_path))
            if tile_img is None:
                raise FileNotFoundError(f"Unable to read tile '{tile_path}'.")

            detections = detector.predict(tile_img, args.threshold)
            projected = _project_detections(detections, offset_x, offset_y)
            aggregated.extend(projected)
            print(
                f"    [{idx}/{len(tiles)}] {tile_path.name} -> {len(detections)} detections "
                f"(offset=({offset_x}, {offset_y}))"
            )
    finally:
        detector.close()

    print(f"[INFO] Total projected detections before suppression: {len(aggregated)}")
    suppressed = _apply_suppression(aggregated, SuppressionParams())
    print(f"[INFO] Detections after suppression: {len(suppressed)}")

    original_image_path = dataset_root / "train" / args.image_name
    output_path = args.output
    _draw_detections(original_image_path, suppressed, output_path, score_threshold=args.threshold)


if __name__ == "__main__":
    main()
