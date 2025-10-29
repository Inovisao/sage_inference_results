from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Set, Tuple

import cv2

from pipeline.coco_utils import load_coco_json
from pipeline.data_prep import parse_tile_filename
from pipeline.detectors import resolve_detector
from pipeline.reconstruction import (
    apply_suppression,
    detect_tile_orientation,
    remap_detections_by_rotation,
)
from pipeline.types import DetectionRecord, SuppressionParams

DEFAULT_SUPPRESSION = SuppressionParams(
    method='cluster_diou_ait',
    affinity_threshold=0.4,
    lambda_weight=0.3,
    score_ratio_threshold=0.85,
    duplicate_iou_threshold=0.5,
)


def _stem_candidates(image_name: str) -> Set[str]:
    """Return possible tile stems that correspond to the original image name."""
    stem = Path(image_name).stem
    candidates: Set[str] = {stem}

    # Handle Roboflow-style names (e.g., 123_jpg.rf.<hash>) by keeping the prefix.
    markers = (
        "_jpg.rf.",
        "_jpeg.rf.",
        "_png.rf.",
        "_bmp.rf.",
        "_tif.rf.",
        "_tiff.rf.",
    )
    for marker in markers:
        if marker in stem:
            candidates.add(stem.split(marker, 1)[0])
            break

    return candidates


def _discover_tiles_for_image(test_dir: Path, target_name: str) -> List[Tuple[Path, int, int]]:
    """Return a sorted list of (tile_path, offset_x, offset_y) for a given original image name."""
    candidate_files = []
    tiles_dir = test_dir
    images_dir = test_dir / "images"
    valid_stems = _stem_candidates(target_name)

    for directory in [tiles_dir, images_dir]:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.jpg")):
            stem, offset_x, offset_y = parse_tile_filename(path.name)
            if stem in valid_stems:
                candidate_files.append((path, offset_x, offset_y))

    if not candidate_files:
        raise FileNotFoundError(
            f"No tiles found for image '{target_name}' under '{test_dir}'. "
            "Ensure the fold/test split contains the target image."
        )

    # sort by offset_y, then offset_x
    candidate_files.sort(key=lambda item: (item[2], item[1]))
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


def _draw_detections(
    image_path: Path,
    detections: Sequence[DetectionRecord],
    ground_truths: Sequence[Tuple[float, float, float, float]],
    output_path: Path,
    score_threshold: float,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(
            f"Unable to read original image at '{image_path}'.")

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

    for bbox in ground_truths:
        gx1 = int(bbox[0])
        gy1 = int(bbox[1])
        gx2 = int(bbox[0] + bbox[2])
        gy2 = int(bbox[1] + bbox[3])
        cv2.rectangle(image, (gx1, gy1), (gx2, gy2), (0, 255, 255), 2)
        cv2.putText(
            image,
            "GT",
            (gx1, max(0, gy1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    print(f"[INFO] Saved visualization to {output_path}")


def _load_ground_truth_boxes(dataset_root: Path, image_name: str) -> List[Tuple[float, float, float, float]]:
    coco_path = dataset_root / "train" / "_annotations.coco.json"
    if not coco_path.exists():
        return []

    coco = load_coco_json(coco_path)
    images = {str(img["file_name"]): int(img["id"])
              for img in coco.get("images", [])}
    image_id = images.get(image_name)
    if image_id is None:
        return []

    boxes: List[Tuple[float, float, float, float]] = []
    for ann in coco.get("annotations", []):
        if int(ann.get("image_id", -1)) != image_id:
            continue
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x, y, w, h = (float(v) for v in bbox)
        boxes.append((x, y, w, h))
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference on all tiles of a single image and visualise the suppressed detections."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--fold", type=str, default="fold_5",
                        help="Fold identifier (e.g., fold_1)")
    parser.add_argument("--image-name", type=str, default="7_jpg.rf.4ac9339927b6e2999dd0cb9815a274e7.jpg",
                        help="Original image file name, e.g., 100.jpg")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8",
        help="Detector name (yolov8, yolov11, yolov5_tph, retinanet, faster)",
    )
    parser.add_argument(
        "--weight",
        type=Path,
        default=Path("model_checkpoints/fold_3/YOLOV8/best.pt"),
        help="Path to the trained weight file (.pt/.pth)",
    )
    parser.add_argument("--threshold", type=float,
                        default=0.25, help="Confidence threshold")
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

    tiles = _discover_tiles_for_image(fold_dir, args.image_name)
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

    print(
        f"[INFO] Total projected detections before suppression: {len(aggregated)}")
    original_image_path = dataset_root / "train" / args.image_name
    original_image = cv2.imread(str(original_image_path))
    if original_image is None:
        raise FileNotFoundError(
            f"Unable to read original image '{original_image_path}'.")
    image_height, image_width = original_image.shape[:2]

    sample_tile_path, sample_offset_x, sample_offset_y = tiles[0]
    angle = detect_tile_orientation(
        original_image,
        tile_path=sample_tile_path,
        offset_x=sample_offset_x,
        offset_y=sample_offset_y,
    )
    if angle != 0:
        print(f"[WARN] Detected a {angle}° rotation between tiles and stored original. "
              "Remapping detections to compensate.")

    suppressed = apply_suppression(
        aggregated,
        image_width=image_width,
        image_height=image_height,
        params=DEFAULT_SUPPRESSION,
    )
    print(f"[INFO] Detections after suppression: {len(suppressed)}")
    suppressed = remap_detections_by_rotation(
        suppressed,
        angle,
        image_width=image_width,
        image_height=image_height,
    )

    ground_truth_boxes = _load_ground_truth_boxes(
        dataset_root, args.image_name)
    if ground_truth_boxes:
        print(
            f"[INFO] Loaded {len(ground_truth_boxes)} ground-truth boxes for comparison.")
    else:
        print("[INFO] No ground-truth boxes found for this image.")

    output_path = args.output
    _draw_detections(original_image_path, suppressed, ground_truth_boxes,
                     output_path, score_threshold=args.threshold)


if __name__ == "__main__":
    main()
