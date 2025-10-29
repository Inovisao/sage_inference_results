from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2


def load_reconstructed_annotations(annotations_path: Path) -> Dict[str, List[Dict[str, float]]]:
    with annotations_path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    images = {int(img["id"]): img["file_name"] for img in coco.get("images", [])}
    detections: Dict[str, List[Dict[str, float]]] = {file_name: [] for file_name in images.values()}

    for ann in coco.get("annotations", []):
        image_id = int(ann["image_id"])
        bbox = ann["bbox"]
        score = ann.get("score", 0.0)
        file_name = images.get(image_id)
        if file_name is None:
            continue
        detections.setdefault(file_name, []).append({"bbox": bbox, "score": score})

    return detections


def _normalize_fold_name(name: str) -> str:
    return name.lower().replace("_", "").replace("-", "")


def resolve_ground_truth_path(dataset_root: Path, fold_name: str) -> Path:
    base_dir = dataset_root / "imagens_originais"
    if not base_dir.exists():
        raise FileNotFoundError(f"Ground-truth directory not found at {base_dir}")

    target_norm = _normalize_fold_name(fold_name)
    for candidate in base_dir.iterdir():
        if not candidate.is_dir():
            continue
        if _normalize_fold_name(candidate.name) == target_norm:
            coco_path = candidate / "_annotations.coco.json"
            if coco_path.exists():
                return coco_path
    raise FileNotFoundError(
        f"Ground-truth annotations not found for fold '{fold_name}'. "
        f"Checked under {base_dir}. Ensure the directory contains a matching fold with '_annotations.coco.json'."
    )


def load_ground_truth_annotations(annotations_path: Path) -> Dict[str, List[List[float]]]:
    with annotations_path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    images = {int(img["id"]): img["file_name"] for img in coco.get("images", [])}
    annotations: Dict[str, List[List[float]]] = {file_name: [] for file_name in images.values()}

    for ann in coco.get("annotations", []):
        image_id = int(ann["image_id"])
        bbox = [float(v) for v in ann["bbox"]]
        file_name = images.get(image_id)
        if file_name is None:
            continue
        annotations.setdefault(file_name, []).append(bbox)

    return annotations


def resolve_tiles_dir(dataset_root: Path, fold_name: str) -> Optional[Path]:
    tiles_root = dataset_root / "tiles"
    if not tiles_root.exists():
        return None

    target_norm = _normalize_fold_name(fold_name)
    for candidate in tiles_root.iterdir():
        if not candidate.is_dir():
            continue
        if _normalize_fold_name(candidate.name) == target_norm:
            test_dir = candidate / "test"
            if test_dir.exists():
                return test_dir
    return None


def _parse_tile_filename(file_name: str) -> Tuple[str, int, int]:
    stem = Path(file_name).stem
    if "_tile_" not in stem:
        raise ValueError(f"Tile file '{file_name}' does not contain '_tile_' segment.")
    prefix, _, suffix = stem.partition("_tile_")
    try:
        offset_x_str, offset_y_str = suffix.split("_", 1)
        offset_x = int(offset_x_str)
        offset_y = int(offset_y_str)
    except ValueError as exc:
        raise ValueError(f"Could not parse offsets from tile file '{file_name}'.") from exc
    return prefix, offset_x, offset_y


def load_tile_boundaries(dataset_root: Path, fold_name: str, image_name: str) -> List[Tuple[int, int, int, int]]:
    tiles_dir = resolve_tiles_dir(dataset_root, fold_name)
    if tiles_dir is None:
        return []

    annotations_path = tiles_dir / "_annotations.coco.json"
    tile_sizes: Dict[str, Tuple[int, int]] = {}
    if annotations_path.exists():
        with annotations_path.open("r", encoding="utf-8") as handle:
            coco = json.load(handle)
        for image_entry in coco.get("images", []):
            file_name = str(image_entry["file_name"])
            width = int(image_entry.get("width", 0))
            height = int(image_entry.get("height", 0))
            tile_sizes[file_name] = (width, height)

    stem = Path(image_name).stem
    candidate_dirs = [tiles_dir, tiles_dir / "images"]
    boundaries: Dict[str, Tuple[int, int, int, int]] = {}
    for directory in candidate_dirs:
        if not directory.exists():
            continue
        for tile_path in directory.glob(f"{stem}_tile_*"):
            if tile_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            _, offset_x, offset_y = _parse_tile_filename(tile_path.name)
            width, height = tile_sizes.get(tile_path.name, (0, 0))
            if width <= 0 or height <= 0:
                tile_img = cv2.imread(str(tile_path))
                if tile_img is None:
                    continue
                height, width = tile_img.shape[:2]
            boundaries[tile_path.name] = (offset_x, offset_y, width, height)

    return list(boundaries.values())


def draw_boxes(
    image_path: Path,
    annotations: List[List[float]],
    detections: List[Dict[str, float]],
    output_path: Path,
    score_threshold: float,
    tile_boundaries: Optional[List[Tuple[int, int, int, int]]] = None,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image at {image_path}")

    if tile_boundaries:
        overlay = image.copy()
        for idx, (offset_x, offset_y, width, height) in enumerate(tile_boundaries):
            color_value = 150 + (idx * 30) % 80
            color = (color_value, color_value, color_value)
            top_left = (int(offset_x), int(offset_y))
            bottom_right = (int(offset_x + width), int(offset_y + height))
            cv2.rectangle(overlay, top_left, bottom_right, color, 1)
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

    for bbox in annotations:
        x, y, w, h = bbox
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)

    for det in detections:
        score = float(det.get("score", 0.0))
        if score < score_threshold:
            continue
        x, y, w, h = det["bbox"]
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{score:.2f}",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    print(f"[INFO] Saved visualization to {output_path}")


def pick_random_image(detections: Dict[str, List[Dict[str, float]]]) -> str:
    candidates = [name for name, dets in detections.items() if dets]
    if not candidates:
        raise ValueError("No detections found in reconstructed annotations.")
    return random.choice(candidates)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Display bounding boxes from reconstructed predictions for a random (or specified) image."
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--model", type=str, default="yolov8")
    parser.add_argument("--fold", type=str, default="fold5")
    parser.add_argument("--image-name", type=str, help="Optional image file name to visualise.")
    parser.add_argument(
        "--image-number",
        type=int,
        help="Optional numeric identifier (image stem) to visualise, e.g., 788 for 788.jpg.",
    )
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/verify_bbox.png"),
        help="Path to save the rendered image.",
    )
    args = parser.parse_args()

    reconstructed_dir = args.results_root / "reconstructed" / args.model / args.fold
    if not reconstructed_dir.exists():
        alt_fold = args.fold.replace("_", "")
        reconstructed_dir = args.results_root / "reconstructed" / args.model / alt_fold
    annotations_path = reconstructed_dir / "_annotations.coco.json"
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found at {annotations_path}")

    detections = load_reconstructed_annotations(annotations_path)

    gt_path = resolve_ground_truth_path(args.dataset_root, args.fold)
    ground_truth = load_ground_truth_annotations(gt_path)

    image_name: Optional[str] = None
    if args.image_number is not None:
        target_stem = str(args.image_number)
        matches = [name for name in detections if Path(name).stem == target_stem]
        if not matches:
            raise FileNotFoundError(
                f"No detections available for image number '{target_stem}'. "
                "Ensure the COCO file contains a matching image."
            )
        image_name = matches[0]
    elif args.image_name:
        image_name = args.image_name
    else:
        image_name = pick_random_image(detections)

    if image_name not in detections:
        raise FileNotFoundError(f"No detections available for image '{image_name}'.")

    image_path_candidates = [
        reconstructed_dir / "images" / image_name,
        args.dataset_root / "train" / image_name,
    ]

    chosen_image_path: Optional[Path] = None
    for candidate in image_path_candidates:
        if candidate.exists():
            chosen_image_path = candidate
            break

    if chosen_image_path is None:
        raise FileNotFoundError(
            f"Could not locate image '{image_name}' in reconstructed images or original dataset."
        )

    print(f"[INFO] Visualising image '{image_name}' from {chosen_image_path}")
    draw_boxes(
        chosen_image_path,
        ground_truth.get(image_name, []),
        detections[image_name],
        args.output,
        score_threshold=args.score_threshold,
        tile_boundaries=load_tile_boundaries(args.dataset_root, args.fold, image_name),
    )


if __name__ == "__main__":
    main()
