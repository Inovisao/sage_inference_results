from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

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


def draw_boxes(
    image_path: Path,
    annotations: List[List[float]],
    detections: List[Dict[str, float]],
    output_path: Path,
    score_threshold: float,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image at {image_path}")

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

    image_name = args.image_name or pick_random_image(detections)
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
    )


if __name__ == "__main__":
    main()
