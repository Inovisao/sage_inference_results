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


def draw_boxes(
    image_path: Path,
    detections: List[Dict[str, float]],
    output_path: Path,
    score_threshold: float,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image at {image_path}")

    for det in detections:
        score = float(det.get("score", 0.0))
        if score < score_threshold:
            continue
        x, y, w, h = det["bbox"]
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 128, 255), 2)
        cv2.putText(
            image,
            f"{score:.2f}",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 128, 255),
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
    parser.add_argument("--model", type=str, default="faster")
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

    image_name = args.image_name or pick_random_image(detections)
    if image_name not in detections:
        raise FileNotFoundError(f"No detections available for image '{image_name}'.")

    image_path_candidates = [
        reconstructed_dir / "images" / image_name,
        Path("dataset") / "train" / image_name,
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
        detections[image_name],
        args.output,
        score_threshold=args.score_threshold,
    )


if __name__ == "__main__":
    main()
