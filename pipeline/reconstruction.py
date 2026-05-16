from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import cv2
import numpy as np
from PIL import Image

from supression.bws import bws as suppression_bws
from supression.cluster_diou_AIT import adaptive_cluster_diou_nms
from supression.cluster_diou_nms import cluster_diou_nms
from supression.nms import nms as suppression_nms
from supression.nms_ioa import nms_ioa as suppression_nms_ioa

from .types import (
    DetectionRecord,
    OriginalImage,
    OriginalToTiles,
    ReconstructionStats,
    SuppressionParams,
    TileDetections,
    TileMetadata,
)


def _clip_detection(det: DetectionRecord, *, width: int, height: int) -> DetectionRecord | None:
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


# Supported suppression method names:
#   - "nms"
#   - "nms_ioa"
#   - "cluster_diou_nms" / "cluster_nms"
def _apply_suppression_by_method(
    detections: Sequence[DetectionRecord],
    *,
    image_width: int,
    image_height: int,
    params: SuppressionParams,
) -> List[DetectionRecord]:
    if not detections:
        return []

    method = getattr(params, "method", "cluster_diou_nms")
    method_key = str(method).lower().replace("-", "_")
    extra = getattr(params, "extra", {}) or {}

    grouped: Dict[int, List[DetectionRecord]] = defaultdict(list)
    for det in detections:
        grouped[det.category_id].append(det)

    final: List[DetectionRecord] = []
    for class_id, class_dets in grouped.items():
        if len(class_dets) == 1:
            clipped = _clip_detection(class_dets[0], width=image_width, height=image_height)
            if clipped:
                final.append(clipped)
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

        def _box_to_detection(box: Sequence[float], score: float) -> DetectionRecord | None:
            x1, y1, x2, y2 = [float(v) for v in box]
            detection = DetectionRecord(
                x=x1,
                y=y1,
                width=float(x2 - x1),
                height=float(y2 - y1),
                score=float(score),
                category_id=class_id,
            )
            return _clip_detection(detection, width=image_width, height=image_height)

        if method_key in {"cluster_diou_ait", "adaptive_cluster_diou", "ait"}:
            keep_indices = adaptive_cluster_diou_nms(
                boxes,
                scores,
                tau_0=float(extra.get("tau_0", params.affinity_threshold)),
                alpha=float(extra.get("alpha", params.lambda_weight)),
                k=int(extra.get("k", 5)),
                tau_min=float(extra.get("tau_min", 0.05)),
                gamma=float(extra.get("gamma", params.score_ratio_threshold)),
                tau_dup=float(extra.get("tau_dup", params.duplicate_iou_threshold)),
            )
            for idx in keep_indices:
                clipped = _box_to_detection(boxes[idx], scores[idx])
                if clipped:
                    final.append(clipped)
            continue

        if method_key in {"nms"}:
            iou_thresh = float(extra.get("iou_threshold", params.iou_threshold))
            suppressed_boxes, suppressed_scores = suppression_nms(boxes, scores, iou_thresh=iou_thresh)
            suppressed_boxes = np.atleast_2d(suppressed_boxes)
            suppressed_scores = np.atleast_1d(suppressed_scores)
            for box, score in zip(suppressed_boxes, suppressed_scores):
                clipped = _box_to_detection(box, score)
                if clipped:
                    final.append(clipped)
            continue

        if method_key in {"nms_ioa"}:
            suppressed_boxes, suppressed_scores = suppression_nms_ioa(
                boxes,
                scores,
                ioa_thresh=float(extra.get("ioa_threshold", params.iou_threshold)),
                conf_threshold=float(extra.get("conf_threshold", 0.4)),
                sigma=float(extra.get("sigma", 0.5)),
            )
            suppressed_boxes = np.atleast_2d(suppressed_boxes)
            suppressed_scores = np.atleast_1d(suppressed_scores)
            for box, score in zip(suppressed_boxes, suppressed_scores):
                clipped = _box_to_detection(box, score)
                if clipped:
                    final.append(clipped)
            continue

        if method_key in {"bws"}:
            iou_thresh = float(extra.get("iou_threshold", params.iou_threshold))
            suppressed_boxes, suppressed_scores = suppression_bws(boxes, scores, iou_thresh=iou_thresh)
            suppressed_boxes = np.atleast_2d(suppressed_boxes)
            suppressed_scores = np.atleast_1d(suppressed_scores)
            for box, score in zip(suppressed_boxes, suppressed_scores):
                clipped = _box_to_detection(box, score)
                if clipped:
                    final.append(clipped)
            continue

        if method_key in {"cluster_diou_nms", "cluster_nms"}:
            diou_thresh = float(extra.get("diou_threshold", params.diou_threshold))
            suppressed_boxes, suppressed_scores = cluster_diou_nms(boxes, scores, diou_thresh=diou_thresh)
            suppressed_boxes = np.atleast_2d(suppressed_boxes)
            suppressed_scores = np.atleast_1d(suppressed_scores)
            for box, score in zip(suppressed_boxes, suppressed_scores):
                clipped = _box_to_detection(box, score)
                if clipped:
                    final.append(clipped)
            continue

        raise ValueError(f"Unsupported suppression method '{method}'.")

    final.sort(key=lambda det: det.score, reverse=True)
    return final


def apply_suppression(
    detections: Sequence[DetectionRecord],
    *,
    image_width: int,
    image_height: int,
    params: SuppressionParams,
) -> List[DetectionRecord]:
    """Public helper to run the configured suppression method."""
    return _apply_suppression_by_method(
        detections,
        image_width=image_width,
        image_height=image_height,
        params=params,
    )


def _project_tile_detections(tile: TileMetadata, detections: Sequence[DetectionRecord]) -> List[DetectionRecord]:
    projected: List[DetectionRecord] = []
    for det in detections:
        projected.append(
            DetectionRecord(
                x=det.x + tile.offset_x,
                y=det.y + tile.offset_y,
                width=det.width,
                height=det.height,
                score=det.score,
                category_id=det.category_id,
            )
        )
    return projected


def _reconstruct_image(original: OriginalImage, tiles: Sequence[TileMetadata], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (int(original.width), int(original.height)))
    for tile in tiles:
        with Image.open(tile.path) as tile_img:
            if tile_img.mode != "RGB":
                tile_img = tile_img.convert("RGB")
            canvas.paste(tile_img, (int(tile.offset_x), int(tile.offset_y)))
    canvas.save(output_path)


def _reconstruct_image_array(original: OriginalImage, tiles: Sequence[TileMetadata]) -> np.ndarray:
    canvas = np.zeros((int(original.height), int(original.width), 3), dtype=np.uint8)
    for tile in tiles:
        tile_image = cv2.imread(str(tile.path))
        if tile_image is None:
            raise FileNotFoundError(f"Unable to read tile image '{tile.path}'.")
        tile_h, tile_w = tile_image.shape[:2]
        y1 = int(tile.offset_y)
        y2 = min(int(tile.offset_y) + tile_h, canvas.shape[0])
        x1 = int(tile.offset_x)
        x2 = min(int(tile.offset_x) + tile_w, canvas.shape[1])
        canvas[y1:y2, x1:x2] = tile_image[: max(0, y2 - y1), : max(0, x2 - x1)]
    return canvas


def _build_visualization_background(
    original: OriginalImage,
    tiles: Sequence[TileMetadata],
    source_images_dir: Optional[Path],
) -> np.ndarray:
    if source_images_dir is not None:
        source_path = source_images_dir / original.file_name
        if source_path.exists():
            image = cv2.imread(str(source_path))
            if image is not None:
                return image
    return _reconstruct_image_array(original, tiles)


def _draw_detections(
    image: np.ndarray,
    detections: Sequence[DetectionRecord],
) -> np.ndarray:
    rendered = image.copy()
    for det in detections:
        x1 = int(round(det.x))
        y1 = int(round(det.y))
        x2 = int(round(det.x + det.width))
        y2 = int(round(det.y + det.height))
        cv2.rectangle(rendered, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det.category_id}:{det.score:.2f}"
        cv2.putText(
            rendered,
            label,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return rendered


def _save_visualization(
    original: OriginalImage,
    tiles: Sequence[TileMetadata],
    detections: Sequence[DetectionRecord],
    *,
    output_path: Path,
    source_images_dir: Optional[Path],
) -> None:
    background = _build_visualization_background(original, tiles, source_images_dir)
    rendered = _draw_detections(background, detections)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), rendered)


def render_visualizations_from_dataset(
    *,
    prediction_dataset: Mapping[str, object],
    fold_original_to_tiles: OriginalToTiles,
    original_images: Mapping[str, OriginalImage],
    output_images_dir: Path,
    source_images_dir: Optional[Path] = None,
    create_mosaics: bool = False,
) -> None:
    """Render final reconstructed images from an already saved COCO prediction dataset."""

    image_id_to_name: Dict[int, str] = {}
    for image_entry in prediction_dataset.get("images", []):
        try:
            image_id = int(image_entry["id"])
            file_name = str(image_entry["file_name"])
        except (KeyError, TypeError, ValueError):
            continue
        image_id_to_name[image_id] = file_name

    detections_by_image: Dict[str, List[DetectionRecord]] = defaultdict(list)
    for ann in prediction_dataset.get("annotations", []):
        try:
            image_id = int(ann["image_id"])
            file_name = image_id_to_name[image_id]
            bbox = ann["bbox"]
            detections_by_image[file_name].append(
                DetectionRecord(
                    x=float(bbox[0]),
                    y=float(bbox[1]),
                    width=float(bbox[2]),
                    height=float(bbox[3]),
                    score=float(ann.get("score", 0.0)),
                    category_id=int(ann["category_id"]),
                )
            )
        except (KeyError, TypeError, ValueError, IndexError):
            continue

    for original_name, tiles in fold_original_to_tiles.items():
        original_meta = original_images.get(original_name)
        if original_meta is None:
            continue

        _save_visualization(
            original_meta,
            tiles,
            detections_by_image.get(original_name, []),
            output_path=output_images_dir / original_name,
            source_images_dir=source_images_dir,
        )

        if create_mosaics:
            mosaic_output_path = output_images_dir.parent / "mosaics" / original_name
            _reconstruct_image(original_meta, tiles, mosaic_output_path)


def detect_tile_orientation(
    original_image: np.ndarray,
    *,
    tile_path: Path,
    offset_x: int,
    offset_y: int,
) -> int:
    """
    Determine whether tiles align with the stored original image (0°) or require a 180° rotation.

    Returns the detected angle (0 or 180 degrees). Defaults to 0 if no better match is found.
    """

    tile_image = cv2.imread(str(tile_path))
    if tile_image is None:
        raise FileNotFoundError(f"Unable to read tile image '{tile_path}'.")
    tile_h, tile_w = tile_image.shape[:2]

    candidates = {
        0: original_image,
        180: cv2.rotate(original_image, cv2.ROTATE_180),
    }

    best_angle = 0
    best_score = float("inf")
    for angle, candidate in candidates.items():
        if offset_y + tile_h > candidate.shape[0] or offset_x + tile_w > candidate.shape[1]:
            continue
        region = candidate[offset_y:offset_y + tile_h, offset_x:offset_x + tile_w]
        if region.shape[:2] != (tile_h, tile_w):
            continue
        diff = cv2.absdiff(region, tile_image)
        score = float(diff.mean())
        if score < best_score:
            best_score = score
            best_angle = angle

    return best_angle


def remap_detections_by_rotation(
    detections: Sequence[DetectionRecord],
    angle: int,
    *,
    image_width: int,
    image_height: int,
) -> List[DetectionRecord]:
    """Remap detections from a rotated frame (currently supports 180°) back to the stored orientation."""
    if angle == 0:
        return list(detections)
    if angle != 180:
        raise NotImplementedError(f"Unsupported rotation angle {angle}.")

    remapped: List[DetectionRecord] = []
    for det in detections:
        new_x = image_width - (det.x + det.width)
        new_y = image_height - (det.y + det.height)
        remapped.append(
            DetectionRecord(
                x=new_x,
                y=new_y,
                width=det.width,
                height=det.height,
                score=det.score,
                category_id=det.category_id,
            )
        )
    return remapped


def build_prediction_dataset(
    *,
    fold_original_to_tiles: OriginalToTiles,
    tile_predictions: TileDetections,
    suppression: SuppressionParams,
    original_images: Mapping[str, OriginalImage],
    base_coco: Mapping[str, object],
    output_images_dir: Path,
    source_images_dir: Optional[Path] = None,
    create_mosaics: bool = False,
    orientation_by_image: Optional[Mapping[str, int]] = None,
) -> tuple[Mapping[str, object], ReconstructionStats]:
    """
    Combine tile detections into original-image predictions and return a COCO-like dict.
    """

    annotations: List[MutableMapping[str, object]] = []
    ann_id = 0
    suppression_time_s = 0.0
    build_start = time.perf_counter()

    for original_name, tiles in fold_original_to_tiles.items():
        original_meta = original_images.get(original_name)
        if original_meta is None:
            raise KeyError(f"Missing metadata for original image '{original_name}'.")

        combined: List[DetectionRecord] = []
        for tile in tiles:
            detections = tile_predictions.get(tile.file_name, [])
            combined.extend(_project_tile_detections(tile, detections))

        suppression_start = time.perf_counter()
        suppressed = apply_suppression(
            combined,
            image_width=original_meta.width,
            image_height=original_meta.height,
            params=suppression,
        )
        suppression_time_s += time.perf_counter() - suppression_start

        if orientation_by_image:
            angle = int(orientation_by_image.get(original_name, 0))
            if angle:
                suppressed = remap_detections_by_rotation(
                    suppressed,
                    angle,
                    image_width=int(original_meta.width),
                    image_height=int(original_meta.height),
                )

        for det in suppressed:
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": original_meta.id,
                    "category_id": det.category_id,
                    "bbox": [det.x, det.y, det.width, det.height],
                    "area": det.width * det.height,
                    "score": det.score,
                }
            )
            ann_id += 1

        visualization_output = output_images_dir / original_name
        _save_visualization(
            original_meta,
            tiles,
            suppressed,
            output_path=visualization_output,
            source_images_dir=source_images_dir,
        )

        if create_mosaics:
            mosaic_output_path = output_images_dir.parent / "mosaics" / original_name
            _reconstruct_image(original_meta, tiles, mosaic_output_path)

    dataset = {
        "info": base_coco.get("info", {}),
        "licenses": base_coco.get("licenses", []),
        "images": base_coco.get("images", []),
        "annotations": annotations,
        "categories": base_coco.get("categories", []),
    }
    total_time_s = time.perf_counter() - build_start
    stats = ReconstructionStats(
        original_image_count=len(fold_original_to_tiles),
        annotation_count=len(annotations),
        suppression_time_s=total_time_s if suppression_time_s > total_time_s else suppression_time_s,
        total_time_s=total_time_s,
    )
    return dataset, stats
