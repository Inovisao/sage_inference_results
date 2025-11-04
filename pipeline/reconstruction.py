from __future__ import annotations

from collections import defaultdict
import time
from pathlib import Path
from typing import Callable, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
from PIL import Image

from .coco_utils import save_coco_json
from .types import (
    DetectionRecord,
    OriginalImage,
    OriginalToTiles,
    SuppressionParams,
    TileDetections,
    TileMetadata,
)


def apply_suppression(
    detections: Sequence[DetectionRecord],
    *,
    image_width: int,
    image_height: int,
    params: SuppressionParams,
) -> List[DetectionRecord]:
    """
    Apply the configured suppression strategy over class-wise detections.
    """
    method = (params.method or "cluster_diou_ait").lower()
    extra = params.extra or {}

    if method == "cluster_diou_ait":
        from pipeline.suppression.adaptive_cluster_diou import adaptive_cluster_diou_nms

        T0 = float(extra.get("T0", 0.45))
        alpha = float(extra.get("alpha", 0.15))
        k = int(extra.get("k", 5))
        score_ratio_thresh = float(extra.get("score_ratio_threshold", extra.get("score_ratio_thresh", 0.85)))
        diou_dup_thresh = float(extra.get("duplicate_iou_threshold", extra.get("diou_dup_thresh", 0.5)))

        return _apply_classwise_suppression(
            detections,
            image_width=image_width,
            image_height=image_height,
            suppression_fn=lambda boxes, scores: adaptive_cluster_diou_nms(
                boxes,
                scores,
                T0=T0,
                alpha=alpha,
                k=k,
                score_ratio_thresh=score_ratio_thresh,
                diou_dup_thresh=diou_dup_thresh,
            ),
            mode="boxes",
        )

    if method == "cluster_ait":
        from supression.cluster_ait import cluster_ait

        T0 = float(extra.get("T0", params.affinity_threshold if params.affinity_threshold is not None else 0.5))
        alpha = float(extra.get("alpha", 0.2))
        k = int(extra.get("k", 5))
        lambda_weight = float(extra.get("lambda_weight", params.lambda_weight if params.lambda_weight is not None else 0.6))

        return _apply_classwise_suppression(
            detections,
            image_width=image_width,
            image_height=image_height,
            suppression_fn=lambda boxes, scores: cluster_ait(
                boxes,
                scores,
                T0=T0,
                alpha=alpha,
                k=k,
                lambda_weight=lambda_weight,
            ),
            mode="boxes",
        )

    if method == "nms":
        from supression.nms import nms

        iou_threshold = float(
            extra.get("iou_threshold", params.iou_threshold if params.iou_threshold is not None else 0.5)
        )
        return _apply_classwise_suppression(
            detections,
            image_width=image_width,
            image_height=image_height,
            suppression_fn=lambda boxes, scores: nms(boxes, scores, iou_thresh=iou_threshold),
            mode="boxes",
        )

    if method == "bws":
        from supression.bws import bws

        iou_threshold = float(
            extra.get("iou_threshold", params.iou_threshold if params.iou_threshold is not None else 0.5)
        )
        return _apply_classwise_suppression(
            detections,
            image_width=image_width,
            image_height=image_height,
            suppression_fn=lambda boxes, scores: bws(boxes, scores, iou_thresh=iou_threshold),
            mode="boxes",
        )

    if method == "cluster_diou_nms":
        from supression.cluster_diou_nms import cluster_diou_nms

        diou_threshold = float(
            extra.get("diou_threshold", params.diou_threshold if params.diou_threshold is not None else 0.5)
        )
        return _apply_classwise_suppression(
            detections,
            image_width=image_width,
            image_height=image_height,
            suppression_fn=lambda boxes, scores: cluster_diou_nms(boxes, scores, diou_thresh=diou_threshold),
            mode="boxes",
        )

    if method == "cluster_diou_bws":
        from supression.cluster_diou_bws import cluster_diou_bws

        affinity_threshold = float(
            extra.get("affinity_threshold", params.affinity_threshold if params.affinity_threshold is not None else 0.5)
        )
        lambda_weight = float(
            extra.get("lambda_weight", params.lambda_weight if params.lambda_weight is not None else 0.6)
        )
        return _apply_classwise_suppression(
            detections,
            image_width=image_width,
            image_height=image_height,
            suppression_fn=lambda boxes, scores: cluster_diou_bws(
                boxes,
                scores,
                affinity_thresh=affinity_threshold,
                lambda_weight=lambda_weight,
            ),
            mode="boxes",
        )

    # Default to adaptive Cluster-DIoU when no known method is provided.
    from pipeline.suppression.adaptive_cluster_diou import adaptive_cluster_diou_nms

    default_k = int(extra.get("k", 5))
    T0 = float(params.affinity_threshold if params.affinity_threshold is not None else 0.45)
    alpha = float(params.lambda_weight if params.lambda_weight is not None else 0.15)
    score_ratio_thresh = float(
        params.score_ratio_threshold if params.score_ratio_threshold is not None else 0.85
    )
    diou_dup_thresh = float(
        params.duplicate_iou_threshold if params.duplicate_iou_threshold is not None else 0.5
    )

    return _apply_classwise_suppression(
        detections,
        image_width=image_width,
        image_height=image_height,
        suppression_fn=lambda boxes, scores: adaptive_cluster_diou_nms(
            boxes,
            scores,
            T0=T0,
            alpha=alpha,
            k=default_k,
            score_ratio_thresh=score_ratio_thresh,
            diou_dup_thresh=diou_dup_thresh,
        ),
        mode="boxes",
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


def _apply_classwise_suppression(
    detections: Sequence[DetectionRecord],
    *,
    image_width: int,
    image_height: int,
    suppression_fn: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray] | Sequence[int]],
    mode: str,
) -> List[DetectionRecord]:
    if not detections:
        return []

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

        result = suppression_fn(boxes, scores)

        if mode == "indices":
            keep_indices = list(result)
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
                    final.append(clipped)
        elif mode == "boxes":
            kept_boxes, kept_scores = result
            if len(kept_boxes) == 0:
                continue
            for box, score in zip(kept_boxes, kept_scores):
                x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                clipped = _clip_detection(
                    DetectionRecord(
                        x=x1,
                        y=y1,
                        width=float(x2 - x1),
                        height=float(y2 - y1),
                        score=float(score),
                        category_id=class_id,
                    ),
                    width=image_width,
                    height=image_height,
                )
                if clipped:
                    final.append(clipped)
        else:
            raise ValueError(f"Unknown suppression mode '{mode}'.")

    final.sort(key=lambda det: det.score, reverse=True)
    return final


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


def collect_projected_detections(
    *,
    fold_original_to_tiles: OriginalToTiles,
    tile_predictions: TileDetections,
    original_images: Mapping[str, OriginalImage],
) -> Tuple[Dict[int, List[DetectionRecord]], Dict[int, OriginalImage]]:
    """
    Project tile detections back to the original image space (pre-suppression).
    Returns both the detections per image id and the associated metadata lookup.
    """
    detections_by_image: Dict[int, List[DetectionRecord]] = {}
    image_meta_by_id: Dict[int, OriginalImage] = {}

    for original_name, tiles in fold_original_to_tiles.items():
        original_meta = original_images.get(original_name)
        if original_meta is None:
            raise KeyError(f"Missing metadata for original image '{original_name}'.")

        image_meta_by_id[original_meta.id] = original_meta
        combined: List[DetectionRecord] = []
        for tile in tiles:
            detections = tile_predictions.get(tile.file_name, [])
            combined.extend(_project_tile_detections(tile, detections))

        clipped: List[DetectionRecord] = []
        for det in combined:
            clipped_det = _clip_detection(det, width=original_meta.width, height=original_meta.height)
            if clipped_det:
                clipped.append(clipped_det)

        detections_by_image[original_meta.id] = clipped

    return detections_by_image, image_meta_by_id


def build_dataset_from_detections(
    *,
    base_coco: Mapping[str, object],
    detections_by_image: Mapping[int, Sequence[DetectionRecord]],
) -> Mapping[str, object]:
    """
    Build a COCO-like dictionary using the provided detections per image id.
    """
    annotations: List[MutableMapping[str, object]] = []
    ann_id = 0

    for image_id, detections in detections_by_image.items():
        for det in detections:
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": det.category_id,
                    "bbox": [det.x, det.y, det.width, det.height],
                    "area": det.width * det.height,
                    "score": det.score,
                }
            )
            ann_id += 1

    dataset = {
        "info": base_coco.get("info", {}),
        "licenses": base_coco.get("licenses", []),
        "images": base_coco.get("images", []),
        "annotations": annotations,
        "categories": base_coco.get("categories", []),
    }
    return dataset


def apply_suppression_to_detections(
    *,
    detections_by_image: Mapping[int, Sequence[DetectionRecord]],
    image_meta_by_id: Mapping[int, OriginalImage],
    params: SuppressionParams,
) -> Dict[int, List[DetectionRecord]]:
    """
    Apply suppression per image using the supplied SuppressionParams.
    """
    suppressed: Dict[int, List[DetectionRecord]] = {}
    for image_id, detections in detections_by_image.items():
        meta = image_meta_by_id.get(image_id)
        if meta is None:
            raise KeyError(f"No metadata registered for image id {image_id}.")
        suppressed[image_id] = apply_suppression(
            detections,
            image_width=meta.width,
            image_height=meta.height,
            params=params,
        )
    return suppressed


def build_raw_detection_dataset(
    *,
    fold_original_to_tiles: OriginalToTiles,
    tile_predictions: TileDetections,
    original_images: Mapping[str, OriginalImage],
    base_coco: Mapping[str, object],
) -> Tuple[Mapping[str, object], Dict[int, List[DetectionRecord]], Dict[int, OriginalImage]]:
    """
    Convenience helper that collects projected detections and builds a raw COCO dataset.
    Returns the dataset along with the projected detections and metadata lookup.
    """
    detections_by_image, image_meta_by_id = collect_projected_detections(
        fold_original_to_tiles=fold_original_to_tiles,
        tile_predictions=tile_predictions,
        original_images=original_images,
    )
    raw_dataset = build_dataset_from_detections(
        base_coco=base_coco,
        detections_by_image=detections_by_image,
    )
    return raw_dataset, detections_by_image, image_meta_by_id


def _reconstruct_image(
    original: OriginalImage,
    tiles: Sequence[TileMetadata],
    output_path: Path,
    *,
    source_images_dir: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_path = source_images_dir / original.file_name
    if base_path.exists():
        with Image.open(base_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            canvas = img.copy()
    else:
        canvas = Image.new("RGB", (int(original.width), int(original.height)))
    for tile in tiles:
        with Image.open(tile.path) as tile_img:
            if tile_img.mode != "RGB":
                tile_img = tile_img.convert("RGB")
            canvas.paste(tile_img, (int(tile.offset_x), int(tile.offset_y)))
    canvas.save(output_path)


def build_prediction_dataset(
    *,
    fold_original_to_tiles: OriginalToTiles,
    tile_predictions: TileDetections,
    suppression: SuppressionParams,
    original_images: Mapping[str, OriginalImage],
    base_coco: Mapping[str, object],
    output_images_dir: Path,
    source_images_dir: Path,
    create_mosaics: bool = False,
    projected_detections: Mapping[int, Sequence[DetectionRecord]] | None = None,
    image_meta_by_id: Mapping[int, OriginalImage] | None = None,
) -> Mapping[str, object]:
    """
    Combine tile detections into original-image predictions and return a COCO-like dict.
    """

    if projected_detections is None or image_meta_by_id is None:
        projected_detections, image_meta_by_id = collect_projected_detections(
            fold_original_to_tiles=fold_original_to_tiles,
            tile_predictions=tile_predictions,
            original_images=original_images,
        )

    suppression_start = time.perf_counter()
    suppressed_by_image = apply_suppression_to_detections(
        detections_by_image=projected_detections,
        image_meta_by_id=image_meta_by_id,
        params=suppression,
    )
    suppression_elapsed = time.perf_counter() - suppression_start
    print(f"        [suppression] Completed in {suppression_elapsed:.2f}s")

    dataset = build_dataset_from_detections(
        base_coco=base_coco,
        detections_by_image=suppressed_by_image,
    )

    if create_mosaics:
        mosaic_start = time.perf_counter()
        total_mosaics = len(fold_original_to_tiles)
        for index, (original_name, tiles) in enumerate(fold_original_to_tiles.items(), start=1):
            original_meta = original_images.get(original_name)
            if original_meta is None:
                raise KeyError(f"Missing metadata for original image '{original_name}'.")
            output_path = output_images_dir / original_name
            _reconstruct_image(
                original_meta,
                tiles,
                output_path,
                source_images_dir=source_images_dir,
            )
            if index % 25 == 0 or index == total_mosaics:
                print(
                    f"        [mosaic] Reconstructed {index}/{total_mosaics} original images "
                    f"({output_path.parent})"
                )
        mosaic_elapsed = time.perf_counter() - mosaic_start
        print(f"        [mosaic] Finished in {mosaic_elapsed:.2f}s")

    return dataset
