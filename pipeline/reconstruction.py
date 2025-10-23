from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence

import numpy as np
from PIL import Image

from supression.cluster_diou_AIT import adaptive_cluster_diou_nms

from .coco_utils import save_coco_json
from .types import (
    DetectionRecord,
    OriginalImage,
    OriginalToTiles,
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


def _apply_nms_suppression(
    detections: Sequence[DetectionRecord],
    *,
    image_width: int,
    image_height: int,
    params: SuppressionParams,
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

        keep_indices = adaptive_cluster_diou_nms(
            boxes,
            scores,
            T0=params.affinity_threshold,
            alpha=params.lambda_weight,
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
                final.append(clipped)

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


def _reconstruct_image(original: OriginalImage, tiles: Sequence[TileMetadata], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    create_mosaics: bool = False,
) -> Mapping[str, object]:
    """
    Combine tile detections into original-image predictions and return a COCO-like dict.
    """

    annotations: List[MutableMapping[str, object]] = []
    ann_id = 0

    for original_name, tiles in fold_original_to_tiles.items():
        original_meta = original_images.get(original_name)
        if original_meta is None:
            raise KeyError(f"Missing metadata for original image '{original_name}'.")

        combined: List[DetectionRecord] = []
        for tile in tiles:
            detections = tile_predictions.get(tile.file_name, [])
            combined.extend(_project_tile_detections(tile, detections))

        suppressed = _apply_nms_suppression(
            combined,
            image_width=original_meta.width,
            image_height=original_meta.height,
            params=suppression,
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

        if create_mosaics:
            output_path = output_images_dir / original_name
            _reconstruct_image(original_meta, tiles, output_path)

    dataset = {
        "info": base_coco.get("info", {}),
        "licenses": base_coco.get("licenses", []),
        "images": base_coco.get("images", []),
        "annotations": annotations,
        "categories": base_coco.get("categories", []),
    }
    return dataset
