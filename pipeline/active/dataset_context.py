from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping

from pipeline.coco_utils import build_image_lookup_by_stem, extract_original_images, load_coco_json
from pipeline.data_prep import build_tile_index, prepare_original_test_split

def load_base_dataset(
    source_images_root: Path,
) -> tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
    base_coco_path = source_images_root / "_annotations.coco.json"
    if not base_coco_path.exists():
        raise FileNotFoundError(f"Base COCO annotations not found at {base_coco_path}")

    base_coco = load_coco_json(base_coco_path)
    original_images = extract_original_images(base_coco)
    original_lookup = build_image_lookup_by_stem(original_images)
    return base_coco, original_images, original_lookup


def resolve_annotations_path(split_dir: Path) -> Path:
    for name in ("_annotations.coco.json", "annotations.coco.json"):
        candidate = split_dir / name
        if candidate.exists():
            return candidate
    return split_dir / "_annotations.coco.json"


def resolve_weight_path(
    models_root: Path,
    model_name: str,
    fold_idx: int,
    model_specs: Mapping[str, Mapping[str, object]],
) -> Path:
    weight_relpath = model_specs[model_name]["weight_relpath"]
    weight_path = models_root / f"fold_{fold_idx}" / weight_relpath
    if not weight_path.exists():
        raise FileNotFoundError(f"Weight for {model_name} fold_{fold_idx} not found at {weight_path}")
    return weight_path


def prepare_fold_context(
    *,
    fold_dir: Path,
    base_coco: Mapping[str, object],
    original_lookup: Mapping[str, object],
    source_images_root: Path,
    originals_root: Path,
) -> tuple[Mapping[str, object], Dict[str, List[object]], Path, int]:
    test_dir = fold_dir / "test"
    _, original_to_tiles = build_tile_index(test_dir, original_lookup)
    filtered_coco_path = prepare_original_test_split(
        base_coco,
        original_to_tiles,
        output_dir=originals_root / fold_dir.name,
        source_images_dir=source_images_root,
    )
    filtered_coco = load_coco_json(filtered_coco_path)
    total_tiles = sum(len(tiles) for tiles in original_to_tiles.values())
    return filtered_coco, original_to_tiles, filtered_coco_path, total_tiles
