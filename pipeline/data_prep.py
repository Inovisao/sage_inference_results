from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Dict, List, Mapping

from . import coco_utils
from .types import OriginalImage, OriginalToTiles, TileIndex, TileMetadata

_FOLD_REGEX = re.compile(r"fold[_\-]?(\d+)", re.IGNORECASE)


def discover_fold_directories(tiles_root: Path) -> List[Path]:
    folds: List[Tuple[int, Path]] = []
    for candidate in tiles_root.iterdir():
        if not candidate.is_dir():
            continue
        match = _FOLD_REGEX.match(candidate.name)
        if not match:
            continue
        index = int(match.group(1))
        folds.append((index, candidate))
    folds.sort(key=lambda item: item[0])
    return [path for _, path in folds]


def parse_tile_filename(file_name: str) -> Tuple[str, int, int]:
    stem = Path(file_name).stem
    if "_tile_" not in stem:
        raise ValueError(f"Tile file '{file_name}' does not contain '_tile_' segment.")
    prefix, _, suffix = stem.partition("_tile_")
    try:
        offset_x_str, offset_y_str = suffix.split("_", 1)
    except ValueError as exc:
        raise ValueError(f"Could not parse offsets from tile file '{file_name}'.") from exc
    try:
        offset_x = int(offset_x_str)
        offset_y = int(offset_y_str)
    except ValueError as exc:
        raise ValueError(f"Offsets extracted from '{file_name}' are not integers.") from exc
    return prefix, offset_x, offset_y


def build_tile_index(
    fold_test_dir: Path,
    train_images_by_stem: Mapping[str, OriginalImage],
) -> Tuple[TileIndex, OriginalToTiles]:
    annotations_path = fold_test_dir / "_annotations.coco.json"
    if not annotations_path.exists():
        annotations_path = fold_test_dir / "annotations.coco.json"
    if not annotations_path.exists():
        raise FileNotFoundError(f"Test annotations not found under {fold_test_dir}")

    coco = coco_utils.load_coco_json(annotations_path)
    tile_index: Dict[str, TileMetadata] = {}
    original_to_tiles: Dict[str, List[TileMetadata]] = {}

    for image_entry in coco.get("images", []):
        file_name = str(image_entry["file_name"])
        stem, offset_x, offset_y = parse_tile_filename(file_name)
        original = train_images_by_stem.get(stem)
        if original is None:
            raise KeyError(f"No original image found for tile '{file_name}' (stem '{stem}').")

        tile_path = fold_test_dir / file_name
        if not tile_path.exists():
            # some datasets keep tiles in a child 'images' folder
            alternative = fold_test_dir / "images" / file_name
            if alternative.exists():
                tile_path = alternative
            else:
                raise FileNotFoundError(f"Tile image '{file_name}' not found in {fold_test_dir}.")

        metadata = TileMetadata(
            id=int(image_entry["id"]),
            file_name=file_name,
            path=tile_path,
            width=int(image_entry.get("width", 0)),
            height=int(image_entry.get("height", 0)),
            offset_x=offset_x,
            offset_y=offset_y,
            original=original,
        )
        tile_index[file_name] = metadata
        original_to_tiles.setdefault(original.file_name, []).append(metadata)

    # Ensure deterministic order per image (top-left to bottom-right)
    for tile_list in original_to_tiles.values():
        tile_list.sort(key=lambda meta: (meta.offset_y, meta.offset_x))

    return tile_index, original_to_tiles


def prepare_original_test_split(
    train_coco: Mapping[str, object],
    original_to_tiles: OriginalToTiles,
    *,
    output_dir: Path,
    source_images_dir: Path,
) -> Path:
    """
    Copy the original images used in the fold's test split and generate a filtered COCO.
    Returns the path to the generated annotations file.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    required_images = set(original_to_tiles.keys())

    train_images_map = {img["file_name"]: img for img in train_coco.get("images", [])}
    image_ids = []
    for file_name in required_images:
        image_entry = train_images_map.get(file_name)
        if image_entry is None:
            raise KeyError(f"Original image '{file_name}' not found in training COCO.")
        image_ids.append(int(image_entry["id"]))

        src_path = source_images_dir / file_name
        if not src_path.exists():
            raise FileNotFoundError(f"Source image '{src_path}' not found.")
        dest_path = output_dir / file_name
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if not dest_path.exists():
            shutil.copy2(src_path, dest_path)

    filtered_coco = coco_utils.filter_coco_dataset(train_coco, image_ids, reassign_ids=False)
    annotations_path = output_dir / "_annotations.coco.json"
    coco_utils.save_coco_json(filtered_coco, annotations_path)
    return annotations_path
