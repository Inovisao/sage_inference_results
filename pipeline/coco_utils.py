from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping

from .types import OriginalImage

JsonDict = MutableMapping[str, object]


def load_coco_json(path: Path) -> JsonDict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_coco_json(data: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def extract_original_images(coco: Mapping[str, object]) -> Dict[str, OriginalImage]:
    images = {}
    for entry in coco.get("images", []):
        file_name = str(entry["file_name"])
        extra = entry.get("extra", {}) if isinstance(entry.get("extra"), dict) else {}
        source_name = extra.get("name") or file_name
        stem = Path(source_name).stem
        images[file_name] = OriginalImage(
            id=int(entry["id"]),
            file_name=file_name,
            width=int(entry.get("width", 0)),
            height=int(entry.get("height", 0)),
            stem=stem,
        )
    return images


def build_image_lookup_by_stem(images: Mapping[str, OriginalImage]) -> Dict[str, OriginalImage]:
    lookup = {}
    for img in images.values():
        lookup[img.stem] = img
    return lookup


def group_annotations_by_image(coco: Mapping[str, object]) -> Dict[int, List[MutableMapping[str, object]]]:
    by_image: Dict[int, List[MutableMapping[str, object]]] = {}
    for ann in coco.get("annotations", []):
        image_id = int(ann["image_id"])
        by_image.setdefault(image_id, []).append(dict(ann))
    return by_image


def filter_coco_dataset(
    coco: Mapping[str, object],
    image_ids: Iterable[int],
    *,
    reassign_ids: bool = True,
) -> JsonDict:
    image_set = set(int(i) for i in image_ids)
    filtered_images: List[MutableMapping[str, object]] = []
    id_mapping: Dict[int, int] = {}

    for entry in coco.get("images", []):
        image_id = int(entry["id"])
        if image_id in image_set:
            id_mapping[image_id] = len(filtered_images)
            new_entry = dict(entry)
            filtered_images.append(new_entry)

    filtered_annotations: List[MutableMapping[str, object]] = []
    for ann in coco.get("annotations", []):
        image_id = int(ann["image_id"])
        if image_id not in image_set:
            continue
        new_ann = dict(ann)
        if reassign_ids:
            new_ann["id"] = len(filtered_annotations)
            new_ann["image_id"] = id_mapping.get(image_id, image_id)
        filtered_annotations.append(new_ann)

    dataset = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": coco.get("categories", []),
    }
    return dataset


def create_empty_coco_template() -> JsonDict:
    return {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }
