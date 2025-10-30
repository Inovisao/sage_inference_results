from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, Tuple


FILENAME_PATTERN = re.compile(r"^(?P<id>\d+)_jpg\.rf\.[A-Za-z0-9]+\.jpg$")


def build_mapping(images_dir: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for image_path in images_dir.glob("*.jpg"):
        match = FILENAME_PATTERN.match(image_path.name)
        if not match:
            continue
        new_name = f"{match.group('id')}.jpg"
        mapping[image_path.name] = new_name
    return mapping


def ensure_no_conflicts(images_dir: Path, mapping: Dict[str, str]) -> None:
    inverted: Dict[str, str] = {}
    for old_name, new_name in mapping.items():
        if new_name in inverted:
            raise RuntimeError(
                f"Conflito detectado: {old_name} e {inverted[new_name]} geram o mesmo nome {new_name}."
            )
        target_path = images_dir / new_name
        if target_path.exists() and target_path.name not in mapping:
            raise RuntimeError(
                f"O arquivo de destino {target_path} já existe e não será renomeado automaticamente."
            )
        inverted[new_name] = old_name


def rename_files(images_dir: Path, mapping: Dict[str, str]) -> None:
    for old_name, new_name in mapping.items():
        if old_name == new_name:
            continue
        source = images_dir / old_name
        target = images_dir / new_name
        source.rename(target)


def update_coco_json(coco_path: Path, mapping: Dict[str, str]) -> Tuple[int, int]:
    with coco_path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    updated_images = 0
    extras_updated = 0

    for image_entry in coco.get("images", []):
        old_name = image_entry.get("file_name")
        if not isinstance(old_name, str):
            continue
        new_name = mapping.get(old_name)
        if not new_name:
            continue
        image_entry["file_name"] = new_name
        updated_images += 1
        extra = image_entry.get("extra")
        if isinstance(extra, dict):
            extra["name"] = new_name
            extras_updated += 1

    with coco_path.open("w", encoding="utf-8") as handle:
        json.dump(coco, handle, ensure_ascii=False, indent=2)

    return updated_images, extras_updated


def main() -> int:
    images_dir = Path("dataset/train")
    coco_path = images_dir / "_annotations.coco.json"

    if not images_dir.exists():
        print(f"Pasta não encontrada: {images_dir}", file=sys.stderr)
        return 1
    if not coco_path.exists():
        print(f"Arquivo COCO não encontrado: {coco_path}", file=sys.stderr)
        return 1

    mapping = build_mapping(images_dir)
    if not mapping:
        print("Nenhum arquivo corresponde ao padrão esperado.", file=sys.stderr)
        return 1

    ensure_no_conflicts(images_dir, mapping)
    rename_files(images_dir, mapping)
    updated_images, extras_updated = update_coco_json(coco_path, mapping)

    print(f"Arquivos renomeados: {len(mapping)}")
    print(f"Entradas atualizadas em '_annotations.coco.json': {updated_images}")
    print(f"Campos 'extra.name' atualizados: {extras_updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

