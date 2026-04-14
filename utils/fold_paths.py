from __future__ import annotations

from pathlib import Path


def normalize_fold_name(name: str) -> str:
    return name.lower().replace("_", "").replace("-", "")


def resolve_ground_truth_path(dataset_root: Path, fold_name: str) -> Path:
    base_dir = dataset_root / "imagens_originais"
    if not base_dir.exists():
        raise FileNotFoundError(f"Ground-truth directory not found at {base_dir}")

    target_norm = normalize_fold_name(fold_name)
    for candidate in base_dir.iterdir():
        if not candidate.is_dir():
            continue
        if normalize_fold_name(candidate.name) != target_norm:
            continue
        coco_path = candidate / "_annotations.coco.json"
        if coco_path.exists():
            return coco_path

    raise FileNotFoundError(
        f"Ground-truth annotations not found for fold '{fold_name}'. "
        f"Checked under {base_dir}. Ensure the directory contains a matching fold with '_annotations.coco.json'."
    )
