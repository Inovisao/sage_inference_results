from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from verify_bboxes import (
    draw_boxes,
    load_ground_truth_annotations,
    load_reconstructed_annotations,
    load_tile_boundaries,
    resolve_ground_truth_path,
)


def _discover_root(reconstructed_root: Path, model: Optional[str]) -> Path:
    if model:
        target = reconstructed_root / model
        if not target.exists():
            raise FileNotFoundError(
                f"Modelo '{model}' não encontrado em {reconstructed_root}")
        return target

    if (reconstructed_root / "_annotations.coco.json").exists():
        return reconstructed_root

    subdirs = [path for path in reconstructed_root.iterdir() if path.is_dir()]
    if not subdirs:
        raise FileNotFoundError(
            f"Nenhuma pasta encontrada em {reconstructed_root}")

    candidate_folds = [path for path in subdirs if (
        path / "_annotations.coco.json").exists()]
    if candidate_folds:
        return reconstructed_root

    if len(subdirs) == 1:
        nested = subdirs[0]
        if (nested / "_annotations.coco.json").exists():
            return nested
        nested_folds = [path for path in nested.iterdir() if path.is_dir() and (
            path / "_annotations.coco.json").exists()]
        if nested_folds:
            return nested

    raise FileNotFoundError(
        "Não foi possível determinar automaticamente os folds. "
        "Informe explicitamente o modelo com --model."
    )


def _iter_fold_dirs(root: Path) -> Iterable[Tuple[str, Path]]:
    if (root / "_annotations.coco.json").exists():
        yield (root.name, root)
        return

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        annotations_path = subdir / "_annotations.coco.json"
        if annotations_path.exists():
            yield (subdir.name, subdir)


def _load_ground_truth(dataset_root: Path, fold_name: str) -> Mapping[str, List[List[float]]]:
    try:
        gt_path = resolve_ground_truth_path(dataset_root, fold_name)
    except FileNotFoundError:
        fallback = dataset_root / "train" / "_annotations.coco.json"
        if not fallback.exists():
            raise
        gt_path = fallback
    return load_ground_truth_annotations(gt_path)


def _select_samples(
    detections: Mapping[str, Sequence[Mapping[str, float]]],
    count: int,
    seed: Optional[int],
) -> List[str]:
    rng = random.Random(seed)
    available = [name for name, dets in detections.items() if dets]
    if not available:
        return []
    if count >= len(available):
        return available
    return rng.sample(available, count)


def _ensure_output_dir(base: Path, fold_name: str) -> Path:
    output_dir = base / fold_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _format_output_name(image_name: str) -> str:
    stem = Path(image_name).stem
    return f"{stem}_reconstructed.png"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seleciona imagens reconstruídas por fold e gera visualizações com detecção e ground truth."
    )
    parser.add_argument("--results-root", type=Path,
                        default=Path("results/reconstructed"))
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--model", type=str, help="Nome do modelo dentro de results/reconstructed (ex.: yolov8).")
    parser.add_argument("--images-per-fold", type=int, default=3)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--output-root", type=Path,
                        default=Path("results/reconstructed_samples"))
    parser.add_argument("--seed", type=int, default=None,
                        help="Semente para reprodutibilidade.")
    args = parser.parse_args()

    if args.images_per_fold <= 0:
        raise SystemExit(
            "O parâmetro --images-per-fold deve ser maior que zero.")

    reconstructed_root = args.results_root
    if not reconstructed_root.exists():
        raise FileNotFoundError(f"Pasta {reconstructed_root} não encontrada.")

    target_root = _discover_root(reconstructed_root, args.model)
    print(f"[INFO] Usando diretório base: {target_root}")

    fold_dirs = list(_iter_fold_dirs(target_root))
    if not fold_dirs:
        raise FileNotFoundError(f"Nenhum fold encontrado em {target_root}")

    total_generated = 0
    for fold_name, fold_dir in fold_dirs:
        annotations_path = fold_dir / "_annotations.coco.json"
        if not annotations_path.exists():
            print(
                f"[WARN] Arquivo {annotations_path} ausente; ignorando fold {fold_name}.")
            continue

        detections = load_reconstructed_annotations(annotations_path)
        if not detections:
            print(
                f"[WARN] Nenhuma detecção em {annotations_path}; ignorando fold {fold_name}.")
            continue

        try:
            ground_truth = _load_ground_truth(args.dataset_root, fold_name)
        except FileNotFoundError:
            print(
                f"[WARN] Ground truth para {fold_name} não encontrado; usando sem GT."
            )
            ground_truth = {}

        samples = _select_samples(detections, args.images_per_fold, args.seed)
        if not samples:
            print(
                f"[WARN] Nenhuma imagem com detecções encontradas em {fold_name}.")
            continue

        output_dir = _ensure_output_dir(args.output_root, fold_name)
        images_dir = fold_dir / "images"

        print(f"[INFO] Fold {fold_name}: selecionando {len(samples)} imagens.")
        for image_name in samples:
            output_name = _format_output_name(image_name)
            output_path = output_dir / output_name

            candidate_paths = [
                images_dir / image_name,
                args.dataset_root / "train" / image_name,
            ]

            chosen_path = next(
                (path for path in candidate_paths if path.exists()), None)
            if chosen_path is None:
                print(f"[WARN] Imagem {image_name} não encontrada; pulando.")
                continue

            if chosen_path.parent == images_dir:
                source_label = "reconstructed"
            else:
                source_label = "dataset/train"

            tile_boundaries = load_tile_boundaries(
                args.dataset_root, fold_name, image_name)
            draw_boxes(
                chosen_path,
                ground_truth.get(image_name, []),
                detections.get(image_name, []),
                output_path,
                score_threshold=args.score_threshold,
                tile_boundaries=tile_boundaries,
            )
            total_generated += 1
            print(f"    -> {output_path} (fonte: {source_label})")

    if total_generated == 0:
        print("[WARN] Nenhuma visualização foi gerada.")
    else:
        print(f"[DONE] Visualizações geradas: {total_generated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
