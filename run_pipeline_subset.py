from __future__ import annotations

import argparse
import random
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import cv2

from pipeline import PipelineSettings, SageInferencePipeline
from pipeline.coco_utils import load_coco_json, save_coco_json
from pipeline.data_prep import build_tile_index, discover_fold_directories, prepare_original_test_split
from pipeline.reconstruction import build_prediction_dataset, collect_projected_detections
from pipeline.types import ModelWeights, SuppressionParams, TileMetadata


_FOLD_REGEX = re.compile(r"fold[_\-]?(\d+)", re.IGNORECASE)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa a pipeline em um subconjunto (N imagens por fold)."
    )
    parser.add_argument("--limit-per-fold", type=int, default=10,
                        help="Quantidade de imagens originais por fold.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Semente opcional para amostragem aleatória.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--models-root", type=Path,
                        default=Path("model_checkpoints"))
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results_subset",
        help="Subpasta dentro do projeto onde os resultados serão gravados.",
    )
    parser.add_argument(
        "--originals-dir",
        type=str,
        default="original_images_test_subset",
        help="Subpasta onde as imagens originais reconstruídas serão salvas.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Lista opcional de modelos (nomes) a serem executados. Ex.: --models Faster YOLOV8",
    )
    parser.add_argument(
        "--create-mosaics",
        action="store_true",
        help="Reconstrói mosaicos RGB (igual ao run_pipeline padrão).",
    )
    parser.add_argument(
        "--no-create-mosaics",
        dest="create_mosaics",
        action="store_false",
        help="Desativa reconstrução de mosaicos.",
    )
    parser.set_defaults(create_mosaics=True)
    return parser.parse_args()


def _select_originals(
    mapping: Mapping[str, Sequence[TileMetadata]],
    limit: int,
    seed: Optional[int],
) -> Dict[str, Sequence[TileMetadata]]:
    names = sorted(mapping.keys())
    if limit <= 0:
        return {}
    if len(names) <= limit:
        return {name: mapping[name] for name in names}

    rng = random.Random(seed)
    selected = rng.sample(names, limit)
    selected.sort()
    return {name: mapping[name] for name in selected}


def _filter_tile_index(
    tile_index: Mapping[str, TileMetadata],
    selected_names: Iterable[str],
) -> Dict[str, TileMetadata]:
    allowed = set(selected_names)
    return {
        tile_name: metadata
        for tile_name, metadata in tile_index.items()
        if metadata.original.file_name in allowed
    }


def _filter_model_specs(
    specs: Sequence[ModelWeights],
    allowed: Optional[Sequence[str]],
) -> List[ModelWeights]:
    if not allowed:
        return list(specs)
    allowed_lower = {name.lower() for name in allowed}
    filtered = [spec for spec in specs if spec.name.lower() in allowed_lower]
    missing = allowed_lower.difference(
        {spec.name.lower() for spec in filtered})
    if missing:
        print(
            f"[WARN] Modelos não encontrados nos pesos: {', '.join(sorted(missing))}")
    return filtered


def main() -> None:
    args = _parse_args()
    if args.limit_per_fold <= 0:
        raise SystemExit("O parâmetro --limit-per-fold deve ser positivo.")

    project_root = Path(__file__).resolve().parent
    results_root = project_root / args.results_dir
    originals_root = project_root / args.originals_dir

    settings = PipelineSettings(
        dataset_root=args.dataset_root.resolve(),
        models_root=args.models_root.resolve(),
        results_root=results_root,
        originals_root=originals_root,
        create_mosaics=args.create_mosaics,
        suppression=SuppressionParams(method="nms", iou_threshold=0.5),
    )

    pipeline = SageInferencePipeline(settings)
    folds = discover_fold_directories(pipeline.tiles_root)
    if not folds:
        print(f"[WARN] Nenhum fold encontrado em {pipeline.tiles_root}.")
        return

    model_specs = pipeline._discover_model_weights()
    model_specs = _filter_model_specs(model_specs, args.models)
    if not model_specs:
        print("[WARN] Nenhum modelo com pesos disponível para executar.")
        return

    start_time = time.time()
    print(
        f"[INFO] Rodando subset da pipeline ({args.limit_per_fold} imagens por fold, "
        f"mosaicos={'on' if args.create_mosaics else 'off'})"
    )

    for fold_dir in folds:
        fold_match = _FOLD_REGEX.match(fold_dir.name)
        if not fold_match:
            continue
        fold_idx = int(fold_match.group(1))
        print(f"\n[INFO] Processando {fold_dir.name} (fold {fold_idx})")

        test_dir = fold_dir / "test"
        tile_index, original_to_tiles = build_tile_index(
            test_dir, pipeline.original_images_by_stem)

        subset_originals = _select_originals(
            original_to_tiles, args.limit_per_fold, args.seed)
        if not subset_originals:
            print(
                f"[WARN] Nenhuma imagem selecionada para {fold_dir.name}. Pulando.")
            continue

        subset_tile_index = _filter_tile_index(
            tile_index, subset_originals.keys())
        total_tiles = len(subset_tile_index)
        print(
            f"[INFO] Selecionadas {len(subset_originals)} imagens ({total_tiles} tiles) "
            f"de um total de {len(original_to_tiles)} imagens."
        )

        originals_output_dir = pipeline.originals_root / f"fold{fold_idx}"
        annotations_path = prepare_original_test_split(
            pipeline.train_coco,
            subset_originals,
            output_dir=originals_output_dir,
            source_images_dir=pipeline.train_images_dir,
        )
        filtered_coco = load_coco_json(annotations_path)

        for spec in model_specs:
            weight_path = spec.get(fold_idx)
            if weight_path is None:
                print(
                    f"[WARN] Modelo '{spec.name}' não possui pesos para fold {fold_idx}.")
                continue

            print(
                f"[INFO]  +- Executando modelo '{spec.name}' com pesos '{weight_path.name}'")
            model_start = time.time()
            tile_predictions: MutableMapping[str, Sequence] = {}
            detector = None
            try:
                detector = pipeline._instantiate_detector(
                    spec.name, weight_path)
                with detector:
                    for idx, (tile_name, metadata) in enumerate(sorted(subset_tile_index.items()), start=1):
                        image = cv2.imread(str(metadata.path))
                        if image is None:
                            raise FileNotFoundError(
                                f"Não foi possível ler a tile '{metadata.path}'.")
                        threshold = pipeline.detection_thresholds.get(
                            spec.name.lower(), detector.threshold)
                        detections = detector.predict(image, threshold)
                        tile_predictions[tile_name] = detections
                        if idx % 50 == 0 or idx == total_tiles:
                            print(
                                f"        [fold {fold_idx}][{spec.name}] {idx}/{total_tiles} tiles processados "
                                f"(último: {tile_name}, detecções={len(detections)})"
                            )
            finally:
                if detector is not None:
                    detector.close()

            if not tile_predictions:
                print(
                    f"[WARN] Nenhuma detecção para '{spec.name}' no fold {fold_idx}. Pulando.")
                continue

            reconstructed_dir = pipeline.results_root / "reconstructed" / spec.name / f"fold{fold_idx}"
            images_dir = reconstructed_dir / "images"
            reconstructed_dir.mkdir(parents=True, exist_ok=True)

            projected_detections, image_meta = collect_projected_detections(
                fold_original_to_tiles=subset_originals,
                tile_predictions=tile_predictions,
                original_images=pipeline.original_images,
            )

            suppression_method = (pipeline.suppression.method or "unknown").lower()
            detection_counts = [len(dets) for dets in projected_detections.values()]
            if detection_counts:
                total_dets = sum(detection_counts)
                avg_dets = total_dets / len(detection_counts)
                print(
                    f"[INFO]  +- Aplicando supressão ({suppression_method}) em {len(detection_counts)} imagens | "
                    f"detecções: total={total_dets}, média={avg_dets:.1f}, "
                    f"mín={min(detection_counts)}, máx={max(detection_counts)}"
                )

            annotations_output = reconstructed_dir / "_annotations.coco.json"
            predictions = build_prediction_dataset(
                fold_original_to_tiles=subset_originals,
                tile_predictions=tile_predictions,
                suppression=pipeline.suppression,
                original_images=pipeline.original_images,
                base_coco=filtered_coco,
                output_images_dir=images_dir,
                source_images_dir=pipeline.train_images_dir,
                create_mosaics=pipeline.create_mosaics,
                projected_detections=projected_detections,
                image_meta_by_id=image_meta,
            )
            save_coco_json(predictions, annotations_output)
            elapsed = time.time() - model_start
            print(
                f"[INFO]  +- Modelo '{spec.name}' concluído em {elapsed:.1f}s. "
                f"Resultados em {annotations_output}"
            )

    total_elapsed = time.time() - start_time
    print(
        f"\n[DONE] Pipeline reduzida concluída em {total_elapsed/60:.1f} min.")


if __name__ == "__main__":
    main()
