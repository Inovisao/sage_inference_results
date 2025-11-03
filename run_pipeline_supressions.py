from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Sequence

from calcula_estatisticas.evaluate_reconstructed import (
    discover_folds as eval_discover_folds,
    discover_models as eval_discover_models,
    evaluate_fold,
    fold_to_gt_path,
    write_details_csv,
    write_results_csv,
)
from pipeline import PipelineSettings, SageInferencePipeline
from pipeline.types import SuppressionParams

AVAILABLE_METHODS = ["cluster_diou_ait", "cluster_diou_nms", "cluster_diou_bws", "nms", "bws"]


def _default_params(method: str) -> SuppressionParams:
    method_lower = method.lower()
    if method_lower == "cluster_diou_ait":
        extra = {
            "T0": 0.45,
            "alpha": 0.15,
            "k": 5,
            "score_ratio_threshold": 0.85,
            "duplicate_iou_threshold": 0.5,
        }
        return SuppressionParams(method=method_lower, extra=extra)
    if method_lower == "cluster_diou_nms":
        return SuppressionParams(method=method_lower, diou_threshold=0.5)
    if method_lower == "cluster_diou_bws":
        return SuppressionParams(method=method_lower, affinity_threshold=0.4, lambda_weight=0.3)
    if method_lower in {"nms", "bws"}:
        return SuppressionParams(method=method_lower, iou_threshold=0.5)
    raise ValueError(f"Método de supressão não suportado: {method}")


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Executa a pipeline completa para cada método de supressão e gera métricas agregadas."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_root / "dataset",
        help="Diretório raiz do dataset.",
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=project_root / "model_checkpoints",
        help="Diretório com os pesos das redes.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=project_root / "results",
        help="Diretório base onde os resultados por método serão gravados.",
    )
    parser.add_argument(
        "--originals-root",
        type=Path,
        default=project_root / "original_images_test",
        help="Diretório base para salvar as imagens originais reconstruídas por método.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        choices=AVAILABLE_METHODS,
        default=AVAILABLE_METHODS,
        help="Lista opcional de métodos de supressão para executar.",
    )
    parser.add_argument(
        "--create-mosaics",
        action="store_true",
        help="Reconstrói mosaicos RGB das imagens originais.",
    )
    parser.add_argument(
        "--no-create-mosaics",
        dest="create_mosaics",
        action="store_false",
        help="Não reconstrói mosaicos RGB (padrão).",
    )
    parser.set_defaults(create_mosaics=False)
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Executa apenas a inferência/reconstrução, pulando a avaliação.",
    )
    return parser.parse_args()


def _ensure_paths(paths: Sequence[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _evaluate_method(dataset_root: Path, results_root: Path, originals_root: Path) -> None:
    try:
        model_dirs = eval_discover_models(results_root, models=None)
    except FileNotFoundError as exc:
        print(f"[WARN] {exc}")
        return

    results_csv = results_root / "results.csv"
    if results_csv.exists():
        results_csv.unlink()

    aggregate_rows: List[List[str]] = []
    for model_dir in model_dirs:
        model_name = model_dir.name
        fold_dirs = eval_discover_folds(model_dir)
        if not fold_dirs:
            print(f"[WARN] Nenhum fold encontrado para o modelo '{model_name}' em {model_dir}.")
            continue

        for fold_dir in fold_dirs:
            pred_path = fold_dir / "_annotations.coco.json"
            if not pred_path.exists():
                print(f"[WARN] Arquivo de predições não encontrado em {pred_path}. Pulando.")
                continue

            try:
                gt_path = fold_to_gt_path(dataset_root, originals_root, fold_dir.name)
            except FileNotFoundError as exc:
                print(f"[WARN] {exc}")
                continue

            print(f"[INFO] Avaliando {model_name} - {fold_dir.name}")
            per_image, summary = evaluate_fold(pred_path, gt_path)
            write_details_csv(results_root, model_name, fold_dir.name, per_image)
            aggregate_rows.append(
                [
                    model_name,
                    fold_dir.name,
                    str(len(per_image)),
                    f"{summary.precision:.6f}",
                    f"{summary.recall:.6f}",
                    f"{summary.f1:.6f}",
                    f"{summary.map_all:.6f}",
                    f"{summary.map50:.6f}",
                    f"{summary.map75:.6f}",
                    f"{summary.mae:.6f}",
                    f"{summary.rmse:.6f}",
                ]
            )

    if not aggregate_rows:
        print(f"[WARN] Nenhuma métrica agregada gerada para {results_root}.")
        return

    write_results_csv(results_root, aggregate_rows)


def main() -> None:
    args = _parse_args()

    dataset_root = args.dataset_root.resolve()
    models_root = args.models_root.resolve()
    results_root = args.results_root.resolve()
    originals_root = args.originals_root.resolve()

    if not args.methods:
        print("[ERROR] Nenhum método informado. Nada a executar.")
        sys.exit(1)

    for method in args.methods:
        params = _default_params(method)
        method_results_root = results_root / method
        method_originals_root = originals_root / method

        _ensure_paths([method_results_root, method_originals_root])

        print("=" * 80)
        print(f"[INFO] Iniciando pipeline para método '{method}'")
        start_time = time.time()

        settings = PipelineSettings(
            dataset_root=dataset_root,
            models_root=models_root,
            results_root=method_results_root,
            originals_root=method_originals_root,
            suppression=params,
            create_mosaics=args.create_mosaics,
        )

        try:
            pipeline = SageInferencePipeline(settings)
            pipeline.run()
        except Exception as exc:
            print(f"[ERROR] Falha ao executar pipeline para '{method}': {exc}")
            continue

        elapsed = time.time() - start_time
        print(f"[INFO] Pipeline concluída para '{method}' em {elapsed:.1f}s")

        if args.skip_evaluation:
            print(f"[INFO] Avaliação ignorada para '{method}'.")
            continue

        eval_start = time.time()
        _evaluate_method(dataset_root, method_results_root, method_originals_root)
        eval_elapsed = time.time() - eval_start
        print(f"[INFO] Avaliação concluída para '{method}' em {eval_elapsed:.1f}s")

    print("=" * 80)
    print("[DONE] Execução finalizada.")


if __name__ == "__main__":
    main()
