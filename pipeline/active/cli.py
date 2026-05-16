from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

from .model_specs import MODEL_SPECS, SUPPRESSIONS


def parse_requested(items: Sequence[str] | None, available: Iterable[str]) -> List[str]:
    available_list = list(available)
    available_by_lower = {item.lower(): item for item in available_list}
    if not items:
        return available_list

    resolved: List[str] = []
    for item in items:
        match = available_by_lower.get(item.lower())
        if match is None:
            raise KeyError(f"Unknown item '{item}'. Available: {', '.join(sorted(available_by_lower.values()))}")
        resolved.append(match)
    return resolved


def parse_folds(items: Sequence[str] | None) -> Sequence[int] | None:
    if not items:
        return None
    resolved = []
    for item in items:
        normalized = item.lower().replace("_", "").replace("-", "")
        if normalized.startswith("fold"):
            normalized = normalized[4:]
        resolved.append(int(normalized))
    return resolved


def build_parser(project_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Principal SAHI-based inference pipeline for YOLOV8 and YOLOV11. "
            "Select the models and suppression methods you want to evaluate."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=project_root / "dataset")
    parser.add_argument("--source-images-root", type=Path, default=project_root / "dataset" / "all")
    parser.add_argument("--models-root", type=Path, default=project_root / "pesos")
    parser.add_argument("--results-root", type=Path, default=project_root / "results")
    parser.add_argument("--reports-root", type=Path, default=None)
    parser.add_argument("--originals-root", type=Path, default=None)
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "configs" / "inference.yaml",
        help="Optional YAML configuration file with SAHI/model/suppression parameters.",
    )
    parser.add_argument("--dataset-name", default="sage")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--models",
        nargs="*",
        help="Subset of models to execute. Defaults to all supported models.",
    )
    parser.add_argument(
        "--suppressions",
        nargs="*",
        help="Subset of suppression methods to execute. Defaults to all supported methods.",
    )
    parser.add_argument("--folds", nargs="*", help="Optional list of folds, e.g. fold_1 fold_3 or 1 3.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing outputs and rebuild everything.")
    return parser

__all__ = ["build_parser", "parse_folds", "parse_requested", "MODEL_SPECS", "SUPPRESSIONS"]
