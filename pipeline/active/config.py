from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

import yaml

from pipeline.types import SuppressionParams

from .model_specs import MIN_ULTRALYTICS_VERSION, MODEL_SPECS, SUPPRESSIONS

DEFAULT_SUPPRESSION_DEFAULTS = {
    "affinity_threshold": 0.5,
    "lambda_weight": 0.6,
    "score_ratio_threshold": 0.85,
    "duplicate_iou_threshold": 0.5,
    "iou_threshold": 0.5,
    "diou_threshold": 0.5,
}

DEFAULT_SUPPRESSION_METHODS = {
    "cluster_diou_nms": {
        "diou_threshold": 0.5,
    },
    "nms": {
        "iou_threshold": 0.5,
    },
    "nms_ioa": {
        "ioa_threshold": 0.5,
        "conf_threshold": 0.4,
        "sigma": 0.5,
    },
}


@dataclass(frozen=True)
class ActiveInferenceConfig:
    model_specs: Mapping[str, Mapping[str, object]]
    suppressions: Sequence[str]
    ultralytics_minimum_versions: Mapping[str, tuple[int, int, int]]
    suppression_defaults: Mapping[str, float] = field(default_factory=dict)
    suppression_methods: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def suppression_params(self, method: str) -> SuppressionParams:
        method_key = method.lower().replace("-", "_")
        extra = dict(self.suppression_methods.get(method_key, {}))
        return SuppressionParams(
            method=method,
            affinity_threshold=float(self.suppression_defaults.get("affinity_threshold", 0.5)),
            lambda_weight=float(self.suppression_defaults.get("lambda_weight", 0.6)),
            score_ratio_threshold=float(self.suppression_defaults.get("score_ratio_threshold", 0.85)),
            duplicate_iou_threshold=float(self.suppression_defaults.get("duplicate_iou_threshold", 0.5)),
            iou_threshold=float(self.suppression_defaults.get("iou_threshold", 0.5)),
            diou_threshold=float(self.suppression_defaults.get("diou_threshold", 0.5)),
            extra=extra,
        )


def _deep_update(base: MutableMapping[str, object], overrides: Mapping[str, object]) -> MutableMapping[str, object]:
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _deep_update(base[key], value)  # type: ignore[index,arg-type]
        else:
            base[key] = value
    return base


def _normalize_model_specs(raw_specs: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, object]]:
    normalized: dict[str, dict[str, object]] = {}
    for model_name, spec in raw_specs.items():
        normalized_spec = dict(spec)
        if "weight_relpath" in normalized_spec:
            normalized_spec["weight_relpath"] = Path(str(normalized_spec["weight_relpath"]))
        normalized[model_name] = normalized_spec
    return normalized


def _normalize_min_versions(raw_versions: Mapping[str, object]) -> dict[str, tuple[int, int, int]]:
    normalized: dict[str, tuple[int, int, int]] = {}
    for model_name, version in raw_versions.items():
        if isinstance(version, str):
            parts = tuple(int(part) for part in version.split("."))
        else:
            parts = tuple(int(part) for part in version)  # type: ignore[arg-type]
        if len(parts) != 3:
            raise ValueError(f"Ultralytics minimum version for {model_name} must have three components.")
        normalized[model_name] = parts
    return normalized


def load_active_config(config_path: Path | None = None) -> ActiveInferenceConfig:
    raw: MutableMapping[str, object] = {
        "sahi": {
            "models": deepcopy(MODEL_SPECS),
        },
        "ultralytics": {
            "minimum_versions": deepcopy(MIN_ULTRALYTICS_VERSION),
        },
        "suppressions": {
            "enabled": list(SUPPRESSIONS),
            "defaults": deepcopy(DEFAULT_SUPPRESSION_DEFAULTS),
            "methods": deepcopy(DEFAULT_SUPPRESSION_METHODS),
        },
    }

    if config_path is not None:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, Mapping):
            raise ValueError(f"Configuration file must contain a mapping at the top level: {config_path}")
        _deep_update(raw, loaded)

    sahi_config = raw.get("sahi", {})
    ultralytics_config = raw.get("ultralytics", {})
    suppression_config = raw.get("suppressions", {})
    if not isinstance(sahi_config, Mapping):
        raise ValueError("The 'sahi' configuration section must be a mapping.")
    if not isinstance(ultralytics_config, Mapping):
        raise ValueError("The 'ultralytics' configuration section must be a mapping.")
    if not isinstance(suppression_config, Mapping):
        raise ValueError("The 'suppressions' configuration section must be a mapping.")

    model_specs = _normalize_model_specs(sahi_config.get("models", {}))  # type: ignore[arg-type]
    minimum_versions = _normalize_min_versions(ultralytics_config.get("minimum_versions", {}))  # type: ignore[arg-type]
    suppressions = [str(item) for item in suppression_config.get("enabled", [])]  # type: ignore[union-attr]
    suppression_defaults = {
        str(key): float(value)
        for key, value in dict(suppression_config.get("defaults", {})).items()  # type: ignore[union-attr]
    }
    suppression_methods = {
        str(method): {str(key): float(value) for key, value in dict(values).items()}
        for method, values in dict(suppression_config.get("methods", {})).items()  # type: ignore[union-attr]
    }

    return ActiveInferenceConfig(
        model_specs=model_specs,
        suppressions=suppressions,
        ultralytics_minimum_versions=minimum_versions,
        suppression_defaults=suppression_defaults,
        suppression_methods=suppression_methods,
    )

