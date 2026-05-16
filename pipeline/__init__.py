"""
Core modules for the SAGE inference pipeline.

This package intentionally avoids eager imports so submodules such as
`pipeline.reporting` and `pipeline.reconstruction` can be imported without
pulling in legacy orchestration code.
"""

from __future__ import annotations

from importlib import import_module

__all__ = ["PipelineSettings", "SageInferencePipeline"]


def __getattr__(name: str):
    if name in {"PipelineSettings", "SageInferencePipeline"}:
        module = import_module("pipeline.orchestrator")
        return getattr(module, name)
    raise AttributeError(f"module 'pipeline' has no attribute {name!r}")
