"""
Utilities to orchestrate the SAGE inference → reconstruction → evaluation pipeline.

The package is organised in the following submodules:

```
pipeline/
    __init__.py          - package marker and convenience exports
    coco_utils.py        - helpers for COCO JSON loading/filtering
    data_prep.py         - discovery of folds, tiles, and original images
    detectors.py         - model-specific inference wrappers
    reconstruction.py    - reprojection of detections + suppression logic
    orchestrator.py      - high-level pipeline coordination
    types.py             - dataclasses shared across modules
```

The entrypoint for most workflows is :class:`pipeline.orchestrator.SageInferencePipeline`.
"""

from .orchestrator import PipelineSettings, SageInferencePipeline

__all__ = ["SageInferencePipeline", "PipelineSettings"]
