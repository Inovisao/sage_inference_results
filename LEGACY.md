# Legacy and Experimental Code

This repository keeps legacy and experimental modules because historical runs
may depend on their exact behavior. Do not remove or rewrite these files unless
the corresponding result-regeneration workflow has been audited.

The canonical active flow is:

```text
run_inference.py -> pipeline.active.runner
```

Legacy or compatibility-oriented areas include:

- `pipeline/orchestrator.py`
- `pipeline/detectors.py`
- `run_all_yolov8_suppressions.py`
- `debug_single_image.py`
- `run_random_single_image_yolov8.py`
- suppression methods that are present on disk but not listed by the active
  `pipeline.active.model_specs.SUPPRESSIONS`

These modules are retained to preserve reproducibility of older experiments.

