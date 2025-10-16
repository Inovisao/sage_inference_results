from __future__ import annotations

from pathlib import Path

from pipeline import PipelineSettings, SageInferencePipeline


def main() -> None:
    project_root = Path(__file__).resolve().parent
    settings = PipelineSettings(
        dataset_root=project_root / "dataset",
        models_root=project_root / "pesos",
        results_root=project_root / "results",
        originals_root=project_root / "original_images_test",
        create_mosaics=True,
    )
    pipeline = SageInferencePipeline(settings)
    pipeline.run()


if __name__ == "__main__":
    main()
