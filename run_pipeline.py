from __future__ import annotations

from pathlib import Path
from pipeline import PipelineSettings, SageInferencePipeline
import time

def main() -> None:

    print ("Iniciando treinamento! ")
    inicio = time.time()

    project_root = Path(__file__).resolve().parent
    settings = PipelineSettings(
        dataset_root=project_root / "dataset",
        models_root=project_root / "model_checkpoints",
        results_root=project_root / "results",
        originals_root=project_root / "original_images_test",
        create_mosaics=True,
    )
    pipeline = SageInferencePipeline(settings)
    pipeline.run()

    fim = time.time()
    print(f"Tempo de execução: {fim-inicio:.2f} segundos")


if __name__ == "__main__":
    main()
