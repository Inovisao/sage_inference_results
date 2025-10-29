from __future__ import annotations

from pathlib import Path
from config_inference import load_config
from pipeline import SageInferencePipeline
import time

def main() -> None:

    print ("Iniciando treinamento! ")
    inicio = time.time()

    project_root = Path(__file__).resolve().parent
    settings = load_config(project_root)
    pipeline = SageInferencePipeline(settings)
    pipeline.run()

    fim = time.time()
    print(f"Tempo de execução: {fim-inicio:.2f} segundos")


if __name__ == "__main__":
    main()
