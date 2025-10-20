# SAGE – Pipeline de Inferência e Reconstrução

Este repositório reúne scripts de treinamento, inferência e avaliação usados no projeto **SAGE** para detecção de objetos em mosaicos de imagens. Ele nasceu a partir de experimentos com **YOLOv8**, **Faster R-CNN** e **YOLOv5-TPH**, além de utilitários para reconstruir resultados em imagens originais e calcular métricas robustas.

## Visão geral

- `main.py` automatiza o treinamento/avaliação por dobra (fold) usando os detectores suportados.
- `pipeline/` concentra uma nova orquestração modular para inferência → reconstrução → avaliação.
- `ResultsDetections.py` coleta métricas (via `torchmetrics`) em cima dos pesos gerados.
- Scripts auxiliares (`run_pipeline.py`, `evaluate_reconstructed.py`, `verify_bboxes.py`, `debug_single_image.py`) oferecem execuções reproduzíveis e ferramentas de inspeção.

> ⚠️ Observação: o módulo `ResultsDetectionsbyclass.py`, referenciado em `main.py`, não está presente no diretório. Os relatórios por classe precisam desse arquivo para funcionar.

## Estrutura esperada de diretórios

```
results_inference_sage/
├── dataset/
│   ├── all/                      # estrutura antiga (filesJSON, YOLO, etc.) usada por main.py
│   ├── tiles/fold_*/test/        # mosaicos (tiles) por dobra
│   ├── train/                    # imagens originais com _annotations.coco.json
│   └── imagens_originais/        # gerado pela pipeline para reconstruções
├── pesos/
│   ├── yolov8/fold_*/...         # pesos nomeados por dobra
│   └── yolov5_tph/fold_*/...
├── results/
│   ├── prediction/               # saídas de deteção por tile
│   └── reconstructed/...         # COCO + imagens reconstruídas
├── detectors/                    # wrappers específicos por arquitetura
└── pipeline/                     # nova camada de orquestração
```

Adapte os caminhos no arquivo `PipelineSettings` (ver abaixo) se a estrutura do seu dataset diferir.

## Configuracao do ambiente

1. **Python**: recomenda-se Python 3.10 (garante compatibilidade com Torch 1.13.x e Ultralytics 8.x).
2. **Ambiente virtual com venv**:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/macOS
   ```

   Alternativa com Conda (GPU):

   ```bash
   conda create -n sage python=3.10
   conda activate sage
   conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.6 -c pytorch -c nvidia
   pip install -r requirements.txt
   ```

   Para CPU apenas, substitua a instalacao do PyTorch por:

   ```bash
   conda install pytorch==1.13.1 torchvision==0.14.1 cpuonly -c pytorch
   ```

3. **Instale as dependencias**:

   ```bash
   pip install -r requirements.txt
   ```

4. **YOLOv5-TPH**: clone o repositorio oficial dentro de `detectors/YOLOV5_TPH/tph-yolov5` e instale as dependencias listadas la (usando o mesmo ambiente). O wrapper `ResultYOLOV5TPH` procura por essa pasta automaticamente.

5. **Pesos pre-treinados**: coloque os arquivos `.pt/.pth/.onnx` em `pesos/<modelo>/fold_X/` respeitando o padrao de nomes (`fold_1`, `fold_2`, ...). A pipeline identifica o indice da dobra a partir do nome do arquivo ou do diretorio.

## Execucao via Docker

O repositorio inclui um `Dockerfile` baseado em `pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime`, que instala as dependencias do projeto e do wrapper YOLOv5-TPH.

1. **Build da imagem**

   ```bash
   docker build -t sage-inference .
   ```

2. **Monte os diretorios grandes como volumes**

   Os diretorios `dataset/`, `pesos/`, `results/` e `original_images_test/` estao listados em `.dockerignore` para evitar camadas enormes. Monte-os ao executar o container:

   ```bash
   docker run --rm \
     --gpus all \  # remova se nao houver GPU/NVIDIA
     -v ${PWD}/dataset:/app/dataset \
     -v ${PWD}/pesos:/app/pesos \
     -v ${PWD}/results:/app/results \
     -v ${PWD}/original_images_test:/app/original_images_test \
     sage-inference
   ```

   O comando padrao executa `python run_pipeline.py`. Para rodar outro script, sobrescreva o comando:

   ```bash
   docker run --rm -it \
     --gpus all \
     -v ${PWD}/dataset:/app/dataset \
     -v ${PWD}/pesos:/app/pesos \
     -v ${PWD}/results:/app/results \
     sage-inference \
     python debug_single_image.py --model faster --weight pesos/faster/fold_3/best.pth
   ```

3. **Execucao em CPU**

   Se nao houver GPU, altere a imagem base para `python:3.10-slim` e instale os wheels CPU do PyTorch antes de `requirements.txt`:

   ```dockerfile
   RUN pip install --upgrade pip && \
       pip install torch==1.13.1+cpu torchvision==0.14.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu && \
       pip install -r requirements.txt
   ```

   Remova `--gpus all` ao executar o container.

## Fluxos principais

### 1. Treinamento/Evaluação tradicional (`main.py`)

O script percorre as dobras em `dataset/all/filesJSON`, treina cada modelo habilitado em `MODELS` e grava métricas consolidadas em `results/results.csv`.

Uso típico:

```bash
python main.py
```

Personalizações relevantes:

- `MODELS`: lista de detectores a treinar/testar (`YOLOV8`, `Faster`, `Detr`, ...).
- Flags como `APENAS_TESTE`, `GeraRult`, `GeraResultByClass`, `save_imgs`.
- `resetar_pasta` limpa artefatos antigos antes de cada execução.

> Importante: alguns caminhos importam módulos com `Detectors.` (D maiúsculo). Em sistemas sensíveis a case (Linux), assegure-se de manter a capitalização original ao mover/copiar a pasta.

### 2. Pipeline moderna (`run_pipeline.py`)

Para processar as dobras com a nova orquestração:

```bash
python run_pipeline.py
```

O script instancia `PipelineSettings` com os diretórios padrão e executa:

1. Descoberta de pesos por modelo (`pipeline.orchestrator.SageInferencePipeline`).
2. Inferência tile a tile (`pipeline.detectors`).
3. Reconstrução e supressão (`pipeline.reconstruction` + `supression/`).
4. Geração de COCO reconstruído em `results/reconstructed/<modelo>/foldX/_annotations.coco.json`.

Parâmetros personalizáveis em `PipelineSettings`:

- `detection_thresholds`: sobrescreva confiança padrão por modelo.
- `model_class_offsets`: ajuste deslocamentos de IDs de classe quando necessários.
- `create_mosaics`: gera mosaicos RGB com as tiles combinadas.
- `suppression`: configure `affinity_threshold` (IoU) e `lambda_weight` para a supressão híbrida.

### 3. Avaliação de reconstruções

Depois da pipeline, calcule métricas por dobra/modelo:

```bash
python evaluate_reconstructed.py --dataset-root dataset --results-root results
```

- Gera `results/pipeline_metrics.csv` com agregados.
- Cria `results/details_<modelo>_<fold>.csv` contendo métricas por imagem.

### 4. Depuração visual

- `verify_bboxes.py`: desenha detecções reconstruídas em uma imagem. Útil para validação manual.
- `debug_single_image.py`: roda todo o fluxo de uma única imagem (via tiles) exibindo supressão intermediária.

Ambos scripts aceitam argumentos para escolher modelo, dobra, limiar de confiança e destino do arquivo de saída.

## Dependências relevantes

- **Torch/Torchvision 1.13.1**: equilíbrio entre compatibilidade com YOLOv5-TPH e suporte ao Ultralytics.
- **Ultralytics 8.x**: interface oficial para YOLOv8.
- **Supervision**: desenhar caixas e utilitários para YOLOv8.
- **Torchmetrics**: métricas de detecção (mAP, Precision, Recall, etc.).
- **OpenCV & Pillow**: manipulação de imagens e construção de mosaicos.
- **PyYAML**: geração de arquivos `data.yaml` para treinos YOLO.

Consulte `requirements.txt` para versão recomendada e pacotes auxiliares.

## Pontos de atenção

- Certifique-se de que a codificação dos arquivos (`UTF-8`) suporte caracteres acentuados presentes nos scripts. Alguns foram exportados com caracteres corrompidos; regrave se necessário.
- O módulo `Detectors.mminference.inference` é referenciado, mas não está presente no repositório. Se a integração com MMDetection for necessária, adicione a pasta correspondente.
- Em `main.py`, funções para DETR dependem de `Detectors/Detr`, também ausente aqui.
- As pastas `dataset/` e `pesos/` não são versionadas; garanta que o conteúdo local siga a convenção descrita.

## Scripts úteis

| Script | Descrição |
|--------|-----------|
| `main.py` | Loop de treinamento/avaliação clássico por dobra. |
| `run_pipeline.py` | Entrada única para a pipeline modular (inferência → reconstrução). |
| `ResultsDetections.py` | Consolida métricas padronizadas e gera CSVs. |
| `evaluate_reconstructed.py` | Calcula métricas em reconstruções geradas pela pipeline. |
| `verify_bboxes.py` | Visualização rápida de detecções (threshold configurável). |
| `debug_single_image.py` | Depuração detalhada de uma imagem original (tile + supressão). |

## Próximos passos sugeridos

1. Revisar/normalizar encoding dos scripts com acentuação corrompida para evitar exceções em ambientes Unix.
2. Completar os módulos ausentes (`ResultsDetectionsbyclass.py`, `Detectors/Detr`, integ. MMDetection) ou remover as referências se não forem mais utilizados.
3. Automatizar testes rápidos (por exemplo, em `pytest`) para garantir que a pipeline identifique corretamente a presença/ausência de pesos por dobra.

---

Para dúvidas ou melhorias, abra uma issue interna descrevendo o cenário (dataset, pesos e scripts executados). Isso facilita reproduzir o ambiente e acelerar a correção.

