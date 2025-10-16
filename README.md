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

## Configuração do ambiente

1. **Python**: recomenda-se Python 3.10 (garante compatibilidade com Torch 1.13.x e Ultralytics 8.x).
2. **Crie um ambiente virtual**:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/macOS
   ```

3. **Instale as dependências**:

   ```bash
   pip install -r requirements.txt
   ```

4. **YOLOv5-TPH**: clone o repositório oficial dentro de `detectors/YOLOV5_TPH/tph-yolov5` e instale as dependências listadas lá (usando o mesmo ambiente). O wrapper `ResultYOLOV5TPH` procura por essa pasta automaticamente.

5. **Pesos pré-treinados**: coloque os arquivos `.pt/.pth/.onnx` em `pesos/<modelo>/fold_X/` respeitando o padrão de nomes (`fold_1`, `fold_2`, ...). A pipeline identifica o índice da dobra a partir do nome do arquivo ou do diretório.

## Fluxos principais

Personalizações relevantes:

- `MODELS`: lista de detectores a treinar/testar (`YOLOV8`, `Faster`, `Detr`, ...).
- Flags como `APENAS_TESTE`, `GeraRult`, `GeraResultByClass`, `save_imgs`.
- `resetar_pasta` limpa artefatos antigos antes de cada execução.

### Pipeline (`run_pipeline.py`)

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

### Avaliação de reconstruções

Depois da pipeline, calcule métricas por dobra/modelo:

```bash
python evaluate_reconstructed.py --dataset-root dataset --results-root results
```

- Gera `results/pipeline_metrics.csv` com agregados.
- Cria `results/details_<modelo>_<fold>.csv` contendo métricas por imagem.

### Depuração visual

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
- As pastas `dataset/` e `pesos/` não são versionadas; garanta que o conteúdo local siga a convenção descrita.

## Scripts úteis

| Script | Descrição |
|--------|-----------|
| `run_pipeline.py` | Entrada única para a pipeline modular (inferência → reconstrução). |
| `ResultsDetections.py` | Consolida métricas padronizadas e gera CSVs. |
| `evaluate_reconstructed.py` | Calcula métricas em reconstruções geradas pela pipeline. |
| `verify_bboxes.py` | Visualização rápida de detecções (threshold configurável). |
| `debug_single_image.py` | Depuração detalhada de uma imagem original (tile + supressão). |


