# SAGE Inference

Pipeline principal de inferência com **SAHI + Ultralytics** para **YOLOV8** e **YOLOV11**.

O fluxo oficial deste repositório agora é único:

1. lê as imagens originais em `dataset/all`
2. identifica quais imagens pertencem ao `test` de cada fold a partir de `dataset/tiles/fold_X/test`
3. executa inferência fatiada com SAHI
4. junta as detecções na imagem original
5. aplica a supressão escolhida
6. salva COCO final, imagens visualizadas e CSVs consolidados

## Modelos suportados

- `YOLOV8`
- `YOLOV11`

Os pesos são esperados em:

```text
pesos/
  fold_1/
    YOLOV8/train/weights/best.pt
    YOLOV11/train/weights/best.pt
  fold_2/
    ...
```

## Supressões suportadas

- `cluster_diou_nms`
- `nms`
- `nms_ioa`

## Estrutura esperada

```text
dataset/
  all/
    _annotations.coco.json
    *.jpg|png|...
  tiles/
    fold_1/
      train/_annotations.coco.json
      val/_annotations.coco.json
      test/_annotations.coco.json
    fold_2/
    ...
pesos/
  fold_1/
  fold_2/
results/
```

## Instalação

Ambiente recomendado:

- Python `3.10`
- PyTorch `2.1.2`
- torchvision `0.16.2`
- SAHI `0.11.36`
- ultralytics `8.3.161`

Exemplo com `venv`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Se precisar instalar PyTorch GPU manualmente antes do `requirements.txt`:

```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Execução

O entrypoint oficial é [run_inference.py](/home/neto/development/sage_inference_results/run_inference.py:1).

Rodar tudo:

```bash
python run_inference.py
```

Rodar apenas um modelo:

```bash
python run_inference.py --models YOLOV8
```

Rodar um modelo em uma fold:

```bash
python run_inference.py --models YOLOV11 --folds fold_1
```

Rodar múltiplas supressões:

```bash
python run_inference.py --models YOLOV8 --suppressions cluster_diou_nms nms nms_ioa
```

Forçar reprocessamento completo sem reaproveitar saídas antigas:

```bash
python run_inference.py --models YOLOV8 --folds fold_1 --no-resume
```

Todos os argumentos disponíveis:

```bash
python run_inference.py --help
```

`run_pipeline.py` e `run_sahi_inference.py` existem apenas como compatibilidade e redirecionam para o mesmo fluxo.

## Saídas

Para cada combinação `supressão/modelo/fold`, o pipeline salva:

```text
results/reconstructed/<suppression>/<model>/foldX/
  _annotations.coco.json
  images/
    *.jpg|png|...
  per_image_metrics.csv
  run_metadata.json
```

As imagens em `images/` saem com:

- **ground truth em amarelo**
- **predição final do modelo em verde**

## CSVs gerados

Relatório principal por `modelo + supressão + fold`:

- [results/reports/fold_results.csv](/home/neto/development/sage_inference_results/results/reports/fold_results.csv:1)
- [results/reports/results.csv](/home/neto/development/sage_inference_results/results/reports/results.csv:1)
- [results/results.csv](/home/neto/development/sage_inference_results/results/results.csv:1)

Detalhado por imagem:

- [results/reports/image_results.csv](/home/neto/development/sage_inference_results/results/reports/image_results.csv:1)

Sumários derivados:

- [results/reports/summary_by_model.csv](/home/neto/development/sage_inference_results/results/reports/summary_by_model.csv:1)
- [results/reports/summary_by_suppression.csv](/home/neto/development/sage_inference_results/results/reports/summary_by_suppression.csv:1)
- [results/reports/summary_by_model_suppression.csv](/home/neto/development/sage_inference_results/results/reports/summary_by_model_suppression.csv:1)

As colunas principais de `fold_results.csv` são:

- `dataset`
- `suppression`
- `model`
- `fold`
- `images`
- `tiles`
- `precision`
- `recall`
- `f1`
- `mAP`
- `mAP50`
- `mAP75`
- `MAE`
- `RMSE`
- `model_load_time_s`
- `tile_inference_time_s`
- `reconstruction_time_s`
- `suppression_time_s`
- `evaluation_time_s`
- `total_time_s`

## Visualização manual

Para inspecionar uma reconstrução:

```bash
python verify_bboxes.py \
  --results-root results \
  --dataset-root dataset \
  --suppression cluster_diou_nms \
  --model YOLOV8 \
  --fold fold_1
```

## Arquitetura

O pipeline principal foi organizado em estilo funcional:

- `run_inference.py`: composição do fluxo
- `pipeline/data_prep.py`: descoberta de folds e preparação do ground truth por fold
- `pipeline/reconstruction.py`: supressão
- `pipeline/reporting.py`: geração dos CSVs
- `calcula_estatisticas/evaluate_reconstructed.py`: métricas por fold e por imagem

As regras principais são:

- um único entrypoint de execução
- um único layout de saída
- uma fonte de verdade para os CSVs
- SAHI como fluxo oficial para inferência em imagem original

## Referências usadas para a stack

- SAHI docs para integração com Ultralytics: https://obss.github.io/sahi/models/ultralytics/
- SAHI PyPI: https://pypi.org/project/sahi/
- Ultralytics docs: https://docs.ultralytics.com/
