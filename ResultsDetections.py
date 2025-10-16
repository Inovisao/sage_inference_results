import os
import argparse
import json
import numpy as np
import cv2
import torch
import torchmetrics
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, PearsonCorrCoef
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassAccuracy
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAccuracy
import shutil
import sys
import csv

# Importações dos modelos de detecção
from detectors.yolov5_tph.DetectionsYOLOV5TPH import ResultYOLOV5TPH
from Detectors.YOLOV8.DetectionsYolov8 import resultYOLO
from Detectors.FasterRCNN.inference import ResultFaster
from Detectors.Detr.inference_image_detect import resultDetr
from Detectors.mminference.inference import runMMdetection

# Constantes
LIMIAR_THRESHOLD = 0.50
IOU_THRESHOLD = 0.50
RESULTS_PATH = os.path.join("..", "results", "prediction")
os.makedirs(RESULTS_PATH, exist_ok=True)  # Garante que a pasta existe

def print_to_file(line='', file_path='../results/results.csv', mode='a'):
    """Função para escrever uma linha em um arquivo."""
    try:
        dir_path = os.path.dirname(file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        original_stdout = sys.stdout  # Salva a referência para a saída padrão original
        with open(file_path, mode) as f:
            sys.stdout = f  # Altera a saída padrão para o arquivo criado
            print(line)
            sys.stdout = original_stdout  # Restaura a saída padrão para o valor original
    except Exception as e:
        print(f"[ERRO] Falha ao escrever no arquivo {file_path}: {e}")

def generate_csv(data):
    """Gera um arquivo CSV com os dados fornecidos."""
    file_name = '../results/counting.csv'
    headers = ['ml', 'weight', 'fold', 'groundtruth', 'predicted', 'TP', 'FP', 'dif', 'fileName']
    try:
        dir_path = os.path.dirname(file_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        with open(file_name, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            for row in data:
                writer.writerow(row)
    except Exception as e:
        print(f"[ERRO] Falha ao salvar CSV de contagem em {file_name}: {e}")

def get_classes(json_path):
    """Extrai as classes de um arquivo JSON no formato COCO."""
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return {category["id"]: category["name"] for category in data["categories"]}

def load_dataset(fold_path):
    """Carrega o dataset a partir de um arquivo JSON."""
    with open(fold_path, 'r') as f:
        data = json.load(f)

    image_info_list = []
    for image in data['images']:
        image_id = image['id']
        file_name = image['file_name']
        annotations = [annotation for annotation in data['annotations'] if annotation['image_id'] == image_id]
        
        bboxes = [annotation['bbox'] for annotation in annotations]
        labels = [annotation['category_id'] for annotation in annotations]
        
        annotation_info = {
            'bboxes': bboxes,
            'labels': labels,
            'bboxes_ignore': np.array([]),
            'masks': [[]],
            'seg_map': file_name
        }
        
        image_info = {
            'image_id': image_id,
            'file_name': file_name,
            'annotations': annotation_info
        }
        image_info_list.append(image_info)

    return image_info_list

def xywh_to_xyxy(bbox):
    """Converte bbox de formato (x, y, w, h) para (x_min, y_min, x_max, y_max)."""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]

def calculate_iou(box1, box2):
    """Calcula a interseção sobre união (IoU) entre duas bboxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x1_min, y1_min, x1_max, y1_max = x1, y1, x1 + w1, y1 + h1
    x2_min, y2_min, x2_max, y2_max = x2, y2, x2 + w2, y2 + h2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height

    area1 = w1 * h1
    area2 = w2 * h2

    union_area = area1 + area2 - inter_area
    iou = inter_area / union_area if union_area > 0 else 0

    return iou

def resolve_dataset_paths(root, fold):
    """
    Resolve os caminhos relevantes para arquivos de anotações e imagens de acordo com a organização do dataset.
    Aceita tanto o layout legado (filesJSON + train) quanto a nova estrutura em tiles por fold.
    """
    legacy_test_json = os.path.join(root, 'filesJSON', f'{fold}_test.json')
    legacy_annotations = os.path.join(root, 'train', '_annotations.coco.json')

    if os.path.exists(legacy_test_json) and os.path.exists(legacy_annotations):
        return {
            'test_json_path': legacy_test_json,
            'annotations_path': legacy_annotations,
            'image_base_dir': os.path.join(root, 'train'),
            'fold_dir': None,
            'structure': 'legacy'
        }

    fold_dir = os.path.join(root, fold)
    test_json_path = os.path.join(fold_dir, 'test', '_annotations.coco.json')
    annotations_path = os.path.join(fold_dir, 'train', '_annotations.coco.json')
    image_base_dir = os.path.join(fold_dir, 'test')

    if not os.path.exists(test_json_path):
        raise FileNotFoundError(
            f"Não foi possível localizar o arquivo de anotações de teste para o fold '{fold}'. "
            f"Caminho verificado: {test_json_path}"
        )

    if not os.path.exists(annotations_path):
        raise FileNotFoundError(
            f"Não foi possível localizar o arquivo de anotações de treino para o fold '{fold}'. "
            f"Caminho verificado: {annotations_path}"
        )

    return {
        'test_json_path': test_json_path,
        'annotations_path': annotations_path,
        'image_base_dir': image_base_dir,
        'fold_dir': fold_dir,
        'structure': 'tiles'
    }

def process_predictions(ground_truth, predictions, classes, save_img, image_base_dir, fold, model_name, weight_alias=None):
    """Processa as previsões e calcula métricas como TP, FP, precisão e recall."""
    ground_truth_list = []
    predict_list = []
    ground_truth_list_count = []
    predict_list_count = []
    data = []
    model_run_id = model_name if not weight_alias else f"{model_name}_{weight_alias}"
    for key in predictions:
        img_path = os.path.join(image_base_dir, key)
        image = cv2.imread(img_path)
        save_current_image = True
        if image is None:
            print(f"[AVISO] Não foi possível carregar a imagem '{img_path}'. Usando um canvas vazio para manter o fluxo das métricas.")
            image = np.zeros((512, 512, 3), dtype=np.uint8)
            save_current_image = False

        gt_count = len(ground_truth[key])
        pred_count = len(predictions[key])
        ground_truth_list_count.append(gt_count)
        predict_list_count.append(pred_count)
        cv2.putText(image, f"GT: {gt_count}", (5, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 1)
        cv2.putText(image, f"PRED: {pred_count}", (5, 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 1)

        true_positives = 0
        false_positives = 0
        matched_gt = set()

        for bbox_pred in predictions[key]:
            x1_max, y1_max = int(bbox_pred[0] + bbox_pred[2]), int(bbox_pred[1] + bbox_pred[3])
            best_iou = 0
            best_gt = None

            for i, bbox_gt in enumerate(ground_truth[key]):
                x2_max, y2_max = int(bbox_gt[0] + bbox_gt[2]), int(bbox_gt[1] + bbox_gt[3])
                iou = calculate_iou(bbox_pred[:4], bbox_gt[:4])
                cv2.putText(image, str(classes[bbox_gt[-1]]), (int(bbox_gt[0]), int(bbox_gt[1]+5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                if iou >= IOU_THRESHOLD and iou > best_iou and i not in matched_gt:
                    best_iou = iou
                    best_gt = i

            if best_gt is not None:
                matched_gt.add(best_gt)
                gt_class = ground_truth[key][best_gt][-1]

                ground_truth_list.append(gt_class)
                predict_list.append(bbox_pred[4])

                color = (0, 255, 0) if gt_class == bbox_pred[4] else (0, 0, 255)
                cv2.rectangle(image, (int(bbox_pred[0]), int(bbox_pred[1])), (int(x1_max), int(y1_max)), color, thickness=2)
                cv2.putText(image, str(classes[bbox_pred[4]]), (int(bbox_pred[0]), int(y1_max)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                if gt_class == bbox_pred[4]:
                    true_positives += 1
                else:
                    false_positives += 1
            else:
                cv2.rectangle(image, (int(bbox_pred[0]), int(bbox_pred[1])), (int(x1_max), int(y1_max)), (0, 0, 255), thickness=2)
                cv2.putText(image, str(classes[bbox_pred[4]]), (int(bbox_pred[0]), int(y1_max)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                ground_truth_list.append(0)  # Falso Positivo
                predict_list.append(bbox_pred[4])
                false_positives += 1

        for i, bbox_gt in enumerate(ground_truth[key]):
            if i not in matched_gt:
                x2_max, y2_max = int(bbox_gt[0] + bbox_gt[2]), int(bbox_gt[1] + bbox_gt[3])
                cv2.rectangle(image, (int(bbox_gt[0]), int(bbox_gt[1])), (x2_max, y2_max), (255, 0, 0), thickness=2)
                cv2.putText(image, str(classes[bbox_gt[-1]]), (x2_max, y2_max), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                ground_truth_list.append(bbox_gt[-1])
                predict_list.append(0)  # Falso Negativo

        precision = round(true_positives / (true_positives + false_positives), 3) if (true_positives + false_positives) > 0 else 0
        recall = round(true_positives / gt_count, 3) if gt_count > 0 else 0

        cv2.putText(image, f"P: {precision}", (5, 90), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255), 1)
        cv2.putText(image, f"R: {recall}", (5, 120), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255), 1)
        
        if save_img and save_current_image:
            try:
                save_path = os.path.join(RESULTS_PATH, model_run_id, fold, 'all_classes')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_path = os.path.join(save_path, key)
                cv2.imwrite(save_path, image)
            except Exception as e:
                print(f"[ERRO] Falha ao salvar imagem de predição em {save_path}: {e}")
        data.append({
            'ml': model_name,
            'weight': weight_alias or '',
            'fold': fold,
            'groundtruth': gt_count,
            'predicted': pred_count,
            'TP': true_positives,
            'FP': false_positives,
            'dif': int(gt_count - pred_count),
            'fileName': key
        })
    generate_csv(data)
    ground_truth_list_count = torch.tensor(ground_truth_list_count)
    predict_list_count = torch.tensor(predict_list_count)

    pearson = PearsonCorrCoef()
    r = pearson(predict_list_count.float(), ground_truth_list_count.float())
    return ground_truth_list, predict_list,r

def compute_metrics(preds, targets, num_classes=1):
    """Calcula métricas de classificação como precisão, recall, F1-score e acurácia."""
    preds = torch.tensor(preds)
    targets = torch.tensor(targets)
    
    if num_classes <= 2:
        precision = BinaryPrecision()(preds, targets)
        recall = BinaryRecall()(preds, targets)
        fscore = BinaryF1Score()(preds, targets)
        accuracy = BinaryAccuracy()(preds, targets)
    else:
        precision = MulticlassPrecision(num_classes=num_classes, average='macro')(preds, targets)
        recall = MulticlassRecall(num_classes=num_classes, average='macro')(preds, targets)
        fscore = MulticlassF1Score(num_classes=num_classes, average='macro')(preds, targets)
        accuracy = MulticlassAccuracy(num_classes=num_classes, average='macro')(preds, targets)
    

            # Para multiclasse, converter logits para probabilidades com softmax
        #preds_prob = preds.float().softmax(dim=1).argmax(dim=1)  # Pegando a classe mais provável

    return precision.item(), recall.item(), fscore.item()

def generate_results(root, fold, model, model_name, save_imgs, weight_alias=None):
    """Gera resultados para um modelo específico e salva as métricas."""
    dataset_paths = resolve_dataset_paths(root, fold)

    classes_dict = get_classes(dataset_paths['annotations_path'])
    coco_test = load_dataset(dataset_paths['test_json_path'])
    predictions = {}
    ground_truth = {}
    for image in coco_test:

        ground_truth_list = []
        for i, bbox in enumerate(image['annotations']['bboxes']):
            x1, y1, width, height = bbox
            label = image["annotations"]['labels'][i]
            ground_truth_list.append([x1, y1, width, height, label])
        ground_truth[image['file_name']] = ground_truth_list
        image_path = os.path.join(dataset_paths['image_base_dir'], image['file_name'])

        frame = cv2.imread(image_path)

        model_name_lower = model_name.lower()
        if model_name_lower == "yolov8":
            result = resultYOLO.result(frame, model,LIMIAR_THRESHOLD)
        elif model_name_lower in {"faster", "fasterrcnn"}:
            print(image_path)
            result = ResultFaster.resultFaster(frame,model,LIMIAR_THRESHOLD)
        elif model_name_lower == "detr":
            print(image_path)
            result = resultDetr(fold,frame,LIMIAR_THRESHOLD)
        elif model_name_lower in {"yolov5_tph", "yolov5tph"}:
            result = ResultYOLOV5TPH.result(frame, model, LIMIAR_THRESHOLD)
        else:
            print(image_path)
            result = runMMdetection(model,frame,LIMIAR_THRESHOLD)
        predictions[image['file_name']] = result

    ground_truth_map = []
    predictions_map = []

    for key in ground_truth:
        bbox_list = []
        label_list = []
        for values in ground_truth[key]:
            bbox = xywh_to_xyxy(values[:4])
            bbox_list.append(bbox)
            label_list.append(values[-1])
        boxes_tensor = torch.tensor(bbox_list, dtype=torch.float32) if bbox_list else torch.empty((0, 4), dtype=torch.float32)
        labels_tensor = torch.tensor(label_list, dtype=torch.int64) if label_list else torch.empty((0,), dtype=torch.int64)
        ground_truth_map.append({"boxes": boxes_tensor, "labels": labels_tensor})
        
    for key in predictions:
        bbox_list = []
        label_list = []
        score_list = []
        for values in predictions[key]:
            bbox = xywh_to_xyxy(values[:4])
            bbox_list.append(bbox)
            label_list.append(values[4])
            score_list.append(values[5])
        boxes_tensor = torch.tensor(bbox_list, dtype=torch.float32) if bbox_list else torch.empty((0, 4), dtype=torch.float32)
        labels_tensor = torch.tensor(label_list, dtype=torch.int64) if label_list else torch.empty((0,), dtype=torch.int64)
        scores_tensor = torch.tensor(score_list, dtype=torch.float32) if score_list else torch.empty((0,), dtype=torch.float32)
        predictions_map.append({"boxes": boxes_tensor, "scores": scores_tensor, "labels": labels_tensor})

    metric = MeanAveragePrecision()
    metric.update(predictions_map, ground_truth_map)
    result_map = metric.compute()

    ground_truth_counts = []
    for key in ground_truth:
        count_classes = [0] * len(classes_dict)
        for bbox in ground_truth[key]:
            count_classes[bbox[-1]] += 1
        ground_truth_counts.append(count_classes)
    ground_truth_counts = torch.tensor(ground_truth_counts)

    prediction_counts = []
    for key in predictions:
        count_classes = [0] * len(classes_dict)
        for bbox in predictions[key]:
            for gt_bbox in ground_truth[key]:
                iou = calculate_iou(bbox[:4], gt_bbox[:4])
                if bbox[4] == gt_bbox[-1] and iou >= IOU_THRESHOLD:
                    count_classes[bbox[4]] += 1
        prediction_counts.append(count_classes)
    prediction_counts = torch.tensor(prediction_counts)

    pred_counts = prediction_counts.sum(dim=1)
    gt_counts = ground_truth_counts.sum(dim=1)

    mae = MeanAbsoluteError()(pred_counts, gt_counts)
    rmse = MeanSquaredError(squared=False)(pred_counts, gt_counts)

    mAP = result_map["map"]
    mAP50 = result_map["map_50"]
    mAP75 = result_map["map_75"]

    ground_truth_list, predict_list, r = process_predictions(
        ground_truth,
        predictions,
        classes_dict,
        save_imgs,
        dataset_paths['image_base_dir'],
        fold,
        model_name,
        weight_alias=weight_alias
    )

    num_classes = len(classes_dict)
    precision, recall, fscore = compute_metrics(predict_list, ground_truth_list, num_classes=num_classes)

    return mAP.item(), mAP50.item(), mAP75.item(), mae.item(), rmse.item(), precision, recall, fscore, r.item()

def create_csv(selected_model, fold, root, model_path, save_imgs, weight_alias=None):
    """Cria um arquivo CSV com os resultados das métricas."""
    try:
        mAP, mAP50, mAP75, MAE, RMSE, precision, recall, fscore, r = generate_results(
            root,
            fold,
            model_path,
            selected_model,
            save_imgs,
            weight_alias=weight_alias
        )
        results_path = os.path.join('..', 'results', 'results.csv')
        file_exists = os.path.isfile(results_path)
        dir_path = os.path.dirname(results_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        with open(results_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["ml", "weight", "fold", "mAP", "mAP50", "mAP75", "MAE", "RMSE", "accuracy", "precision", "recall", "fscore"])
            writer.writerow([selected_model, weight_alias or '', fold, mAP, mAP50, mAP75, MAE, RMSE, r, precision, recall, fscore])
        print(f"[INFO] Resultados salvos com sucesso em {results_path}")
    except Exception as e:
        print(f"[ERRO] Falha ao salvar resultados em {results_path}: {e}")


def _iter_weight_files(weights_dir, extensions):
    """
    Itera por todos os arquivos de pesos dentro da pasta fornecida.
    Considera cada subpasta imediata como o nome do modelo de detecção.
    """
    for entry in sorted(os.listdir(weights_dir)):
        model_dir = os.path.join(weights_dir, entry)
        if not os.path.isdir(model_dir):
            continue

        weight_paths = []
        for dirpath, _, filenames in os.walk(model_dir):
            for filename in filenames:
                if os.path.splitext(filename)[1].lower() in extensions:
                    weight_paths.append(os.path.join(dirpath, filename))

        if not weight_paths:
            print(f"[AVISO] Nenhum arquivo de peso suportado encontrado em {model_dir}.")
            continue

        for weight_path in sorted(weight_paths):
            rel_path = os.path.relpath(weight_path, model_dir)
            alias = os.path.splitext(rel_path)[0].replace(os.sep, "__")
            yield entry, alias, weight_path


def evaluate_weight_directory(weights_dir, dataset_root, save_imgs=False, folds=None, extensions=None):
    """
    Percorre uma pasta com múltiplos pesos de modelos, aplicando a inferência e o cálculo das métricas
    para cada fold de teste presente em dataset_root.
    """
    if extensions is None:
        extensions = {'.pt', '.pth', '.bin'}

    weights_dir = os.path.abspath(weights_dir)
    dataset_root = os.path.abspath(dataset_root)

    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(f"A pasta de pesos '{weights_dir}' não existe ou não é um diretório.")

    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"A pasta de dataset '{dataset_root}' não existe ou não é um diretório.")

    if folds is None:
        folds = sorted([
            entry for entry in os.listdir(dataset_root)
            if entry.startswith('fold_') and os.path.isdir(os.path.join(dataset_root, entry))
        ])

    if not folds:
        raise ValueError(f"Nenhum fold encontrado em '{dataset_root}'. Verifique se a estrutura segue o padrão fold_X.")

    # Valida se os arquivos de anotações existem antes de iniciar as inferências
    for fold in folds:
        try:
            resolve_dataset_paths(dataset_root, fold)
        except FileNotFoundError as err:
            raise FileNotFoundError(f"Falha ao preparar o fold '{fold}': {err}") from err

    for model_name, alias, weight_path in _iter_weight_files(weights_dir, extensions):
        print(f"[INFO] Avaliando modelo '{model_name}' com pesos '{alias}' ({weight_path}).")
        for fold in folds:
            create_csv(
                selected_model=model_name,
                fold=fold,
                root=dataset_root,
                model_path=weight_path,
                save_imgs=save_imgs,
                weight_alias=alias
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Gera métricas de detecção para vários pesos e folds.")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Caminho para a pasta contendo subpastas com pesos (ex.: YOLOV8/best.pt, Faster/best.pth, ...)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(".", "dataset", "tiles"),
        help="Caminho para a pasta com os folds (padrão: ./dataset/tiles)."
    )
    parser.add_argument(
        "--folds",
        nargs="*",
        help="Lista opcional de folds a serem processados (ex.: fold_1 fold_2). Se omitido, todos os folds serão utilizados."
    )
    parser.add_argument(
        "--save-imgs",
        action="store_true",
        help="Se definido, salva as visualizações com as detecções em ../results/prediction."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_weight_directory(
        weights_dir=args.weights,
        dataset_root=args.dataset,
        save_imgs=args.save_imgs,
        folds=args.folds
    )
