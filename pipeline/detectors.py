from __future__ import annotations

import importlib
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Type

import numpy as np

from .types import DetectionRecord


def _resolve_torch_device(device: Optional[str] = None):
    try:
        import torch  # type: ignore
    except ImportError:  # pragma: no cover - torch not available
        return None, device or "cpu"

    if device is None:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        torch_device = torch.device(device)
    else:
        torch_device = device
    return torch_device, str(torch_device)


class BaseDetector:
    """Common interface for all detector wrappers."""

    model_name: str = "base"
    default_threshold: float = 0.25

    def __init__(self, weight_path: Path, *, device: Optional[str] = None, class_id_offset: int = 0, **kwargs):
        self.weight_path = Path(weight_path)
        self.class_id_offset = class_id_offset
        self.torch_device, self.device_label = _resolve_torch_device(device)
        self.kwargs = kwargs
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        raise NotImplementedError

    def predict(self, image: np.ndarray, threshold: float) -> List[DetectionRecord]:
        raise NotImplementedError

    def close(self) -> None:
        self._model = None

    # context manager helpers
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class YOLOv8Detector(BaseDetector):
    model_name = "yolov8"

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("ultralytics is required for YOLOv8 inference.") from exc

        self._model = YOLO(str(self.weight_path))
        try:
            self._model.fuse()
        except Exception:  # pragma: no cover - fuse may fail on some builds
            warnings.warn("YOLOv8 fuse() call failed; continuing without fusion.", RuntimeWarning)

    def predict(self, image: np.ndarray, threshold: float) -> List[DetectionRecord]:
        if self._model is None:
            raise RuntimeError("YOLOv8 model not loaded.")
        results = self._model.predict(
            image,
            conf=threshold,
            verbose=False,
            device=self.device_label,
        )
        detections: List[DetectionRecord] = []
        for result in results:
            if not hasattr(result, "boxes"):
                continue
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), score, cls in zip(xyxy, scores, classes):
                width = max(0.0, float(x2 - x1))
                height = max(0.0, float(y2 - y1))
                detections.append(
                    DetectionRecord(
                        x=float(x1),
                        y=float(y1),
                        width=width,
                        height=height,
                        score=float(score),
                        category_id=int(cls) + self.class_id_offset,
                    )
                )
        return detections


class YOLOv5TPHDetector(BaseDetector):
    model_name = "yolov5_tph"

    def _load_model(self) -> None:
        module_path = "detectors.YOLOV5_TPH.DetectionsYOLOV5TPH"
        module = importlib.import_module(module_path)
        self._impl = module.ResultYOLOV5TPH  # type: ignore[attr-defined]

    def predict(self, image: np.ndarray, threshold: float) -> List[DetectionRecord]:
        results = self._impl.result(image, str(self.weight_path), threshold)  # type: ignore[attr-defined]
        detections = [
            DetectionRecord(
                x=float(det[0]),
                y=float(det[1]),
                width=float(det[2]),
                height=float(det[3]),
                score=float(det[4]),
                category_id=int(det[5]) + self.class_id_offset,
            )
            for det in results
        ]
        return detections


class RetinaNetDetector(BaseDetector):
    model_name = "retinanet"
    default_threshold = 0.3

    def _load_model(self) -> None:
        module_path = "detectors.RetinaNet.DetectionsRetinaNet"
        module = importlib.import_module(module_path)
        self._impl = module.ResultRetinaNet  # type: ignore[attr-defined]

    def predict(self, image: np.ndarray, threshold: float) -> List[DetectionRecord]:
        results = self._impl.result(image, str(self.weight_path), threshold)  # type: ignore[attr-defined]
        detections: List[DetectionRecord] = []
        for det in results:
            x, y, w, h, label, score = det
            detections.append(
                DetectionRecord(
                    x=float(x),
                    y=float(y),
                    width=float(w),
                    height=float(h),
                    score=float(score),
                    category_id=int(label) + self.class_id_offset,
                )
            )
        return detections


class YOLOv11Detector(BaseDetector):
    model_name = "yolov11"

    def _load_model(self) -> None:
        module_path = "detectors.YOLOV11.DetectionsYOLOV11"
        module = importlib.import_module(module_path)
        self._impl = module.ResultYOLOV11  # type: ignore[attr-defined]

    def predict(self, image: np.ndarray, threshold: float) -> List[DetectionRecord]:
        results = self._impl.result(image, str(self.weight_path), threshold)  # type: ignore[attr-defined]
        detections: List[DetectionRecord] = []
        for det in results:
            x, y, w, h, label, score = det
            detections.append(
                DetectionRecord(
                    x=float(x),
                    y=float(y),
                    width=float(w),
                    height=float(h),
                    score=float(score),
                    category_id=int(label) + self.class_id_offset,
                )
            )
        return detections


class FasterRCNNDetector(BaseDetector):
    model_name = "faster"

    def _load_model(self) -> None:
        try:
            import torch  # type: ignore
            import torchvision  # type: ignore
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("torch and torchvision are required for Faster R-CNN inference.") from exc

        num_classes = int(self.kwargs.get("num_classes", 2))
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        checkpoint = torch.load(self.weight_path, map_location=self.torch_device or "cpu")

        # === Handle multiple checkpoint formats ===
        state_dict = None
        if isinstance(checkpoint, dict):
            # Case 1: plain state_dict
            first_key = next(iter(checkpoint.keys())) if checkpoint else ""
            if any(first_key.startswith(prefix) for prefix in ["backbone.", "rpn.", "roi_heads.", "transform."]):
                state_dict = checkpoint

            # Case 2: wrapped checkpoints
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                model_obj = checkpoint["model"]
                state_dict = (
                    model_obj.state_dict()
                    if hasattr(model_obj, "state_dict")
                    else model_obj
                )

        # Caso 3: YOLOv8 ou outro modelo salvo inteiro
        elif hasattr(checkpoint, "state_dict"):
            state_dict = checkpoint.state_dict()

        # Caso não identificado
        if state_dict is None:
            raise RuntimeError(
                f"Unsupported checkpoint format at {self.weight_path}. "
                f"Found type={type(checkpoint)}, keys={list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'N/A'}"
            )

        if state_dict is not None:
            prefixes = ("model.", "module.")
            cleaned_state_dict = OrderedDict()
            for key, value in state_dict.items():
                new_key = key
                for prefix in prefixes:
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix):]
                cleaned_state_dict[new_key] = value
            state_dict = cleaned_state_dict

        # === Load the state_dict ===
        load_result = model.load_state_dict(state_dict, strict=False)
        missing = getattr(load_result, "missing_keys", [])
        unexpected = getattr(load_result, "unexpected_keys", [])
        if missing:
            warnings.warn(f"Faster R-CNN checkpoint missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            warnings.warn(f"Faster R-CNN checkpoint has unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

        model.to(self.torch_device or "cpu")
        model.eval()
        self._model = model


    def predict(self, image: np.ndarray, threshold: float) -> List[DetectionRecord]:
        if self._model is None:
            raise RuntimeError("Faster R-CNN model not loaded.")
        try:
            import torch  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("torch is required for Faster R-CNN inference.") from exc

        image_rgb = image[:, :, ::-1].copy()  # BGR to RGB (copy to avoid negative strides)
        tensor = torch.from_numpy(image_rgb).float() / 255.0
        tensor = tensor.permute(2, 0, 1).to(self.torch_device or "cpu")
        with torch.no_grad():
            outputs = self._model([tensor])[0]

        boxes = outputs.get("boxes")
        scores = outputs.get("scores")
        labels = outputs.get("labels")
        if boxes is None or scores is None or labels is None:
            return []

        boxes_np = boxes.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        if boxes_np.size == 0:
            return []

        height, width = image.shape[:2]
        max_coord = float(boxes_np.max()) if boxes_np.size else 0.0
        if max_coord <= 1.0 + 1e-6:
            scale = np.array([width, height, width, height], dtype=np.float32)
            boxes_np = boxes_np * scale

        boxes_np[:, [0, 2]] = np.clip(boxes_np[:, [0, 2]], 0.0, float(width))
        boxes_np[:, [1, 3]] = np.clip(boxes_np[:, [1, 3]], 0.0, float(height))

        detections: List[DetectionRecord] = []
        for (x1, y1, x2, y2), score, label in zip(boxes_np, scores_np, labels_np):
            if float(score) < threshold:
                continue
            width_box = max(0.0, float(x2 - x1))
            height_box = max(0.0, float(y2 - y1))
            if width_box == 0.0 or height_box == 0.0:
                continue
            detections.append(
                DetectionRecord(
                    x=float(x1),
                    y=float(y1),
                    width=width_box,
                    height=height_box,
                    score=float(score),
                    category_id=int(label) + self.class_id_offset,
                )
            )
        return detections


DETECTOR_REGISTRY: Dict[str, Type[BaseDetector]] = {
    YOLOv8Detector.model_name: YOLOv8Detector,
    YOLOv5TPHDetector.model_name: YOLOv5TPHDetector,
    FasterRCNNDetector.model_name: FasterRCNNDetector,
    "fasterrcnn": FasterRCNNDetector,
    RetinaNetDetector.model_name: RetinaNetDetector,
    "retina": RetinaNetDetector,
    YOLOv11Detector.model_name: YOLOv11Detector,
    "yolo11": YOLOv11Detector,
}


def resolve_detector(model_name: str) -> Type[BaseDetector]:
    key = model_name.lower()
    if key not in DETECTOR_REGISTRY:
        raise KeyError(f"No detector registered under name '{model_name}'.")
    return DETECTOR_REGISTRY[key]
