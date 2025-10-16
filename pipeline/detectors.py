from __future__ import annotations

import importlib
import warnings
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
        state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else checkpoint
        if state_dict is None:
            raise RuntimeError(f"Unexpected checkpoint format for {self.weight_path}")
        model.load_state_dict(state_dict)
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

        image_rgb = image[:, :, ::-1]  # BGR to RGB
        tensor = torch.from_numpy(image_rgb).float() / 255.0
        tensor = tensor.permute(2, 0, 1).to(self.torch_device or "cpu")
        with torch.no_grad():
            outputs = self._model([tensor])[0]

        boxes = outputs.get("boxes")
        scores = outputs.get("scores")
        labels = outputs.get("labels")
        if boxes is None or scores is None or labels is None:
            return []

        detections: List[DetectionRecord] = []
        for (x1, y1, x2, y2), score, label in zip(boxes.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy()):
            if float(score) < threshold:
                continue
            detections.append(
                DetectionRecord(
                    x=float(x1),
                    y=float(y1),
                    width=max(0.0, float(x2 - x1)),
                    height=max(0.0, float(y2 - y1)),
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
}


def resolve_detector(model_name: str) -> Type[BaseDetector]:
    key = model_name.lower()
    if key not in DETECTOR_REGISTRY:
        raise KeyError(f"No detector registered under name '{model_name}'.")
    return DETECTOR_REGISTRY[key]
