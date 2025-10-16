import importlib
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

_MODULE_DIR = Path(__file__).resolve().parent


def _detect_repo_path() -> Path:
    candidates = [
        _MODULE_DIR / "tph-yolov5",
        _MODULE_DIR / "tph-yolov5" / "tph-yolov5",
        _MODULE_DIR / "TPH-YOLOV5",
        _MODULE_DIR / "TPH-YOLOV5" / "tph-yolov5",
    ]
    for candidate in candidates:
        if (candidate / "models").is_dir():
            return candidate
    return candidates[0]


REPO_DIR = _detect_repo_path()
_IMPORT_ERROR: Optional[Exception] = None

if REPO_DIR.exists():
    repo_path = str(REPO_DIR)
    if repo_path in sys.path:
        sys.path.remove(repo_path)
    sys.path.insert(0, repo_path)

try:
    from models.experimental import attempt_load  # type: ignore
    from utils.augmentations import letterbox  # type: ignore
    from utils.general import non_max_suppression, scale_coords  # type: ignore
except (ModuleNotFoundError, ImportError) as exc:  # pragma: no cover - captured for diagnostics
    attempt_load = None  # type: ignore
    letterbox = None  # type: ignore
    non_max_suppression = None  # type: ignore
    scale_coords = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


MOSTRAIMAGE = False


def xyxy_to_xywh(boxes: List[List[float]]) -> List[List[float]]:
    coco_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max, conf, cls = box
        w = x_max - x_min
        h = y_max - y_min
        coco_boxes.append([x_min, y_min, w, h, conf, cls])
    return coco_boxes


class ResultYOLOV5TPH:
    _model_cache = {}

    @classmethod
    def _ensure_repo_available(cls) -> None:
        global attempt_load, letterbox, non_max_suppression, scale_coords, _IMPORT_ERROR
        if attempt_load is not None and letterbox is not None and non_max_suppression is not None and scale_coords is not None:
            return

        repo_path = str(REPO_DIR)
        if REPO_DIR.exists() and repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        try:
            models_exp = importlib.import_module("models.experimental")
            utils_aug = importlib.import_module("utils.augmentations")
            utils_general = importlib.import_module("utils.general")
        except (ModuleNotFoundError, ImportError) as exc:
            _IMPORT_ERROR = exc
            details = f" ({exc})"
            raise ImportError(
                "Cannot import YOLOv5 TPH modules. Clone the tph-yolov5 repository into "
                "detectors/YOLOV5_TPH/tph-yolov5 and install its requirements." + details
            ) from exc

        attempt_load = getattr(models_exp, "attempt_load", None)
        letterbox = getattr(utils_aug, "letterbox", None)
        non_max_suppression = getattr(utils_general, "non_max_suppression", None)
        scale_coords = getattr(utils_general, "scale_coords", None)

        if None in (attempt_load, letterbox, non_max_suppression, scale_coords):
            missing = []
            if attempt_load is None:
                missing.append("models.experimental.attempt_load")
            if letterbox is None:
                missing.append("utils.augmentations.letterbox")
            if non_max_suppression is None:
                missing.append("utils.general.non_max_suppression")
            if scale_coords is None:
                missing.append("utils.general.scale_coords")
            raise ImportError(
                "YOLOv5 TPH repository is missing expected symbols: " + ", ".join(missing)
            )

        _IMPORT_ERROR = None

        try:
            from torch import serialization  # type: ignore

            safe_funcs = [
                np.core.multiarray._reconstruct,  # type: ignore[attr-defined]
            ]
            if hasattr(serialization, "add_safe_globals"):
                serialization.add_safe_globals(safe_funcs)  # type: ignore[arg-type]
        except Exception:
            pass

        if not getattr(torch.load, "_yolov5_tph_patched", False):
            original_load = torch.load

            def _patched_torch_load(*args, **kwargs):
                kwargs.setdefault("weights_only", False)
                return original_load(*args, **kwargs)

            _patched_torch_load._original = original_load  # type: ignore[attr-defined]
            _patched_torch_load._yolov5_tph_patched = True  # type: ignore[attr-defined]
            torch.load = _patched_torch_load  # type: ignore[assignment]

    @classmethod
    def _load_model(cls, model_path: str):
        cls._ensure_repo_available()
        cached = cls._model_cache.get(model_path)
        if cached:
            return cached
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = attempt_load(model_path, map_location=device)  # type: ignore[arg-type]
        model.eval()
        cls._model_cache[model_path] = (model, device)
        return model, device

    @classmethod
    def result(cls, frame, model_path: str, threshold: float):
        model, device = cls._load_model(model_path)
        img0 = frame.copy()
        stride = int(model.stride.max()) if hasattr(model, 'stride') else 32
        img = letterbox(img0, 640, stride=stride, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        img_tensor = torch.from_numpy(img).to(device)
        img_tensor = img_tensor.float() / 255.0
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            pred = model(img_tensor)[0]
        pred = non_max_suppression(pred, threshold, 0.45, None, False)[0]

        results = []
        if pred is not None and len(pred):
            pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], img0.shape).round()
            for *xyxy, conf, cls in pred.cpu().numpy():
                x1, y1, x2, y2 = xyxy
                results.append([int(x1), int(y1), int(x2), int(y2), int(cls + 1), float(conf)])
        return xyxy_to_xywh(results)
