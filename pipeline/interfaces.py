from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from .types import DetectionRecord


class DetectorProtocol(Protocol):
    """Structural interface expected from detector wrappers."""

    model_name: str
    default_threshold: float
    weight_path: Path
    class_id_offset: int
    device_label: str

    def predict(self, image: np.ndarray, threshold: float) -> list[DetectionRecord]:
        ...

    def close(self) -> None:
        ...


class DetectorFactoryProtocol(Protocol):
    """Callable interface for detector classes."""

    model_name: str

    def __call__(
        self,
        weight_path: Path,
        *,
        device: Optional[str] = None,
        class_id_offset: int = 0,
        **kwargs: object,
    ) -> DetectorProtocol:
        ...

