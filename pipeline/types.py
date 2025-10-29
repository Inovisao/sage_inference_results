from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Mapping, MutableMapping, Optional, Sequence


@dataclass(frozen=True)
class OriginalImage:
    """Metadata for an original (não recortada) image from the base dataset."""

    id: int
    file_name: str
    width: int
    height: int
    stem: str


@dataclass(frozen=True)
class TileMetadata:
    """All the contextual information required to map a tile back to its original image."""

    id: int
    file_name: str
    path: Path
    width: int
    height: int
    offset_x: int
    offset_y: int
    original: OriginalImage


@dataclass(frozen=True)
class DetectionRecord:
    """Single detection expressed in COCO (x, y, w, h) format."""

    x: float
    y: float
    width: float
    height: float
    score: float
    category_id: int

    def to_bbox(self) -> List[float]:
        return [self.x, self.y, self.width, self.height]


@dataclass(frozen=True)
class SuppressionParams:
    """Parameters that control the suppression stage."""

    method: str = "cluster_diou_ait"
    affinity_threshold: float = 0.5
    lambda_weight: float = 0.6
    score_ratio_threshold: float = 0.85
    duplicate_iou_threshold: float = 0.5
    iou_threshold: float = 0.5
    diou_threshold: float = 0.5
    extra: Mapping[str, float] = field(default_factory=dict)


@dataclass
class ModelWeights:
    """Mapping of folds to weight files for a particular detector."""

    name: str
    fold_to_path: MutableMapping[int, Path] = field(default_factory=dict)

    def available_folds(self) -> Sequence[int]:
        return sorted(self.fold_to_path.keys())

    def get(self, fold: int) -> Optional[Path]:
        return self.fold_to_path.get(fold)


TileIndex = Mapping[str, TileMetadata]
OriginalToTiles = Mapping[str, List[TileMetadata]]
TileDetections = Mapping[str, Sequence[DetectionRecord]]
OriginalDetections = Mapping[str, List[DetectionRecord]]
