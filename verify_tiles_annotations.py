from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2


try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


@dataclass(frozen=True)
class LabeledBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seleciona imagens aleatórias dos tiles de treino e desenha as anotações YOLO."
    )
    parser.add_argument(
        "--tiles-root",
        type=Path,
        default=Path("dataset/tiles"),
        help="Diretório que contém os folds gerados em tiles.",
    )
    parser.add_argument(
        "--fold",
        type=str,
        default="fold_1",
        help="Nome do fold alvo (por exemplo, fold_1).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Quantidade de imagens aleatórias a serem verificadas.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("verification_tiles"),
        help="Diretório onde as imagens anotadas serão salvas.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semente para o gerador de números aleatórios.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Exibe as imagens anotadas em uma janela interativa.",
    )
    return parser.parse_args(argv)


def resolve_train_dirs(tiles_root: Path, fold: str) -> Tuple[Path, Path]:
    base_dir = tiles_root / fold / "YOLOV5_TPH" / "train"
    images_dir = base_dir / "images"
    labels_dir = base_dir / "labels"
    if not images_dir.exists():
        raise FileNotFoundError(f"Pasta de imagens não encontrada: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Pasta de rótulos não encontrada: {labels_dir}")
    return images_dir, labels_dir


def load_class_names(tiles_root: Path, fold: str) -> List[str]:
    yaml_path = tiles_root / fold / "data_yolov5_tph.yaml"
    if yaml is None or not yaml_path.exists():
        return []
    try:
        with yaml_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except Exception:
        return []
    names = data.get("names")
    if isinstance(names, Iterable):
        return [str(name) for name in names]
    return []


def gather_images(images_dir: Path) -> List[Path]:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    images: List[Path] = []
    for pattern in patterns:
        images.extend(sorted(images_dir.glob(pattern)))
    if not images:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {images_dir}")
    return images


def read_labels(label_path: Path) -> List[LabeledBox]:
    boxes: List[LabeledBox] = []
    if not label_path.exists():
        return boxes
    with label_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"Linha inválida em {label_path} (linha {line_number}): '{line}'")
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = map(float, parts[1:5])
            boxes.append(LabeledBox(class_id, x_center, y_center, width, height))
    return boxes


def yolo_to_xyxy(box: LabeledBox, image_width: int, image_height: int) -> Tuple[int, int, int, int]:
    x_c = box.x_center * image_width
    y_c = box.y_center * image_height
    half_w = (box.width * image_width) / 2
    half_h = (box.height * image_height) / 2
    x1 = max(int(round(x_c - half_w)), 0)
    y1 = max(int(round(y_c - half_h)), 0)
    x2 = min(int(round(x_c + half_w)), image_width - 1)
    y2 = min(int(round(y_c + half_h)), image_height - 1)
    return x1, y1, x2, y2


def draw_annotations(
    image_path: Path,
    labels_dir: Path,
    output_path: Path,
    class_names: Sequence[str],
) -> Tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Não foi possível abrir {image_path}")
    height, width = image.shape[:2]

    label_path = labels_dir / f"{image_path.stem}.txt"
    boxes = read_labels(label_path)
    anomalies = 0

    for box in boxes:
        x1, y1, x2, y2 = yolo_to_xyxy(box, width, height)
        if x1 >= x2 or y1 >= y2:
            anomalies += 1
            continue
        color_seed = (box.class_id * 47) % 255
        color = (color_seed, 255 - color_seed, (color_seed * 2) % 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = class_names[box.class_id] if 0 <= box.class_id < len(class_names) else str(box.class_id)
        cv2.putText(
            image,
            label,
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            lineType=cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return len(boxes), anomalies


def pick_samples(images: Sequence[Path], count: int, seed: Optional[int]) -> List[Path]:
    rng = random.Random(seed)
    if count >= len(images):
        return list(images)
    return rng.sample(list(images), count)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.count <= 0:
        print("O parâmetro --count deve ser positivo.", file=sys.stderr)
        return 2

    try:
        images_dir, labels_dir = resolve_train_dirs(args.tiles_root, args.fold)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    try:
        images = gather_images(images_dir)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    samples = pick_samples(images, args.count, args.seed)
    if not samples:
        print("Nenhuma imagem selecionada.", file=sys.stderr)
        return 1

    class_names = load_class_names(args.tiles_root, args.fold)

    print(f"Total de imagens no treino: {len(images)}")
    print(f"Imagens selecionadas ({len(samples)}):")

    for idx, image_path in enumerate(samples, start=1):
        output_path = args.output / f"{image_path.stem}_checked{image_path.suffix}"
        boxes_count, anomalies = draw_annotations(image_path, labels_dir, output_path, class_names)
        status = "OK" if boxes_count and anomalies == 0 else "VERIFICAR"
        label_summary = f"{boxes_count} caixas"
        if anomalies:
            label_summary += f" ({anomalies} fora dos limites)"
        print(f"{idx}. {image_path.name}: {label_summary} → {output_path.name} [{status}]")
        if args.show:
            annotated = cv2.imread(str(output_path))
            if annotated is not None:
                cv2.imshow(output_path.name, annotated)
                cv2.waitKey(0)
                cv2.destroyWindow(output_path.name)

    if args.show:
        cv2.destroyAllWindows()

    print(f"As imagens anotadas foram salvas em {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

