from pathlib import Path
from typing import Tuple

from PIL import Image

from .config import DEFAULT_IMAGE_SIZE


def load_image(image_path: str, image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> Image.Image:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    image = Image.open(path).convert("RGB")
    return image.resize(image_size)


def save_image(image: Image.Image, output_path: str) -> None:
    path = Path(output_path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    image.save(path)
