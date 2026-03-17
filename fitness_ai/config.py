from dataclasses import dataclass
from typing import Tuple

DEFAULT_IMAGE_SIZE: Tuple[int, int] = (512, 512)


@dataclass(frozen=True)
class ModelConfig:
    """Model and runtime settings for Stable Diffusion img2img."""

    model_id: str = "runwayml/stable-diffusion-v1-5"
    device: str = "cpu"
    dtype: str = "float32"
    disable_safety_checker: bool = True
    use_dpm_solver: bool = True
