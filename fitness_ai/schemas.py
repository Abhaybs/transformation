from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class GenerationRequest:
    input_image_path: str
    output_image_path: str
    goal: str
    strength: Optional[float] = None
    guidance_scale: Optional[float] = None
    num_inference_steps: Optional[int] = None
    prompt_suffix: str = ""
    seed: Optional[int] = None


@dataclass(frozen=True)
class GenerationResult:
    output_image_path: str
    goal: str
    strength: float
    guidance_scale: float
    num_inference_steps: int
    latency_seconds: float


@dataclass(frozen=True)
class BenchmarkRequest:
    input_image_path: str
    grid_output_path: str
    goal: str = "muscle_gain"
    strengths: Sequence[float] = (0.3, 0.45, 0.6)
    guidance_scale: Optional[float] = None
    num_inference_steps: Optional[int] = None
    prompt_suffix: str = ""
    seed: Optional[int] = None
    annotate_tiles: bool = True


@dataclass(frozen=True)
class BenchmarkItem:
    strength: float
    latency_seconds: float
