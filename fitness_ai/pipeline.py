from typing import Dict

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline

from .config import ModelConfig

_DTYPE_MAP: Dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.strip().lower()
    if normalized not in _DTYPE_MAP:
        valid_types = ", ".join(sorted(_DTYPE_MAP.keys()))
        raise ValueError(f"Unsupported dtype '{dtype_name}'. Use one of: {valid_types}")

    return _DTYPE_MAP[normalized]


def create_pipeline(config: ModelConfig) -> StableDiffusionImg2ImgPipeline:
    pipeline_kwargs = {"torch_dtype": _resolve_dtype(config.dtype)}
    if config.disable_safety_checker:
        pipeline_kwargs["safety_checker"] = None
        pipeline_kwargs["requires_safety_checker"] = False

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(config.model_id, **pipeline_kwargs)

    if config.use_dpm_solver:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to(config.device)

    if str(config.device).startswith("cpu"):
        pipe.enable_attention_slicing()

    return pipe
