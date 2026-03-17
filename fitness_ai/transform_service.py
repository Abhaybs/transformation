import time
from typing import Optional, Tuple

import torch
from PIL import Image

from .goals import GoalPreset, get_goal_preset
from .image_io import load_image, save_image
from .schemas import GenerationRequest, GenerationResult


class TransformationService:
    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline

    def generate(self, request: GenerationRequest) -> GenerationResult:
        init_image = load_image(request.input_image_path)
        preset = get_goal_preset(request.goal)

        image, latency, strength, guidance_scale, steps = self.generate_from_image(
            init_image=init_image,
            preset=preset,
            strength=request.strength,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            prompt_suffix=request.prompt_suffix,
            seed=request.seed,
        )

        save_image(image, request.output_image_path)

        return GenerationResult(
            output_image_path=request.output_image_path,
            goal=preset.key,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            latency_seconds=latency,
        )

    def generate_from_image(
        self,
        init_image: Image.Image,
        preset: GoalPreset,
        strength: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        prompt_suffix: str = "",
        seed: Optional[int] = None,
    ) -> Tuple[Image.Image, float, float, float, int]:
        prompt = preset.positive_prompt
        if prompt_suffix.strip():
            prompt = f"{prompt}, {prompt_suffix.strip()}"

        resolved_strength = strength if strength is not None else preset.recommended_strength
        resolved_guidance = (
            guidance_scale
            if guidance_scale is not None
            else preset.recommended_guidance_scale
        )
        resolved_steps = (
            num_inference_steps
            if num_inference_steps is not None
            else preset.recommended_steps
        )

        generator = self._build_generator(seed)

        start_time = time.time()
        output = self.pipeline(
            prompt=prompt,
            negative_prompt=preset.negative_prompt,
            image=init_image,
            strength=resolved_strength,
            guidance_scale=resolved_guidance,
            num_inference_steps=resolved_steps,
            generator=generator,
        )
        latency = time.time() - start_time

        return output.images[0], latency, resolved_strength, resolved_guidance, resolved_steps

    def _build_generator(self, seed: Optional[int]) -> Optional[torch.Generator]:
        if seed is None:
            return None

        device = getattr(self.pipeline, "device", torch.device("cpu"))
        if isinstance(device, torch.device):
            device_name = device.type
        else:
            device_name = str(device)

        return torch.Generator(device=device_name).manual_seed(seed)
