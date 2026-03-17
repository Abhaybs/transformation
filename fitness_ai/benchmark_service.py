from typing import List, Tuple

from PIL import Image, ImageDraw

from .goals import get_goal_preset
from .image_io import load_image, save_image
from .schemas import BenchmarkItem, BenchmarkRequest
from .transform_service import TransformationService


class BenchmarkService:
    def __init__(self, transformation_service: TransformationService) -> None:
        self.transformation_service = transformation_service

    def run(self, request: BenchmarkRequest) -> Tuple[List[BenchmarkItem], str]:
        if not request.strengths:
            raise ValueError("At least one strength value is required for benchmarking.")

        init_image = load_image(request.input_image_path)
        preset = get_goal_preset(request.goal)

        benchmark_items: List[BenchmarkItem] = []
        generated_images: List[Image.Image] = []

        for strength in request.strengths:
            image, latency, _, _, _ = self.transformation_service.generate_from_image(
                init_image=init_image,
                preset=preset,
                strength=strength,
                guidance_scale=request.guidance_scale,
                num_inference_steps=request.num_inference_steps,
                prompt_suffix=request.prompt_suffix,
                seed=request.seed,
            )

            if request.annotate_tiles:
                self._annotate_tile(image=image, strength=strength, latency=latency)

            benchmark_items.append(BenchmarkItem(strength=strength, latency_seconds=latency))
            generated_images.append(image)

        grid_image = self._create_grid(generated_images)
        save_image(grid_image, request.grid_output_path)

        return benchmark_items, request.grid_output_path

    @staticmethod
    def _annotate_tile(image: Image.Image, strength: float, latency: float) -> None:
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 10, 260, 80), fill=(0, 0, 0))
        text = f"Strength: {strength:.2f}\nLatency: {latency:.2f}s"
        draw.text((18, 18), text, fill=(255, 255, 255))

    @staticmethod
    def _create_grid(images: List[Image.Image]) -> Image.Image:
        if not images:
            raise ValueError("Cannot build benchmark grid from an empty image list.")

        tile_width, tile_height = images[0].size
        grid = Image.new("RGB", (tile_width * len(images), tile_height), color=(30, 30, 30))

        for index, image in enumerate(images):
            grid.paste(image, (index * tile_width, 0))

        return grid
