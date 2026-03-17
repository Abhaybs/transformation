import argparse
from typing import Optional, Sequence

from .benchmark_service import BenchmarkService
from .config import ModelConfig
from .goals import get_goal_preset, list_goals
from .pipeline import create_pipeline
from .schemas import BenchmarkRequest, GenerationRequest
from .transform_service import TransformationService


def _add_model_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-id",
        default="runwayml/stable-diffusion-v1-5",
        help="Hugging Face model identifier for Stable Diffusion img2img.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Runtime device (for example: cpu, cuda, cuda:0).",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "bfloat16"],
        help="Torch data type used to load model weights.",
    )
    parser.add_argument(
        "--enable-safety-checker",
        action="store_true",
        help="Enable Diffusers safety checker. Disabled by default for deterministic outputs.",
    )
    parser.add_argument(
        "--disable-dpm-solver",
        action="store_true",
        help="Use default scheduler instead of DPM-Solver++.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI-Based Fitness Transformation Generator (Diffusion img2img)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate one transformed image from an input photo.",
    )
    generate_parser.add_argument("--input", default="image2.jpg", help="Input image path.")
    generate_parser.add_argument(
        "--output",
        default="output_transformation.png",
        help="Output image path.",
    )
    generate_parser.add_argument(
        "--goal",
        default="muscle_gain",
        choices=list_goals(),
        help="Transformation goal preset.",
    )
    generate_parser.add_argument(
        "--strength",
        type=float,
        default=None,
        help="img2img noise strength. Lower values preserve identity better.",
    )
    generate_parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Classifier-free guidance scale.",
    )
    generate_parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Inference steps.",
    )
    generate_parser.add_argument(
        "--prompt-suffix",
        default="",
        help="Optional prompt extension appended to the selected goal prompt.",
    )
    generate_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducible generation.",
    )
    _add_model_runtime_args(generate_parser)

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run a strength sweep benchmark and save a comparison grid.",
    )
    benchmark_parser.add_argument("--input", default="image2.jpg", help="Input image path.")
    benchmark_parser.add_argument(
        "--grid-output",
        default="fitness_benchmark_results.png",
        help="Output path for grid visualization.",
    )
    benchmark_parser.add_argument(
        "--goal",
        default="muscle_gain",
        choices=list_goals(),
        help="Transformation goal preset.",
    )
    benchmark_parser.add_argument(
        "--strengths",
        nargs="+",
        type=float,
        default=[0.3, 0.45, 0.6],
        help="One or more strength values to benchmark.",
    )
    benchmark_parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Override guidance scale for all benchmark runs.",
    )
    benchmark_parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override inference steps for all benchmark runs.",
    )
    benchmark_parser.add_argument(
        "--prompt-suffix",
        default="",
        help="Optional prompt extension appended to the selected goal prompt.",
    )
    benchmark_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducible benchmark runs.",
    )
    benchmark_parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Disable tile labels for strength and latency.",
    )
    _add_model_runtime_args(benchmark_parser)

    return parser


def _create_transform_service(args: argparse.Namespace) -> TransformationService:
    model_config = ModelConfig(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        disable_safety_checker=not args.enable_safety_checker,
        use_dpm_solver=not args.disable_dpm_solver,
    )
    pipeline = create_pipeline(model_config)
    return TransformationService(pipeline)


def _run_generate(args: argparse.Namespace) -> None:
    preset = get_goal_preset(args.goal)
    service = _create_transform_service(args)

    request = GenerationRequest(
        input_image_path=args.input,
        output_image_path=args.output,
        goal=args.goal,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        prompt_suffix=args.prompt_suffix,
        seed=args.seed,
    )

    print(f"Generating transformation for goal: {preset.display_name}")
    result = service.generate(request)

    print("Transformation complete.")
    print(f"Saved: {result.output_image_path}")
    print(
        f"Strength={result.strength:.2f}, "
        f"Guidance={result.guidance_scale:.2f}, "
        f"Steps={result.num_inference_steps}, "
        f"Latency={result.latency_seconds:.2f}s"
    )


def _run_benchmark(args: argparse.Namespace) -> None:
    preset = get_goal_preset(args.goal)
    service = _create_transform_service(args)
    benchmark_service = BenchmarkService(service)

    request = BenchmarkRequest(
        input_image_path=args.input,
        grid_output_path=args.grid_output,
        goal=args.goal,
        strengths=args.strengths,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        prompt_suffix=args.prompt_suffix,
        seed=args.seed,
        annotate_tiles=not args.no_annotate,
    )

    print(f"Benchmarking goal: {preset.display_name}")
    items, grid_path = benchmark_service.run(request)

    for item in items:
        print(f"Strength={item.strength:.2f} | Latency={item.latency_seconds:.2f}s")

    avg_latency = sum(item.latency_seconds for item in items) / len(items)
    print(f"Average latency: {avg_latency:.2f}s")
    print(f"Grid saved: {grid_path}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate":
        _run_generate(args)
        return

    if args.command == "benchmark":
        _run_benchmark(args)
        return

    parser.error(f"Unsupported command: {args.command}")


def run_interactive_generate(
    input_path: str = "image2.jpg",
    output_path: str = "output_transformation1.png",
) -> None:
    print("Select goal:")
    print("1 - Muscle Gain")
    print("2 - Fat Loss")
    choice = input("Enter choice: ").strip()

    choice_to_goal = {
        "1": "muscle_gain",
        "2": "fat_loss",
    }

    if choice not in choice_to_goal:
        raise SystemExit("Invalid choice. Use 1 for Muscle Gain or 2 for Fat Loss.")

    main(
        [
            "generate",
            "--input",
            input_path,
            "--output",
            output_path,
            "--goal",
            choice_to_goal[choice],
        ]
    )


if __name__ == "__main__":
    main()
