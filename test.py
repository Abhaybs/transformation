from fitness_ai.cli import main


if __name__ == "__main__":
    main(
        [
            "benchmark",
            "--input",
            "image2.jpg",
            "--goal",
            "muscle_gain",
            "--strengths",
            "0.3",
            "0.45",
            "0.6",
            "--grid-output",
            "fitness_benchmark_results.png",
        ]
    )