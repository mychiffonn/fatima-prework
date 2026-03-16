"""Run blind-spot evaluations against Qwen base model and save results.

Usage:
    uv run scripts/run_model.py                  # run all scenarios
    uv run scripts/run_model.py reversal math    # run specific scenarios
    uv run scripts/run_model.py --list           # list available scenarios
"""

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_lm import load
from src.runner import save_results, seed_rng

MODEL_ID = "Qwen/Qwen3.5-4B-Base"

SCENARIOS: dict[str, tuple[str, str, str]] = {
    # name: (module_path, function_name, output_filename)
    "reversal": ("src.reversal", "run_reversal", "reversal.json"),
    "math": ("src.math_reasoning", "run_math", "math.json"),
    "mcq": ("src.mcq", "run_mcq", "mcq.json"),
    # "trolley": ("src.trolley", "run_trolley", "trolley.json"),
    # "trolley_cultural": (
    #     "src.trolley_cultural",
    #     "run_trolley_cultural",
    #     "trolley_cultural.json",
    # ),
    # "trolley_multilingual": (
    #     "src.trolley_multilingual",
    #     "run_trolley_multilingual",
    #     "trolley_multilingual.json",
    # ),
    "hallucination": ("src.hallucination", "run_hallucination", "hallucination.json"),
}


def main() -> None:
    """Load model and run selected evaluations."""
    parser = argparse.ArgumentParser(description="Run blind-spot evaluations.")
    parser.add_argument(
        "scenarios",
        nargs="*",
        choices=list(SCENARIOS),
        help="Scenarios to run (default: all)",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available scenarios and exit"
    )
    args = parser.parse_args()

    if args.list:
        for name in SCENARIOS:
            print(f"  {name}")
        return

    selected = args.scenarios if args.scenarios else list(SCENARIOS)

    seed_rng()
    print(f"Loading {MODEL_ID} ...")
    model, tokenizer = load(MODEL_ID)  # type: ignore[reportCallIssue]
    print("Model loaded.\n")

    for name in selected:
        module_path, func_name, filename = SCENARIOS[name]
        module = importlib.import_module(module_path)
        run_fn = getattr(module, func_name)
        results = run_fn(model, tokenizer)
        save_results(results, filename)
        print()


if __name__ == "__main__":
    main()
