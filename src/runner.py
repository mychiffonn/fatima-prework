"""Common infrastructure for running blind-spot evaluations."""

import json
import re
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SEED = 42
_GREEDY_SAMPLER = make_sampler(temp=0.0)

_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def seed_rng(seed: int = SEED) -> None:
    """Set the MLX random seed for reproducibility."""
    mx.random.seed(seed)


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks; if unclosed, remove from <think> onward."""
    result = _THINK_PATTERN.sub("", text)
    if "<think>" in result:
        result = result[: result.index("<think>")]
    return result.strip()


def run_prompt(
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    prompt: str,
    max_tokens: int = 256,
) -> str:
    """Run a prompt and return first-line cleaned response (think tags stripped)."""
    raw = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=_GREEDY_SAMPLER,
        verbose=False,
    )
    cleaned = strip_think_tags(raw)
    if not cleaned:
        return raw.strip().split("\n")[-1].strip()
    return cleaned.split("\n")[0].strip()


def run_single(  # noqa: PLR0913
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    prompt: str,
    *,
    check_fn: Callable[[str], bool],
    label: str,
    expected: str,
    format_output: Callable[[str], str] | None = None,
    strip_think: bool = True,
    max_tokens: int = 256,
) -> tuple[bool, str]:
    """Run a prompt once (deterministic, greedy). Returns (passed, output)."""
    raw = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=_GREEDY_SAMPLER,
        verbose=False,
    )
    if strip_think:
        output = strip_think_tags(raw) or raw.strip()
    else:
        output = raw.strip()

    passed = check_fn(output)
    status = "PASS" if passed else "FAIL"
    display = format_output(output) if format_output else output
    print(f"[{status}] {label}")
    print(f"  Expected: {expected}")
    print(f"  Got:      {display}")
    print()

    return passed, output


def save_results(results: list[dict], filename: str) -> None:
    """Save results to a JSON file in data/."""
    path = DATA_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} results to {path}")
