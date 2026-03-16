"""Mathematical reasoning: ConvexBench and arithmetic.

Sources:
- ConvexBench: https://arxiv.org/abs/2602.01075
"""

import re
from dataclasses import dataclass, field

from mlx import nn
from mlx_lm.tokenizer_utils import TokenizerWrapper

from src.runner import run_single


@dataclass
class MathCase:
    """A single math reasoning test case."""

    question: str
    expected: str
    tags: list[str] = field(default_factory=lambda: ["math"])


CONVEX_INSTRUCTION = """\
Determine whether the given function is convex, concave, or neither.

Explain your reasoning step by step using composition rules and second-derivative tests, \
then state your final answer: "convex", "concave", or "neither".
"""

ARITHMETIC_INSTRUCTION = """\
Solve the math problem step by step. Show your work, \
then state your final answer as a single number after "Answer:".
"""


def build_prompt(question: str, instruction: str) -> str:
    """Build a zero-shot instruction prompt for math reasoning."""
    return f"{instruction}Question: {question}\nAnswer:"


def _extract_convexity(text: str) -> str:
    """Extract convex/concave/neither classification from model output."""
    lower = text.lower()
    if "answer:" in lower:
        lower = lower.split("answer:")[-1]
    if "neither" in lower:
        return "neither"
    if "concave" in lower:
        return "concave"
    if "convex" in lower:
        return "convex"
    return ""


def _check_pass(case: MathCase, output: str) -> bool:
    """Check whether model output matches the expected answer."""
    if "convexbench" in case.tags:
        return _extract_convexity(output) == case.expected
    pattern = r"\b" + re.escape(case.expected) + r"\b"
    return bool(re.search(pattern, output))


CONVEX_CASES: list[MathCase] = [
    # Depth 4, convex: affine→exp→negate→log→negate+scale→exp
    MathCase(
        question=(
            "Is f(x) = exp(-2*log(3*(1 - exp(0.5*x - 5)) + 1) + 0.5)"
            " convex, concave, or neither?"
        ),
        expected="convex",
        tags=["convexbench"],
    ),
    # Depth 4, convex: affine→exp→negate→log→negate+scale→exp
    MathCase(
        question=(
            "Is f(x) = exp(1 - 2*log(5 - exp(2 - x))) convex, concave, or neither?"
        ),
        expected="convex",
        tags=["convexbench"],
    ),
    # Depth 5, convex: affine→exp→negate→scale→log→negate+scale→exp
    MathCase(
        question=(
            "Is f(x) = exp(-3*log(2*(1 - exp(-0.5*x + 1)) + 1) + 0.5)"
            " convex, concave, or neither?"
        ),
        expected="convex",
        tags=["convexbench"],
    ),
    # Depth 4, concave: affine→exp→negate→sqrt→log
    MathCase(
        question=(
            "Is f(x) = log(1 + sqrt(3 - exp(0.5*x - 1))) convex, concave, or neither?"
        ),
        expected="concave",
        tags=["convexbench"],
    ),
    # Depth 2, concave: negated log-sum-exp
    MathCase(
        question=(
            "Is f(x) = -log(exp(-2*x) + exp(3*x)) convex, concave, or neither on R?"
        ),
        expected="concave",
        tags=["convexbench"],
    ),
    # Depth 2, neither: f''(x) = (2-2x^2)/(x^2+1)^2 changes sign
    MathCase(
        question="Is f(x) = log(x^2 + 1) convex, concave, or neither on R?",
        expected="neither",
        tags=["convexbench"],
    ),
    # Depth 4, neither: sin causes oscillation
    MathCase(
        question=(
            "Is f(x) = exp(sin(log(x^2 + 1) - x)) convex, concave, or neither on R?"
        ),
        expected="neither",
        tags=["convexbench"],
    ),
]

ARITHMETIC_CASES: list[MathCase] = [
    MathCase(
        question="What is 3^5 - 2^7 + 15 * 4?",
        expected="175",
        tags=["math_arithmetic"],
    ),
    MathCase(
        question="What is the remainder when 7^100 is divided by 13?",
        expected="9",
        tags=["math_arithmetic"],
    ),
    MathCase(
        question="If f(x) = 2*x + 1 and g(x) = x^2 - 3, what is f(g(f(2)))?",
        expected="45",
        tags=["math_arithmetic"],
    ),
]


def run_math(model: nn.Module, tokenizer: TokenizerWrapper) -> list[dict]:
    """Run math reasoning cases with deterministic greedy decoding."""
    failures: list[dict] = []

    groups: list[tuple[str, list[MathCase], str]] = [
        ("ConvexBench", CONVEX_CASES, CONVEX_INSTRUCTION),
        ("Arithmetic", ARITHMETIC_CASES, ARITHMETIC_INSTRUCTION),
    ]
    for group_name, cases, instruction in groups:
        print(f"  --- {group_name} ---")
        for case in cases:
            prompt = build_prompt(case.question, instruction)
            is_convex = "convexbench" in case.tags
            passed, output = run_single(
                model,
                tokenizer,
                prompt,
                check_fn=lambda out, c=case: _check_pass(c, out),
                label=case.question,
                expected=case.expected,
                format_output=lambda out, cv=is_convex: (
                    f"{out} (parsed: {_extract_convexity(out)})" if cv else out
                ),
                max_tokens=2048,
            )

            if not passed:
                parsed = _extract_convexity(output) if is_convex else output
                failures.append(
                    {
                        "input": prompt,
                        "expected": case.expected,
                        "output": output,
                        "tags": case.tags,
                        "metadata": {
                            "question": case.question,
                            "parsed_answer": parsed,
                        },
                    }
                )

    return failures
