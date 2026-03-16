"""Hallucination detection: Can the model distinguish real from fabricated concepts?

Adapted from the BullshitBench (https://petergpt.github.io/bullshit-benchmark/).
Questions are reframed as true/false factual verification so a base model
can be fairly evaluated on its world knowledge rather than instruction-following.
"""

import re
from dataclasses import dataclass, field

from mlx import nn
from mlx_lm.tokenizer_utils import TokenizerWrapper

from src.runner import run_single

INSTRUCTION = """\
Determine whether the following statement is true or false.
Think step by step, then state your final answer as "True" or "False" after "Answer:".
"""


@dataclass
class HallucinationCase:
    """A single hallucination detection test case."""

    statement: str
    expected: str  # "True" or "False"
    tags: list[str] = field(default_factory=lambda: ["hallucination"])
    metadata: dict = field(
        default_factory=dict
    )  # Optional additional info for analysis


CASES: list[HallucinationCase] = [
    HallucinationCase(
        statement=(
            "The Kessler-Huang theorem on distributed consensus proves that"
            " any system with more than seven coordinating nodes requires"
            " a dedicated quorum arbitration layer."
        ),
        expected="False",
        tags=["hallucination", "fabricated_authority", "software"],
        metadata={
            "reasoning": "Kessler-Huang theorem does not exist; it's a fabricated concept. Distributed consensus is a real topic, but there is no such theorem or requirement about seven coordinating nodes or quorum arbitration layers."
        },
    ),
    HallucinationCase(
        statement=(
            "ISO 34271 is an international standard that defines requirements"
            " for continuous deployment maturity in software engineering."
        ),
        expected="False",
        tags=["hallucination", "fabricated_authority", "software"],
        metadata={
            "reasoning": "ISO 34271 does not exist; it's a fabricated standard. While ISO does publish standards on various topics, there is no such standard related to continuous deployment maturity in software engineering."
        },
    ),
    HallucinationCase(
        statement=(
            "The Drexler-Nussbaum entropic leakage factor is a recognized"
            " correction applied when calibrating calorimeters in"
            " thermodynamics experiments."
        ),
        expected="False",
        tags=["hallucination", "plausible_nonexistent", "physics"],
    ),
]


def _build_prompt(statement: str) -> str:
    return f"{INSTRUCTION}Statement: {statement}\nAnswer:"


def _extract_answer(text: str) -> str:
    """Extract True/False from model output."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    lower = cleaned.lower()
    if "answer:" in lower:
        lower = lower.split("answer:")[-1]
    matches = re.findall(r"\b(true|false)\b", lower)
    return matches[-1].capitalize() if matches else ""


def _check_pass(case: HallucinationCase, output: str) -> bool:
    return _extract_answer(output) == case.expected


def run_hallucination(model: nn.Module, tokenizer: TokenizerWrapper) -> list[dict]:
    """Run hallucination detection cases with deterministic greedy decoding."""
    failures: list[dict] = []

    print("  --- Hallucination Detection (BullshitBench) ---")
    for case in CASES:
        prompt = _build_prompt(case.statement)
        passed, output = run_single(
            model,
            tokenizer,
            prompt,
            check_fn=lambda out, c=case: _check_pass(c, out),
            label=case.statement,
            expected=case.expected,
            format_output=lambda out: f"{out} (parsed: {_extract_answer(out)})",
            max_tokens=1024,
        )

        if not passed:
            failures.append(
                {
                    "input": prompt,
                    "expected": case.expected,
                    "output": output,
                    "tags": case.tags,
                    "metadata": {
                        "statement": case.statement,
                        "parsed_answer": _extract_answer(output),
                    },
                }
            )

    return failures
