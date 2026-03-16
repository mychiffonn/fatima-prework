"""Reversal Curse: models that know "A's child is B" often fail "B's parent is?".

Source: https://arxiv.org/pdf/2309.12288.
"""

from dataclasses import dataclass, field

from mlx import nn
from mlx_lm.tokenizer_utils import TokenizerWrapper

from src.runner import run_prompt, run_single


@dataclass
class ReversalCase:
    """A single reversal-curse test case."""

    question: str
    expected: str
    forward_question: str = ""
    tags: list[str] = field(default_factory=lambda: ["reversal_curse"])


def build_prompt(question: str) -> str:
    """Minimal Q/A prompt — lets the model reason naturally."""
    return f"Q: {question}\nA:"


CASES: list[ReversalCase] = [
    ReversalCase(
        question="Who is Mary Lee Pfeiffer's son?",
        expected="Tom Cruise",
        forward_question="Who is Tom Cruise's mother?",
    ),
    ReversalCase(
        question="Name a child of Donna Chomsky.",
        expected="Noam Chomsky",
        forward_question="Who is Noam Chomsky's mother?",
    ),
    ReversalCase(
        question="Who is Stanley Ann Dunham's son?",
        expected="Barack Obama",
        forward_question="Who is Barack Obama's mother?",
    ),
    ReversalCase(
        question="Name a child of Celine Dion's husband.",
        expected="René-Charles Angélil",
        forward_question="Who is René-Charles Angélil's mother?",
    ),
    ReversalCase(
        question="Who is Ambati Rao's son?",
        expected="Balamurali Ambati",
        forward_question="Who is Balamurali Ambati's father?",
    ),
]


def run_reversal(model: nn.Module, tokenizer: TokenizerWrapper) -> list[dict]:
    """Run reversal-curse cases with deterministic greedy decoding."""
    failures: list[dict] = []

    for case in CASES:
        prompt = build_prompt(case.question)
        passed, output = run_single(
            model,
            tokenizer,
            prompt,
            check_fn=lambda out, exp=case.expected: exp.lower() in out.lower(),
            label=case.question,
            expected=case.expected,
        )

        if not passed:
            metadata: dict = {}
            if case.forward_question:
                fwd_prompt = build_prompt(case.forward_question)
                forward_output = run_prompt(model, tokenizer, fwd_prompt)
                metadata["forward_question"] = case.forward_question
                metadata["forward_output"] = forward_output
                print(f"  Forward:  {case.forward_question} -> {forward_output}")
                print()

            failures.append(
                {
                    "input": prompt,
                    "expected": case.expected,
                    "output": output,
                    "tags": case.tags,
                    "metadata": metadata,
                }
            )

    return failures
