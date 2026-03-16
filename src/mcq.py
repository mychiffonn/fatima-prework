"""Multiple-choice questions testing cultural and historical knowledge.

Questions probe knowledge that may be underrepresented in training data:
1. Vietnamese compound-word reversal (linguistics)
2. Hoa Lo prison escape history (Vietnamese history)
"""

import re
from dataclasses import dataclass, field

from mlx import nn
from mlx_lm.tokenizer_utils import TokenizerWrapper

from src.runner import run_single

INSTRUCTION = """\
Read the question carefully and choose the best answer from the options.
Think step by step, then state your final answer as a single letter (A, B, C, D, or E) after "Answer:".
"""


@dataclass
class MCQCase:
    """A single multiple-choice test case."""

    question: str
    options: list[str]
    correct: str  # "A", "B", "C", "D", or "E"
    tags: list[str] = field(default_factory=lambda: ["mcq"])


CASES: list[MCQCase] = [
    MCQCase(
        question=(
            "Vietnamese language has compound words that are reversible, "
            "which means the new reversed words also exist and have meaning. "
            "Some reversed words have same meanings as their original compound "
            "words some different (example in English: stop bus != bus stop). "
            "Which one of the following words is reversible, and has exactly the "
            "same meaning as its reversed word?"
        ),
        options=[
            "A. loại từ",
            "B. cảm tình",
            "C. mặt tiền",
            "D. tâm tư",
            "E. than thở",
        ],
        correct="E",
        tags=["mcq", "vietnamese linguistics", "multi-hop reasoning"],
    ),
    MCQCase(
        question=(
            "In which of the following years there were NO attempted escapes "
            "by political prisoners from Hoa Lo prison during French "
            "colonization of Indochina?"
        ),
        options=[
            "A. 1945",
            "B. 1946",
            "C. 1950",
            "D. 1951",
            "E. 1932",
        ],
        correct="B",
        tags=["mcq", "vietnamese history", "negation"],
    ),
]


def _build_prompt(case: MCQCase) -> str:
    options_str = "\n".join(case.options)
    return f"{INSTRUCTION}Question: {case.question}\n{options_str}\nAnswer:"


_ANSWER_RE = re.compile(r"\b([A-E])\b")


def _extract_answer(text: str) -> str:
    """Extract the chosen letter (A-E) from model output."""
    lower = text.lower()
    if "answer:" in lower:
        after = text[lower.rindex("answer:") + len("answer:") :]
        match = _ANSWER_RE.search(after)
        if match:
            return match.group(1)
    matches = _ANSWER_RE.findall(text)
    return matches[-1] if matches else ""


def _check_pass(case: MCQCase, output: str) -> bool:
    return _extract_answer(output) == case.correct


def run_mcq(model: nn.Module, tokenizer: TokenizerWrapper) -> list[dict]:
    """Run MCQ cases with deterministic greedy decoding."""
    failures: list[dict] = []

    print("  --- Multiple-Choice Questions ---")
    for case in CASES:
        prompt = _build_prompt(case)
        passed, output = run_single(
            model,
            tokenizer,
            prompt,
            check_fn=lambda out, c=case: _check_pass(c, out),
            label=case.question[:80],
            expected=case.correct,
            format_output=lambda out: f"{out} (parsed: {_extract_answer(out)})",
            max_tokens=1024,
        )

        if not passed:
            failures.append(
                {
                    "input": prompt,
                    "expected": case.correct,
                    "output": output,
                    "tags": case.tags,
                    "metadata": {
                        "question": case.question,
                        "options": case.options,
                        "parsed_answer": _extract_answer(output),
                    },
                }
            )

    return failures
