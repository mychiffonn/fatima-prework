# Fatima Fellowship 2026 Technical Challenge

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Linting: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv) [![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

Systematic evaluation of [Qwen/Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) blind spots. The resulting dataset of rejected completions is on [Hugging Face](https://huggingface.co/datasets/chiffonng/fatima-prework).

## Setup

Requires Python 3.12+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/mychiffonn/fatima-prework.git
cd fatima-prework
uv sync
```

The model ([Qwen/Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base)) is loaded locally via [mlx-lm](https://github.com/ml-explore/mlx-lm) (Apple Silicon). It downloads automatically on first run.

## Usage

### Run evaluations

```bash
uv run scripts/run_model.py                  # run all scenarios
uv run scripts/run_model.py reversal math    # run specific scenarios
uv run scripts/run_model.py --list           # list available scenarios
```

Available scenarios: `reversal`, `math`, `mcq`, `hallucination`, `trolley`.
Results are saved as JSON files in `data/`.

### Merge and push to Hugging Face

```bash
uv run scripts/merge_data.py
```

This merges all `data/*.json` into a HuggingFace Dataset, saves to `data/merged_hf/`, and pushes to the Hub. Requires `HF_TOKEN` set in environment or `.env`.

## Project structure

```
src/
  runner.py           # Common infrastructure (model loading, generation, answer parsing)
  hallucination.py    # Fabricated concept verification tests
  reversal.py         # Reversal curse tests
  math_reasoning.py   # Convexity and modular arithmetic tests
  mcq.py              # Multiple-choice factual questions
  trolley.py          # Trolley problem with cultural profiles
scripts/
  run_model.py        # CLI entry point for running evaluations
  merge_data.py       # Merge JSON results and push to HuggingFace Hub
data/
  merged_hf/          # HuggingFace Dataset (Arrow format + dataset card)
```

## Blind spots

The 13 examples span 5 distinct failure modes across different domains:

| Category         | Domain                          | Reasoning type                  | Language           |
| ---------------- | ------------------------------- | ------------------------------- | ------------------ |
| Hallucination    | Software engineering, Standards | Factual verification            | English            |
| Reversal curse   | Biography, Family relations     | Bidirectional recall            | English            |
| Math reasoning   | Mathematics                     | Multi-step composition rules    | English            |
| MCQ              | Linguistics, History            | Vietnamese knowledge + negation | English/Vietnamese |
| Trolley cultural | Ethics                          | Instruction-following + logic   | English            |

## Evaluation methodology

Each blind spot is tested via a reproducible mini-evaluation pipeline (see `src/runner.py` and `scripts/run_model.py`):

- **Fair prompting**: Minimal `Q: ... / A:` format with no few-shot examples.
- **Deterministic generation**: Greedy decoding (`temperature=0.0`), task-appropriate `max_tokens` (256 for factual, 2048 for math reasoning). Each case runs once (in actual evaluation we should run at least 3 times due to non-determinism).
- **Parseable answers**: Post-processing strips `<think>...</think>` blocks, then extracts the answer via regex or substring matching against the expected value.
- **Forward verification** (reversal curse): For each failed reversal, the forward question is also run to confirm the model knows the fact in the trained direction.

## License

MIT
