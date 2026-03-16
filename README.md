---
language:
  - en
  - vi
license: mit
task_categories:
  - text-generation
tags:
  - blind-spots
  - hallucination
  - reversal-curse
  - math-reasoning
  - convexity
  - trolley-problem
  - evaluation
size_categories:
  - n<1K
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/merged_hf/data-*.arrow
dataset_info:
  features:
    - name: input
      dtype: string
    - name: expected
      dtype: string
    - name: output
      dtype: string
    - name: tags
      sequence: string
    - name: metadata
      dtype: string
    - name: source_file
      dtype: string
---

# Fatima Fellowship 2026 Technical Challenge: Qwen3.5-4B-Base Rejected Outputs

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Linting: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv) [![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

**Every example in this dataset is a rejected completion.** Each row contains a prompt (`input`), the correct answer (`expected`), and the model's incorrect output (`output`). These are failure cases from [Qwen/Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) identified through systematic evaluation.

Full code for evaluation and dataset generation is on [GitHub](https://github.com/mychiffonn/fatima-prework).

## Model and loading code

[Qwen/Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) — a 4B-parameter base language model. I loaded it locally via [mlx-lm](https://github.com/ml-explore/mlx-lm) (Apple Silicon optimized) for faster iteration on local machine.

```python
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

model, tokenizer = load("Qwen/Qwen3.5-4B-Base")
output = generate(model, tokenizer, prompt="Q: ...\nA:", max_tokens=256, sampler=make_sampler(temperature=0.0))
```

Core module for model is [src/runner.py](https://github.com/mychiffonn/fatima-prework/blob/main/src/runner.py). Full code for evaluation and dataset generation is on [GitHub](https://github.com/mychiffonn/fatima-prework).

## Blind spots

I tested the model across diverse failure categories and collected 13 data points (across 5 categories) where it consistently produces incorrect outputs:

- **Hallucination of fabricated concepts** (2 examples): The model confidently affirms fake theorems and standards. When asked to verify "the Kessler-Huang theorem" or "ISO 34271" — both entirely fabricated — it treats them as real and outputs "True" instead of "False".

- **Reversal curse** (2 examples) — [Berglund et al. 2023](https://arxiv.org/abs/2309.12288): The model knows "Tom Cruise's mother is Mary Lee Pfeiffer" but fails "Mary Lee Pfeiffer's son is \_\_\_", hallucinating "John Pfeiffer" instead. The forward direction is confirmed correct, so the knowledge exists but cannot be accessed in reverse. Failure scales inversely with fame (Obama/Chomsky reversals pass).

- **Math reasonings** (3 examples) — 2 from [ConvexBench 2025](https://arxiv.org/abs/2602.01075) where the model has to apply composition rules to determine convexity, and 1 for determining the last digit of a large modular exponentiation. In all cases, the model misapplies the rules or makes a critical error in the multi-step reasoning, leading to an incorrect final answer.

- **Multiple-choice factual questions** (2 examples): One requires Vietnamese language knowledge with 2-hop reasoning (compound word reversibility), and one requires negation reasoning about Vietnamese history ("which year had NO escapes").

- **Trolley problems with different cultural profiles** (4 examples): Given explicit ranked ethical preferences (e.g., "Strong: spare younger over older"), the model misreads which outcome spares whom, or ignores the preference ordering entirely. For example, it says "continuing ahead spares the child" when continuing actually kills the elderly person.

### Diversity of data points

The 13 examples span 5 distinct failure modes across different domains:

| Category         | Domain                          | Reasoning type                  | Language           |
| ---------------- | ------------------------------- | ------------------------------- | ------------------ |
| Hallucination    | Software engineering, Standards | Factual verification            | English            |
| Reversal curse   | Biography, Family relations     | Bidirectional recall            | English            |
| Convexity        | Mathematics                     | Multi-step composition rules    | English            |
| MCQ              | Linguistics, History            | Vietnamese knowledge + negation | English/Vietnamese |
| Trolley cultural | Ethics                          | Instruction-following + logic   | English            |

## Dataset structure

- **`input`** (`string`): The prompt sent to the model.
- **`expected`** (`string`): The ground-truth correct answer.
- **`output`** (`string`): The model's full (incorrect) response — the **rejected** completion.
- **`tags`** (`list[string]`): Categories for the failure (e.g., `reversal_curse`, `hallucination`, `convexbench`).
- **`metadata`** (`string`): JSON-encoded extra info (parsed answer, forward verification, etc.).
- **`source_file`** (`string`): Which `data/*.json` file the example came from.

## Fine-tuning recommendations

### What kind of dataset?

To fix these blind spots, a fine-tuning dataset should include:

1. **Reversal pairs** (~2k–5k examples): Bidirectional factual associations — for each "A's child is B" fact, include both "Who is A's child?" and "Who is B's parent?". The [reversal curse paper](https://arxiv.org/abs/2309.12288) shows that even seeing the reversed fact once during training can fix the failure, so moderate-sized datasets should suffice.

2. **Convexity chain-of-thought** (~1k–2k examples): Step-by-step convexity proofs using composition rules (Boyd & Vandenberghe Ch. 3). Each example should include the function, the decomposition into elementary functions, monotonicity and convexity of each piece, and the correct composition rule application.

3. **Modular arithmetic** (~1k examples): Multi-step modular exponentiation with verified intermediate steps showing each reduction via Fermat's little theorem / Euler's theorem.

4. **Instruction-following with ranked constraints** (~1k examples): Scenarios where the model must apply an explicit priority ordering of rules, with distractors that test whether it respects the ranking.

### How to assemble or find such a dataset?

- **Reversal pairs**: Source from **Wikidata** family-relation triples (P22/P25 parent properties, P40 child), filtered to entities with Wikipedia pages. Programmatically generate both forward and reverse question-answer pairs.
- **Convexity chain-of-thought**: **Generate programmatically** by composing random elementary functions and computing second derivatives symbolically via SymPy. Verify each label (convex/concave/neither) numerically.
- **Modular arithmetic**: **Generate programmatically** by sampling random bases, exponents, and moduli, computing correct answers, and producing step-by-step solution traces.
- **Instruction-following**: Adapt from existing benchmarks like [IFEval](https://arxiv.org/abs/2311.07911) or generate synthetic scenarios with explicit constraint rankings.

### How big of a dataset?

- **Reversal curse**: ~2k–5k bidirectional pairs.
- **Math reasoning**: ~1k–2k worked examples per category, with chain-of-thought traces showing correct rule application.
- **Instruction-following**: ~1k examples with ranked constraints.
- **Total**: ~5k–10k examples for a targeted fine-tune on a 4B model. This is small enough to be practical but sufficient for targeted corrections — the reversal curse paper shows that even limited exposure to the correct pattern can fix the failure mode.

## Evaluation methodology

Each blind spot is tested via a reproducible mini-evaluation pipeline (See `src/runner.py` and `scripts/run_model.py`):

- **Fair prompting**: Minimal `Q: ... / A:` format with no few-shot examples.
- **Deterministic generation**: Greedy decoding (`temperature=0.0`), task-appropriate `max_tokens` (256 for factual, 2048 for math reasoning). Each case runs once (in actual evaluation we should run at least 3 times due to non-determinism).
- **Parseable answers**: Post-processing strips `<think>...</think>` blocks, then extracts the answer via regex or substring matching against the expected value.
- **Forward verification** (reversal curse): For each failed reversal, the forward question is also run to confirm the model knows the fact in the trained direction.

## License

MIT
