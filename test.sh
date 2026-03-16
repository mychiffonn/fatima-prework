#!/usr/bin/env bash
set -euo pipefail

PROMPT="${1:-Hello, world!}"

uv run mlx_lm.generate --model "Qwen/Qwen3.5-4B-Base" --prompt "$PROMPT"
