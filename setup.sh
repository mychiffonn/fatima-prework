#!/bin/sh

cp .env.example .env
source .venv/bin/activate
uv sync --locked --dev
pre-commit install
