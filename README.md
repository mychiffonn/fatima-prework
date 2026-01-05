# Python Project Template

[![Python 3.12+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Linting: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv) [![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

## Get started

### Requirements

- Require Python>=3.12 (as specified in .python-version)
- Get [uv](https://github.com/astral-sh/uv) if you haven't to manage packages and run scripts.

### Setup

1. Fork/clone the repo. Either works, as long as you can `git pull` from our remote repo.
2. Run the set up code in the terminal

```bash
sh setup.sh
```

You should be in the virtual environment named "cot-monitor-scaling". If not, manually do `source .venv/bin/activate`

3. Go to `.env` and fill in the secrets.

#### Git commit / push failed

Pre-commit hooks may block your commit/push if one of these fails:

1. code style or linting issues (ruff)

Please fix the issues shown in the terminal. If the commit/push fails due to (2) or (3), they're automatically fixed, just try committing/pushing again.

## File tree
