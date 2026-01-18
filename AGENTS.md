# Repository Guidelines

## Project Structure & Module Organization
- `kvpress/` contains the library code; most compression methods live in `kvpress/presses/` and the pipeline in `kvpress/pipeline.py`.
- `tests/` holds unit and integration tests, with press-specific tests under `tests/presses/`.
- `evaluation/` provides benchmarking scripts and configs (see `evaluation/README.md`).
- `kvzap/` contains training/eval utilities for KVzap.
- `notebooks/` includes demos and experimentation notebooks.
- `script/` hosts helper scripts such as `script/test-environment.sh`.

## Build, Test, and Development Commands
- `uv sync --all-groups` installs dev dependencies; use `uv sync --extra eval` for evaluation tooling.
- `make format` runs `isort` and `black` to auto-format code.
- `make style` runs `flake8`, `mypy`, and SPDX header checks; outputs logs in `reports/`.
- `make test` runs the full pytest suite with coverage and fails on skipped tests.
- Evaluation: `cd evaluation && python evaluate.py --dataset loogle --press_name expected_attention`.

## Coding Style & Naming Conventions
- Python formatting: `black` with line length 120; import ordering via `isort`.
- Lint/type checks: `flake8` and `mypy` (see `pyproject.toml`).
- Files should include SPDX headers (`SPDX-FileCopyrightText:`) or `make style` will fail.
- Naming: presses follow `*_press.py` with classes like `MyPress`; add exports to `kvpress/presses/__init__.py`.

## Testing Guidelines
- Tests use `pytest`; keep new tests under `tests/` and name files `test_*.py`.
- Add coverage for new presses and behaviors, and include them in `tests/default_presses.py` when appropriate.
- Run focused tests with `uv run pytest tests/test_generate.py` before full `make test`.

## Commit & Pull Request Guidelines
- Commit messages are short, imperative, and often include PR numbers, e.g., `Add KVzapPress (#171)`.
- All commits must be signed off (`git commit -s`) per the DCO in `CONTRIBUTING.md`.
- PRs should include a clear description, link issues, and follow `.github/PULL_REQUEST_TEMPLATE.md` (format, tests, SPDX headers).
- For new presses, update README “Available presses” and add a test entry as noted in the PR checklist.

## Security & Configuration Tips
- Evaluation and some tests require optional deps (e.g., `flash-attn`, `optimum-quanto`); install via `uv` extras or follow `Makefile`.
- Use `evaluation/evaluate_config.yaml` for reproducible benchmark settings.
