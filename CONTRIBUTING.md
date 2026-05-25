# Contributing to SAM

Thank you for contributing. SAM is orchestration-only; quantitative logic belongs in [ML4T](https://github.com/orgs/ml4t/repositories) libraries or in `sam.strategies` when strategy-specific.

## Development setup

```bash
cd sam
uv sync --extra backtest --extra live --extra data --group dev
cp .env.example .env
uv run python scripts/generate_fixtures.py
```

Optional full stack (research extras; may require CMake for numba/llvmlite):

```bash
uv sync --extra all --group dev
```

## Running checks

```bash
uv run ruff check .
uv run pytest -q
uv run pytest -q tests/integration/test_backtest_ma.py
uv run pytest -q tests/integration/test_live_shadow.py
```

Integration tests that need Alpaca credentials or network are marked `integration` and skipped by default (`addopts = -m 'not integration'` in `pyproject.toml`).

## Pull requests

1. Keep changes focused — SAM should remain a thin layer over ML4T.
2. Update `README.md`, `docs/ARCHITECTURE.md`, or `docs/RUNBOOK.md` when CLI behavior, configs, or promotion flow changes.
3. Add or extend tests for new behavior; prefer unit tests with fixtures under `tests/fixtures/`.
4. Do not commit secrets (`.env`), run outputs (`artifacts/`, `state/`, `data/`), or local virtualenvs.

CI runs on push to `main` and on pull requests (see `.github/workflows/ci.yml`).

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — layers, diagrams, promotion model
- [Runbook](docs/RUNBOOK.md) — operator promotion checklist
