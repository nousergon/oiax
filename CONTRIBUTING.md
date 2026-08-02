# Contributing to oiax

Thanks for your interest. oiax is a young project — contributions are welcome.

## Getting started

```bash
git clone https://github.com/nousergon/oiax.git
cd oiax
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Tests

```bash
pytest                              # all tests
ruff check src/ tests/              # lint
mypy src/oiax --python-version=3.13 --no-site-packages  # types
```

CI runs pytest against Python 3.11, 3.12, and 3.13. All three must pass before merge.

## Conventions

- Type annotations on all public functions. `mypy --strict` with `ignore_missing_imports` for sklearn/fastembed deps.
- Ruff (E, F, I, W, UP rules). Line length 100.
- Tests are hermetic — no network calls, no filesystem outside `tmpdir`.
- Public API is additive-only. No renames or removals without a migration period.
- No NE-specific paths, policy names, or absolute filesystem references in `src/oiax/`.

## Pull requests

- Open an issue first for anything beyond a bug fix.
- Keep changes focused. One PR, one concern.
- Update tests for behavior changes.
- CI must be green.

## License

By contributing, you agree that your contributions will be licensed under the AGPL-3.0 license.
