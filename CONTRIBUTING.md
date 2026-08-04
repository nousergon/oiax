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

## Changing the selection configuration

The selection configuration — admission floors, fusion parameters, scorers, the embedding model — is the router's decision rule. Every change ships as an entry in `src/oiax/eval/corpora/ARMS.jsonl`, an append-only record. The rule, and the reasoning:

1. **A challenger is promoted only on a run against the same labelled set as the incumbent**, with both results recorded. "Better on a different set" is not a comparison.
2. **The incumbent's entry is marked `superseded_by`, never deleted**, so the history is recoverable from the repo rather than from commit archaeology.
3. **The shipped configuration names its arm id** in `calibration.py`, so the running system and the record cannot disagree.
4. **`route_eval calibrate` writes the arm entry** when passed `--arm-id` and `--arms` flags. Running it is the default path, not an extra discipline to remember.

To view the arms record: `python -m oiax.eval.route_eval arms`.

## License

By contributing, you agree that your contributions will be licensed under the MIT license.
