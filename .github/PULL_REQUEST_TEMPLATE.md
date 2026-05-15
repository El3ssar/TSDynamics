<!--
Thanks for contributing! Keep PRs focused — one feature / fix per PR.
-->

## Summary

<!-- One or two sentences describing what changes and why. -->

## Changes

- [ ]

## Test plan

<!-- How did you verify? Paste relevant `pytest` output or example session. -->

```
$ uv run pytest -m "not slow" --no-cov
```

## Checklist

- [ ] Conventional Commit messages (`feat:`, `fix:`, `docs:`, ...)
- [ ] `uv run ruff check src/ tests/` is clean
- [ ] `uv run pytest --no-cov` is green locally
- [ ] Public API changes documented in the relevant docstrings / README
- [ ] New systems added to `tests/test_ode_systems.py` (or DDE/map equivalent)
