---
description: Contributing to TSDynamics — dev setup with uv, quality gates, conventional commits, and the end-to-end path for adding a system.
---

<span class="ts-kicker">Project</span>

# Contributing

Clear math, minimal API, no hacks. The full text lives in
[CONTRIBUTING.md](https://github.com/El3ssar/TSDynamics/blob/main/CONTRIBUTING.md);
this page is the short version.

## Dev setup

You need Python ≥ 3.12, a C compiler (see [Install](../start/install.md)),
and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/El3ssar/TSDynamics.git
cd TSDynamics
uv sync --group dev          # editable install + pytest, ruff, mypy, pre-commit
uv run pre-commit install    # optional: style enforced at commit time
```

## Quality gates

Run before pushing — CI rejects PRs that fail any of these:

```bash
uv run ruff check src/ tests/            # lint (zero errors)
uv run ruff format --check src/ tests/   # formatting (line length 100)
uv run pytest -m "not slow" --no-cov     # fast suite, ~2 s — no C compilation
uv run pytest --no-cov                   # full suite, ~35 s — compiles + Lyapunov
```

Docstrings follow the NumPy convention; commits follow
[Conventional Commits](https://www.conventionalcommits.org/)
(`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `ci:`).

## Releases

PRs are **squash-merged**, and the PR title becomes the commit message —
so write PR titles as conventional commits too. Releases are automated
with python-semantic-release: `feat:` bumps minor, `fix:`/`perf:` bump
patch, `BREAKING CHANGE:`/`!` bumps major. The version is rewritten in
`src/tsdynamics/__init__.py`, tagged, published to PyPI, and the release
notes are generated — nothing to do by hand.

## Adding a system, end to end

The whole pipeline is driven by the class definition:

1. **Write the class** in the right module under
   `src/tsdynamics/systems/continuous/` or `.../discrete/`, following the
   [subclass contract](../start/concepts.md#the-three-contracts) for its
   family. Add the name to the module's `__all__`.
2. **The registry picks it up** automatically at import
   (`registry.get("MyAttractor")` now works, and the class is re-exported
   from the top level).
3. **The test suite sweeps it** — the bulk tests iterate the registry, so
   your system gets smoke, signature, and Jacobian checks without writing
   a test. Add literature Lyapunov values via `known_lyapunov` to opt in
   to the known-value tests.
4. **Its docs page auto-generates** — equations rendered from the
   symbolic definition, defaults, and a figure, on the next docs build.

### Optional metadata that improves everything downstream

```python
class MyAttractor(ContinuousSystem):
    params = {"a": 0.2, "b": 0.2, "c": 5.7}
    dim = 3
    variables = ("x", "y", "z")                  # named traj["x"] access + labelled figures
    reference = "Rössler (1976), Phys. Lett. A 57, 397-398"   # surfaced in the docs
    default_ic = (1.0, 0.0, 0.0)                 # only if random ICs miss the basin
    known_lyapunov = {                           # opt in to known-value tests
        "spectrum": (0.0714, 0.0, -5.39),
        "atol": (0.06, 0.06, 1.5),
        "kwargs": {"dt": 0.1, "burn_in": 100.0, "final_time": 500.0},
        "source": "Sprott (2003), Chaos and Time-Series Analysis",
    }
```

!!! tip "Checklist for the PR"
    - [ ] Class follows the family contract (symbolic ops only for flows;
          positional param order for maps)
    - [ ] Added to the module `__all__`
    - [ ] `variables` and `reference` declared
    - [ ] `uv run pytest --no-cov` green locally
    - [ ] PR title is a conventional commit
