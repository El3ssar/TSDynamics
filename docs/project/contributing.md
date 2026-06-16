---
description: Contributing to TSDynamics — dev setup with uv, quality gates, conventional commits, and the end-to-end path for adding a system.
---

<span class="ts-kicker">Project</span>

# Contributing

Clear math, minimal API, no hacks. The full text lives in
[CONTRIBUTING.md](https://github.com/El3ssar/TSDynamics/blob/main/CONTRIBUTING.md);
this page is the short version.

## Dev setup

You need Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/). The native engine
ships as a prebuilt wheel, so no compiler is needed to install
(see [Install](../start/install.md)):

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
uv run pytest -m "not slow" --no-cov     # fast suite, ~2 s
uv run pytest --no-cov                   # full suite, ~35 s — integration + Lyapunov
```

Docstrings follow the NumPy convention; commits follow
[Conventional Commits](https://www.conventionalcommits.org/)
(`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `ci:`).

## Documentation

```bash
uv sync --group docs                              # mkdocs + material + mkdocstrings
TSD_DOCS_FIGURES=0 uv run mkdocs build --strict   # fast, figure-less validation
uv run mkdocs serve                               # live preview at 127.0.0.1:8000
```

The build must pass `--strict` (CI enforces it). Two build-time conventions
are worth knowing:

- **The system catalogue is auto-generated.** `hooks/docs_autogen.py` renders
  one page per registered system — equations from the symbolic definition,
  a parameter table, the `reference`, and a cached phase portrait — so a new
  system documents itself. Set `TSD_DOCS_FIGURES=0` to skip the (slow) figure
  rendering during local previews.
- **Citations, never competitors.** `hooks/citation_lint.py` fails the build
  if a published page names a competing dynamical-systems library or the
  `*.jl` ecosystem. Cite the **original paper** for every method — that is
  both the scholarly norm and a hard rule here. (The bare word "Julia" is
  fine — "Julia set", a person's name — only library/ecosystem references are
  blocked.)

When you add an analysis or transform, add its prose page under
`docs/analysis/` or `docs/transforms/`, an mkdocstrings stanza on the matching
`docs/reference/*` page, and a `nav` entry in `mkdocs.yml`.

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
