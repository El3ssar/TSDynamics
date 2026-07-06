---
description: Contributing to TSDynamics — dev setup with uv, the quality gates (ruff, mypy --strict, the two test tiers), change-scoped testing, Conventional Commits, and the automated release flow.
---

<span class="ts-kicker">Project · Contributing</span>

# Contributing

Clear math, minimal API, no hacks. Contributions are welcome — a new attractor,
a sharper docstring, a bug fix, a whole analysis. The full text lives in
[CONTRIBUTING.md](https://github.com/El3ssar/TSDynamics/blob/main/CONTRIBUTING.md);
this page is the working version.

## Dev setup

You need Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/). The native Rust
engine ships as a prebuilt `abi3` wheel, so **no compiler is needed** to install
or work on the pure-Python side (see [Install](../start/install.md)). Building
from source — only needed if you edit the Rust engine — needs a
[Rust toolchain](https://rustup.rs/).

```bash
git clone https://github.com/El3ssar/TSDynamics.git
cd TSDynamics
uv sync --group dev          # editable install + pytest, ruff, mypy, pre-commit
uv run pre-commit install    # optional: ruff lint + format enforced at commit time
```

## The everyday loop — change-scoped testing

The test suite is **registry-driven**: every per-system test is parametrised
over all 171 built-in systems, and every analysis / transform test over the
whole toolkit. A plain `uv run pytest` is therefore thousands of items and takes
minutes. **Don't reach for the full suite as your inner loop.** Instead run only
what your diff touches:

```bash
make test          # change-scoped FAST tier for your diff — the everyday loop
make test-slow     # change-scoped SLOW tier (long sims), if you touched heavy code
```

`make test` diffs your working tree against `origin/main` and selects only the
tests your change can affect — a touched system module runs that module's
systems (plus the catalogue-correctness gates), a touched analysis area runs
that area's tests, a touched test file runs itself. It is deliberately biased to
**over-select**: any foundational change (the engine, a family base, the
registry, `pyproject`, or any Rust crate) disables selection and runs the full
tier, so a mis-scoped run can never *ship* a regression — the full suite runs on
every merge to `main` and nightly. The selector prints exactly what it kept and
why.

Override the diff base when you need to (defaults to `origin/main`):

```bash
make test BASE=HEAD~3
uv run pytest --changed --changed-since=HEAD~3 -m "not slow" --no-cov -n auto
```

For a final pre-push sanity check over *everything* (parallel, not the nightly
exhaustive sweep):

```bash
make test-all      # full FAST tier over every system / analysis
make test-full     # fast + slow tiers over everything
```

## The quality gates

CI rejects a PR that fails any of these — run them before pushing:

```bash
uv run ruff check src/ tests/            # lint (E, F, I, N, UP, B, SIM, D rules)
uv run ruff format --check src/ tests/   # formatting (line length 100)
uv run mypy --strict src/tsdynamics      # static types — CI-blocking, must be clean
make test                                # change-scoped fast tier
```

- **`ruff`** is both linter and formatter; `ruff check --fix` auto-fixes the safe
  issues.
- **`mypy --strict`** is a hard gate. The core library is fully strict; the
  system *catalogue* (`tsdynamics.systems.*`) relaxes exactly three codes
  inherent to its framework-contract kernels — `override`, `no-untyped-def`,
  `no-untyped-call` — via a documented `[tool.mypy.overrides]` block, because the
  `_equations` / `_step` / `_drift` bodies receive their parameters positionally.
- **Docstrings** follow the NumPy convention. Cite the **original paper** for any
  method — the scholarly norm (and it keeps the API docs pointing at the source,
  not at whichever library we happened to look at).

## Documentation

```bash
uv sync --group docs                              # mkdocs + material + mkdocstrings
TSD_DOCS_FIGURES=0 uv run mkdocs build --strict   # fast, figure-less validation
uv run mkdocs serve                               # live preview at 127.0.0.1:8000
```

The build must pass `--strict` (CI enforces it). One build-time convention is
worth knowing:

- **The system catalogue documents itself.** `hooks/docs_autogen.py` renders one
  page per registered system — equations from the symbolic definition, a
  parameter table, the `reference`, and a cached phase portrait — so a new system
  needs no hand-written page. `TSD_DOCS_FIGURES=0` skips the (slow) figure
  rendering during local previews.

When you add an analysis or transform, add its prose page under
`docs/analysis/`, an mkdocstrings stanza on the matching `docs/reference/*` page,
and a `nav` entry in `mkdocs.yml`.

## Commits & the PR flow

Commits follow [Conventional Commits](https://www.conventionalcommits.org/) —
the changelog and release notes are generated from them:

```
feat: add Sprott-N attractor to chaotic_attractors
fix(dde): reject zero or negative delay parameters
perf(engine): reuse the frozen Jacobian across SDIRK substages
docs: clarify n_exp behaviour in DelaySystem.lyapunov_spectrum
```

The prefix decides the release: `feat:` → minor, `fix:` / `perf:` → patch,
`!` or `BREAKING CHANGE:` → major; `chore` / `ci` / `docs`-only release nothing.

1. **Branch from `main`** — `git switch -c feat/my-thing main`.
2. **Edit, then run the gates** above until green.
3. **Open a PR.** GitHub Actions runs the linters, `mypy --strict`, the
   change-scoped test matrix (Python 3.12 + 3.13 on Linux and macOS), and a docs
   build.
4. **The PR title must be a conventional commit** (enforced by `pr-title.yml`):
   PRs are **squash-merged**, so the title becomes the commit that decides the
   next release. Write it accordingly.

## Releases

Releases are **fully automated** by
[python-semantic-release](https://python-semantic-release.readthedocs.io/) —
nobody bumps a version or pushes a tag by hand. Every push to `main` runs the
full suite with coverage, computes the version bump from the conventional-commit
history, rewrites `__version__` (and `pyproject.toml`), tags `vX.Y.Z`, publishes
the `abi3` wheels to PyPI via [Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
(OIDC — no API tokens), and generates the [release notes](changelog.md). The
static `[project].version` is kept in lock-step because the build backend
([maturin](https://www.maturin.rs/)) cannot read a Python `__version__`.

## The two extension paths

Most contributions are one of two well-trodden recipes, each with its own page:

- [**Adding a system**](adding-a-system.md) — a new attractor, map, DDE, or SDE
  for the catalogue. One class; the registry, tests, and docs follow.
- [**Adding a solver**](adding-a-solver.md) — a new integration kernel for the
  Rust engine, mirrored in the Python registry.

## PR checklist

!!! tip "Before you open the PR"
    - [ ] Focused, minimal change.
    - [ ] `make test` (and `make test-slow` if you touched heavy code) green
          locally.
    - [ ] `ruff check` and `ruff format --check` clean.
    - [ ] `mypy --strict src/tsdynamics` clean.
    - [ ] Public API changes documented in docstrings; new methods cite the
          original paper.
    - [ ] Conventional-commit PR title.
    - [ ] New system in the module / category `__all__`; a new DDE also has a
          history in `tests/_sampling.py::DDE_HISTORIES`, a new built-in SDE a
          sample in `SDE_SAMPLES` (guard tests remind you).
