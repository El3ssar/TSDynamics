# Contributing to TSDynamics

Clear math, minimal API, no hacks. Thanks for helping.

---

## Prerequisites

- Python **≥ 3.12**
- The native Rust engine ships as a prebuilt `abi3` wheel, so **no compiler is
  needed** to install or run TSDynamics. Only building from the sdist needs a
  [Rust toolchain](https://rustup.rs/) (the build backend is
  [maturin](https://www.maturin.rs/)).
- [uv](https://docs.astral.sh/uv/) is the recommended package manager (pip / conda also work).

---

## Setup

```bash
git clone https://github.com/El3ssar/TSDynamics.git
cd TSDynamics

# Editable install with dev dependencies (pytest, ruff, mypy, ...)
uv sync --group dev
```

Optional: install pre-commit hooks so style is enforced before you commit:

```bash
uv run pre-commit install                # ruff lint + format on each commit
```

---

## Workflow

### 1. Branch from `main`

```bash
git switch -c feat/my-thing main
```

Use the [Conventional Commits](https://www.conventionalcommits.org/) prefix as your branch name when convenient: `feat/...`, `fix/...`, `docs/...`, `refactor/...`, `test/...`, `ci/...`.

### 2. Edit, then run the quick checks locally

```bash
# Formatter + linter (auto-fix safe issues)
uv run ruff check --fix src/ tests/
uv run ruff format src/ tests/
uv run mypy --strict src/tsdynamics

# The everyday loop: only the tests your diff can affect (see below)
make test
```

**Run `make test`, not the whole suite.** The suite is registry-driven — every
per-system test is parametrized over all 171 built-in systems, so the full fast
tier is ~5 400 items and takes a few minutes even in parallel. `make test` wraps
`pytest --changed`, which diffs your working tree against `origin/main` and keeps
only the tests that diff can affect (touch one system → that system's sweeps;
touch one analysis area → that area's files; touch anything foundational — the
engine/solver/family layers, the registry, a lockfile, any crate, any workflow —
and it falls back to the full tier). It prints exactly what it kept and why.

| Command | What it runs |
|---|---|
| `make test` | Change-scoped fast tier — the loop |
| `make test-slow` | Change-scoped slow tier (long simulations) |
| `make test-all` | Whole fast tier, every system — pre-push sanity |
| `make test-full` | Fast + slow tiers over everything |

The exhaustive nightly sweep is `uv run pytest -m full --no-cov`; you do not need
to run it locally.

The test suite is registry-driven: a new system is swept automatically — no test-file edits needed.  Only new DDEs need one extra line (a non-equilibrium history in `tests/_sampling.py::DDE_HISTORIES`; a guard test reminds you).

### 3. Commit

Use [Conventional Commits](https://www.conventionalcommits.org/) — the changelog and release notes are generated from these messages.

```
feat: add Sprott-N attractor to chaotic_attractors
fix(dde): reject zero or negative delay parameters
docs: clarify n_exp behaviour in DelaySystem.lyapunov_spectrum
test: cover Tinkerbell default_ic path
ci: pin ruff to 0.15
```

### 4. Open a PR

GitHub Actions runs `ruff check`, `ruff format --check`, `mypy --strict`, the test matrix (Python 3.12 / 3.13 / 3.14 on Linux and macOS) and a docs build for every PR.  The CI jobs are themselves change-scoped: a docs-only PR skips the Python jobs, and the ones that do run use `pytest --changed`.  Gate merges on the single **CI summary** check — it is green when every required job either passed or was skipped as out of scope.  **The PR title must be a conventional commit** (enforced by the `pr-title.yml` check): PRs are squash-merged, so the title becomes the commit message that decides the next release.

---

## Release process

Releases are **fully automated** via [python-semantic-release](https://python-semantic-release.readthedocs.io/): nobody bumps versions or pushes tags by hand.

Every push to `main` runs `.github/workflows/release.yml`, which:

1. Runs lint + the full test suite.
2. Parses the conventional-commit messages since the last release: `feat:` → minor, `fix:`/`perf:` → patch, `!`/`BREAKING CHANGE` → major.  Chore/ci/docs-only pushes release nothing.
3. If a release is due: rewrites `__version__` in `src/tsdynamics/__init__.py`, commits, tags `vX.Y.Z`, creates the GitHub Release with generated notes, and publishes to PyPI via [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC — no API tokens).

Release notes land on [GitHub Releases](https://github.com/El3ssar/TSDynamics/releases) and python-semantic-release keeps CHANGELOG.md up to date.

**One-time PyPI setup** (only the project owner needs to do this once):

1. Create the project on https://pypi.org or claim the `tsdynamics` name.
2. In the project's **Publishing** settings, add a pending publisher:
   - Owner: `El3ssar`
   - Repository: `TSDynamics`
   - Workflow filename: `release.yml`
   - Environment: `pypi`
3. In GitHub repo settings → **Environments** → create an environment named `pypi` (no secrets required). Optionally restrict it to the `main` branch or to tag pushes.

---

## Style

| Tool | Purpose | Config |
|---|---|---|
| `ruff check` | Lint (E, F, I, N, UP, B, SIM, D rules) | `pyproject.toml [tool.ruff]` |
| `ruff format` | Formatter (line length 100) | `pyproject.toml [tool.ruff.format]` |
| `mypy --strict` | Static type checking (**required**) | `pyproject.toml [tool.mypy]` |
| `pr-title.yml` | Conventional-commit PR titles | `.github/workflows/pr-title.yml` |
| Docstrings | NumPy convention | `pyproject.toml [tool.ruff.lint.pydocstyle]` |

CI will reject a PR with any `ruff` error **or any `mypy --strict` error**: run
`uv run mypy --strict src/tsdynamics` before pushing (the `typecheck` job in
`ci.yml` gates it). The system *catalogue* (`tsdynamics.systems.*`) relaxes exactly
the codes inherent to its framework-contract kernels — `override`, `no-untyped-def`,
`no-untyped-call` (see the documented `[tool.mypy.overrides]` block); the core
library is fully strict.

---

## Adding a new system

### ODE — `ContinuousSystem`

```python
from symengine import sin, cos
from tsdynamics import ContinuousSystem

class MyAttractor(ContinuousSystem):
    """Short one-liner describing the system."""

    params = {"a": 1.0, "b": 2.0, "c": 3.0}
    dim = 3

    @staticmethod
    def _equations(y, t, *, a, b, c):       # kwargs, names matching params keys
        x, yv, z = y(0), y(1), y(2)
        return (
            a * (yv - x),
            x * (b - z) - yv,
            x * yv - c * z,
        )

    @staticmethod                            # optional — helpful for stiff solvers
    def _jacobian(y, t, *, a, b, c):
        ...
```

- Use **only** symbolic operations (`symengine.sin`, `cos`, `+`, `*`, `**`, `abs`, ...). No NumPy, no `math`, no Python control flow over `y`.
- Return a sequence of exactly `dim` expressions.
- Add the class name to the module's `__all__` and to `src/tsdynamics/systems/continuous/__init__.py` `__all__` — a registry consistency test fails if you forget.
- That's it: the registry sweeps it into the bulk tests and the docs build generates its page (equations + attractor figure) automatically.  Optional metadata ClassVars: `variables`, `reference`, `known_lyapunov`.

### DDE — `DelaySystem`

```python
from tsdynamics import DelaySystem

class MyDDE(DelaySystem):
    params = {"k": 2.0, "tau": 1.5}
    dim = 1
    _delay_params = ("tau",)        # default — override for non-"tau" names or multi-delay

    @staticmethod
    def _equations(y, t, k, tau):
        return [k * y(0, t - tau) - y(0)]
```

`y(i, t - tau_expr)` is the delayed accessor. If your delay parameter isn't called `tau`, list its name(s) in `_delay_params`.

### Discrete map — `DiscreteMap`

```python
from tsdynamics import DiscreteMap

class MyMap(DiscreteMap):
    params = {"a": 1.4, "b": 0.3}
    dim = 2

    @staticmethod
    def _step(X, a, b):              # ⚠ params arrive POSITIONALLY in params-dict order
        x, y = X
        return (1 - a*x**2 + y, b*x)

    @staticmethod
    def _jacobian(X, a, b):
        x, y = X
        return ((-2*a*x, 1.0), (b, 0.0))
```

Keep the function signature order in sync with the `params` dict insertion order; the engine feeds the parameters positionally when it lowers `_step` to its IR (`__init_subclass__` validates the match at import time).

### Variable-dimension systems

Anything whose `_equations` has a loop length or list comprehension driven by a parameter (e.g. `range(N)` in Lorenz-96, `KuramotoSivashinsky`, or `MultiChua`) must declare the dimension-controlling parameter as **structural** so it gets baked into the engine IR tape:

```python
class Lorenz96(ContinuousSystem):
    params = {"f": 8.0, "N": 20}
    _structural_params = frozenset({"N"})
```

Non-structural parameters are read live from the system on every run, so they can be changed at runtime without re-lowering. Structural ones are baked into the tape, so changing one re-lowers (there is no on-disk cache).

### Initial conditions for tough basins

If random `U[0, 1)^dim` ICs don't land in your map's basin of attraction (e.g. Tinkerbell), set a class-level fallback:

```python
class MyMap(DiscreteMap):
    ...
    default_ic = np.array([-0.72, -0.64])
```

`resolve_ic` will consult `default_ic` before falling back to `U[0, 1)^dim`.

---

## Reporting bugs

Open an issue with:

- Minimal reproducer (Python script + expected vs observed)
- OS, architecture, Python version, `uv` / `pip` version
- `uv pip list | grep -iE "tsdynamics|symengine|sympy|numpy|scipy"` output
- Full traceback / warning text

---

## PR checklist

- [ ] Focused, minimal change.
- [ ] Tests cover new behavior; `make test` (and `make test-slow` if you touched anything heavy) is green locally.
- [ ] `ruff check`, `ruff format --check` and `mypy --strict src/tsdynamics` are clean.
- [ ] Public API changes documented in docstrings and `README.md`.
- [ ] Conventional Commit messages.
- [ ] If a new system is added: it's in the module/category `__all__` (the registry sweeps it automatically); a new DDE also has a history in `tests/_sampling.py::DDE_HISTORIES`.
