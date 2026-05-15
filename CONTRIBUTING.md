# Contributing to TSDynamics

Clear math, minimal API, no hacks. Thanks for helping.

---

## Prerequisites

- Python **≥ 3.12**
- A C/C++ compiler (needed because JiTCODE / JiTCDDE compile RHS bodies to C):
  - Linux: `sudo apt-get install build-essential python3-dev`
  - macOS: `xcode-select --install`
  - Windows: MSVC Build Tools
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
uv run pre-commit install                # standard hook (runs on each commit)
uv run pre-commit install --hook-type commit-msg   # commitizen on the commit message
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

# Fast tests — these don't compile any C code (≈ 2 s for the full fast suite)
uv run pytest -m "not slow" --no-cov

# Full suite — exercises JiTCODE/JiTCDDE compilation + Lyapunov spectra (≈ 35 s)
uv run pytest --no-cov
```

If you add a new system, also add a row to `tests/test_ode_systems.py::ALL_ODE_SYSTEMS` (or its DDE / map equivalents) — see the *Adding a new system* section below for the full template.

### 3. Commit

Use [Conventional Commits](https://www.conventionalcommits.org/) — the changelog and release notes are generated from these messages.

```
feat: add Sprott-N attractor to chaotic_attractors
fix(dde): reject zero or negative delay parameters
docs: clarify n_exp behaviour in DelaySystem.lyapunov_spectrum
test: cover Tinkerbell default_ic path
ci: pin ruff to 0.15
```

If pre-commit hooks are installed, `commitizen` will validate the message on commit.

### 4. Open a PR

GitHub Actions runs `ruff check`, `ruff format --check`, and the full test matrix (Python 3.12 + 3.13 on Linux and macOS) for every PR. The PR must be green before review.

---

## Release process

Versioning is **fully Git-tag driven** via `hatch-vcs`. Releases require no local builds — pushing the tag triggers everything.

```bash
# Bump version (commitizen reads commits since the last tag and decides
# the semver bump; --yes accepts the suggestion). This:
#   - bumps the tag
#   - updates CHANGELOG.md
#   - commits both
#   - creates the new annotated tag
uv run cz bump --yes

git push origin main
git push origin --tags
```

The push of the new `v*.*.*` tag triggers `.github/workflows/release.yml`, which:

1. Builds an sdist and a wheel with `hatch build`.
2. Publishes them to PyPI using [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC — no API tokens needed).
3. Creates a GitHub Release with auto-generated notes and the built artifacts attached.

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
| `mypy` | Optional type checking | `pyproject.toml [tool.mypy]` |
| `commitizen` | Conventional Commits enforcement | `pyproject.toml [tool.commitizen]` |
| Docstrings | NumPy convention | `pyproject.toml [tool.ruff.lint.pydocstyle]` |

CI will reject a PR with any `ruff` error. Type errors are advisory for now.

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
- Add the class name to the module's `__all__` and to `src/tsdynamics/systems/continuous/__init__.py` `__all__`.

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
from tsdynamics.utils import staticjit

class MyMap(DiscreteMap):
    params = {"a": 1.4, "b": 0.3}
    dim = 2

    @staticjit
    def _step(X, a, b):              # ⚠ params arrive POSITIONALLY in params-dict order
        x, y = X
        return (1 - a*x**2 + y, b*x)

    @staticjit
    def _jacobian(X, a, b):
        x, y = X
        return ((-2*a*x, 1.0), (b, 0.0))
```

Keep the function signature order in sync with the `params` dict insertion order; that's how Numba feeds parameters in.

### Variable-dimension systems

Anything whose `_equations` has a loop length or list comprehension driven by a parameter (e.g. `range(N)` in Lorenz-96, `KuramotoSivashinsky`, or `MultiChua`) must declare the dimension-controlling parameter as **structural** so it gets baked into the compiled C code:

```python
class Lorenz96(ContinuousSystem):
    params = {"f": 8.0, "N": 20}
    _structural_params = frozenset({"N"})
```

Non-structural parameters become JiTCODE `control_pars` and can be changed at runtime without recompiling. Structural ones force a recompile (cached on disk).

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
- `uv pip list | grep -iE "jitcode|jitcdde|symengine|numba"` output
- Full traceback / warning text

---

## PR checklist

- [ ] Focused, minimal change.
- [ ] Tests cover new behavior; both `pytest -m "not slow"` and `pytest` are green locally.
- [ ] `ruff check` and `ruff format --check` are clean.
- [ ] Public API changes documented in docstrings and `README.md`.
- [ ] Conventional Commit messages.
- [ ] If a new system is added: row in `tests/test_ode_systems.py` / equivalent.
