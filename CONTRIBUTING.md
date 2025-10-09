# Contributing to tsdynamics

We try to keep it clean: **clear math, minimal API, no hacks**. Thanks for helping.

## Toolchain prerequisites

You’ll need a C/C++ compiler:

* Linux: `build-essential` + `python3-dev`
* macOS: `xcode-select --install`
* Windows: “MSVC Build Tools”

Python **≥ 3.12**.

## Setup

### Using uv (recommended)

```bash
git clone https://github.com/El3ssar/TSDynamics.git
cd TSDynamics

uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# dev install (tests, lint)
uv pip install -e ".[dev]"

# docs extras
uv pip install -e ".[docs]"
```

### Using pip

```bash
python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]" ".[docs]"
```

### Using conda

```bash
conda create -n tsdynamics python=3.12 -y
conda activate tsdynamics

pip install -e ".[dev]" ".[docs]"
```

## Project layout (typical)

```
src/tsdynamics/
  base/
    base.py            # BaseDyn
    ode_base.py        # DynSys
    dde_base.py        # DynSysDelay
    continuous/        # Continuous systems (Lorenz, L96, MG, KS)
      ...
    discrete/          # Discrete systems (Henon, Ikeda, Tinkerbell, etc.)
      ...
```

## Style & quality

* **Formatting:** `black` (line length 100)
* **Import sort:** `isort` (or `ruff`’s built-ins)
* **Lint:** `ruff` (keep zero warnings)
* **Typing:** `mypy` where feasible
* **Docstrings:** NumPy style (`pydocstyle` config already in `pyproject.toml`)
* **Commit messages:** Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`)

## Tests

* Add unit tests for new behavior (`pytest`).
* For Lyapunov exponents, compare to **ranges** (not exact scalars); convergence varies with tolerances and dt.
* If a warning is expected and harmless, suppress it **locally** with `warnings.catch_warnings()` in the test.

## Building & releasing

Version is derived from **Git tags** via `hatch-vcs`.

* Bump by tagging:

```bash
git tag -a v0.1.1 -m "v0.1.1"
git push --tags
```

* Build wheels/sdist (no global installs needed):

```bash
# with uvx (ephemeral tool runner)
uvx hatch build
# artifacts in dist/
```

> If you prefer Hatch locally: `pipx install hatch` then `hatch build`.

## PR checklist

* [ ] Focused change; minimal API surface
* [ ] Tests cover new behavior; `pytest -q` is green
* [ ] `ruff`, `black`, `mypy` all pass
* [ ] Docstrings/examples updated
* [ ] No noisy global warnings
* [ ] Changelog entry if you keep one

## Code guidelines (systems)

* **ODE (`DynSys`)**: implement `_rhs(y, t, **params)` using **scalar** `y(i)` calls; no NumPy ops inside `_rhs`. Keep expression size O(N).
* **DDE (`DynSysDelay`)**: `_rhs(y, t, **params)` with delays via `y(i, t - tau_k)`. Provide a sensible example **history** (avoid equilibria if you show chaos).
* Don’t add optional “knobs” unless they’re necessary. Tolerances and method selection are enough for most users.

---

If something is unclear or brittle, open an issue with:

* Minimal repro
* Expected vs observed
* OS/arch, Python, uv/pip/conda, and package versions
* Full tracebacks / warnings

Thanks for keeping it sharp.