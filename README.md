# TSDynamics

[![Python](https://img.shields.io/pypi/pyversions/tsdynamics)](https://pypi.org/project/tsdynamics/)
[![CI](https://github.com/El3ssar/TSDynamics/actions/workflows/ci.yml/badge.svg)](https://github.com/El3ssar/TSDynamics/actions/workflows/ci.yml)
[![Release](https://github.com/El3ssar/TSDynamics/actions/workflows/release.yml/badge.svg)](https://github.com/El3ssar/TSDynamics/actions/workflows/release.yml)
[![Docs](https://github.com/El3ssar/TSDynamics/actions/workflows/docs.yml/badge.svg)](https://el3ssar.github.io/TSDynamics/)
[![PyPI](https://img.shields.io/pypi/v/tsdynamics)](https://pypi.org/project/tsdynamics/)

**Dynamical systems in Python: 149 built-in systems, compiled integration, and
chaos analysis — with the simplest system-definition contract anywhere.**

You write the math (one symbolic method); TSDynamics handles compilation,
caching, integration, Lyapunov spectra, bifurcation diagrams, Poincaré
sections, and even the documentation page for your system.

```python
import tsdynamics as ts

lor = ts.Lorenz()
traj = lor.integrate(final_time=100.0, dt=0.01)
traj["x"]                              # named component access
lor.lyapunov_spectrum()                # → [0.91, ~0, -14.57]
ts.kaplan_yorke_dimension(_)           # → ~2.06
```

📖 **Documentation: <https://el3ssar.github.io/TSDynamics/>**

---

## Highlights

- **Three families, one interface** — ODEs (compiled via JiTCODE),
  delay-differential equations (JiTCDDE — including **DDE Lyapunov spectra**),
  and discrete maps (Numba). All implement one stepping protocol, so every
  analysis tool works on every system.
- **149 built-in systems** with literature parameters: Lorenz, Rössler, Chua,
  21 Sprott flows, Mackey–Glass, Hénon, ... each with an auto-generated docs
  page showing its equations and attractor.
- **Compile once, sweep forever** — parameters are runtime control values;
  changing them never recompiles. Compiled modules persist across sessions.
- **Composition** — a `PoincareMap` of a flow *is* a discrete map, so
  `orbit_diagram(PoincareMap(Rossler(), (1, 0.0)), "c", values)` draws the
  bifurcation diagram of a flow with one line.
- **Analysis toolkit** — orbit/bifurcation diagrams, Poincaré sections
  (root-refined), maximal Lyapunov exponent without Jacobians, Kaplan–Yorke
  dimension, fixed points with stability.
- **Experimental Rust backend** — `integrate(backend="diffsol")` JIT-compiles
  your equations through LLVM and solves them with Rust kernels; no C
  compiler needed (`pip install tsdynamics[diffsol]`).

## Install

```bash
pip install tsdynamics            # or: uv add tsdynamics
```

A C toolchain is required for the default compiled backends
(`build-essential` + `python3-dev` on Debian/Ubuntu, `xcode-select --install`
on macOS, MSVC Build Tools on Windows).

Optional extras: `tsdynamics[plot]` (matplotlib), `tsdynamics[diffsol]`
(Rust solver backend).

## Define your own system

```python
import tsdynamics as ts

class MySystem(ts.ContinuousSystem):
    params = {"a": 0.2, "b": 0.2, "c": 5.7}
    dim = 3
    variables = ("x", "y", "z")            # optional niceties

    @staticmethod
    def _equations(y, t, *, a, b, c):
        x, yv, z = y(0), y(1), y(2)
        return (-yv - z, x + a * yv, b + z * (x - c))
```

That's the whole contract. The class auto-registers: the bulk test-suite
sweeps it, and the docs build renders its equations (LaTeX, straight from the
symbolics) and its attractor figure — zero extra steps. Delay systems use
`y(0, t - tau)`; maps implement `_step`/`_jacobian` (signature order is
validated at import).

## A taste of the analysis layer

```python
import numpy as np
import tsdynamics as ts

# Bifurcation diagram of the logistic map
od = ts.orbit_diagram(ts.Logistic(), "r", np.linspace(2.5, 4.0, 600))
x, y = od.flat()

# Poincaré section of the Rössler attractor (root-refined crossings)
section = ts.poincare_section(ts.Rossler(), plane=(1, 0.0), steps=500)

# Fixed points of the Hénon map, with stability
ts.fixed_points(ts.Henon())
# [FixedPoint([-1.131354  -0.339406], unstable, ...),
#  FixedPoint([ 0.631354   0.189406], unstable, ...)]

# Maximal Lyapunov exponent, no Jacobian needed
ts.max_lyapunov(ts.Lorenz(ic=[1, 1, 1]), dt=0.05)    # ≈ 0.91
```

## Development

```bash
git clone https://github.com/El3ssar/TSDynamics && cd TSDynamics
uv sync --group dev --group docs
uv run pytest -m "not slow" --no-cov     # fast tier
uv run pytest --no-cov                   # full local suite
TSD_DOCS_FIGURES=0 uv run mkdocs serve   # docs preview
```

Releases are automated: conventional-commit PR titles drive
[semantic-release](https://python-semantic-release.readthedocs.io/) on merge —
see [CONTRIBUTING](https://el3ssar.github.io/TSDynamics/project/contributing/).

## License

MIT © Daniel Estevez
