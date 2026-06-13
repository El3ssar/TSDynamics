---
description: The 26 built-in discrete maps — Numba-compiled iteration and QR-based Lyapunov spectra, across five categories.
---

<span class="ts-kicker">Systems · Discrete</span>

# Discrete maps

**26 iterated maps**, subclasses of
[`DiscreteMap`](../../reference/base.md). A map is defined by two plain
numeric static methods — `_step` and `_jacobian` — decorated with
`@staticjit`, which applies Numba's `njit` (and degrades gracefully to
pure Python when Numba is unavailable):

```python
from tsdynamics.utils import staticjit

class Henon(DiscreteMap):
    params = {"a": 1.4, "b": 0.3}
    dim = 2

    @staticjit
    def _step(X, a, b):
        x, y = X[0], X[1]
        return (1.0 - a * x**2 + y, b * x)

    @staticjit
    def _jacobian(X, a, b):
        ...
```

Parameters arrive **positionally, in `params`-dict order** — a mismatched
signature raises a `TypeError` at import time.

## Categories

| Category | Count | Examples |
| -------- | ----- | -------- |
| `chaotic_maps` | 9 | Hénon, Ikeda, standard map, ... |
| `exotic_maps` | 7 | Less common chaotic recurrences |
| `geometric_maps` | 4 | Baker, cat-map-style area transformations |
| `polynomial_maps` | 3 | Quadratic and cubic recurrences |
| `population_maps` | 3 | Logistic, Ricker, and relatives |

Each map has a generated page in this section with its recurrence,
defaults, and an orbit figure.

## Iterating

```python
import tsdynamics as ts

h = ts.Henon()
traj = h.iterate(steps=10_000)     # Trajectory; traj.t is arange(steps)
```

On first call, a Numba loop is compiled with the current parameter values
inlined and cached per `(class, params)`; changing a parameter costs one
quick re-JIT. If an orbit diverges (random ICs can land outside the
attractor basin), `iterate` retries with fresh random ICs up to
`max_retries` times. Maps whose basin is small declare a class-level
`default_ic` so the first try lands inside.

## Lyapunov spectrum

```python
h.lyapunov_spectrum(steps=5000)    # ≈ [0.42, -1.62]
```

Computed by QR decomposition of the running Jacobian product in a single
forward pass — trajectory and tangent dynamics advance together. A
`reortho_interval` larger than 1 trades a little accuracy for speed on
high-dimensional maps. The math is laid out in
[Theory · Lyapunov exponents](../../theory/lyapunov.md).

## Maps are the analysis workhorse

The protocol-level tools written for maps also run on flows through the
derived wrappers (`PoincareMap`, `StroboscopicMap`) — that composition is
what turns an [orbit diagram](../../analysis/orbit-diagrams.md) over a
wrapped flow into its bifurcation diagram. Map-specific tools like
[fixed-point finding](../../analysis/fixed-points.md) use the explicit
`_jacobian` and apply to `DiscreteMap` subclasses directly.
