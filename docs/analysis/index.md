---
description: The TSDynamics analysis toolkit — trajectories, Lyapunov spectra, orbit diagrams, Poincaré sections, and fixed points, in one-liners.
---

<span class="ts-kicker">Analysis</span>

# Analysis

Every quantifier in the toolkit consumes the same
[`System` protocol](../start/concepts.md#one-protocol-for-everything-that-steps),
so the same call works on a map, a flow, or a derived view of a flow.
One line each:

```python
import numpy as np
import tsdynamics as ts
from tsdynamics import PoincareMap

# Trajectories — compiled integration / iteration
traj = ts.Lorenz().integrate(final_time=100.0, dt=0.01)
orbit = ts.Henon().iterate(steps=10_000)

# Lyapunov quantifiers
spec = ts.lyapunov_spectrum(ts.Lorenz())                     # ≈ [0.906, 0, -14.57]
lam  = ts.max_lyapunov(ts.Lorenz(ic=[1, 1, 1]), dt=0.05)     # Jacobian-free estimate
d_ky = ts.kaplan_yorke_dimension(spec)                       # ≈ 2.06

# Orbit / bifurcation diagrams
od = ts.orbit_diagram(ts.Logistic(), "r", np.linspace(2.5, 4.0, 600))
od = ts.orbit_diagram(PoincareMap(ts.Rossler(), (1, 0.0)), "c",
                      np.linspace(2.0, 6.0, 80))             # bifurcations of a flow

# Poincaré sections
section = ts.poincare_section(ts.Rossler(), plane=(1, 0.0), steps=500)

# Fixed points of maps, with stability
fps = ts.fixed_points(ts.Henon())
```

## The pages

| Page | What it covers |
| ---- | -------------- |
| [Integrate & iterate](integrate.md) | Solvers, tolerances, the `Trajectory` object, the stepping API |
| [Lyapunov spectra](lyapunov.md) | Spectra per family, `max_lyapunov`, Kaplan–Yorke dimension, `TangentSystem` |
| [Orbit & bifurcation diagrams](orbit-diagrams.md) | Parameter sweeps, attractor following, flows via `PoincareMap` |
| [Poincaré sections](poincare.md) | Sections from systems (root-refined) or from trajectory data |
| [Fixed points](fixed-points.md) | Multi-start Newton for maps, eigenvalue stability |

For the underlying math — variational equations, QR re-orthonormalization,
the Kaplan–Yorke conjecture — see [Theory](../theory/index.md).
