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

# Lyapunov quantifiers + chaos indicators
spec = ts.lyapunov_spectrum(ts.Lorenz())                     # ≈ [0.906, 0, -14.57]
d_ky = ts.kaplan_yorke_dimension(spec)                       # ≈ 2.06
K    = ts.zero_one_test(ts.Logistic().with_params(r=4.0).iterate(2000).y[:, 0])  # ≈ 1

# Orbit / bifurcation diagrams, Poincaré sections, fixed points
od = ts.orbit_diagram(ts.Logistic(), "r", np.linspace(2.5, 4.0, 600))
section = ts.poincare_section(ts.Rossler(), plane=(1, 0.0), n=500)
fps = ts.fixed_points(ts.Henon())

# Quantify a measured series — dimension, recurrence, nonlinearity test
dim = ts.correlation_dimension(ts.Henon().iterate(8000).y)   # ≈ 1.2
x = ts.Henon().trajectory(8000, transient=500).y[:, 0]
quant = ts.rqa(x, recurrence_rate=0.1)                       # DET / LAM / L_max / ENTR
test  = ts.surrogate_test(x, ts.time_reversal_asymmetry)     # rejects the linear null

# Global stability — attractors & basins (see the tutorial linked below)
attractors = ts.find_attractors                              # → basins, fractions, entropy …
```

## The pages

| Page | What it covers |
| ---- | -------------- |
| [Integrate & iterate](integrate.md) | Solvers, tolerances, the `Trajectory` object, the stepping API |
| [Lyapunov spectra](lyapunov.md) | Spectra per family, `max_lyapunov`, `lyapunov_from_data`, Kaplan–Yorke, `TangentSystem` |
| [Orbit & bifurcation diagrams](orbit-diagrams.md) | Parameter sweeps, attractor following, flows via `PoincareMap`, return maps |
| [Poincaré sections](poincare.md) | Sections from systems (root-refined) or from trajectory data |
| [Fixed points & periodic orbits](fixed-points.md) | Multi-start Newton, Schmelcher–Diakonos / Davidchack–Lai, shooting for flows |
| [Chaos indicators](chaos.md) | GALI, the 0–1 test, expansion entropy — "is this orbit chaotic?" |
| [Recurrence & RQA](recurrence.md) | Recurrence plots and determinism / laminarity / entropy measures |
| [Fractal dimensions](dimensions.md) | Correlation, box-counting, generalized-Rényi, fixed-mass; scaling-region fits |
| [Delay embeddings](embedding.md) | Takens reconstruction; delay (MI/ACF) and dimension (FNN/Cao) selection |
| [Entropy & complexity](entropy.md) | Permutation / dispersion / sample / multiscale entropy and LZ76 |
| [Surrogates & nonlinearity tests](surrogate.md) | Shuffle / FT / AAFT / IAAFT surrogates and the hypothesis test |
| [Attractors & basins](basins.md) | Attractor location, basins, basin entropy, continuation, tipping, resilience |

New to the toolkit? The [equations → basins](../tutorials/equations-to-basins.md)
tutorial wires several of these together on one system. For the underlying math
— variational equations, QR re-orthonormalization, the Kaplan–Yorke conjecture
— see [Theory](../theory/index.md). The signal layer that feeds analysis lives
under [Transforms](../transforms/index.md).
