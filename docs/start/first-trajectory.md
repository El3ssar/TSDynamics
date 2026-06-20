---
description: Integrate the Lorenz system, compute Lyapunov exponents, iterate the Hénon map, and integrate the Mackey–Glass delay equation.
---

<span class="ts-kicker">Start · 02</span>

# First trajectory

One example per system family: a continuous flow, a discrete map, and a
delay equation. Together they cover most of what the library does.

Every family produces a trajectory through the one verb **`run`** — flows,
maps, DDEs and SDEs all answer the same call, dispatching on whether the
system is discrete. The family-specific spellings (`integrate` for flows and
delay equations, `iterate` for maps) and the protocol method `trajectory`
remain as permanent aliases.

## A flow: Lorenz

```python
import tsdynamics as ts

lor = ts.Lorenz()                                  # sigma=10, rho=28, beta=8/3
traj = lor.run(final_time=100.0, dt=0.01)          # `integrate` is a permanent alias

traj.t.shape, traj.y.shape                         # ((10001,), (10001, 3))
```

The right-hand side is lowered to the Rust engine in-process and runs with no
warmup. `dt` is only the *output grid* — the internal stepper is adaptive, so a
coarse `dt` does not lose accuracy.

`run` returns a [`Trajectory`](../analysis/integrate.md#the-trajectory-object).
Because `Lorenz` declares `variables = ("x", "y", "z")`, components are
accessible by name:

```python
x = traj["x"]               # shape (10001,)
xz = traj[["x", "z"]]       # shape (10001, 2)
settled = traj.after(20.0)  # drop the transient before t = 20
t, y = traj                 # tuple-unpacking also works
```

The Lyapunov spectrum comes from the variational equations:

```python
lor.lyapunov_spectrum()     # ≈ [0.906, 0.0, -14.57]
```

One positive exponent, one zero (the flow direction), one strongly
negative — the signature of a chaotic attractor.

## A map: Hénon

Discrete maps iterate instead of integrate, lowering to the same engine. The
same `run` verb works — a map takes `n` (the number of iterations) instead of
`final_time`/`dt`:

```python
h = ts.Henon()                        # a=1.4, b=0.3
traj = h.run(n=5000)                  # `iterate(steps=5000)` is a permanent alias
traj.t                                # array([0, 1, 2, ...]) — step indices

h.lyapunov_spectrum(steps=5000)       # ≈ [0.42, -1.62]
```

The map spectrum is computed by QR decomposition of the Jacobian product
in a single forward pass.

## A delay equation: Mackey–Glass

The state of a delay system is a whole *history function*, not a point.
Pass one as a callable `h(s) → sequence` defined for `s ≤ 0`:

```python
import numpy as np

mg = ts.MackeyGlass()                 # beta=0.2, gamma=0.1, tau=17, n=10
hist = lambda s: [1.0 + 0.1 * np.sin(0.2 * s)]
traj = mg.integrate(final_time=500.0, dt=0.5, history=hist)
```

If you omit `history`, a constant past equal to the initial condition is
used — fine for exploration, but a constant past at an equilibrium gives
trivial dynamics.

DDE Lyapunov spectra use a two-step pattern — integrate first, then start
the Lyapunov run from the end state, which sits on the attractor:

```python
exps = mg.lyapunov_spectrum(n_exp=1, dt=0.5, ic=traj.y[-1])
```

The reason for the two steps is explained on the
[delay systems page](../systems/delay/index.md).

!!! note "DDE tolerances"
    Delay systems default to `rtol=atol=1e-3`. Resist the urge to tighten
    them to ODE-style values — overly tight tolerances stall the DDE
    solver.

## Next

[**03 · The mental model**](concepts.md) — what a system *is*, and the
handful of ideas the whole library is built on.
