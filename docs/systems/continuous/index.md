---
description: The 142 built-in continuous flows — ODE systems run on the native Rust engine, organized into eight categories.
---

<span class="ts-kicker">Systems · Continuous</span>

# Continuous systems

The largest family: **142 ODE systems**, all subclasses of
[`ContinuousSystem`](../../reference/base.md). Each declares its parameters
and dimension at class level and defines the vector field in one symbolic
`_equations` method; it is lowered to the native Rust engine in-process and
JIT-compiled on first use, with no build step (see
[the compilation pipeline](../../theory/compilation.md)).

```python
import tsdynamics as ts

ross = ts.systems.Rossler()                              # a=0.2, b=0.2, c=5.7
traj = ross.run(final_time=200.0, dt=0.02)
traj["x"]                                        # named components
```

## Browse the catalogue

The built documentation groups ODE models by subject, with equations, defaults,
illustrations and measured diagnostics on each model page. Use
`ts.systems.find(family="ode")` to inspect the installed catalogue.

## Integrating

```python
# skip-doctest — the signature, not a runnable call
sys.run(
    final_time=100.0,    # end of the window
    dt=0.02,             # OUTPUT grid only — the stepper is adaptive
    t0=0.0,              # start time (warm restarts allowed)
    ic=None,             # initial state; falls back to self.ic, then random
    solver="rk45",       # "rk45" (default), "dop853", "tsit5", "rk4", "bdf", ...
    rtol=1e-9, atol=1e-12,   # local error tolerances
) -> Trajectory
```

For adaptive ODE solvers, `rtol` and `atol` control local error while `dt` sets
output spacing. For fixed-step methods, `dt` is the numerical step too. Check
convergence for the observable you plan to measure; dense output does not make
coarse saved data adequate for every analysis.

`ts.analysis.lyapunov_spectrum(system, final_time=200.0, dt=0.1, transient=50.0, k=None, ...)`
computes the spectrum from the variational equations — see
[Lyapunov spectra](../../analysis/lyapunov.md).

## Variable-dimension systems

A few systems (Lorenz-96, Kuramoto–Sivashinsky, MultiChua) have a
parameter that sets the *number of equations*. Such parameters are
**structural** — they change the shape of the equations rather than being
adjustable at runtime:

```python
l96 = ts.systems.Lorenz96(N=10, f=8.0)     # N is structural; f is a control param
```

Changing `f` reuses the lowered equations; changing `N` re-lowers the equations for the new size.

## See also

- [The mental model](../../start/concepts.md) — the `_equations` contract in full
- [Integration & methods](../../analysis/integration-and-methods.md) — methods, tolerances, the `Trajectory` object
