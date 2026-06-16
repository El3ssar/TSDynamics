---
description: The 118 built-in continuous flows — ODE systems run on the native Rust engine, organized into eight categories.
---

<span class="ts-kicker">Systems · Continuous</span>

# Continuous systems

The largest family: **118 ODE systems**, all subclasses of
[`ContinuousSystem`](../../reference/base.md). Each declares its parameters
and dimension at class level and defines the vector field in one symbolic
`_equations` method; it is lowered to the native Rust engine in-process and
runs with no warmup (see
[the compilation pipeline](../../theory/compilation.md)).

```python
import tsdynamics as ts

ross = ts.Rossler()                              # a=0.2, b=0.2, c=5.7
traj = ross.integrate(final_time=200.0, dt=0.02)
traj["x"]                                        # named components
```

## Categories

| Category | Count | Flavor |
| -------- | ----- | ------ |
| `chaotic_attractors` | 47 | The classics: Lorenz, Rössler, Chen, Chua, Sprott flows, Duffing, Thomas, ... |
| `chem_bio_systems` | 14 | Chemical oscillators and excitable media: Brusselator, FitzHugh–Nagumo, ... |
| `climate_geophysics` | 9 | Atmosphere and geodynamo models: Lorenz-84, Lorenz-96, Rikitake, ... |
| `coupled_systems` | 14 | Coupled and driven oscillator networks |
| `exotic_systems` | 15 | Hyperchaos, multiscroll, and other unusual flows |
| `oscillatory_systems` | 8 | Tori, Lissajous figures, stick–slip and relaxation oscillators |
| `physical_systems` | 8 | Mechanical and electrical models: pendula, circuits |
| `population_dynamics` | 3 | Ecological flows: predator–prey and competition models |

Each system has its own generated page in this section with equations,
defaults, and a phase portrait.

## Integrating

```python
sys.integrate(
    final_time=100.0,    # end of the window
    dt=0.02,             # OUTPUT grid only — the stepper is adaptive
    t0=0.0,              # start time (warm restarts allowed)
    ic=None,             # initial state; falls back to self.ic, then random
    method="rk45",       # "rk45" (default), "dop853", "tsit5", "rk4", "bdf", ...
    rtol=1e-6, atol=1e-9,
) -> Trajectory
```

`lyapunov_spectrum(final_time=200.0, dt=0.1, burn_in=50.0, n_exp=None, ...)`
computes the spectrum from the variational equations — see
[Lyapunov spectra](../../analysis/lyapunov.md).

## Variable-dimension systems

A few systems (Lorenz-96, Kuramoto–Sivashinsky, MultiChua) have a
parameter that sets the *number of equations*. Such parameters are
**structural** — they change the shape of the equations rather than being
adjustable at runtime:

```python
l96 = ts.Lorenz96(N=10, f=8.0)     # N is structural; f is a control param
```

Changing `f` is free; changing `N` re-lowers the equations for the new size.

## See also

- [The mental model](../../start/concepts.md) — the `_equations` contract in full
- [Integrate & iterate](../../analysis/integrate.md) — methods, tolerances, the `Trajectory` object
