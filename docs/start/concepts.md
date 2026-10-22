---
description: Understand the four system families, their shared stepping methods, execution backends and derived systems.
---

<span class="ts-kicker">Start · 03</span>

# The mental model

A **system** declares state variables, parameters and equations or an update
rule. Its family supplies the corresponding simulation methods. Analysis
functions and plots then work with the system or its recorded trajectories,
subject to their mathematical and numerical requirements.

## The four families

A system belongs to one of four families, chosen by the base class you subclass.
ODEs, DDEs and maps need one dynamics method; SDEs need both drift and diffusion.

### ODE — `ContinuousSystem`

A flow $\dot{\mathbf{u}} = \mathbf{f}(\mathbf{u}, t)$. Fill in `_equations`:

```python
import tsdynamics as ts

class MyODE(ts.ContinuousSystem):
    params = {"a": 1.0}
    dim = 2

    @staticmethod
    def _equations(y, t, *, a):
        return (
            a * y(0) - y(1),
            y(0) * y(1) - y(1),
        )
```

`y(i)` is the symbolic accessor for state component `i`; `t` is the time symbol.
The body builds **symbolic expressions** — use `symengine.sin`, `cos`, `exp`, and
plain arithmetic. No NumPy, no `math`, and no Python `if` on the state: the
expressions are lowered to an engine tape, not evaluated point by point.

### DDE — `DelaySystem`

A delay equation, where the rate depends on the state at an earlier time
$t - \tau$. The only new ingredient is the *delayed* accessor `y(i, t - tau)`:

```python
class MyDDE(ts.DelaySystem):
    params = {"k": 2.0, "tau": 1.5}
    dim = 1
    _delay_params = ("tau",)        # which params hold delay values (the default)

    @staticmethod
    def _equations(y, t, *, k, tau):
        return [k * y(0, t - tau) - y(0)]
```

`_delay_params` names the parameters that carry delay values, used to size the
history buffer. Its default already covers a single parameter named `tau`, so you
override it only for a differently named or multiple delays. A delay system's
*state* is a whole history **function**, not a point — which is why it is
integrated from a `history=` callable rather than a bare `ic`.

### SDE — `StochasticSystem`

A diagonal-Itô stochastic equation
$dX_k = f_k\,dt + g_k\,dW_k$ with independent noise per component. Fill in a
symbolic **drift** and a symbolic **diffusion** (one coefficient per component):

```python
class MySDE(ts.StochasticSystem):
    params = {"theta": 1.0, "mu": 0.0, "sigma": 0.3}
    dim = 1

    @staticmethod
    def _drift(y, t, theta, mu, sigma):
        return [theta * (mu - y(0))]      # the deterministic skeleton

    @staticmethod
    def _diffusion(y, t, theta, mu, sigma):
        return [sigma]                    # the noise coefficient
```

Integrating an SDE runs a fixed-step scheme (`euler_maruyama` by default,
`milstein` for strong order 1 with componentwise diffusion
$g_k = g_k(X_k,t)$). Cross-component diffusion coefficients require
`euler_maruyama`; this Milstein implementation does not simulate Lévy areas.
Here `dt` sets both the discretisation and the output grid; Wiener increments
have standard deviation $\sqrt{dt}$. Pass `seed=` for a reproducible realisation.

### Map — `DiscreteMap`

A discrete map $\mathbf{x}_{n+1} = \mathbf{F}(\mathbf{x}_n)$. Fill in `_step`;
its Jacobian is derived automatically. An explicit `_jacobian` is optional:

```python
import numpy as np

class MyMap(ts.DiscreteMap):
    params = {"a": 1.4, "b": 0.3}
    dim = 2

    @staticmethod
    def _step(X, a, b):
        x, y = X
        return (1 - a * x**2 + y, b * x)

    @staticmethod
    def _jacobian(X, a, b):
        x, y = X
        return ((-2 * a * x, 1.0), (b, 0.0))
```

Maps are the one family whose body uses **NumPy** functions (`np.sin`, `np.cos`,
`np.exp`, …) rather than `symengine` — the map `_step` is traced symbolically
through those ufuncs to build its tape.

!!! warning "Map parameters are positional"
    A map's `_step` and `_jacobian` receive parameters in the insertion order of
    `params`. Their signatures must match that order; a mismatch raises
    `TypeError` when the class is defined. ODE/DDE `_equations` and SDE
    `_drift`/`_diffusion` receive parameters by name instead.

The full worked example — from a blank class to an integrated, plotted,
auto-registered system — is on [Defining systems](defining-systems.md).

## Reuse compiled equations across parameter sweeps

A continuous system lowers its equations to the Rust engine in-process, with
nothing cached on disk and no ahead-of-time build step. Ordinary parameters
become *control parameters* of the lowered tape: updating their values reuses
the compiled equations rather than compiling a separate program for every value.

```python
lor = ts.systems.Lorenz()
lor.run(final_time=10)     # lowers + compiles on first use
lor.rho = 35.0            # update a value without changing the equations
lor.run(final_time=10)     # same tape, new parameter value
```

Only *structural* parameters — integer loop bounds that change the *shape* of the
equations, like Lorenz-96's `N`, declared in `_structural_params` — require a
fresh lowering. A parameter sweep can reuse the cached tape. Delay values are
the exception: a DDE embeds
its delay into the tape, so a delay sweep re-lowers per value.

## One protocol for everything that steps

All four families — and every derived wrapper below — implement the same
`System` protocol. Analyses use this common surface while declaring their own
supported families and mathematical assumptions:

```python
# skip-doctest — the protocol surface, not a runnable snippet (`sys` is any system)
sys.run(...)           # run and return a Trajectory
sys.step(n_or_dt)      # advance by n iterations / dt of time → new state
sys.state()            # current state vector (a copy)
sys.set_state(u)       # overwrite the state (absent on DDEs — by design)
sys.time()             # current time / iteration count
sys.reinit(u, t=0.0)   # restart the internal stepper from a fresh state
sys.dim                # the state-space dimension (read-only)
sys.family             # "ode" | "dde" | "map" | "sde"
```

Stepping is lazy — the first `step()` on a fresh system performs an implicit
`reinit()`:

```python
lor = ts.systems.Lorenz()
lor.reinit([1.0, 1.0, 1.0])
u1 = lor.step(0.01)          # integrate one dt chunk from the live state
u2 = lor.step(0.01)          # continue from there
```

`run()` starts a fresh simulation from the initial condition; `step()` continues
from the current state. An automatically resolved initial condition
is retained on the instance, so repeated deterministic runs reuse that start.
For reproducibility across fresh instances, fix the initial condition or pass
`seed=`; SDEs also need a fixed noise seed to reproduce their realisation.

!!! note "`family` replaces `is_discrete`"
    `family` identifies `"ode"`, `"dde"`, `"sde"` or `"map"`.
    Use `sys.family == "map"` to identify a discrete map.

## Backends: `jit`, `interp`, `reference`

Choose how the equations are evaluated with `backend=`. Available backends
depend on the family:

| `backend` | What it is | When to use it |
| --------- | ---------- | -------------- |
| `"jit"` | Cranelift compiles the expression tape to native code; compiled evaluators are cached | The default for repeated simulations and parameter sweeps |
| `"interp"` | The Rust interpreter evaluates the same tape without JIT compilation | Short runs, profiling or comparison with JIT execution |
| `"reference"` | A separate Python implementation for ODEs, SDEs and maps | Cross-checks; this path is usually slower and has no DDE integrator |

```python
traj = ts.systems.Lorenz().run(final_time=100.0, dt=0.01, backend="interp")
```

`interp` and `jit` evaluate the same tape using the same numerical kernels and
are tested for bitwise agreement on the same platform. The separate `reference`
implementation provides a cross-check. Its rounding and solver behavior can
differ, and chaotic trajectories amplify small differences over time. Delay
systems reject `backend="reference"` because no Python DDE integrator is provided.

## Derived systems: composition

A **derived system** exposes a different view of an existing system while
keeping the protocol intact. Analyses can consume the wrapped view when its
family and capabilities meet their requirements, and it still answers `run`.

Two common derived systems are available as methods:

```python
pmap = ts.systems.Rossler().poincare("y", 0.0)        # flow → a discrete map
strobe = ts.systems.Duffing().poincare(period=4.488)  # ...sampled once per forcing period
band = ts.systems.Lorenz().ensemble(np.random.rand(100, 3))   # 100 copies, one object
```

`poincare` takes **either** a section plane **or** a strobe period — an affine
surface $g(\mathbf{u}) = \mathbf{n}\cdot\mathbf{u} - c$ and a sampling of the
phase circle are two different sections of the same flow, so giving both raises
and names the choice. One `step()` of the result is one section crossing (or one
forcing period).

Because the result **is** a discrete system, compatible map tools apply to a flow —
an [orbit diagram](../analysis/orbit-diagrams.md) over a Poincaré map **is** a
bifurcation diagram of the flow.

`band.run(final_time=…)` returns a `TrajectoryBatch` — a read-only sequence of trajectories
that also carries `.final`, the `(n, dim)` array of end states.

Tangent dynamics and state projection are available through two further wrappers:

```python
from tsdynamics.derived import TangentSystem, ProjectedSystem

tang = TangentSystem(ts.systems.Lorenz(), k=2)      # flow + variational dynamics
proj = ProjectedSystem(ts.systems.Lorenz(), (0, 2))  # observe (x, z) only
```

## The registry

Every concrete subclass of a family base **auto-registers at class-definition
time** — built-ins and your own classes alike:

```python
from tsdynamics import registry

registry.families()        # {'ode': 142, 'dde': 6, 'sde': 3, 'map': 26}
registry.get("Lorenz")     # SystemEntry('Lorenz', family='ode', category=..., dim=3)
```

Define `class MyODE(ts.ContinuousSystem)` anywhere in your code and it appears in
the registry immediately. For built-in systems the registry is also what the
bulk test suite sweeps and what generates the per-system pages under
[Systems](../systems/index.md). New catalogue entries participate in the shared
tests and documentation generation; model-specific checks are still needed to
verify their equations and expected behavior.

---

## See also

- [Defining systems](defining-systems.md) — the four contracts as one worked example
- [Systems](../systems/index.md) — the catalogue of 177 built-ins
- [Analysis](../analysis/index.md) — what to do with a system once you have one
- [Integration & methods](../analysis/integration-and-methods.md) — solvers, tolerances, and the backends in depth
