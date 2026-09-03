---
description: The TSDynamics mental model — four families sharing one stepping protocol, three execution backends, and derived wrappers that re-present a system through a new lens.
---

<span class="ts-kicker">Start · 03</span>

# The mental model

The whole library rests on a small set of ideas. A **system** is a class that
declares its parameters, its dimension, and *one method* holding the dynamics.
Everything downstream — lowering to the engine, integration, output grids, the
Lyapunov machinery, the analysis toolkit — is inherited from a family base
class. This page is the map of those ideas.

## The four families

A system belongs to one of four families, chosen by the base class you subclass.
Each has exactly one method to fill in.

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
`milstein` for strong order 1). Here `dt` *is* the noise scale $\sqrt{dt}$, so it
sets both the discretisation and the output grid. Pass `seed=` for a reproducible
realisation.

### Map — `DiscreteMap`

A discrete map $\mathbf{x}_{n+1} = \mathbf{F}(\mathbf{x}_n)$. Fill in `_step`
(and, for the exact Lyapunov spectrum, `_jacobian`):

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

!!! warning "Parameter order is positional for maps and SDEs"
    `_step`/`_jacobian` (and `_drift`/`_diffusion`) receive parameters
    **positionally**, in the insertion order of the `params` dict — whereas an
    ODE/DDE `_equations` takes them keyword-only (`*, a`). A map signature that
    disagrees with the `params` order is rejected with a `TypeError` at import,
    so this can never fail silently.

The full worked example — from a blank class to an integrated, plotted,
auto-registered system — is on [Defining systems](defining-systems.md).

## Lower once, sweep for free

A continuous system lowers its equations to the Rust engine in-process, with no
warmup and nothing cached on disk. Ordinary parameters become *control
parameters* of the lowered tape — changing them is free:

```python
lor = ts.systems.Lorenz()
lor.integrate(final_time=10)     # runs immediately (no warmup)
lor.rho = 35.0                   # zero cost
lor.integrate(final_time=10)     # same tape, new parameter value
```

Only *structural* parameters — integer loop bounds that change the *shape* of the
equations, like Lorenz-96's `N`, declared in `_structural_params` — require a
fresh lowering. A parameter sweep (a continuation, a bifurcation diagram) reuses
one cached tape, so it stays cheap. Delay values are the exception: a DDE bakes
its delay into the tape, so a delay sweep re-lowers per value.

## One protocol for everything that steps

All four families — and every derived wrapper below — implement the same
`System` protocol. This is the surface the analysis toolkit consumes, so writing
one analysis makes it work on *every* system:

```python
sys.step(n_or_dt)      # advance by n iterations / dt of time → new state
sys.state()            # current state vector (a copy)
sys.set_state(u)       # overwrite the state (DDEs raise — by design)
sys.time()             # current time / iteration count
sys.reinit(u, t=0.0)   # restart the internal stepper from a fresh state
sys.trajectory(...)    # run and return a uniform-grid Trajectory
sys.is_discrete        # True for maps and derived discrete views
```

Stepping is lazy — the first `step()` on a fresh system performs an implicit
`reinit()`:

```python
lor = ts.systems.Lorenz()
lor.reinit([1.0, 1.0, 1.0])
u1 = lor.step(0.01)          # integrate one dt chunk from the live state
u2 = lor.step(0.01)          # continue from there
```

`integrate` and `iterate` are the convenience verbs on top of this protocol for
the common "run a whole trajectory" case.

## Backends: `interp`, `jit`, `reference`

Orthogonal to *what* a system is is *how* the engine executes its tape. The same
run works on any of three backends, chosen with `backend=`:

| `backend` | What it is | When to use it |
| --------- | ---------- | -------------- |
| `"interp"` | The Rust tape interpreter — the default | Everyday integration; no warmup, no compile step |
| `"jit"` | The Cranelift JIT, which compiles the tape to native code | Long or repeated runs where the compiled speed pays off; **bit-for-bit identical** to `interp` |
| `"reference"` | A dependency-light pure-Python oracle (ODEs, SDEs, maps) | Cross-validation and wheel-free environments — the answer key, not the fast path |

```python
traj = ts.systems.Lorenz().integrate(final_time=100.0, dt=0.01, backend="jit")
```

`interp` and `jit` lower the *same* tape, so they agree to the last bit.
`reference` is an independent implementation kept as a correctness oracle; it
tracks the compiled path closely at early times and, being a separate float
computation, diverges only as roundoff accumulates on a chaotic orbit — exactly
as any two independent integrators would. Not every family supports it: delay
systems have no pure-Python integrator and reject `backend="reference"` loudly
rather than silently degrading.

## Derived systems: composition

A **derived system** re-presents an existing system through a new lens while
keeping the protocol intact — so every analysis keeps working on the wrapped
view. The wrappers live at the top level:

```python
from tsdynamics import PoincareMap, StroboscopicMap

pmap = PoincareMap(ts.systems.Rossler(), plane=("y", 0.0, "up"))   # flow → discrete map
smap = StroboscopicMap(ts.systems.Duffing(), period=2 * 3.14159 / 1.4)
```

One `step()` of a `PoincareMap` is one section crossing; one `step()` of a
`StroboscopicMap` is one forcing period. Because the wrappers *are* discrete
systems (`is_discrete == True`), every map tool applies to a flow — an
[orbit diagram](../analysis/orbit-diagrams.md) over a `PoincareMap` **is** a
bifurcation diagram of the flow. The other wrappers follow the same pattern:

- **`TangentSystem`** — the flow plus its variational (tangent) dynamics; the
  Lyapunov engine.
- **`EnsembleSystem`** — many initial conditions advanced together as one batch.
- **`ProjectedSystem`** — the same dynamics observed through a lower-dimensional
  projection.

## The registry

Every concrete subclass of a family base **auto-registers at class-definition
time** — built-ins and your own classes alike:

```python
from tsdynamics import registry

registry.families()        # {'ode': 136, 'dde': 6, 'sde': 3, 'map': 26}
registry.get("Lorenz")     # SystemEntry(name='Lorenz', family='ode', ...)
```

Define `class MyODE(ts.ContinuousSystem)` anywhere in your code and it appears in
the registry immediately. For built-in systems the registry is also what the
bulk test suite sweeps and what generates the per-system pages under
[Systems](../systems/index.md): adding a system to the library *is* adding its
tests and its documentation.

---

## See also

- [Defining systems](defining-systems.md) — the four contracts as one worked example
- [Systems](../systems/index.md) — the catalogue of 171 built-ins
- [Analysis](../analysis/index.md) — what to do with a system once you have one
- [Integration & methods](../analysis/integration-and-methods.md) — solvers, tolerances, and the backends in depth
