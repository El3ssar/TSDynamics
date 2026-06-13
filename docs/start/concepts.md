---
description: The TSDynamics mental model — a system is a params dict, a dimension, and one method; compile once, analyze through one protocol.
---

<span class="ts-kicker">Start · 03</span>

# The mental model

A system in TSDynamics is three things: a **params dict**, a **dimension**,
and **one method** that defines the dynamics. Everything else — compilation,
caching, integration, output grids, Lyapunov machinery — is inherited from
one of three base classes.

## The three contracts

### ODE — `ContinuousSystem`

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

`y(i)` is the symbolic accessor for state component `i`; `t` is the time
symbol. The body must build **symbolic expressions** — use `symengine.sin`,
`cos`, `exp`, and plain arithmetic. No NumPy, no `math`, no Python `if`
over the state: the expressions are compiled to C, not evaluated.

### DDE — `DelaySystem`

```python
class MyDDE(ts.DelaySystem):
    params = {"k": 2.0, "tau": 1.5}
    dim = 1
    _delay_params = ("tau",)        # which params are delays (default)

    @staticmethod
    def _equations(y, t, *, k, tau):
        return [k * y(0, t - tau) - y(0)]
```

The only new ingredient is the delayed accessor `y(i, t - tau)`.
`_delay_params` names the parameters that hold delay values, used to size
the history buffer; `("tau",)` is the default, so you only override it for
differently named or multiple delays.

### Map — `DiscreteMap`

```python
from tsdynamics.utils import staticjit

class MyMap(ts.DiscreteMap):
    params = {"a": 1.4, "b": 0.3}
    dim = 2

    @staticjit
    def _step(X, a, b):
        x, y = X
        return (1 - a * x**2 + y, b * x)

    @staticjit
    def _jacobian(X, a, b):
        x, y = X
        return ((-2 * a * x, 1.0), (b, 0.0))
```

Maps are plain numeric functions, JIT-compiled by Numba via `@staticjit`.

!!! warning "Parameter order is positional for maps"
    `_step` and `_jacobian` receive parameters **positionally**, in the
    insertion order of the `params` dict. A signature that disagrees with
    the dict order is rejected with a `TypeError` at import time, so this
    can no longer fail silently.

## Compile once, sweep for free

The first run of a `ContinuousSystem` compiles its equations to a shared
library cached under `~/.cache/tsdynamics/`. Ordinary parameters become
*control parameters* of the compiled module — changing them needs **no
recompile**:

```python
lor = ts.Lorenz()
lor.integrate(final_time=10)     # compiles once (first ever run)
lor.rho = 35.0                   # zero recompile cost
lor.integrate(final_time=10)     # reuses the same binary
```

Only *structural* parameters (integer loop bounds like Lorenz-96's `N`,
declared in `_structural_params`) are baked into the binary, keyed into the
cache name. DDEs recompile per parameter set; maps re-JIT in milliseconds.
Details in [the compilation pipeline](../theory/compilation.md).

## One protocol for everything that steps

All three families (and every wrapper below) implement the same `System`
protocol, which is what the analysis toolkit consumes:

```python
sys.step(n_or_dt)    # advance by n iterations / dt time → new state
sys.state()          # current state vector (copy)
sys.set_state(u)     # overwrite the state (DDEs raise — by design)
sys.time()           # current time / iteration count
sys.reinit(u, t=0.0) # restart the internal stepper
sys.trajectory(...)  # uniform-grid Trajectory
```

Stepping is lazy: the first `step()` on a fresh system performs an
implicit `reinit()`.

## Derived systems: composition

A derived system re-presents an existing system through a new lens while
keeping the protocol intact:

```python
from tsdynamics import PoincareMap, StroboscopicMap

pmap = PoincareMap(ts.Rossler(), plane=(1, 0.0))   # flow → discrete map
smap = StroboscopicMap(ts.Duffing(), period=2 * 3.14159 / 1.4)
```

One `step()` of a `PoincareMap` is one section crossing; one `step()` of a
`StroboscopicMap` is one forcing period. Because the wrappers *are*
discrete systems, every map tool now applies to flows — an
[orbit diagram](../analysis/orbit-diagrams.md) over a `PoincareMap` is a
bifurcation diagram of the flow. `TangentSystem`, `EnsembleSystem` and
`ProjectedSystem` follow the same pattern.

## The registry

Every concrete subclass of a family base auto-registers at class-definition
time — built-ins and your own classes alike:

```python
from tsdynamics import registry

registry.families()        # {'ode': 118, 'dde': 5, 'map': 26}
registry.get("Lorenz")     # SystemEntry('Lorenz', family='ode', ...)
```

Define `class MyODE(ts.ContinuousSystem)` anywhere and it appears in the
registry immediately. For built-in systems, the registry is also what the
bulk test suite sweeps and what generates the per-system pages in
[Systems](../systems/index.md) — adding a system to the library *is* adding
its docs and tests.

---

## See also

- [Systems](../systems/index.md) — the catalogue of 149 built-ins
- [Analysis](../analysis/index.md) — what to do with a system once you have one
- [Theory](../theory/index.md) — compilation and the Lyapunov math, precisely
