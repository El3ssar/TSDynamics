---
description: TSDynamics conventions stated precisely — time vs. steps, array shapes, IC resolution priority, map parameter order, and metadata history.
---

<span class="ts-kicker">Theory · 03</span>

# Conventions

The contracts the implementation commits to, stated once and precisely.

## Time: `t` for flows, steps for maps

- **Flows** (`ContinuousSystem`, `DelaySystem`, `StochasticSystem`) live in
  continuous time: `run(final_time, dt, t0=0.0)` produces a uniform grid from
  `t0` to `final_time` *inclusive* (the final point is appended if the grid does
  not land on it exactly). `dt` is the output sampling interval only; the
  internal stepper is adaptive. `dt=None` resolves to the family default
  (`0.02`); `dt=final_time` is the explicit two-point grid.
- **Maps** (`DiscreteMap`) live in iteration count: `run(steps)` produces
  `traj.t == arange(steps)` — integer step indices, not float times. A map
  handed `final_time=` raises and says why: a map has no continuous time.
- The protocol mirrors this: `step(n_or_dt)` means a *time increment* for
  flows (defaults 0.01 ODE / 0.1 DDE) and a *number of iterations* for
  maps (default 1); `time()` returns continuous time or the iteration
  count respectively. Derived wrappers choose their own semantics —
  a `PoincareMap.time()` is the continuous time of the last crossing.
- Lyapunov exponents follow the same units: **per unit time** for flows,
  **per iteration** for maps.

## Shapes

A `Trajectory` is always `t: (T,)` and `y: (T, dim)` — time along the
first axis, components along the second, including `dim = 1` systems
(shape `(T, 1)`, not `(T,)`). Row indexing (`traj[10:]`, boolean masks)
slices `t` and `y` together and returns a new `Trajectory`; string
indexing returns bare component arrays.

## IC resolution priority

Everywhere an initial condition is needed, `resolve_ic` applies the same
order:

1. the explicit `ic=` argument,
2. `self.ic` (set by the constructor or by a *previous* run),
3. the class-level `_default_ic` (declared by systems with small basins),
4. random `U[0, 1)^{dim}`.

An **auto-resolved** IC (cases 3 and 4) is written back to `self.ic`, so a bare
`run()` repeated reproduces the same start — a random IC is drawn once, not per
call. An **explicit** `ic=` argument does *not* latch: `run(ic=[3, 3, 3])` runs
from that point and leaves `self.ic` alone, so a later bare `run()` still starts
where it always did.

And `run()` is always a fresh integration from the resolved IC; `step()` is the
verb that continues from the live state.

## Map parameter order

`_step(X, a, b)` and `_jacobian(X, a, b)` receive parameters
**positionally, in the insertion order of the class-level `params`
dict**. The two must agree; since the names are visible in the signature,
the base class checks them at class-definition time and raises a
`TypeError` on mismatch (names *and* order). This is enforced precisely
because a silent swap once produced plausible-but-wrong dynamics.

## `meta`: provenance travels with the data, not with the system

A system carries no analysis log. Provenance lives on the **things that were
produced**: `traj.meta` is a plain dict frozen at creation (the system, the
parameters, the solver, `dt`, the tolerances, the IC, the library version), and
every analysis result carries its own `result.meta` recording the horizon it
actually used.

```python
lor = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0])

traj = lor.run(final_time=50.0, dt=0.05)
traj.meta["method"], traj.meta["dt"]           # 'rk45', 0.05

spec = ts.analysis.lyapunov_spectrum(lor, dt=0.1, final_time=200.0)
spec.meta["final_time"]                        # 200.0 — what this number came from
```

Re-running an analysis therefore never destroys anything: the previous result
object still holds its own numbers and its own settings.

## Cloning

`system.copy()` and `system.with_params(**overrides)` return *new*
instances with independent `params` and `meta`; nothing in the library
mutates a system you pass in (sweeps clone per value). Parameter keys are
fixed at class definition: values may change, keys may not, and unknown
keys raise immediately.

## See also

- [Integration & methods](../analysis/integration-and-methods.md) — these conventions in action
- [Reference · Base classes](../reference/base.md) — the implementing code
