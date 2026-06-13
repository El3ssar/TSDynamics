---
description: TSDynamics conventions stated precisely — time vs. steps, array shapes, IC resolution priority, map parameter order, and metadata history.
---

<span class="ts-kicker">Theory · 03</span>

# Conventions

The contracts the implementation commits to, stated once and precisely.

## Time: `t` for flows, steps for maps

- **Flows** (`ContinuousSystem`, `DelaySystem`) live in continuous time:
  `integrate(final_time, dt, t0=0.0)` produces a uniform grid from `t0` to
  `final_time` *inclusive* (the final point is appended if the grid does
  not land on it exactly). `dt` is the output sampling interval only; the
  internal stepper is adaptive.
- **Maps** (`DiscreteMap`) live in iteration count: `iterate(steps)`
  produces `traj.t == arange(steps)` — integer step indices, not float
  times.
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
3. the class-level `default_ic` (declared by systems with small basins),
4. random `U[0, 1)^{dim}`.

The resolved IC is **written back to `self.ic`**, so repeating a call
without arguments reproduces the same start — a random IC is drawn once,
not per call.

## Map parameter order

`_step(X, a, b)` and `_jacobian(X, a, b)` receive parameters
**positionally, in the insertion order of the class-level `params`
dict**. The two must agree; since the names are visible in the signature,
the base class checks them at class-definition time and raises a
`TypeError` on mismatch (names *and* order). This is enforced precisely
because a silent swap once produced plausible-but-wrong dynamics.

## `meta`: append-with-history

`system.meta` is a `MetaStore` — dict-like for everyday use, but every
write **appends** rather than overwrites:

```python
lor.lyapunov_spectrum(dt=0.1)            # records value + settings + timestamp
lor.lyapunov_spectrum(dt=0.05)           # appends a second record

lor.meta["lyapunov_spectrum"]            # the LATEST value
lor.meta.history("lyapunov_spectrum")    # every record, with its context
lor.meta.latest()                        # plain dict of latest values
```

Re-running an analysis never destroys the previous result or the settings
that produced it. (Trajectory `meta`, by contrast, is a plain provenance
dict frozen at creation.)

## Cloning

`system.copy()` and `system.with_params(**overrides)` return *new*
instances with independent `params` and `meta`; nothing in the library
mutates a system you pass in (sweeps clone per value). Parameter keys are
fixed at class definition: values may change, keys may not, and unknown
keys raise immediately.

## See also

- [Integrate & iterate](../analysis/integrate.md) — these conventions in action
- [Reference · Base classes](../reference/base.md) — the implementing code
