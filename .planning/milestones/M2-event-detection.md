# Milestone M2 — Event & section detection

Status: TODO
Depends on: M1
Estimated scope: one chat
Design doc: none

## Motivation

Poincaré sections, return maps, interspike intervals, transit times, threshold
crossings, period detection — all of these reduce to "find the times at which
`g(t, y(t)) = 0` along a trajectory, with sign discipline and bracketed
refinement." This milestone ships that one primitive plus a small set of
named-event helpers. Every later bifurcation / phase-space milestone (M4, M12,
M13, M14) consumes it.

## API sketch

```python
from tsdynamics.analysis.events import (
    EventCondition, Plane, Threshold, LocalExtremum, detect_events,
)
from tsdynamics.analysis import poincare_section, return_map

# Generic event detection
events = detect_events(
    traj,
    condition=Plane(axis=2, value=27, direction="up"),
)
# events.t: (K,) — refined crossing times
# events.y: (K, dim) — refined state at crossings

# Poincaré section convenience
section = poincare_section(traj, plane=Plane(axis=2, value=27))
# returns a Trajectory in (dim-1)-dim subspace of crossings

# Return map (1D scalar observable evaluated at successive crossings)
rmap = return_map(traj, plane=Plane(axis=2, value=27), observable=0)
# rmap.x: (K-1,)   values at crossing k
# rmap.y: (K-1,)   values at crossing k+1
```

5-line example:

```python
lor = ts.Lorenz().integrate(final_time=400, dt=0.005).after(50.0)
sec = poincare_section(lor, plane=Plane(axis=2, value=27, direction="up"))
print(f"{sec.n_steps} Poincaré crossings; first three z-times: {sec.t[:3]}")
```

## Design

### `EventCondition` protocol

```python
class EventCondition(Protocol):
    def evaluate(self, t: float, y: ndarray) -> float: ...
    direction: Literal["up", "down", "either"]
```

Library ships:

- `Plane(axis: int, value: float, direction="either")` — `y[axis] - value`.
- `LinearPlane(normal: ndarray, offset: float, direction="either")` — general
  hyperplane.
- `Threshold(component: int, value: float, direction="up")` — scalar threshold.
- `LocalExtremum(component: int, kind="max"|"min")` — defined via the sign of
  the discrete derivative; refined via parabolic fit (3-point).
- Generic `Custom(fn)` for `fn(t, y) -> float`.

### Detection algorithm

For a discrete trajectory `(t_k, y_k)`:

1. Evaluate `g_k = condition.evaluate(t_k, y_k)`.
2. Find sign-change indices `k` where `g_k * g_{k+1} < 0` and the sign change
   matches `direction`.
3. For each, refine using bisection on a cubic Hermite interpolant of
   `(t, y)` between `k` and `k+1`. (cubic Hermite needs slopes — use central
   differences on neighbors.)
4. Tolerance: refine until `|t_new - t_old| < rtol * (t_{k+1} - t_k)`. Default
   `rtol = 1e-8`.

For ODE/DDE trajectories, M2 uses the trajectory samples (the JiTCODE-internal
dense output isn't exposed to us yet). After N2 we'll add an overload that uses
true dense output for higher accuracy — but the public API stays the same.

### `poincare_section`

Returns a `Trajectory` where:
- `t` = refined crossing times
- `y` = interpolated state at crossings, with the section axis collapsed if the
  user supplies an `Plane` (or kept if `LinearPlane` — we project orthogonally
  to the normal).

### `return_map`

A new small result type `ReturnMap(x, y, observable_meta)`. Lives in
`src/tsdynamics/analysis/return_map.py`. Has `.to_dataspec(kind="return_map")`
for V2.

## Files to create / modify

Create:
- `src/tsdynamics/analysis/events.py`
- `src/tsdynamics/analysis/sections.py`
- `src/tsdynamics/analysis/return_map.py`
- `tests/test_events.py`
- `tests/test_poincare.py`

Modify:
- `src/tsdynamics/analysis/__init__.py` — re-export the public symbols.
- `src/tsdynamics/__init__.py` — no change (analysis already exposed in M1).

## Acceptance criteria

- [ ] `Plane`, `LinearPlane`, `Threshold`, `LocalExtremum`, `Custom` all
      implement the `EventCondition` protocol with unit tests.
- [ ] `detect_events` on a `sin(ω t)` synthetic trajectory finds threshold-zero
      crossings with refined-time error < `1e-6` for `dt = 1e-2`.
- [ ] `poincare_section(Lorenz trajectory, Plane(axis=2, value=27, "up"))`
      produces ≥ 100 points on a 400-time-unit trajectory with `dt=0.005`.
- [ ] `return_map` on the Lorenz z-section reproduces the classical "tent-like"
      Lorenz return map (visual sanity check via a regenerated reference plot
      committed to `tests/fixtures/lorenz_return_map.png`).
- [ ] `uv run pytest -m "not slow" --no-cov` passes.
- [ ] `uv run ruff check src/ tests/` clean.
- [ ] Docstrings on every public symbol with example.

## Out of scope

- Higher-order interpolants from dense output (after N2).
- Adaptive refinement of the underlying integrator's step size to land exactly
  on the event (that's a future milestone, post-N2).
- Multi-event composite conditions (AND/OR). Defer until a real user demands it.
- Plotting Poincaré sections (V2's job).

## Open questions for the user

- Should `LocalExtremum` go through `scipy.signal.find_peaks` (consistent with
  M1's `local_maxima`) or use the sign-change-on-derivative approach? The
  sign-change-on-derivative path generalises naturally to multi-dim and
  arbitrary `g(t,y)`. Recommended: sign-change for the M2 primitive,
  `find_peaks` for the M1 convenience wrapper. Document the difference.
- Should `Plane` direction default to `"either"` or `"up"`? Standard Poincaré
  usage is `"up"`. Recommended: `"either"` for the primitive, `"up"` for
  `poincare_section` (its callers want canonical sections).
- Should `return_map` support N-step returns (`y_{k+N} vs y_k`)? Recommended:
  yes, with `step=1` default. One-liner.
