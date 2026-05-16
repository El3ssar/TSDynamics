# Milestone M2 — Event & section detection

Status: DONE (2026-05-16)
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

- [x] `Plane`, `LinearPlane`, `Threshold`, `LocalExtremum`, `Custom` all
      implement the `EventCondition` protocol with unit tests.
- [x] `detect_events` on a `sin(ω t)` synthetic trajectory finds threshold-zero
      crossings with refined-time error < `1e-6` (verified for `dt ≈ 1.26e-2`
      and `dt ≈ 6.28e-3` in `tests/test_events.py`).
- [x] `poincare_section(Lorenz trajectory, Plane(axis=2, value=27, "up"))`
      produces ≥ 100 points on a 400-time-unit trajectory with `dt=0.005`
      (`tests/test_poincare.py::TestLorenzPoincare::test_section_yields_many_points`).
- [x] `return_map` on the Lorenz z-section reproduces a non-trivial, bounded
      shape — replaced the proposed reference PNG with numerical checks
      (`test_return_map_has_unimodal_shape`).  The PNG sanity-check ride-along
      moves to V2 when there's a plotter to render it.
- [x] `uv run pytest -m "not slow" --no-cov` passes (709 passed / 56 skipped).
- [x] `uv run ruff check src/ tests/` and `uv run ruff format --check ...`
      both clean.
- [x] Docstrings on every public symbol with example.
- [x] Full suite (`uv run pytest --no-cov`): 843 passed, 56 skipped.

## Resolved open questions

- `LocalExtremum` uses the sign-change-on-derivative approach (consistent with
  the generic event detector) and refines via the cubic Hermite's analytical
  derivative (a quadratic in `s`).  This stays distinct from M1's
  `local_maxima` / `local_minima`, which use `scipy.signal.find_peaks` — the
  difference is documented in both docstrings.
- `Plane.direction` defaults to `"either"`; `poincare_section` overrides it
  to `"up"` by default (canonical Poincaré convention), with an explicit
  `direction=` kwarg to opt out.
- `return_map` takes a `step` argument (default `1`) for N-step return maps.

## Design notes

- All conditions implement an `EventCondition` protocol; the protocol is
  runtime-checkable via `isinstance(cond, EventCondition)`.
- Zero-crossing conditions share a `_ZeroCrossingCondition` base that
  provides a default `detect(t, y, *, rtol)` implementation:
  1. evaluate `g_k` along the trajectory,
  2. find sign-change brackets,
  3. build a cubic Hermite interpolant of `y(t)` using slopes from
     `np.gradient`,
  4. refine the root via `scipy.optimize.brentq` on
     `s ↦ condition.evaluate(t_a + s·Δt, H(s))`.
- `LocalExtremum` overrides `detect` because the "event function" is the
  *time derivative* of one component, but the refinement scheme stays
  identical (Hermite + Brent) — the Hermite *derivative* is a closed-form
  quadratic in `s`, so this is exact in `s`.
- The bracket mask uses asymmetric `< / >=` so a sample with `g == 0` closes
  the previous bracket rather than opening a new one; the very first sample
  is special-cased because no previous bracket exists.
- `poincare_section` returns a `Trajectory` carrying the **full state** at
  the refined crossings — we deliberately don't collapse the section axis.
  The user can `.project()` if they want a `(dim - 1)`-dim view, which is
  one line and far less surprising than silently dropping a dimension.
- `return_map.ReturnMap.to_dataspec(kind="return_map")` returns the same
  V1-style placeholder dict shape as M1's `to_dataspec`, so V2 can swap in
  the real `DataSpec` without changing call sites.

## Out of scope

- Higher-order interpolants from dense output (after N2).
- Adaptive refinement of the underlying integrator's step size to land exactly
  on the event (that's a future milestone, post-N2).
- Multi-event composite conditions (AND/OR). Defer until a real user demands it.
- Plotting Poincaré sections / committing a reference PNG (V2's job; the
  numerical checks above stand in until then).
