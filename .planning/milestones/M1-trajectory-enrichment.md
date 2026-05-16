# Milestone M1 — Trajectory enrichment

Status: DONE (2026-05-16)
Depends on: M0
Estimated scope: one chat
Design doc: none (small, self-contained)

## Motivation

The `Trajectory` class today exposes `t`, `y`, `dim`, `n_steps`, slicing,
`component(i)`, and `after(t0)`. Everything beyond integration produces bare
ndarrays. This milestone makes `Trajectory` rich enough to be the canonical
container that every analysis primitive (M2 onwards) consumes and returns —
which in turn is what lets the visualization layer (V1) build `DataSpec` objects
without each primitive re-inventing structure.

This is intentionally pure-Python and small. It sets the pattern.

## API sketch

```python
traj = lorenz.integrate(final_time=200, dt=0.005)

# existing
traj.t, traj.y, traj.dim, traj.n_steps
traj[100:200]
traj.component(0)
traj.after(50.0)

# NEW in M1
traj.decimate(every=10)                 # → Trajectory, every 10th sample
traj.resample(dt_new=0.01)              # → Trajectory, interp1d-based
traj.project(dims=(0, 2))               # → Trajectory with reduced dim
traj.window(t0=50.0, t1=100.0)          # → Trajectory in [t0, t1]
traj.derivative(order=1)                # → Trajectory, central differences
traj.norm(axis=1)                       # → ndarray (T,), ||y(t)||
traj.local_maxima(component=0)          # → indices, values
traj.local_minima(component=0)
traj.return_times(component=0)          # → ndarray of inter-peak intervals
traj.to_dataspec(kind="phase_portrait_3d", dims=(0,1,2))   # used by V1
```

5-line example:

```python
import tsdynamics as ts
lor = ts.Lorenz()
traj = lor.integrate(final_time=200, dt=0.005).after(50.0)
peaks_t, peaks_y = traj.decimate(every=2).local_maxima(component=2)
print(f"{len(peaks_t)} peaks; mean ISI = {traj.return_times(component=2).mean():.3f}")
```

## Design

All new methods live on `Trajectory` in [src/tsdynamics/base/base.py](../../src/tsdynamics/base/base.py)
unless the implementation is non-trivial, in which case they delegate to a
helper module:

- `src/tsdynamics/analysis/__init__.py` (new package)
- `src/tsdynamics/analysis/trajectory_ops.py` (new) — the actual algorithms

`Trajectory` methods are thin wrappers; the heavy logic is unit-testable
independently.

Implementation notes per method:

- `decimate(every)` — `y[::every], t[::every]`. Validate `every ≥ 1`.
- `resample(dt_new)` — `scipy.interpolate.interp1d` (cubic by default). Build new
  `t` grid by `np.arange(t[0], t[-1], dt_new)`. Defensive: error on non-monotonic
  `t`.
- `project(dims)` — `y[:, list(dims)]`. Returns Trajectory with same `system`
  ref but its `dim` mismatches the source system's `dim` — that's fine, the
  trajectory carries its own dim.
- `window(t0, t1)` — combine `after` semantics with a `before` mask.
- `derivative(order=1)` — `np.gradient(y, t, axis=0)` for order 1; recurse for
  higher. Edges use forward/backward differences (default `np.gradient`
  behavior).
- `norm(axis=1)` — `np.linalg.norm(y, axis=axis)`.
- `local_maxima(component)` — `scipy.signal.find_peaks(y[:, component])`. Same
  for `local_minima` on `-y[:, component]`. Returns `(t_at_peaks, y_at_peaks)`.
- `return_times(component)` — diff of `t` at successive maxima.
- `to_dataspec(kind, **kwargs)` — placeholder until V1 lands; for M1 it returns
  a plain dict matching the schema. V1 swaps in the real DataSpec class without
  breaking the call sites.

`Trajectory` stays immutable in spirit: every method returns a new instance, no
in-place mutation.

## Files to create / modify

Create:
- `src/tsdynamics/analysis/__init__.py`
- `src/tsdynamics/analysis/trajectory_ops.py`
- `tests/test_trajectory_enrichment.py`

Modify:
- `src/tsdynamics/base/base.py` — add the new methods on Trajectory
- `src/tsdynamics/__init__.py` — re-export `analysis` submodule (mirroring
  `utils`)

## Acceptance criteria

- [x] Every new method has a docstring with a NumPy-style "Examples" block.
- [x] `tests/test_trajectory_enrichment.py` covers:
  - decimate / resample on a synthetic ramp
  - project with valid + invalid dims
  - window edge cases (t0 < t.min, t1 > t.max)
  - derivative against analytic derivative of `sin(omega t)`
  - local_maxima / local_minima on `sin`
  - return_times reproduces 2π/ω on `sin(ω t)` within tolerance
  - immutability: source `traj.y is not result.y` for every transformation
- [x] `uv run pytest -m "not slow" --no-cov` passes (660 passed).
- [x] `uv run ruff check src/ tests/` clean.
- [x] `uv run ruff format --check src/ tests/` clean.
- [x] Existing tests still pass (full suite: 792 passed, 56 skipped).

## Resolved open questions

- `resample` default is **cubic**; ``kind="linear"`` available.  Implemented
  via `scipy.interpolate.make_interp_spline` (k=3 / k=1) — replaces the
  deprecated `interp1d` and works the same for both kinds.
- `local_maxima` / `local_minima` / `return_times` accept `**kwargs` that
  are forwarded verbatim to `scipy.signal.find_peaks` (`prominence`,
  `distance`, `height`, `width`, …).
- Kept the name `decimate` (matches numpy `[::n]` semantics).

## Implementation notes

- Heavy logic lives in `src/tsdynamics/analysis/trajectory_ops.py` as pure
  `(t, y) → (t', y')` functions; `Trajectory` methods are thin wrappers.
- `src/tsdynamics/analysis/__init__.py` re-exports the functions; the
  subpackage is itself re-exported from `tsdynamics` top-level.
- `to_dataspec` returns a plain dict with keys `{kind, t, y, dims, ...}`.
  V1 will swap in the real `DataSpec` class without breaking call sites.
- `derivative` uses repeated `np.gradient`; higher orders are recursive
  applications, so edge accuracy degrades.  Tests check the interior only.
- All transformations return new buffers (`copy()` on slices) so the source
  `Trajectory` is never aliased — verified by a dedicated immutability test
  class.

## Out of scope

- DataSpec class proper (V1).
- Anything that needs event detection / Poincaré (M2).
- Anything that needs spectral methods (M7) — `peaks`/`return_times` is the
  poor man's spectral; the real thing is later.
- Anything that needs scipy methods we don't already depend on.

## Open questions for the user

- Should `resample` default to cubic or linear interpolation? Recommended: cubic
  for smoothness; user can override with `kind="linear"`.
- Should `local_maxima` accept `prominence` / `distance` kwargs (forwarded to
  `find_peaks`)? Recommended: yes — accept `**kwargs` and forward.
- Naming: `decimate` vs `subsample`? Numpy uses `[::n]` which is decimation
  semantics. Stick with `decimate`.
