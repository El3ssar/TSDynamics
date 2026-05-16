# Status — updated 2026-05-16 (M2 + analysis-layer registry refactor)

Current milestone: **none — M2 closed. Pick the next from ROADMAP.md.**
Phase: pick. Track A (M4 — but it needs R2 + M3 first; M5 / M6 / M7 / M10 /
M11 are independent), Track B (V1), Track C (R2), and Track E (N2) are all
unblocked and can run in parallel chats.
Last-touched files:
`src/tsdynamics/analysis/_registry.py` (new — the @trajectory_op decorator
+ install_methods),
`src/tsdynamics/analysis/__init__.py`,
`src/tsdynamics/analysis/_trajectory_ops.py`,
`src/tsdynamics/analysis/events.py`,
`src/tsdynamics/analysis/sections.py`,
`src/tsdynamics/analysis/return_map.py`,
`src/tsdynamics/base/base.py` (~300 lines of wrapper methods deleted,
replaced by a single `install_methods(Trajectory)` call),
`tests/test_events.py`, `tests/test_poincare.py`,
`tests/test_trajectory_enrichment.py`,
`.planning/**`, `CHANGELOG.md`, `CLAUDE.md`.

## What's done

- **M0**: planning framework bootstrapped.
- **R1**: Rust toolchain + maturin + CI wheels — landed 2026-05-16.
- **N1**: Rust map stepper — landed 2026-05-16.
- **M1**: Trajectory enrichment — landed 2026-05-16.
- **M2**: Event & section detection — landed 2026-05-16.
  - New modules under `src/tsdynamics/analysis/`: `events.py`, `sections.py`,
    `return_map.py`.
  - `EventCondition` protocol (runtime-checkable) + the five built-in
    conditions: `Plane`, `LinearPlane`, `Threshold`, `LocalExtremum`,
    `Custom`.  All zero-crossing conditions share a `_ZeroCrossingCondition`
    base that implements bracket-find + cubic-Hermite + Brent refinement.
  - `LocalExtremum` refines via the cubic Hermite's *analytical* derivative
    (a closed-form quadratic in the local parameter), which is exact in `s`
    and avoids the 3-point parabolic fit the milestone sketch proposed.
  - `EventResult` carries `(t, y, indices, direction, condition)`; unpacks
    as `(t, y)` like a `Trajectory`.
  - `detect_events(traj, condition, *, rtol=1e-8)` accepts either a
    `Trajectory` or a bare `(t, y)` tuple — keeps the function unit-testable
    without spinning up an integrator.
  - `poincare_section(traj, plane, *, direction="up", rtol=1e-8)` returns a
    `Trajectory` of refined crossings (full state preserved; users
    `.project()` to drop the section axis if they want).
  - `return_map(traj, plane, observable=0, *, step=1, direction="up")`
    returns a `ReturnMap` (`x`, `y`, `t`, `step`, `observable_meta`) with a
    `to_dataspec(kind="return_map")` shim matching the V1 dataspec shape.
  - Bracket-detection convention: asymmetric `< / >=` so a sample with
    `g == 0` closes the previous bracket rather than opening a new one.
    The very first sample is special-cased because no previous bracket
    exists.
  - All new public symbols re-exported from `tsdynamics.analysis`.
  - **Tests**: 49 new fast cases in `tests/test_events.py` (33) +
    `tests/test_poincare.py` (16), plus 2 slow Lorenz Poincaré-section /
    return-map tests.  Full fast suite: **709 passed / 56 skipped**.  Full
    suite (slow included): **843 passed / 56 skipped** in ~118 s.
    `ruff check` and `ruff format --check` both clean.

## Analysis-layer registry refactor (post-M2)

The post-M2 UX pass had ended with M1 methods written by hand on
`Trajectory` (10+ near-identical wrappers) plus duplicate doc/wrapper
methods for the three new M2 ops.  That was wrong — every primitive
needed two near-identical declarations: the function in
`_trajectory_ops.py` (or `events.py` / `sections.py` / `return_map.py`)
and a wrapper method on the class.  Rewritten as a **single decorator
registry**:

- **New `analysis/_registry.py`** holds the design.  The
  `@trajectory_op(returns=...)` decorator on a free function
  `fn(t, y, *args, **kw)` does two things at import time:
  - Wraps the function so it accepts a `Trajectory`, a `(t, y)` tuple,
    or bare arrays as the leading argument(s).  This is the public
    free-function form.
  - Appends an entry to a module-level registry.
  At Trajectory-class-init time, `install_methods(Trajectory)` (called
  from the bottom of `base/base.py`) drains the registry and installs
  one method per entry.  Method signatures are surgically adjusted via
  `inspect.Signature` so `help(traj.decimate)` shows
  `(self, every)` instead of the underlying `(t, y, every)`.
- **All 13 analysis primitives** (M1 + M2: `decimate`, `resample`,
  `project`, `window`, `derivative`, `norm`, `local_maxima`,
  `local_minima`, `return_times`, `to_dataspec`, `detect_events`,
  `poincare_section`, `return_map`) now have **one** definition each.
  No wrapper methods anywhere.  `base/base.py` shrank by ~300 lines.
- **Both call forms are exposed equally**:
  - `decimate(traj, every=5)` / `decimate((t, y), every=5)` /
    `decimate(t, y, every=5)` — polymorphic free function.
  - `traj.decimate(every=5)` — method (installed at import time).
- **`returns=`** declares wrapping shape: `"trajectory"` (fn returns
  `(t_new, y_new)`), `"ndarray_keep_t"` (fn returns ndarray on same
  time axis, only `derivative`), `"passthrough"` (no wrapping, used by
  `detect_events`, `return_map`, peak-finding helpers, `to_dataspec`).
- **`Trajectory` output uniformly** for `trajectory` / `ndarray_keep_t`
  ops regardless of input shape, with `system` preserved from a
  `Trajectory` input and `None` otherwise.  `Trajectory` is
  tuple-unpackable so legacy `t, y = decimate(...)` call sites still
  work.
- **Adding a new analysis primitive** is now exactly one decorated
  function + one re-export from `analysis/__init__.py`.  No class to
  touch, no docstring to copy.
- **Tests**: 852 passed / 56 skipped (full suite, slow included).
  `ruff check` + `ruff format --check` clean.

## What's in progress

- Nothing. M2 is closed.

## Next action

Pick one of:

- **R2** (Rust parameter-sweep kernel): rayon-backed. Unlocks M3, which
  unlocks M4 (bifurcation diagrams).  Track A → C handoff.
- **V1** (Viz skeleton): DataSpec/Transform/Plotter live here.  M2 left
  three `to_dataspec`-style placeholder dicts (`timeseries`,
  `phase_portrait_*`, `return_map`); V1 should preserve those keys.
- **N2** (Rust ODE stepper): biggest single Track E milestone but the N1
  IR is in place, so this is now multi-chat but tractable.
- **M5** (Equilibria & local stability): independent of M2 outputs.
- **M6** (Embedding utilities): independent.
- **M7** (Spectral toolkit): independent.

Default recommendation: **V1**, since M2 stacked up *three* placeholder
dataspec shapes (`timeseries`, `phase_portrait_*`, `return_map`) and the
sooner V1 turns those into a real `DataSpec` class the less churn there is
later.  R2 is the runner-up because it directly unblocks the bifurcation
track.

## R1 follow-ups (still parked)

- SCM-driven versioning via `update_version.py`.
- Collapse `publish.yml` + `release.yml` + the wheel-emitting half of
  `wheels.yml` into a single tag-triggered publish workflow.
- Windows wheels: add to `wheels.yml` matrix after N2 proves no blockers.

## N1 follow-ups (parked)

- **Performance**: add a constant-folding pass on the IR (lift
  `1.4 * x**2` to `Mul(Const(1.4), Pow(Var(0), 2))` evaluated with
  fewer ops; the win is small per step but compounds). Optional —
  N4 cranelift makes it moot.
- **Disk-backed IR cache**: process-local for N1; revisit in N4 when
  cranelift codegen makes recompilation per-process expensive.
- **DynamicalSystems.jl comparison**: not yet recorded. Add when P1's
  benchmark harness lands.

## M1 follow-ups (parked)

- `to_dataspec` is a dict shim. V1 must replace it with the real
  `DataSpec` class while keeping the `{kind, t, y, dims, ...}` shape so
  callers don't change.
- `resample`'s cubic spline at the array edges inherits boundary
  artifacts from `make_interp_spline` (natural-spline endpoints). If a
  user reports issues, expose a `bc_type` kwarg.

## M2 follow-ups (parked)

- **Dense-output refinement**: today the cubic Hermite is built from
  central-difference slopes of the sampled trajectory — accurate enough
  for `dt = 5e-3` Lorenz, but post-N2 we should use the integrator's
  native dense output (`dop853` / `dopri5` continuous extension) for
  events found in flight.  Public API stays unchanged.
- **Reference return-map PNG**: deferred to V2 when there's a plotter
  that can render it deterministically.  Current tests check
  bounded-ness + non-trivial spread numerically instead.
- **Composite conditions** (AND/OR over multiple conditions): not yet
  needed; defer until a user demands it.
- **Optional section-axis collapse** in `poincare_section`: today we
  keep the full state; a `keep_axis=False` flag for the Plane case
  would be a one-liner if asked.

## How to resume

A future chat should:

1. Read this file.
2. Pick a milestone from the "Next action" list (or whatever the
   user asks for).
3. Open `.planning/milestones/<chosen>.md` (or create it from the
   template if it doesn't yet exist).
4. After landing the milestone: tick it in `ROADMAP.md`, rewrite this
   `STATUS.md`, commit `.planning/` together with the code.
