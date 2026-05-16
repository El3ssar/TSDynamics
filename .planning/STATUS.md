# Status — updated 2026-05-16 after M1 landed

Current milestone: **none — M1 closed. Pick the next from ROADMAP.md.**
Phase: pick. Track A (M2), Track B (V1), Track C (R2), and Track E (N2) are
all unblocked and can run in parallel chats.
Last-touched files: `src/tsdynamics/analysis/__init__.py`,
`src/tsdynamics/analysis/trajectory_ops.py`,
`src/tsdynamics/base/base.py`, `src/tsdynamics/__init__.py`,
`tests/test_trajectory_enrichment.py`, `.planning/**`.

## What's done

- **M0**: planning framework bootstrapped.
- **R1**: Rust toolchain + maturin + CI wheels — landed 2026-05-16.
- **N1**: Rust map stepper — landed 2026-05-16.
- **M1**: Trajectory enrichment — landed 2026-05-16.
  - New `src/tsdynamics/analysis/` subpackage with pure `(t, y) → (t', y')`
    functions; `Trajectory` methods are thin wrappers so the algorithms
    stay independently unit-testable.
  - `Trajectory` gained: `decimate`, `resample`, `project`, `window`,
    `derivative`, `norm`, `local_maxima`, `local_minima`, `return_times`,
    `to_dataspec`.
  - `resample` uses `scipy.interpolate.make_interp_spline(k=1|3)` —
    replaces the deprecated `interp1d`, supports `kind="linear"` and
    `kind="cubic"`.
  - `local_maxima`/`local_minima`/`return_times` forward `**kwargs` to
    `scipy.signal.find_peaks` (`prominence`, `distance`, `height`, …).
  - `to_dataspec` returns a `{kind, t, y, dims, ...}` dict — placeholder
    for V1's eventual `DataSpec` class.
  - Existing `Trajectory` API (slicing, `component`, `after`, tuple-unpack)
    is preserved; the new methods are purely additive.
  - `tsdynamics.analysis` is re-exported at the top level (in `__all__`).
  - **Tests**: 44 new cases in `tests/test_trajectory_enrichment.py`. Full
    fast suite: 660 passed / 56 skipped. Full suite (slow included):
    **792 passed / 56 skipped**. `ruff check` and `ruff format --check`
    both clean.

## What's in progress

- Nothing. M1 is closed.

## Next action

Pick one of:

- **M2** (Event & section detection): powers Poincaré, return maps. Track A.
  Now the natural successor to M1 — most call sites for events expect a
  `Trajectory` and now `Trajectory` has all the slice/decimate/project
  helpers needed downstream.
- **R2** (Rust parameter-sweep kernel): rayon-backed. Unlocks M3+.
- **V1** (Viz skeleton): DataSpec/Transform/Plotter live here. Track B.
  Note: M1's `to_dataspec(kind, **kwargs)` already returns the placeholder
  dict V1 will replace with a real class — V1 should preserve those keys.
- **N2** (Rust ODE stepper): biggest single Track E milestone but the N1
  IR is in place, so this is now multi-chat but tractable.

Default recommendation: **M2**, since it's the smallest remaining Track A
item and several downstream milestones (M4 bifurcation, M13 periodic
orbits) need event/section detection.

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

## How to resume

A future chat should:

1. Read this file.
2. Pick a milestone from the "Next action" list (or whatever the
   user asks for).
3. Open `.planning/milestones/<chosen>.md` (or create it from the
   template if it doesn't yet exist).
4. After landing the milestone: tick it in `ROADMAP.md`, rewrite this
   `STATUS.md`, commit `.planning/` together with the code.
