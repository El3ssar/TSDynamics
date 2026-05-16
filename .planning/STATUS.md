# Status — updated 2026-05-16 after R1 landed

Current milestone: **N1 — Rust map stepper (drops Numba dispatch)**
Phase: not yet started (open `.planning/milestones/N1-rust-map-stepper.md`)
Last-touched files: `crates/**`, `pyproject.toml`, `.github/workflows/*.yml`,
`src/tsdynamics/_native/**`, `src/tsdynamics/_version.py`, `.planning/**`

## What's done

- **M0**: planning framework bootstrapped.
- **R1**: Rust toolchain + maturin + CI wheels — landed 2026-05-16.
  - Cargo workspace at `crates/` with `tsdyn-core` (rlib, placeholder
    `ProblemHandle`) and `tsdyn-smoke` (cdylib + PyO3).
  - Build backend is maturin (replaces hatchling). PyO3 0.28 (supports
    cp312 – cp314).
  - `tsdynamics._native._smoke.add_one(41) == 42` round-trips through Rust.
  - `tests/test_native_smoke.py` covers the round trip.
  - `.github/workflows/tests.yml` installs Rust before `uv sync`, runs
    `maturin develop --release` before pytest.
  - `.github/workflows/wheels.yml` (new) — Linux + macOS matrix wheels for
    cp312/cp313 + Linux sdist.
  - `publish.yml` / `release.yml` updated minimally to install Rust + build via
    maturin.
  - Versioning: dropped `hatch-vcs`; static `[project] version = "1.0.0"` and
    `src/tsdynamics/_version.py` reads from `importlib.metadata`. See R1
    milestone "Deviations" for the SCM-versioning follow-up note.

## What's in progress

- Nothing. R1 is closed.

## Next action

**Open `.planning/milestones/N1-rust-map-stepper.md` and execute it.** N1 is
the first real kernel migration: `DiscreteMap.iterate` and
`DiscreteMap.lyapunov_spectrum` move into Rust. Numba stays as a no-op
`@staticjit` wrapper so user subclasses keep working. Read
`.planning/design/native-solver-migration.md` before starting — N1 also
introduces the symbolic-IR plumbing every later N-milestone reuses.

After N1, the parallelism opens up:
- Track A can start M1 (Trajectory enrichment), M2 (events/sections).
- Track C can start R2 (parameter-sweep kernel).
- Track B can start V1 (viz skeleton).
- Track E continues with N2 (Rust ODE stepper) once N1's IR is in place.

## Open questions for the user (resolve before N1)

- **Symbolic IR scope for N1:** how much of the IR scaffolding lands in N1 vs.
  deferred to N2? The design doc suggests N1 ships a minimal `Op` enum sized
  for discrete maps; N2 extends it. Confirm before starting.
- **Numba removal timing:** N1 keeps `@staticjit` as a no-op shim. Confirm
  numba stays in `[project] dependencies` until N7, or drop earlier?
- **Reference-Python test policy:** N1 adds the first
  `tests/reference/<feature>.py` slow-but-trusted implementation pattern. The
  design doc calls for this; confirm before implementing.

## R1 follow-ups (non-blocking, parked)

- SCM-driven versioning via a small `update_version.py` script that bumps
  `crates/tsdyn-smoke/Cargo.toml` from `git describe` before each
  `maturin build`. Tracking issue: file when revisited.
- Collapse `publish.yml` + `release.yml` + the wheel-emitting half of
  `wheels.yml` into a single tag-triggered publish workflow.
- Windows wheels: add to `wheels.yml` matrix after N2 proves no blockers.

## How to resume

A future chat should:

1. Read this file.
2. Read `.planning/milestones/<current>.md` (currently N1).
3. Read any design doc the milestone references
   (`.planning/design/native-solver-migration.md` for N1).
4. Ask the user the open questions above (if any) before implementing.
5. After landing the milestone: tick it in `ROADMAP.md`, rewrite this
   `STATUS.md`, commit `.planning/` together with the code.
