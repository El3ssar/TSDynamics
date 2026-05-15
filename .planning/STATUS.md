# Status — updated 2026-05-16 by M0 bootstrap chat

Current milestone: **R1 — Rust toolchain + maturin + CI wheels**
Phase: design (not yet started)
Last-touched files: `.planning/*` (whole tree just created)

## What's done

- M0: planning framework bootstrapped.
  - `.planning/ROADMAP.md`, `STATUS.md`, `CONTRIBUTING-CLAUDE.md`
  - Four design docs under `.planning/design/`
  - Four detailed milestone files under `.planning/milestones/` (M0 record + R1, N1, M1, M2)
  - `CLAUDE-PLANNING.md` at repo root
  - `CLAUDE.md` updated to mention `.planning/` as session entry point

## What's in progress

- Nothing. M0 is closed.

## Next action

**Open `.planning/milestones/R1-rust-toolchain.md` and execute it.** R1 is one chat,
docs + skeleton crates + CI plumbing, no Python behavior changes. Acceptance is that
`import tsdynamics._native.smoke` works after `pip install -e .` and `pytest` is still
green.

After R1, the natural next chat is **N1 — Rust map stepper** (`.planning/milestones/N1-rust-map-stepper.md`),
which replaces Numba inside `DiscreteMap.iterate` and the QR Lyapunov loop. Read both
N1 and the design doc `.planning/design/native-solver-migration.md` before starting.

## Open questions for the user (resolve before R1)

- **PyO3 / maturin layout:** put `crates/` at repo root with maturin building into
  `src/tsdynamics/_native/`, or use a `python/` + `rust/` split? (Recommended: keep
  `src/tsdynamics/` as the Python package and add `crates/` at repo root; maturin's
  `[tool.maturin]` config can target an installed location inside the package.)
- **MSRV (minimum supported Rust version):** pin to stable, declare in
  `rust-toolchain.toml`? Default proposal: latest stable, no MSRV promises pre-1.0.
- **Windows support from day one or Linux+macOS first?** Adds cibuildwheel complexity.
  Recommended: Linux + macOS for R1; add Windows in a follow-up if N2 reveals no
  blockers.

## How to resume

A future chat should:

1. Read this file.
2. Read `.planning/milestones/<current>.md`.
3. Read any design doc the milestone references.
4. Ask the user the open questions above (if any) before implementing.
5. After landing the milestone: tick it in `ROADMAP.md`, rewrite this `STATUS.md`,
   commit `.planning/` together with the code.
