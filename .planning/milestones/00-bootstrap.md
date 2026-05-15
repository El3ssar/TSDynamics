# Milestone M0 — Bootstrap `.planning/` framework

Status: DONE (2026-05-16)
Depends on: nothing
Estimated scope: one chat (this one)

## Motivation

Establish the persistent planning framework so every future chat can resume
without re-deriving context. The framework lives in-repo, is versioned, and is
the canonical entry point for all subsequent work.

## What was delivered

- `.planning/ROADMAP.md` — master list of milestones with status badges.
- `.planning/STATUS.md` — current state, pointer to next milestone, open
  questions.
- `.planning/CONTRIBUTING-CLAUDE.md` — protocol for every future chat.
- `.planning/design/viz-architecture.md` — Track B (visualization) design.
- `.planning/design/rust-acceleration.md` — Track C (analysis kernels) design.
- `.planning/design/native-solver-migration.md` — Track E (solver migration)
  design.
- `.planning/design/parallelism-and-sweeps.md` — stub, fleshed out when R2 / M3
  start.
- `.planning/milestones/00-bootstrap.md` — this file.
- `.planning/milestones/R1-rust-toolchain.md` — detailed next-up milestone.
- `.planning/milestones/N1-rust-map-stepper.md` — detailed.
- `.planning/milestones/M1-trajectory-enrichment.md` — detailed.
- `.planning/milestones/M2-event-detection.md` — detailed.
- `CLAUDE-PLANNING.md` at repo root — pointer into `.planning/`.
- `CLAUDE.md` updated to mention `.planning/` as session entry point.

## Acceptance criteria

- [x] `.planning/` directory tree exists and matches the layout in the master plan.
- [x] `STATUS.md` points at R1 as the next milestone and lists open questions.
- [x] `ROADMAP.md` lists all 35 milestones in execution order with TODO badges
      (M0 marked DONE).
- [x] Each of the four detailed milestone files (R1, N1, M1, M2) follows the
      template (Motivation / API sketch / Design / Files / Acceptance / Out of
      scope / Open questions).
- [x] `CLAUDE.md` updated.
- [x] `CLAUDE-PLANNING.md` exists at repo root.
- [x] Commit lands on `chore/planning-bootstrap` branch.
- [x] `uv run pytest -m "not slow" --no-cov` still passes (M0 changes no code).
- [x] `uv run ruff check src/ tests/` still clean.

## Out of scope

- Any actual code changes — M0 is docs only.
- Writing detailed milestone files for everything beyond R1/N1/M1/M2. Those land
  as their respective chats start.

## Why the next chat opens R1

R1 is the gating dependency for **every Rust-backed milestone in Track C and
Track E**. Until R1 lands, R2-R6, N1-N7 are all blocked. R1 itself ships no
user-visible features and changes no Python behaviour — it's purely build
plumbing. That makes it both the lowest-risk and most-unblocking item to do
next.

After R1, the parallel structure of the roadmap opens up: a chat can pick any
of {N1, M1, M2, V1, M5, M6, M7, M10, M11} without dependencies on each other.
The recommended order remains `R1 → N1 → M1 → M2 → R2 → M3 → V1` to lock in the
Rust toolchain proof and the Trajectory + viz substrate early.
