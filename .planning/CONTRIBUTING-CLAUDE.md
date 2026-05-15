# How a Claude chat works on TSDynamics

This file is the protocol every chat follows. It is short on purpose.

## On opening a chat

1. **Read [`STATUS.md`](STATUS.md) first.** It points at the current milestone and the
   literal next action.
2. **Read the milestone file** under `milestones/` that STATUS points at, end to end.
3. **Read any design doc** the milestone references (paths are listed inside the
   milestone file).
4. **Read [`ROADMAP.md`](ROADMAP.md)** only if you need to understand sequencing or
   dependencies — not required for execution.
5. **Read [`/CLAUDE.md`](../CLAUDE.md)** for codebase conventions (ruff, line length,
   commit format, base-class contract).

If the milestone has *Open questions*, ask the user before writing code.

## While working

- Keep one milestone in progress at a time. Don't drift across milestones.
- Use the existing test patterns in `tests/`. Every new public symbol gets a test and
  a docstring with example.
- Run `uv run ruff check src/ tests/` and `uv run ruff format src/ tests/` before
  committing.
- Run `uv run pytest -m "not slow" --no-cov` for the fast smoke; full suite before
  PR.
- Conventional Commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`.

## On closing a chat

1. **Tick the milestone** in `ROADMAP.md` (`TODO` → `WIP` → `DONE`).
2. **Rewrite [`STATUS.md`](STATUS.md)** with:
   - the new current milestone (the next one in the roadmap),
   - the literal next action a future chat should take,
   - any new open questions for the user.
3. **Commit `.planning/`** in the same commit as the code, so reading the diff tells
   the whole story.
4. If you're leaving the milestone unfinished, update the milestone file's *Status* to
   `WIP` and write what's left in STATUS's *What's in progress*.

## Non-negotiables (from the master plan)

- The `_equations` / `_jacobian` / `params` / `dim` contract is sacred. Never require
  the user to change a system definition to get a new feature.
- `Trajectory` slicing/projection returns new objects. Analysis results live in their
  own result types, never bare ndarrays for non-trivial outputs.
- **No backend choice surfaces to users.** No `backend="..."` kwarg, no
  `use_native=True`, no env var override of the implementation. The implementation
  language is an internal detail.
- Every Rust-backed analysis primitive has a benchmark recorded under `bench/` against
  the DynamicalSystems.jl equivalent.

## Files you may edit

- Anything under `src/`, `tests/`, `bench/`, `docs/`, `examples/`.
- `pyproject.toml`, `Cargo.toml`, `rust-toolchain.toml` (once they exist).
- `.planning/STATUS.md`, `.planning/ROADMAP.md`, the milestone file you're executing.

## Files you should leave alone unless you have a reason

- Other milestone files (they're owned by their respective chats).
- Design docs (edit only with an ADR-style commit explaining why).
- The `decisions/` ADR directory is append-only; never rewrite a past ADR.

## When you're unsure

- If a milestone open question hasn't been answered, **ask the user** rather than
  guess. Architectural choices made silently rot fastest.
- If a milestone seems wrong, write a new ADR proposing the change before
  re-planning.
