# Parallel v3 development — streams, worktrees & CI

> **This file is documentation, not a claim board.** The *live* claim state lives
> in **GitHub issues** (one `stream`-labelled issue per stream) plus open
> `stream/*` branches — never in a checked-in file (a board file would
> merge-conflict and only reflect reality after its PR merges). The authoritative
> protocol is **ROADMAP.md §5 and §6.0**; this file is the contributor-facing
> how-to and the entry point to the helper tooling. Stream **F4** owns it.

The v3 program (Rust sole-engine + parity moat) is built by **many concurrent
sessions at once**. The architecture is deliberately one-file-per-thing with
auto-discovery so streams edit disjoint paths and rarely touch a shared file.
This page is how a fresh session goes from "what's free?" to an open PR without
colliding with anyone.

---

## TL;DR for a new session

```bash
make streams                 # what's FREE & UNBLOCKED right now (ROADMAP §6.0)
make claim ID=E1             # assign yourself + comment + re-check the race window
make worktree ID=E1 SLUG=interp   # → .claude/worktrees/tsd-E1 on branch stream/E1-interp
# ... build only within your stream's `owns` paths, against the frozen interfaces ...
# open ONE PR titled  'feat: [E1] …'  with  'Closes #<issue>'  in the body.
```

Everything above is the §6.0 claiming protocol, automated. You can run the steps
by hand with `gh` if you prefer (see ROADMAP §6.0) — the helper just removes the
boilerplate and bakes in the assign-then-recheck race mitigation.

---

## Worktree & branch conventions

| Thing | Convention | Why |
|---|---|---|
| Worktree path | `.claude/worktrees/tsd-<ID>` | Inside the project, under the **gitignored** `.claude/worktrees/`. The parent directory holds unrelated projects and must stay clean. |
| Branch | `stream/<ID>-<slug>` off `main` | One worktree per stream → no two sessions ever share a working tree. The `stream/<ID>-` prefix is the second "taken" signal the helper scans for. |
| PR title | `<type>: [<ID>] …` | The `[<ID>]` ties the PR to its stream; the `<type>:` Conventional-Commit prefix is **required** by `pr-title.yml`. |
| PR body | `Closes #<issue>` | Closes the stream's claim ticket automatically on merge. |
| Merge | green CI, rebased on `main`, **maintainer merges** | Never force-push `main`; rebase (don't merge-commit) when `main` moves under you. |

**Release safety:** PRs are squash-merged and the title becomes the
release-deciding commit. For engine-scaffolding streams that should **not** cut a
PyPI release, use a non-releasing prefix — `build:`, `chore:`, `ci:`, or `docs:`
(only `feat:`/`fix:`/`perf:`/`!` bump the version). See CONTRIBUTING.md → Release.

**Frozen interfaces:** the Foundation milestone (M0: F0–F4) freezes the shared
traits / registries / IR / package layout. Build against them; do not edit a
shared interface inside a feature PR. If an interface is wrong, raise a separate
`[interface]` PR and flag the maintainer (ROADMAP §5).

---

## The CI matrix (what runs on your PR)

CI is **path-scoped**: only the jobs relevant to your diff run; the rest skip
(and skipping is green). Merges gate on the single always-on **`CI summary`**
check in `ci.yml`.

| Workflow | Fires on | Does |
|---|---|---|
| `ci.yml` (**python job**) | `src/**`, `tests/**`, `pyproject.toml`, `uv.lock` | ruff lint/format + the pytest matrix (3.12/3.13 × Linux/macOS). |
| `rust-workspace.yml` | `crates/tsdyn-*/**`, `crates/Cargo.*` | `cargo build`/`fmt`/`clippy`/`test` over the v3 `tsdyn-*` engine workspace. |
| `engine-bindings.yml` | `crates/tsdyn-core/**`, `tests/xval_harness.py`, `tests/test_xval*.py` | `tsdyn-core` fmt/clippy/cargo-test + the engine-marked Python tests (incl. the catalogue cross-validation gate `tests/test_xval_catalogue.py`). |
| `pr-title.yml` | every PR | Enforces the `<type>: …` title. |
| `docs.yml` / `nightly.yml` / `release.yml` | docs / schedule / push-to-`main` | Pages deploy / full sweep / semantic-release. |

The three engine-era jobs (**rust-workspace + python + engine-bindings**) are
the F4 matrix split.

---

## The cross-validation scaffold

The migration's removal gate (D1 / I-XVAL) has landed: cross-validation now
targets the real `tsdynamics._rust` engine. The reusable harness lives in
[`tests/xval_harness.py`](tests/xval_harness.py) and the catalogue gate is
[`tests/test_xval_catalogue.py`](tests/test_xval_catalogue.py) — RHS lowering
within tolerance, `interp`==`jit` bit-exact, `reference`==engine, and engine
Lyapunov against the literature. It runs in `engine-bindings.yml`.

The harness is **backend-agnostic**: a backend is anything with `name`,
`available()`, and `integrate_dense(system, ic, t_eval)`. `ScipyReference` (the
pure-Python numeric oracle, always available) and the compiled engine plug in
through the same interface.

---

## Helper reference (`scripts/claim-stream.sh`)

| Command | Effect |
|---|---|
| `list` (default) | Classify every stream: FREE-&-UNBLOCKED / BLOCKED (with open blockers) / TAKEN / DONE. Read-only. |
| `claim <ID\|#n>` | Assign `@me`, comment, then re-check once and auto-release on a race (§6.0 step 4). |
| `worktree <ID> [slug]` | `git worktree add .claude/worktrees/tsd-<ID> -b stream/<ID>-<slug> main`. |
| `unclaim <ID\|#n>` | Remove your assignment + comment, returning the stream to the pool. |

`list`/`claim` read blockers from each issue's `Depends on:` line and resolve
them to issue state, so a stream only shows FREE once **all** its dependency
issues are closed. Requires an authenticated `gh`, `git`, and `python3`.

> If nothing is FREE: don't start a BLOCKED stream to get ahead. Report which
> streams are blocked and on what, then either stop or assist an active stream
> (review its PR; add tests/benches/docs). — ROADMAP §6.0 step 7.
