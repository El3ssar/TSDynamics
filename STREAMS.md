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
| `cross-validation.yml` | `tests/xval_harness.py`, `tests/test_xval*.py`, `crates/tsdynamics-core/**`, `rustcore.py` | Builds the v2-seed accelerator and runs the **Rust-vs-v2 xval** scaffold on Lorenz. |
| `rust-core.yml` | `crates/tsdynamics-core/**`, `rustcore.py`, `tests/test_rustcore*.py` | Builds the v2 accelerator + its numeric kernel tests. |
| `pr-title.yml` | every PR | Enforces the `<type>: …` title. |
| `docs.yml` / `nightly.yml` / `release.yml` | docs / schedule / push-to-`main` | Pages deploy / full sweep / semantic-release. |

The three engine-era jobs (**rust-workspace + python + cross-validation**) are
the F4 matrix split; `rust-core.yml` stays scoped to the v2 accelerator until
I-XVAL retires it (ROADMAP §9, milestone M3).

---

## The cross-validation scaffold

Before any v2 backend is deleted (D1), every system's Rust trajectory must match
its v2 trajectory within tolerance. The reusable harness lives in
[`tests/xval_harness.py`](tests/xval_harness.py); [`tests/test_xval.py`](tests/test_xval.py)
demonstrates it on Lorenz.

```bash
make xval         # runs the scaffold; the Rust path skips if the accelerator isn't installed
make xval-build   # builds+installs the v2-seed accelerator first, so the Rust path runs too
```

The harness is **backend-agnostic**: a backend is anything with `name`,
`available()`, and `integrate_dense(system, ic, t_eval)`. Two ship today —
`ScipyReference` (the v2 numeric truth, always available) and `RustCore` (the
accelerator). When the new engine lands (streams E1–E7) it plugs in as one more
backend with no new plumbing; **I-XVAL** sweeps the harness over the whole
catalogue to build the removal gate.

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
