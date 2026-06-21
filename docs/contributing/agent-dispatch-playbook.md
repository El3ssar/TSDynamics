# Agent dispatch playbook

How one or many Claude (or human) sessions take a ticket from the board to an open PR —
**in parallel, without colliding** — and how to know which tickets need extra care.

This is the canonical, durable protocol. It formalizes and extends the proven v3/v4 stream
protocol ([STREAMS.md](../../STREAMS.md), `scripts/claim-stream.sh`, the GitHub-issues claim
board). It is written for **future sessions** — read it before dispatching work.

> **One source of truth.** The live claim state is **GitHub issues** (one `stream`-labelled
> issue per ticket). The work *plan* is a `tickets.tsv` under `planning/<program>/` (e.g.
> [`planning/production/tickets.tsv`](../../planning/production/tickets.tsv)). Never keep a
> second copy of claim state in a checked-in file — it would merge-conflict and lie.

---

## 1. The ticket schema (`tickets.tsv`)

13 tab-separated columns; header row required. The first 10 are the proven v4 schema; the
last 3 are the **dispatch metadata** that tells a session *how* to work a ticket.

| # | Column | Meaning |
|---|---|---|
| 1 | `id` | Stable ticket id, `^[A-Z0-9][A-Z0-9-]*$`, unique repo-wide. Drives branch / worktree / `[ID]` PR prefix. |
| 2 | `title` | One-line imperative summary → `[ID] title` issue title. |
| 3 | `track` | Work track: `CORRECTNESS · FINISH-V4 · VIZ-GROUND · DOCS-IA · DOCS-CONTENT · DOCS-ENG · EXAMPLES · PROD-HARDEN · RELEASE` (+ the legacy `STRUCT/IFACE/PERF/POLISH`). |
| 4 | `tier` | Dependency tier `1..5` (lower = more foundational). Ordering hint, **not** mode. |
| 5 | `effort` | `S · M · L · XL`. `XL` is excluded from Mode A by rule. |
| 6 | `slug` | kebab branch suffix → `stream/<id>-<slug>`. |
| 7 | `phase` | Release bucket `P0..P3` (when, not how). |
| 8 | `depends` | Comma-separated upstream ids. **A ticket is UNBLOCKED only when every dep is a *merged* PR on origin/main.** |
| 9 | `owns` | `;`-separated glob paths this ticket may edit — **the collision key.** Two tickets in one Mode-A batch MUST have disjoint `owns`. Always list the test file(s) the ticket touches. |
| 10 | `acceptance` | Concrete, checkable done-criteria (the PR's definition of done). |
| 11 | **`mode`** | `A` (parallel fan-out) or `B` (dedicated session). See the decision rule. |
| 12 | **`care`** | `routine · adversarial · security` — the verification depth a dispatcher MUST apply regardless of mode. |
| 13 | **`verify`** | Free text: the explicit checks this ticket needs *beyond* `make test` (copied verbatim into the agent prompt **and** the PR checklist). |

**Mode decision rule** (derive it, don't guess):

> `mode = B` if **any** of: `care ∈ {adversarial, security}`; `owns` touches `crates/**`,
> `engine/**`, `families/**`, or `solvers/**`; `effort = XL`; `depends` is cross-cutting (≥2
> deps in different tracks); or it is a **correctness fix on a headline analysis**.
> Otherwise `mode = A`. A dispatcher may upgrade A→B, but never downgrade B→A without
> re-reading the audit.

**`care` → verification depth:** `routine` = author + one adversarial pass; `adversarial` =
spawn ≥1 fresh skeptic subagent + a slow-tier regression; `security` = also run the
`/security-review` skill and add a threat-model note to the PR.

**New GitHub labels** (the bootstrapper creates them): `mode:A` / `mode:B`,
`care:routine` / `care:adversarial` / `care:security`, and the new `track:*` values, alongside
the existing `stream`, `tier:N`, `phase:PN`.

---

## 2. The two modes

Both modes use the **same** claim/worktree/branch/PR/merge conventions (§4). They differ only
in **how many tickets run at once** and **how deep the verification goes**.

### Mode A — ultracode Workflow parallel fan-out (the default for routine work)

Clear a **batch** of routine, well-scoped, mutually file-disjoint tickets in **one**
orchestrator session. Each ticket is worked by a subagent in its **own git worktree**
(`isolation: 'worktree'`), through an *implement → adversarial-verify → report* pipeline. The
orchestrator never writes product code — it **selects, dispatches, collects, and PRs.**

1. **Select the batch.** `make streams` → keep `FREE ∧ UNBLOCKED ∧ mode=A ∧ care=routine`.
   Greedily build the **maximal file-disjoint** subset (no two chosen tickets share an `owns`
   glob — two tickets that both touch `analysis/_result.py` cannot batch together; defer one).
   Cap at ≈6 tickets so one session can supervise. Print the chosen batch **and** the
   deferred-because-overlap list.
2. **Fan out** with the template at
   [`scripts/workflows/dispatch-batch.mjs`](../../scripts/workflows/dispatch-batch.mjs)
   (`Workflow({ scriptPath, args: ["<ID1>", "<ID2>", …] })`). Per ticket, in its worktree:
   - **Stage 1 — Implement:** read acceptance + `verify` + `owns`; implement **only** within
     `owns`; follow CLAUDE.md (ruff, NumPy docstrings, no `.jl` refs, one-file-per-thing).
     Editing outside `owns` is an auto-fail the agent must self-report.
   - **Stage 2 — Adversarial-verify:** a *fresh* skeptic subagent (no implementation context)
     reads the diff + `verify`, runs `make test` (scoped), runs the `verify` checks, and greps
     the diff for **weakened/deleted assertions or loosened tolerances** → PASS/FAIL. On FAIL,
     bounded retry to Stage 1.
   - **Stage 3 — Report:** structured `{id, branch, files_changed, acceptance_checklist,
     verify_results, test_tail, verdict}`. **No PR yet** (the orchestrator serializes git/gh).
3. **Collect → PR** (serialized, after fan-out). For each PASSED ticket:
   `claim-stream.sh claim <ID>` (race re-check; drop on race) → push the worktree branch →
   `gh pr create` (`<type>: [ID] …`, `Closes #n`, report + `verify` checklist in the body) →
   **leave OPEN for the maintainer.** FAILED-after-retries → comment the failure on the issue,
   unclaim, mark **escalate to Mode B**. End with a `{id → PR | escalate | deferred}` table.

### Mode B — dedicated single session (high-care)

One whole session for **one** `mode=B` ticket. Same conventions; the difference is **depth**.

Choose Mode B when the schema rule fires: `care ∈ {adversarial, security}`; core files
(`crates/** · engine/** · families/** · solvers/**`); `effort=XL`; cross-track deps; or a
correctness fix on a headline analysis (these ship *silently wrong* answers, so they get the
full grill).

The loop:

1. **Claim + worktree** as in §4 (`make claim` → `make worktree`, off origin/main).
2. **Grill the design** (if the spec is fuzzy): run `grill-with-docs` / `grill-me` against the
   acceptance + the audit entry, resolving every decision branch **before** coding; update
   CLAUDE.md / glossary / ADR inline as decisions crystallize.
3. **Implement** within `owns`.
4. **Adversarial self-critique:** spawn ≥1 fresh **skeptic** subagent (no implementation
   context) to *try to refute* the change, fed the `verify` field. For a correctness fix,
   demand a **failing-first regression test** — it must fail on the *old* code and assert the
   literature/known value (e.g. Lorenz `fixed_points` → 3 equilibria; MackeyGlass DDE
   Lyapunov → +0.0072, not −0.0125).
5. **Slow-tier regression:** `make test-slow` (scoped) + any ticket-named slow gate.
   **Mandatory** for `care=adversarial` — the WS-STEPBUF bug was slow-only and a fast run
   missed it.
6. **Security review:** if `care=security`, run `/security-review` over the diff and add a
   threat-model note to the PR (entry-point/plugin discovery, FFI boundary, packaging).
7. **Invariant gates:** for engine tickets assert `interp == jit` bit-for-bit and, where an
   oracle is claimed, `reference == engine` under a fixed seed. *(The audit found the
   "bit-for-bit SDE oracle" claim was false — any ticket asserting an oracle must add the test
   that proves it, or downgrade the wording.)*
8. **PR** as in §4; the body additionally carries the skeptic verdict, the slow-tier output
   tail, the regression-test names, and (if security) the threat-model note. Rust tickets
   rebuild the extension (no symlink shortcut) and note that the `engine`-marker tests ran.

**Why not always Mode B?** It is one session per ticket — expensive. The board is mostly
routine, file-disjoint S/M work Mode A clears in one session. Reserve B for tickets where a
wrong answer ships silently or a core file changes behavior.

---

## 3. The future-session runbook

1. **Read the board:** `make streams`. Empty? → step 8.
2. **Filter to actionable:** `FREE` (unassigned, no open PR/branch) ∧ `UNBLOCKED` (every
   `depends` id is a **merged** PR on origin/main — verify with `gh pr list --state merged`,
   not local main). Order by `tier`, then `phase`, then audit priority (Tier-0 correctness
   first).
3. **Pick:** read `mode`/`care`/`verify`. `mode=B` → step 4. `mode=A` → build the maximal
   file-disjoint batch → step 5.
4. **Dispatch Mode B:** `make claim ID=X` → `make worktree ID=X SLUG=slug` → run the §2 Mode-B
   loop → push → `gh pr create` → leave OPEN. → step 6.
5. **Dispatch Mode A:** run `dispatch-batch.mjs` over the batch (implement→verify→report,
   `isolation:'worktree'`, off origin/main, `owns`-locked) → collect → PR each PASS, escalate
   each FAIL. → step 6.
6. **Hand to maintainer:** PRs stay OPEN (the deploy guardrail denies self-merge to main). The
   maintainer merges; the version bump derives from the PR title prefix.
7. **Record:** add one line to the auto-memory MEMORY index per landed/escalated ticket.
8. **Loop:** back to step 1 — newly-merged PRs unblock downstream deps.
9. **Nothing free?** Either (a) everything blocked → work the critical-path upstream ticket
   (most dependents) in Mode B to unblock the wave; or (b) board exhausted → mint new tickets
   (step 10).
10. **Add a ticket:** append a 13-column row to `tickets.tsv` (choose `mode`/`care` via the
    rule, write concrete `acceptance` + `verify`, keep `owns` disjoint if Mode-A-batchable) →
    `scripts/bootstrap-issues.sh planning/<program>/tickets.tsv` (dry run) then `--create`.

---

## 4. Conventions both modes obey (the guardrails — non-negotiable)

1. **Never weaken, delete, skip, or `xfail` an existing assertion, or loosen a tolerance, to
   make a test pass.** The adversarial stage greps the diff for removed `assert`, changed
   `atol/rtol`, added `xfail/skip` and FAILs on any — unless the ticket's `acceptance`
   explicitly sanctions a named carve-out.
2. **Branch off `origin/main`, never stale local main.** `git fetch origin` first; worktree
   from `origin/main`; rebase before the PR if it drifted. *(Past failure: local main 3
   commits behind.)*
3. **Scoped tests are the loop, the full suite is the net.** `make test` (= `pytest --changed
   -m "not slow and not full" --no-cov -n auto`) inner loop. `care=adversarial` also requires
   `make test-slow`. The full suite runs on the maintainer's merge (release.yml) + nightly.
   `--changed` is biased to over-select (any foundational/Rust/CI path → full tier), so engine
   tickets get the full run anyway.
4. **`owns` disjointness is the collision contract.** Edit only `owns` globs (incl. the test
   file). Mode-A batches are pairwise `owns`-disjoint. Enforceable because the repo is
   one-file-per-thing + auto-discovery — a new system/analysis/solver/renderer needs no edit
   to a shared dispatch file.
5. **One PR per ticket, maintainer merges.** Title `<type>: [ID] …` (passes `pr-title.yml`);
   `Closes #n`; non-releasing prefix (`build/chore/ci/docs`) unless a genuine `feat/fix/perf`.
   Self-merge to main is denied by the deploy guardrail — leave PRs OPEN.
6. **Claim is race-checked.** `claim-stream.sh claim` assigns, comments, sleeps, re-checks, and
   auto-unclaims on a detected race.
7. **Doc lock-step.** A rename or behavior change updates CLAUDE.md + the relevant docs **in
   the same PR**.
8. **Correctness fixes need a failing-first regression test** that asserts the literature/known
   value (the audit gives the numbers), so a future regression is caught.

**Worktree / branch / PR table** (unchanged from the proven protocol):

| Thing | Convention |
|---|---|
| Worktree | `.claude/worktrees/tsd-<ID>` (under gitignored `.claude/worktrees/`) |
| Branch | `stream/<ID>-<slug>` off **origin/main** |
| PR title | `<type>: [<ID>] …` (`build/chore/ci/docs` = non-releasing; `feat/fix/perf/!` bump) |
| PR body | `Closes #<issue>` + acceptance checklist + `verify` results (+ Mode-B: skeptic verdict, slow tail, regression names, security note) |
| Merge | green CI, rebased on origin/main, **maintainer merges** — never self-merge |

**Rust/worktree speed gotcha:** Mode-A tickets are by construction *not* Rust tickets, so
symlink the prebuilt `_rust*.so` into the worktree and set `PYTHONPATH=<worktree>/src` to skip
a slow maturin rebuild per worktree. Mode-B Rust tickets must rebuild the extension.

---

## 5. Tooling

- **`scripts/bootstrap-issues.sh [tickets.tsv] [--create]`** — generalized seeder; reads the
  13-column schema (tolerates 10-column legacy rows), creates the labels, mints one labeled
  issue per row, idempotent + dry-run by default.
- **`scripts/claim-stream.sh`** — `list | claim | worktree | unclaim` (race re-check; worktree
  off origin/main). `make streams | claim | worktree | unclaim` wrap it.
- **`scripts/workflows/dispatch-batch.mjs`** — the Mode-A Workflow template; edit the id list
  via `args` and run `Workflow({ scriptPath, args })`.
- **`make test` / `make test-slow`** — the scoped inner loop (never the full `uv run pytest`).

*This page belongs in the docs `Project → Contributing` nav; `DOCS-NAV-SKELETON` wires it in.*
