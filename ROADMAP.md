# TSDynamics Roadmap — v3 "Rust Engine + Parity" Program

**Mission.** Make TSDynamics *the* reference platform for nonlinear dynamics in
Python: every capability the established Julia dynamical-systems ecosystem
offers — reproduced with our own design — plus first-class DDEs, SDEs, a
zero-warmup Rust engine, and the simplest system-definition contract in any
language. Then go beyond it.

The bar is not "works" — it is **robust, deep, and ready-to-publish**. A
researcher should define a system in three lines, integrate it on a fast and
trustworthy engine, and run the full chaos-quantification and parity-moat suite
to produce literature-validated, reproducible, citable results — without leaving
Python. **§0** lists the aspirational perks that define "superior"; **§13** holds
the live consolidation status and the continuation notes independent sessions
build from.

---

## 0. Vision — what "superior" means (the perks we are building toward)

These are commitments, not nice-to-haves. Every stream should advance at least
one and regress none. They are the yardstick for "are we better than the
benchmark yet?".

1. **The simplest system-definition contract in any language — and never
   regress it.** `params` + `dim` + one symbolic `_equations` (or `_step` +
   `_jacobian` for maps; `_drift` + `_diffusion` for SDEs). The user writes the
   math; the library handles compilation, caching, output grids, provenance and
   docs. Proven over 149 systems — the moat a monolith can't copy.
2. **One uniform interface across every family.** ODE, DDE, SDE and maps all
   implement the same `System` protocol and compose through the same derived
   wrappers. **First-class DDEs and SDEs** — integration *and* Lyapunov — in that
   one interface is a standing differentiator.
3. **A real numerical-solver collection, pluggable by name.** Explicit RK family
   (rk4 / rk45 / tsit5 / dop853), L-stable implicit/stiff family
   (Rosenbrock-W / TR-BDF2, analytic-Jacobian), SDE kernels (Euler–Maruyama /
   Milstein), DDE method-of-steps — with error control, dense output, event
   detection, auto-stiffness selection, and **third-party solvers registerable
   from outside**. Symplectic/geometric integrators, higher-order stiff
   (BDF/Radau) and adaptive-order methods are on the roadmap, not the ceiling.
4. **Zero warmup *and* native speed on demand.** The SSA-tape interpreter starts
   instantly; the pure-Rust Cranelift JIT delivers native throughput for
   large/hot problems — both behind one `Evaluator`, numerically identical.
5. **A complete, literature-validated analysis moat.** Lyapunov (spectra / max /
   from-data), chaos indicators (GALI, 0–1 test, expansion entropy), fixed &
   periodic orbits for maps *and* flows, **attractors / basins / global
   continuation / tipping / resilience** (the deepest part of the moat), fractal
   dimensions, delay embeddings, entropy & complexity (incl. LZ76), recurrence /
   RQA, surrogate hypothesis tests. Each lands only with a test reproducing a
   published value.
6. **Transforms / feature extraction feeding analysis.** Spectral (PSD, spectral
   entropy), detrend / filter / normalize, generic feature extractors — the
   signal layer that turns raw trajectories into analyzable features.
7. **Publication-grade reproducibility, baked in.** Every result carries
   provenance (`Trajectory.meta`: system, params, solver, tolerances, ic, seed,
   version); every stochastic/sampling entry is seeded so parallel == serial
   bit-for-bit; quantifiers cite the *original* literature (never a competitor).
8. **Python ecosystem gravity.** numpy / pandas / scikit-learn / the ML stack one
   import away; `Trajectory` is the lingua franca everything consumes.
9. **Pluggable everything.** Systems, solvers, analyses and transforms
   self-register (class hooks + directory scans + entry-point plugins) — an
   external ecosystem a monolith can't match.
10. **Performance you can publish.** Benchmarked vs SciPy and (internally) the
    Julia baseline on Lorenz / Rössler / Mackey–Glass / Lorenz-96 N=128 /
    Robertson; time-to-first-result tracked as a first-class metric.
11. **Visualization (deferred to 3.x, designed for now).** Result objects are
    shaped so a `.plot()` / dashboard layer can grow on top without a redesign
    (D6).

**This file is the single source of truth for the v3 program.** It is written
for *autonomous coding sessions* (including future instances of the assistant)
who will build it **in parallel**. Read the banner below before doing anything.

---

## ⛳ READ THIS FIRST — you are one of many concurrent sessions

Several sessions (other instances of you) are working this roadmap **at the same
time**. To not collide:

1. **You own exactly one Work Stream.** Your launcher may tell you your stream ID
   (e.g. `E2`, `A7`). If it didn't, **follow the Claiming Protocol (§6.0)**: query
   the GitHub `stream` issues, pick a free, unblocked one, assign yourself, and
   say which you took. Never start two streams; never start a `blocked` one.
2. **Work in a dedicated git worktree**, never on `main` directly:
   `git worktree add .claude/worktrees/tsd-<ID> -b stream/<ID>-<slug> main`. Keep
   the worktree *inside* the project under `.claude/worktrees/` (gitignored) — not
   in the parent directory, which holds unrelated projects and must stay clean.
   One worktree per stream → no two sessions ever share a working tree.
3. **Touch only the paths your stream `owns`** (listed per stream). If you need
   something outside them, you need an *interface*, not an edit — see §5.
4. **Interfaces are frozen after the Foundation milestone (M0).** Build against
   them; do not change a shared trait/registry/contract in a feature stream. If
   an interface is wrong, open a separate `[interface]` PR and flag it.
5. **One PR per stream, titled `[<ID>] …`. Merge only on green CI.** Rebase onto
   `main` before merging; never force-push `main`.
6. **Shared files are append-only** (registries, workspace members, CI matrix,
   `__init__` re-exports). Add your line; never reorder or reformat others'. The
   architecture is deliberately built so most work needs *no* shared-file edits
   (one-file-per-thing + auto-discovery — see §4e).
7. **Never name the Julia ecosystem (or any competitor) in shipped code, docs,
   or comments.** Cite the *original papers*. (This ROADMAP is an internal
   planning doc and may name the benchmark; nothing here ships.)
8. **Keep `main` always releasable.** Every merged PR must leave the full test
   suite green and the package importable.

---

## 1. The decisions that govern this program

These were made deliberately; treat them as fixed unless this file changes.

| # | Decision | Consequence |
|---|---|---|
| **D1** | **Rust is the sole integration engine** (total replacement). | JiTCODE, JiTCDDE, Numba-maps and the diffsol bridge are migrated onto the Rust engine and **removed** once parity + cross-validation are proven. One engine to maintain; no Python-only fallback long-term. |
| **D2** | **Tiered execution: SSA-tape interpreter + optional Cranelift JIT.** | Interpreter = zero warmup (sweeps, tests, small/medium systems). Cranelift JIT = native-code RHS for large/hot problems. **Pure-Rust** (no LLVM) → wheels stay trivial. Both implement one `Evaluator` interface. |
| **D3** | **Clean break, no compatibility shims.** | v3 reorganizes the package freely (modular submodules, Cargo workspace). Old import paths may simply move. Audience is small; we optimize for the right design, not back-compat. |
| **D4** | **Traits + registries + external plugins.** | A Rust trait per extensible kind (`Evaluator`, `Solver`, `Problem`); name registries; **Python entry-point discovery** so third parties register their own systems / solvers / analyses / transforms **without forking**. |
| **D5** | **Parallel work via git worktrees, PR-per-stream.** | See the banner. Worktree isolation + interface-first design is what makes ~20 streams safe to run at once. |
| **D6** | **Visualization is deferred.** | No `viz` work in the v3 gate. A thin `tsdynamics.viz` stub exists so result objects *can* grow a `.plot()` later, but every stream concentrates on the engine, numerics, and data extraction. |
| **D7** | **v3.0 ships on big-bang parity.** | v3.0 is **not** cut until the new architecture + Rust sole-engine **and** the parity moat (basins, fractal dimensions, delay embeddings, recurrence/RQA, surrogates, the full chaos-quantification suite) are all in and literature-validated. One dramatic launch. Internal milestones (M0…M6, §7) stage the work; they are not separate PyPI releases unless we choose to pre-release `3.0.0bN`. |

**Still-open sub-decisions** (do **not** block on these unless your stream needs
them; flagged in §11): the Cranelift JIT trigger heuristic; the exact public
top-level `__all__`; whether `tsdynamics` ships as one maturin wheel or keeps a
separable accelerator. Resolve via an `[interface]`/`[decision]` PR + a note to
the maintainer. *(The SDE noise contract is now **decided** — see §11.)*

---

## 2. Status — where we are at the start of v3 (post v2.2.x)

**Shipped to PyPI (v2 line):** v2.0–v2.2.x. Release + docs deploy automated on
every push to `main` (python-semantic-release + GitHub Pages).

**The v2 Rust seed (already merged — this is what the v3 engine grows from):**
- `crates/tsdynamics-core/` — a PyO3/maturin crate, **optional accelerator**.
- An **SSA expression tape VM**: the symbolic `_equations` (and the analytic
  Jacobian) lower to a flat instruction tape a Rust stack machine evaluates
  GIL-free. **All 118 ODEs lower and match the symbolic RHS to 1e-15.**
- **Three solver kernels**, each cross-validated vs SciPy: fixed-step **RK4**,
  adaptive **Dormand-Prince RK45** (Hermite dense output), and an L-stable
  linearly-implicit **stiff** kernel using the analytic Jacobian (vs Radau/LSODA
  on Van der Pol/Robertson/Oregonator/Chua).
- **rayon ensemble** integration: race-free, ~90× a SciPy per-trajectory loop;
  diverging trajectories raise / return `NaN` (never silent garbage).
- CI: `rust-core.yml` builds the wheel + runs the numeric suite. Every increment
  was adversarially reviewed.

**v2 still in place (to be replaced under D1):** JiTCODE (ODE), JiTCDDE (DDE),
Numba (maps), the diffsol bridge. 149 built-in systems (118 ODE + 5 DDE +
26 maps). `System` protocol + derived wrappers + `Trajectory` + registry +
`tsdynamics.sampling`.

**The v3 job** is to (a) promote that seed into a proper modular engine that is
*the* backend for **all** families, (b) reorganize the library into separated,
pluggable modules, and (c) build the analysis/parity layer on top — all in
parallel.

---

## 3. Architecture at a glance

```
                    ┌──────────────────────────────────────────────┐
   user writes →    │  System subclass: params + dim + _equations    │   (contract UNCHANGED)
                    └───────────────┬──────────────────────────────┘
                                    │  symbolic (SymEngine)
                        ┌───────────▼────────────┐
   Python: tsdynamics  │  compile → IR (tape)     │  tsdynamics.engine.compile
                        └───────────┬────────────┘
   ───────────────────────────────  FFI (PyO3, zero-copy)  ───────────────────
                        ┌───────────▼────────────┐
   Rust workspace       │  tsdyn-ir   (instruction set / tape)            │
                        │  tsdyn-vm   (interpreter)  ─┐                    │
                        │  tsdyn-jit  (Cranelift)   ──┴► Evaluator trait   │
                        │  tsdyn-solvers (Solver trait: RK*, stiff, SDE,…) │
                        │  tsdyn-engine (problem + ensembles + rayon + RNG)│
                        │  tsdyn-core (PyO3 → tsdynamics._rust)            │
                        └──────────────────────────────────────────────┘
```

Everything above the FFI line is Python and pure-Python-importable; everything
below is the compiled wheel. The **Evaluator trait** (interpreter or JIT) and the
**Solver trait** are the two seams that make the engine pluggable; the
**Python registries + entry points** make systems/solvers/analyses/transforms
pluggable from outside.

---

## 4. Target design (the contracts every stream builds against)

### 4a. Rust: a Cargo **workspace** of small crates

`crates/` becomes a workspace (`crates/Cargo.toml` with `[workspace] members`).
One concern per crate so streams don't collide:

| Crate | Responsibility | Depends on |
|---|---|---|
| `tsdyn-ir` | The instruction set + tape data structure (ops, args, immediates, outputs, jac_outputs). Pure data + a builder. **The frozen contract** between Python-compile and Rust-eval. | — |
| `tsdyn-vm` | The interpreter `Evaluator` over `tsdyn-ir` (today's stack machine, generalized). | `tsdyn-ir` |
| `tsdyn-jit` | A Cranelift `Evaluator` that compiles a tape to native code. Same trait as `tsdyn-vm`. | `tsdyn-ir`, `cranelift-*` |
| `tsdyn-solvers` | The `Solver` trait + one module per kernel (explicit RK family, implicit/stiff family, SDE, symplectic…). Dense output, error control, step adaption live here. | `tsdyn-ir` (calls `Evaluator`) |
| `tsdyn-engine` | Problem definition (ODE/DDE/SDE/map), the integrate loop, **ensembles (rayon)**, RNG/Wiener substrate, DDE history buffers, event detection. Wires an `Evaluator` + a `Solver` to a `Problem`. | all above |
| `tsdyn-core` | PyO3 bindings → builds the `tsdynamics._rust` extension module. Thin: marshal numpy ↔ slices, release GIL, dispatch. | `tsdyn-engine` |

**Trait sketches (Foundation stream freezes the real signatures):**

```rust
// tsdyn-ir
pub struct Tape { /* ops, a, b, imm, outputs, jac_outputs, n_state, n_param */ }

// the seam D2 hangs on:
pub trait Evaluator {            // implemented by tsdyn-vm AND tsdyn-jit
    fn eval(&self, u: &[f64], p: &[f64], t: f64, deriv: &mut [f64]);
    fn eval_jac(&self, u: &[f64], p: &[f64], t: f64, deriv: &mut [f64], jac: &mut [f64]);
}

// tsdyn-solvers — one impl per kernel, registered by name
pub trait Solver {
    fn name(&self) -> &'static str;
    fn caps(&self) -> Caps;       // explicit/implicit, adaptive, needs_jacobian, supports: ODE|SDE|…
    fn step(&mut self, ev: &dyn Evaluator, st: &mut SolverState, h: f64) -> StepOutcome;
}
```

### 4b. Python: a clean modular package (D3 — things MOVE)

```
src/tsdynamics/
├── __init__.py            # curated re-exports (append-only per stream)
├── _rust/                 # the compiled extension (built by tsdyn-core)
├── registry.py            # registries: systems / solvers / analyses / transforms
├── plugins.py             # entry-point discovery (D4)
├── engine/                # the Rust-facing engine API
│   ├── compile.py         #   symbolic _equations (+Jacobian) → IR/tape
│   ├── problem.py         #   Problem builders per family
│   └── run.py             #   integrate / ensemble entry points, backend select (interp|jit)
├── solvers/               # solver registry + per-solver Python metadata; plugin hook
├── families/              # base classes + the System protocol
│   ├── protocol.py, base.py
│   ├── continuous.py, delay.py, discrete.py, stochastic.py   # all → Rust engine
├── derived/               # PoincareMap, StroboscopicMap, TangentSystem, EnsembleSystem, …
├── data/                  # Trajectory, state-space regions/samplers, set distances, KD-tree
├── systems/               # the built-in catalogue (one module per category, as today)
├── analysis/              # the quantification layer (see Track C)
│   ├── lyapunov/  chaos/  fixedpoints/  orbits/
│   ├── basins/    dimensions/  embedding/  entropy/  recurrence/  surrogate/
├── transforms/            # signal/data transforms feeding analysis (spectral, filters, features)
└── viz/                   # DEFERRED — stub only (D6)
```

Old paths (`tsdynamics.base.ode_base`, `tsdynamics.backends.*`,
`tsdynamics.sampling`) **move**; update imports, no shims (D3).

### 4c. The execution engine (D1 + D2)

- The **IR/tape** (`tsdyn-ir`) is the universal lowering target. The Python
  `engine.compile` produces it from any system's symbolic form (RHS + Jacobian
  for stiff). Maps lower their `_step`; DDEs lower RHS with delayed-state slots;
  SDEs lower drift + diffusion (see S-ENG-SDE).
- Two `Evaluator`s: interpreter (default, zero warmup) and Cranelift JIT (opt-in
  / auto for large or long runs). **Numerical results must match** between them
  (a cross-check test is mandatory in the JIT stream).
- One integrate path per family in `tsdyn-engine`; ensembles are rayon fan-out
  over independent `Evaluator`+`Solver` workers (already proven race-free).
- Determinism: every stochastic/sampling entry takes a seed; ensembles seed
  per-trajectory-index so parallel == serial bit-for-bit.

### 4d. Solvers as separate, registered units (the modularity the maintainer asked for)

Each solver is **its own module/file** implementing `Solver`, registered by name
in a registry that is **auto-populated** (Rust: an `inventory`/`linkme`-style
registry or an explicit `register!` per file; Python: a `solvers/` directory
scanned at import). Adding a solver = adding one file + one registry line in
*that file* — **no central table to edit**, so two solver streams never
conflict. Capability flags drive `method=` resolution and an
**auto-stiffness-detection** layer (pick implicit when the Jacobian spectrum /
rejected-step ratio says stiff).

### 4e. Why this parallelizes (the anti-collision design)

- **One file per thing**: one crate per concern, one module per solver, one
  module per system category, one module per analysis. Streams edit disjoint
  files.
- **Auto-discovery over central registries**: systems self-register
  (`__init_subclass__`, already true); solvers/analyses/transforms discovered by
  directory scan + entry points. Few hand-edited shared tables.
- **Append-only shared files**: `crates/Cargo.toml` members, top-level
  `__init__.py` re-exports, the CI job matrix — add your line at the end.
- **Interface-first milestone (M0)** freezes the traits/registry/IR so feature
  streams compile against stable seams from day one.

---

## 5. Working in parallel — the protocol (D5)

**Lifecycle of a stream session:**
1. Confirm your stream ID; read its row in §6 (goal, `owns`, `depends-on`,
   `parallel-with`, acceptance).
2. `git worktree add .claude/worktrees/tsd-<ID> -b stream/<ID>-<slug> main`
   (inside the project, under the gitignored `.claude/worktrees/`).
3. Build only within `owns`. Consume frozen interfaces from §4; do not modify
   them. Need a new interface? Smallest possible `[interface]` PR first, merged
   before dependents rely on it.
4. Tests: add your stream's tests; keep the whole suite green. Rust streams:
   `cargo fmt --check` + `cargo clippy -- -D warnings` + tests. Python streams:
   `ruff check` + `ruff format --check` + `pytest`.
5. **Adversarially review** non-trivial numeric/Rust work before the PR (spawn a
   review pass; it has repeatedly caught real defects). Fix; regression-test.
6. PR titled `[<ID>] …`; green CI; rebase; merge; `git worktree remove`.

**If you finish early or are blocked:** pick another `unclaimed` stream (respect
`depends-on`), or extend test/bench/doc coverage of a `done` stream. Announce
what you took.

**Conflict rules:** never reorder/reformat code you don't own; never edit a
frozen interface in a feature PR; if `main` moved under you, rebase (don't
merge-commit); if two streams genuinely need the same file, the later one waits
or coordinates via the maintainer.

**Definition of done (every stream):** code + tests green in CI; public API
documented (NumPy docstrings, paper citations, **no competitor names**);
`main` still releasable; acceptance bullets met.

---

## 6. The Stream Board

Status legend: `done` · `active` · `unclaimed` · `blocked(by …)`.
Streams in the **same tier** are designed to run **simultaneously**.

### 6.0 Claiming protocol — how a fresh session knows what's free

There is **no shared runtime memory** between sessions. The only state every
session can observe is **git + GitHub**, so claims live there, not in chat.

**Source of truth = GitHub Issues (one ticket per stream).** Each stream has an
issue titled `[<ID>] <name>`, label `stream` (+ a `tier:N` label), its row in the
body, and its blockers as `Depends on #…` task-list links. A stream is:

- **free** — issue open & **unassigned**, and all its `depends-on` issues closed;
- **taken** — issue **assigned**, or an open `stream/<ID>-*` branch / PR exists;
- **done** — issue **closed** (its PR merged).

**A self-directed session ("read the ROADMAP and start working") MUST, before
touching code:**

```bash
gh issue list --label stream --state open \
    --json number,title,assignees,labels         # candidate tickets
gh pr list --state open --json headRefName        # 2nd signal: branches in flight
```
1. Drop tickets whose `depends-on` issues aren't closed (blocked).
2. Drop tickets already **assigned** or with an open `stream/<ID>-*` PR/branch.
3. Pick the highest-priority remaining (lowest tier first; Foundation before all).
4. **Claim before coding:** `gh issue edit <n> --add-assignee @me` **and** comment
   "claimed by <session label>". Then **re-list once** — if it was taken in the
   gap, release and pick another. (Assign-then-recheck closes most of the race.)
5. `git worktree add .claude/worktrees/tsd-<ID> -b stream/<ID>-<slug> main`
   (inside the project, under the gitignored `.claude/worktrees/`); open the PR
   with **`Closes #<n>`** in the body so the ticket closes on merge.
6. If you abandon a stream, **unassign + comment** so it returns to the pool.

**Race reality (be honest):** GitHub has no atomic claim — two sessions launched
in the same second *could* grab one ticket. Two mitigations: the
assign-then-recheck step above; and the **zero-race path — the human assigns
streams at launch** ("you are `A-DIM`"). Use self-service pickup for "give me
anything free", explicit assignment when firing several at once.

**Why issues, not a board file:** a `STREAMS.md` claim file merge-conflicts and a
claim only becomes visible *after* its PR merges; issues are conflict-free,
queryable, and assignable. The §6 tables stay the human-readable definition;
the **issues are the live claim state**. *(Generate/refresh the issues from this
board with the maintainer's `gh`-based bootstrap — one issue per row.)*

### Tier 0 — Foundation (small, fast, mostly sequential; UNBLOCKS EVERYONE)

> Goal: in a handful of tight PRs, freeze every interface so Tiers 1–3 fan out.
> Prefer to land these first; coordinate closely (few sessions, or one).

| ID | Stream | Owns | Depends | Acceptance |
|----|--------|------|---------|------------|
| **F0** | Cargo **workspace** scaffold | `crates/Cargo.toml`, crate skeletons (`tsdyn-ir/vm/jit/solvers/engine/core`) with empty-but-compiling lib targets | — | `cargo build` over the workspace; CI builds it; existing `tsdynamics-core` behavior preserved or re-homed into `tsdyn-core`. |
| **F1** | **IR contract** (`tsdyn-ir`) | the `Tape` types + builder + opcode set (migrate from today's vm.rs) | F0 | Opcodes documented; Python↔Rust round-trip test; the v2 tape semantics preserved (still matches symbolic RHS to 1e-15). |
| **F2** | **Evaluator + Solver traits** | trait defs in `tsdyn-ir`/`tsdyn-solvers`, the registry mechanism (Rust auto-register + Python `solvers/` scan + `plugins.py` entry points), `Caps` flags | F1 | A dummy solver registers and is discoverable by name from Python; entry-point discovery loads an out-of-tree toy plugin in a test. |
| **F3** | **Python package reorg skeleton** | the new directory tree (§4b) with modules created and imports re-pointed; `registry.py` extended to 4 registries | F0 | `import tsdynamics` works; full v2 test suite passes against moved paths (tests updated, no shims). |
| **F4** | **Parallel-dev harness** | `STREAMS`/board automation, CI matrix split (rust-workspace job + python job + cross-validation job), worktree/branch conventions doc, the migration cross-validation scaffold (Rust-vs-v2 trajectory compare) | F0 | CI green on an empty change; a `make claim`/docs path for sessions; cross-validation harness runs on Lorenz. |

### Tier 1 — Engine (Rust + the Python compile/run seam) — parallel after F2

| ID | Stream | Owns | Depends | Parallel-with | Acceptance |
|----|--------|------|---------|----------|------------|
| **E1** | Interpreter evaluator (`tsdyn-vm`) | `crates/tsdyn-vm/**` | F1 | E2,E3,… | RHS+Jac eval matches v2 tape to 1e-15 over all 118 ODEs. |
| **E2** | **Cranelift JIT** evaluator (`tsdyn-jit`) | `crates/tsdyn-jit/**` | F1,F2 | E1,E3,… | JIT eval == interpreter eval to ~1e-12 across catalogue; compile latency benchmarked; wheels still build with no LLVM. |
| **E3** | Explicit solver family | `crates/tsdyn-solvers/explicit/**` (RK4, DP45/RK45, DOP853, Tsit5…) | F2 | E4,E5 | Each matches SciPy at tolerance; one file per method; all auto-registered. |
| **E4** | Implicit/stiff solver family | `crates/tsdyn-solvers/implicit/**` (Rosenbrock, SDIRK/TR-BDF2, BDF) | F2 | E3,E5 | vs Radau/LSODA on Robertson/VdP/Oregonator; analytic-Jacobian path. |
| **E5** | Engine core: integrate + ensembles + RNG | `crates/tsdyn-engine/**` | E1 | E3,E4 | single + rayon ensemble paths; seeded determinism (parallel==serial). |
| **E6** | **Symbolic → IR compiler** (Python) | `tsdynamics/engine/compile.py` (migrate from rustcore.py emitter), `problem.py`, `run.py` | F1,F3 | E1–E5 | all families lower; Jacobian lowering with a.e. abs/sign resolution; backend select `interp|jit`. |
| **E7** | PyO3 bindings (`tsdyn-core`) | `crates/tsdyn-core/**`, `tsdynamics/_rust` | E5,E6 | — | numpy zero-copy; GIL released; the documented Python engine API. |
| **E-SDE** | **SDE** engine + solvers (NEW family, **diagonal-Itô**, §11) | `tsdyn-solvers/sde/**`, Wiener/RNG in engine, `families/stochastic.py` | E5 | E-DDE | `_drift`+`_diffusion` (per-component); Euler–Maruyama + Milstein; converges to known SDE moments (OU process, geometric BM); seeded ensembles. |
| **E-DDE** | **DDE** engine (method of steps) | `tsdyn-engine/dde.rs` (history ring buffer + dense interp), `families/delay.py` | E5 | E-SDE | Mackey–Glass matches JiTCDDE within tol; constant + state-dependent delays. |
| **E-MAP** | Maps on the engine | `families/discrete.py` + `tsdyn-engine` map loop | E5 | — | all 26 maps iterate natively; matches v2 Numba within fp tol. |

### Tier 2 — Python core on the new engine — parallel after E6/E7 land

| ID | Stream | Owns | Depends | Acceptance |
|----|--------|------|---------|------------|
| **C-FAM** | Family base classes on Rust engine | `families/continuous.py, delay.py, discrete.py, stochastic.py`, protocol | E6,E7 | subclass contract UNCHANGED; 149 systems integrate via Rust; registry family detection incl. `stochastic`. |
| **C-SOLV** | Solver registry + selection + **auto-stiffness** | `tsdynamics/solvers/**` | F2,E3,E4 | `method=` resolves by name/caps; unknown → clear error; auto-detect picks implicit on stiff RHS; plugin solvers selectable. |
| **C-DATA** | `Trajectory` + state-space data | `tsdynamics/data/**` (migrate sampling, set distances, KD-tree) | F3 | feature-parity with v2 `Trajectory`/`sampling`; the lingua franca every analysis consumes. |
| **C-DERIV** | Derived systems on new engine | `tsdynamics/derived/**` | C-FAM,C-DATA | Poincaré/Stroboscopic/Tangent/Ensemble/Projected/Wrapped all green; `TangentSystem` is the one Lyapunov engine. |

### Tier 3 — Analysis & data extraction — **the wide parallel front** (one session each)

> These depend on Tier 2 contracts (`System`, `Trajectory`) but **not on each
> other**. This is where many sessions run at once.

| ID | Stream | Owns (`tsdynamics/…`) | Depends | Acceptance (literature-validated) |
|----|--------|------------------------|---------|-----------|
| **A-LYAP** | Lyapunov suite | `analysis/lyapunov/**` | C-DERIV | spectrum via TangentSystem; `max_lyapunov`; `lyapunov_from_data` (Kantz); Kaplan–Yorke. Lorenz D_KY≈2.06. |
| **A-CHAOS** | Chaos indicators | `analysis/chaos/**` | C-DERIV | GALI_k (Skokos), 0–1 test (Gottwald–Melbourne), expansion entropy (Hunt–Ott). |
| **A-FP** | Fixed points & periodic orbits | `analysis/fixedpoints/**`, `orbits/**` | C-DERIV | maps (have) + flows; Schmelcher–Diakonos / Davidchack–Lai; shooting for flows; `estimate_period`. |
| **A-ORBIT** | Orbit/bifurcation + sections | `analysis/orbits/**` | C-DERIV | `orbit_diagram` over map/Poincaré/Stroboscopic; `poincare_section`. Logistic + Rössler-via-Poincaré. |
| **A-BASIN** | **Attractors & basins (the moat)** | `analysis/basins/**` | C-DERIV, E5 | `find_attractors` (recurrence-FSM + featurize-cluster + proximity), `basins_of_attraction`, `basin_fractions`, basin entropy/uncertainty-exponent/Wada, global continuation + matching, tipping, resilience. Duffing & magnetic-pendulum basins; basin-fraction error <1% on ≥2 literature systems. *(Large — may sub-split into A-BASIN-find / -basins / -continuation.)* |
| **A-DIM** | Fractal dimensions | `analysis/dimensions/**` | C-DATA | correlation sum + GP (Theiler window), box-assisted, generalized/Rényi, fixed-mass, automated scaling-region fit. Lorenz corr-dim≈2.05. |
| **A-EMBED** | Delay embeddings | `analysis/embedding/**` | C-DATA | `embed`; delay (ACF/MI); dimension (Cao AFNN, Kennel FNN); multivariate. Reconstruct Rössler from x only. |
| **A-ENT** | Entropy & complexity | `analysis/entropy/**` | C-DATA | permutation/sample/dispersion entropy + multiscale; composable (outcome-space × estimator × measure); integrate **lzcomplexity** (our LZ76 lib) as the LZ provider. |
| **A-RQA** | Recurrence & RQA | `analysis/recurrence/**` | C-DATA | recurrence matrices (fixed ε / rate), RQA (DET, LAM, L_max, entropy, trapping), windowed; sparse + Rust kernel. |
| **A-SURR** | Surrogates | `analysis/surrogate/**` | C-DATA | shuffle/FT/AAFT/IAAFT + `SurrogateTest(stat, x, method) → p`. Rejects linearity for Lorenz. |
| **T-XFORM** | Transforms / feature extraction | `tsdynamics/transforms/**` | C-DATA | spectral (PSD, spectral entropy — migrate from lzcomplexity bridge), detrend/filter/normalize, generic feature extractors feeding analysis. |

### Tier 4 — Infrastructure & migration — continuous, parallel-safe

| ID | Stream | Owns | Depends | Acceptance |
|----|--------|------|---------|------------|
| **I-WHEEL** | Cross-platform wheels + packaging | maturin-action CI; decide one-wheel vs accelerator (§11) | E7 | manylinux + macOS(arm64/x86_64) + Windows wheels; `pip install tsdynamics` needs **no compiler**. |
| **I-BENCH** | Benchmarks + perf tracking | `benches/**`, CI perf job | E5 | vs SciPy + (internally) the Julia baseline on Lorenz/Rössler/MackeyGlass/Lorenz-96 N=128/Robertson; time-to-first-result tracked. |
| **I-XVAL** | Migration cross-validation + **removal** of v2 backends | the xval suite; delete JiTCODE/JiTCDDE/diffsol/Numba paths once gated | C-FAM, E-DDE, E-MAP | every system: Rust vs v2 within tol; Lyapunov vs literature; then old backends removed, C-compiler dep gone, `~/.cache/tsdynamics` retired. **Gated — runs last.** |
| **I-DOCS** | Docs restructure (NOT viz) | `docs/**`, autogen hook updated to new layout | F3 | site builds `--strict`; per-system pages; tutorial "equations → basins"; citation lint rule. |
| **I-QA** | Test/property/known-value harness | `tests/**` shared fixtures, hypothesis tests, known-value catalogue | F3 | registry-driven sweeps over new families; property tests for embeddings/dimensions. |

---

## 7. Milestones → the v3.0 gate (D7)

Internal milestones stage the program; **v3.0 ships only when M6 is complete**
(optionally preceded by `3.0.0bN` pre-releases off M4/M5).

- **M0 — Foundations frozen:** F0–F4. ✅ **DONE.** Workspace builds;
  traits/registry/IR frozen; package reorg in place; CI split + xval harness live.
  *Unblocks all.*
- **M1 — Engine parity (ODE):** E1,E3,E4,E5,E6,E7 + C-FAM(ODE) + C-SOLV. 🟡
  **engine ✅, Python wiring open.** E1–E7 merged; ODEs integrate on Rust via
  `backend="interp"` — but **C-FAM/C-SOLV not started**, so the v2 backends are
  still the default runtime. (See §13b.)
- **M2 — JIT + all families:** E2 (Cranelift), E-MAP, E-DDE, E-SDE, C-DERIV. 🟡
  **kernels ✅, reach open.** All merged, but the **SDE FFI and map-ensemble
  bindings are missing and the JIT is hard-rejected at the bridge** (E-WIRE), and
  C-DERIV is not started. JIT==interpreter holds in Rust but is untested through
  Python (I-XVAL).
- **M3 — Old engine retired:** I-XVAL green → remove JiTCODE/JiTCDDE/diffsol/
  Numba. **D1 complete.** (Big internal moment.) ⚠️ **Blocked on evidence:** the
  xval gate must validate `tsdynamics._rust` over the catalogue first (§13b), and
  the piecewise-map opcodes (E-OPS) must land or those maps die at deletion.
- **M4 — Chaos quantification:** A-LYAP, A-CHAOS, A-FP, A-ORBIT + C-DATA.
- **M5 — Parity moat:** A-BASIN, A-DIM, A-EMBED, A-ENT, A-RQA, A-SURR, T-XFORM.
- **M6 — Launch readiness:** I-WHEEL, I-BENCH, I-DOCS, I-QA all green; parity
  matrix (§8) all ✅ or consciously-deferred. **Cut v3.0.**

```
M0 ─┬─► M1 ─► M2 ─► M3 (engine done)
    └─► (Tier-3 analysis A-* can start as soon as C-DERIV/C-DATA from M1/M2 land)
M4 + M5 run largely in parallel across many sessions ─► M6 ─► v3.0
```

---

## 8. Parity matrix — reproduce-then-surpass

The benchmark ecosystem exposes ~550 functions around one idea: a single system
abstraction every analysis consumes, plus composable derived systems. We match
the **capabilities** with our own design; we do **not** copy code or names.

| Capability cluster | Our home | Stream | v3.0? |
|---|---|---|---|
| Unified system interface + derived systems | `families/`, `derived/` | C-FAM, C-DERIV | ✅ (exists, re-homed) |
| ODE/map integration, zero-warmup | `engine/`, `tsdyn-*` | E1–E7 | ✅ |
| **DDE** integration + Lyapunov | `families/delay`, engine | E-DDE | ✅ (**differentiator**) |
| **SDE** family | `families/stochastic` | E-SDE | ✅ (**competition is shallow here**) |
| Lyapunov spectra / max / from-data | `analysis/lyapunov` | A-LYAP | ✅ |
| GALI, 0–1 test, expansion entropy | `analysis/chaos` | A-CHAOS | ✅ |
| Orbit/bifurcation diagrams, Poincaré | `analysis/orbits` | A-ORBIT | ✅ |
| Fixed/periodic orbits | `analysis/fixedpoints` | A-FP | ✅ |
| **Attractors, basins, continuation, tipping, resilience** | `analysis/basins` | A-BASIN | ✅ (**the moat**) |
| Fractal dimensions | `analysis/dimensions` | A-DIM | ✅ |
| Delay embeddings | `analysis/embedding` | A-EMBED | ✅ |
| Entropy/complexity (+ LZ via lzcomplexity) | `analysis/entropy` | A-ENT | ✅ |
| Recurrence / RQA | `analysis/recurrence` | A-RQA | ✅ |
| Surrogates + tests | `analysis/surrogate` | A-SURR | ✅ |
| Visualization | `viz/` | — | ❌ deferred (D6) → 3.x |

**Our additions over the benchmark (build into the design now, fill later):**
first-class DDE+SDE in the *same* interface; zero session warmup (interpreter)
with native speed on demand (JIT); Python ecosystem gravity
(numpy/pandas/sklearn/ML interop free); the simplest definition contract; 149+
built-in systems; an **external plugin ecosystem** (D4) the benchmark lacks.

---

## 9. Migration plan (D1 — total replacement, no shims D3)

1. New engine lands **alongside** v2 backends behind the new `engine/run.py`.
2. **I-XVAL** continuously compares Rust vs v2 trajectories + Lyapunov across the
   whole catalogue; gate = all within tolerance, literature values reproduced.
3. On a green gate (M3), **delete** JiTCODE, JiTCDDE, Numba-map and diffsol code
   paths and dependencies; drop the C-compiler requirement and the
   `~/.cache/tsdynamics` compile cache. Update docs/CLAUDE.md.
4. `tsdynamics` becomes a **compiled (maturin) package** shipping wheels
   (I-WHEEL). Confirm the one-wheel-vs-accelerator packaging in §11 before M3.

---

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Interfaces churn → breaks parallel streams | M0 freezes them; changes only via `[interface]` PRs with a heads-up. |
| Cranelift JIT diverges numerically from interpreter | mandatory eval-equality test in E2; interpreter is the reference. |
| DDE in Rust is hard (history, dense interp) | E-DDE migrated last (M2); keep JiTCDDE until xval-green; vendor/borrow a vetted method-of-steps. |
| Removing v2 backends regresses a system | I-XVAL gate is mandatory before any deletion; deletion is its own reviewed PR. |
| Scope explosion (parity = many functions) | milestones gated by literature-validated acceptance; analysis lands only with a validated test. |
| Parallel sessions collide | worktrees + one-file-per-thing + append-only shared files + auto-discovery (§4e). |
| SDE contract is a permanent public API | **resolved: diagonal-Itô** (§11) — `_drift`+`_diffusion`, EM+Milstein; matrix/scalar/Stratonovich are later opt-in variants behind the same family, not a redesign. |

---

## 11. Open decisions (resolve via `[decision]` PR + maintainer note; don't block siblings)

- **SDE noise contract — ✅ DECIDED: diagonal-Itô.** Contract:
  `_drift(y, t, **params)` (the deterministic part, exactly like `_equations`) +
  `_diffusion(y, t, **params)` returning **one noise coefficient per state
  component**, each multiplying an independent Wiener increment. **Itô**
  interpretation. Solvers: **Euler–Maruyama** (order 0.5) and **Milstein**
  (order 1.0 — uses `∂g/∂u`, which the tape Jacobian already provides). This
  covers additive and multiplicative diagonal noise — the large majority of
  applied SDE models. Matrix (`dim×m`), scalar, and Stratonovich variants may be
  added later as opt-in modes **behind the same family**, so this is not a future
  redesign. E-SDE builds exactly this.
- **Cranelift JIT trigger** *(E2/E6)* — manual `backend="jit"` first; later an
  auto-heuristic (system size × run length × sweep count). Ship manual, then
  auto.
- **Packaging shape** *(I-WHEEL/M3)* — one `tsdynamics` maturin wheel vs keeping
  a separable accelerator. Total replacement (D1) points to one wheel; confirm
  before M3.
- **Top-level `__all__`** *(F3)* — finalize the curated public surface during the
  reorg; keep it small and obvious.
- **lzcomplexity integration** *(A-ENT/T-XFORM)* — vendor vs optional dependency
  for the LZ76/spectral providers.

---

## 12. Differentiators to protect (never regress)

1. **DDEs and SDEs as first-class citizens** in the one interface — integration
   *and* Lyapunov. The benchmark has no DDEs and shallow SDEs.
2. **Zero warmup** (interpreter) *plus* native speed on demand (Cranelift JIT) —
   vs minutes of session-start compilation elsewhere.
3. **Simplest system definition in any language** — `params` + `dim` +
   `_equations`/`_step`. Proven over 149 systems. Never make it harder.
4. **Python ecosystem gravity** — numpy/pandas/sklearn/ML one import away.
5. **Pluggable everything** (D4) — an external ecosystem the benchmark can't
   match from a monolith.

---

## 13. Consolidation status & continuation notes (review of 2026-06-14)

A full cross-subsystem review after the engine streams landed (M0 + the Tier-1
engine + C-DATA) found a **strong substrate with a mid-migration gap**: the hard
parts (IR, interpreter, JIT, RK + stiff + SDE kernels, DDE method-of-steps) are
built and tested, but a large surface of finished code is **not yet reachable
from Python**, the runtime still defaults to the v2 backends, and the
cross-validation evidence layer does not yet cover the shipping engine. A
consolidation PR (`chore/v3-consolidation`) landed the safe correctness and
hygiene fixes; the structural work is carved into the streams below so
independent sessions continue cleanly.

### 13a. Landed in the consolidation PR (done)

- **Stiff-solver safety (CRITICAL):** implicit kernels (`rosenbrock`/`trbdf2`) on
  the engine now **refuse a Jacobian-less tape** instead of silently degrading to
  an unstable forward-Euler step — guarded both at the kernel and at the
  `tsdyn-core` boundary (`require_jacobian_if_needed`), mirroring the DDE guard.
- **Named-component correctness:** `ProjectedSystem` names its *projected*
  columns (was mislabelling / IndexError-ing); `orbit_diagram` resolves names via
  the instance (was leaking the `variables` property descriptor → AttributeError
  over Poincaré/Stroboscopic wrappers).
- **Robustness guards:** backward (`t1 < t0`) ensemble requests and zero-dim
  tapes are rejected loudly instead of returning stale/NaN results.
- **Oracle honesty:** the Python reference `OP_POWI` now uses square-and-multiply
  (matching Rust `f64::powi`), not NumPy `pow`; the "mirrors reference.rs exactly"
  claim is softened (bit-exactness is I-XVAL's gate).
- **Hygiene:** dropped the unused `tsdyn-jit` dep from `tsdyn-engine` and
  `cranelift-native` from `tsdyn-jit` (cranelift no longer enters the wheel build
  tree; `tsdyn-core/Cargo.lock` regenerated); fixed the `gen_fixtures.py` pre-F3
  import + added a regenerator smoke test.

### 13b. Carved out for the named streams (extend their issues)

- **C-FAM** (issue #34) — the seam C-FAM was meant to pre-stage does not exist: only
  `DiscreteMap` routes through `engine.run.integrate`; DDE/SDE reimplement
  integration inline and ODE never calls the seam. Before flipping the default to
  Rust: add a **shared engine-dispatch seam** on `SystemBase` (a
  `_default_backend` ClassVar + a thin `_run`/`_dispatch` template), route all
  four families through `run.integrate` (add `_run_dde`/`_run_sde` branches),
  **hoist the four byte-identical `_make_t_eval` copies and the divergent
  `_provenance`** into one place (and thread `ic`/`t0` into engine-path
  provenance). Also add **SDE registry detection** (`'sde'` in `Family`,
  `'StochasticSystem'` in `_FAMILY_BASES`, broaden `_has_concrete_rhs` to accept
  `_drift`) + an SDE history/fixture hook analogous to `DDE_HISTORIES`.
- **C-SOLV** (issue #35) — register the **in-tree solver specs** (the `solvers/` directory
  ships zero today, so `method=` resolves against an empty table) and pair the
  engine's new Jacobian guard with a **Python-side auto-set of `with_jacobian`**
  when the resolved method is implicit, so the common stiff path "just works"
  instead of raising. Auto-stiffness selection sits on top.
- **I-XVAL** (issue #51) — the gate currently validates the **v2-seed `tsdynamics-core`** on
  Lorenz only, not `tsdynamics._rust`. Add a `RustEngine` backend to
  `xval_harness.py` (`run.integrate(backend="interp"/"jit")`), point
  `cross-validation.yml` at the compiled extension over the **registry
  catalogue**, add Python-level **interp==jit** and **reference==engine** parity
  tests (incl. bit-exact `OP_POWI`), and replace `engine-bindings.yml`'s
  hand-maintained file list with a marker/glob + a meta-test asserting every
  `importorskip("tsdynamics._rust")` module is covered (the skip-as-success
  blind-spot that required out-of-band patch #72).
- **C-DERIV / A-LYAP** (issues #37 / #38) — `TangentSystem` is billed as "the one Lyapunov engine"
  but the family `lyapunov_spectrum` methods triplicate the QR/`jitcode_lyap`
  loop, and its ODE mode is JiTCODE-only (deleted at M3) with no Rust variational
  replacement. Unify into one **backend-neutral variational core** so ODE
  Lyapunov survives M3.

### 13c. New streams to add to the board (open issues for these)

- **E-WIRE (issue #75) — reach the finished engine code from Python.** Add the SDE FFI
  (`integrate_sde_dense` / `_ensemble`) and the map-ensemble binding to
  `tsdyn-core`, and **wire the Cranelift JIT** (replace the `Unsupported` stub in
  `bridge.rs::guard_continuous` with a `JitEvaluator` dispatched through the
  existing `&dyn Evaluator` seam). Today: SDEs run pure-Python, `backend="jit"` is
  a dead option, and `map.rs::iterate_ensemble_final` has no binding. *(Depends:
  E5, E6, E7. Unblocks the SDE/JIT perks and parts of C-FAM/I-XVAL.)*
- **E-OPS (issue #76) — non-smooth / piecewise opcodes.** The IR has no
  comparison/select/min/max/floor/ceil/mod, so built-in modular/piecewise maps
  (Circle, Tent, Baker, Bernoulli) cannot lower and have **zero Rust path** — a
  hard blocker for the M3 Numba deletion. Add an additive opcode block in the
  **reserved wire-value range 50–69** (preserving the F1 freeze), document the
  a.e.-derivative conventions, and teach the map-lowering tracer to emit them.
  *(Depends: F1. `[interface]`-class change — coordinate; it extends, never
  renumbers, the frozen IR.)*
- **E-EVENT (issue #77) — event functions + dense output.** Poincaré-section root-finding,
  recurrence and event detection live entirely in Python today. Add an optional
  **dense-output / event channel** to the `Solver` trait and an event-expression
  hook to the IR/engine, so A-ORBIT / A-BASIN / A-RQA can drive events natively.
  *(Depends: F2, E3/E5. Additive to the frozen `Solver` trait — design as an
  optional capability, not a breaking signature change.)*
- **A-LAYOUT (issue #78) — analysis subpackage restructure (run before the A-\* fan-out).**
  `analysis/` is still the flat v2 quartet; the A-streams each own a *subpackage*.
  Move, with re-exports preserving the public API **and** updating the
  `docs/reference/*` mkdocstrings paths (so `mkdocs --strict` stays green):
  `lyapunov.py → lyapunov/`, `fixed_points.py → fixedpoints/` (note the rename),
  `orbit_diagram.py` **+** `poincare.py → orbits/`; create empty
  `chaos/ basins/ dimensions/ embedding/ entropy/ recurrence/ surrogate/`
  packages. Also wire `analyses`/`transforms` discovery+registration mirroring
  `solvers/` (two of four D4 plugin kinds have no consumer yet). *(Depends: F3.
  Mechanical but cross-cutting — land it as one focused PR so the A-* streams
  branch off the target layout conflict-free.)*

### 13d. Reserved contract space (additive — preserves the M0 freeze)

- **IR opcodes 50–69** are reserved for the E-OPS non-smooth/piecewise block
  (comparison, select, min/max, floor/ceil, rem/mod). Structural ops occupy
  0–3/10–15/20–21 and elementary functions 30–46; the 50–69 range extends the
  frozen `Op` enum without renumbering anything.
- The **`Solver` trait will gain an optional event/dense-output capability**
  (E-EVENT) behind a `Caps` flag, so existing kernels need no change.
- The **Jacobian is `dim×dim` (state-only)** today; parameter-sensitivity
  (`∂f/∂p`) and DDE variational dynamics will need a generalized jac block — call
  it out in any tape-shape work so it lands additively, not as a re-freeze.

---

*Keep this file current: when a stream flips state, edit its row in §6 (one
line). When a milestone completes, tick it in §7. The plan below the decisions
is the destination; §2 and the board are the position. §13 is the live
consolidation ledger — fold its items into §6 rows as their issues are filed.*
