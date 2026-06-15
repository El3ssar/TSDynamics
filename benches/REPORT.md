# Rust engine vs v2 backends — performance decision gate (I-BENCH)

**Question this report answers.** The maintainer is about to retire the v2
backends (JiTCODE / JiTCDDE / Numba / diffsol) in favour of the Rust engine at
milestone **M3** (stream I-XVAL flips `SystemBase._default_backend`). I-XVAL
proves *correctness* (Rust == v2); it says nothing about *speed*. This report is
the missing speed evidence: a head-to-head of the **shipping** Rust engine
(`tsdynamics._rust`, reached through `integrate(backend="interp")` /
`backend="jit"` and `iterate(backend=...)`) against the v2 defaults and a SciPy
baseline, across three regimes with opposite winners, and a go/no-go
recommendation.

> **Bottom line: GO.** The Rust engine is fast enough to be the default for
> **every** family — ODE (incl. stiff), DDE, map and SDE. **Cold
> time-to-first-result** (no compiler, flat ~0.3 s) and **ensemble/sweep
> throughput** (15–4000×) are decisive engine wins everywhere. **Warm
> steady-state** is now also an engine win essentially across the board: small
> ODEs 30–40×, **stiff ODEs ~2× (the variable-order BDF kernel turned the old
> 3–6× loss into a win)**, DDE ~2×, SDE ~400×, and on the 128-dim adversarial
> case Cranelift `jit` **beats** gcc-`O3` JiTCODE. The single remaining sub-case
> where the engine trails is *single-orbit iteration of the cheapest 2-D maps*
> (Hénon ~4× vs Numba), which is ~5 ms in absolute terms and reverses in map
> ensembles. There is no longer a conditional blocker for the flip.

This is a *measurement-only* report (I-BENCH owns `benches/**` + the CI perf
job). It does **not** flip the default or touch any backend.

> **Update history.** This is the **second** edition. The first (PR #94) found
> two warm losses and read **GO-WITH-CONDITIONS**; both conditions have since
> landed and are reflected here: **E-BDF (#98)** added the variable-order BDF
> stiff kernel (flips stiff from a loss to a win), and **E-JITOPT (#97)** wired
> maps onto the Cranelift JIT (they previously fell back to the interpreter, so
> the old report's map `jit` column was really `interp`) and set the JIT
> opt-level. The verdict is now an unconditional **GO**. See §7 for the
> before/after.

---

## 1. Machine & methodology

| | |
|---|---|
| CPU | Intel Core i7-10870H (8 cores / 16 threads, 2.20–5.00 GHz) |
| OS / kernel | Arch Linux, `7.0.12-zen1-1-zen` x86_64 (glibc 2.43) |
| Python | 3.14.2 · NumPy 2.4.4 · SciPy 1.17.1 |
| tsdynamics | engine `tsdynamics._rust` 0.1.0, built from `main` after E-JITOPT (#97) + E-BDF (#98) |
| engine solvers | `rk4 rk45 tsit5 dop853 rosenbrock trbdf2 **bdf**` |
| engine build | `cargo build --release --features extension-module` (Cranelift JIT, opt-level set per E-JITOPT) |

**Harness:** [`benches/bench_engine.py`](bench_engine.py) (run with
`uv run python benches/bench_engine.py`). Each cell is the **best of N** wall
times (N = 5 warm, 3 ensemble, 2 cold) — best-of, not mean, to suppress
scheduler noise. **Tolerances are pinned equal across every backend** for a given
system; maps and SDEs use identical step and seed. `interp` and `jit` are
reported **separately** — they are different products. Stiff systems use the
engine's `bdf` kernel (the auto-stiffness default) vs JiTCODE's LSODA. Numbers
are machine-dependent; regenerate locally.

**Backends.** `interp` = zero-warmup SSA-tape interpreter (`tsdyn-vm`); `jit` =
Cranelift native-code evaluator (`tsdyn-jit`); `jitcode`/`jitcdde`/`numba` = the
v2 defaults; `reference` = the pure-Python tape oracle (the *current* SDE
default); `scipy` = `solve_ivp` over the numeric RHS (baseline).

**Correctness anchor (load-bearing for the verdict):** `interp` and `jit` are
**bit-for-bit identical** (`max|interp − jit| = 0.0`) on ODE, **map** (now that
maps actually JIT, E-JITOPT) and DDE trajectories, including the `bdf` stiff
path — so "use `jit` for the hot path" never changes a result. Full Rust-vs-v2
correctness is I-XVAL's gate, separate from speed.

---

## 2. Regime (a) — cold time-to-first-result (the first-class metric)

Fresh process, the **first** integration, including all compile/JIT/warmup.
Times are the integration call only (seconds); a **shared `import tsdynamics`
floor of ~1.0 s** sits under every backend and is not in the table.
`jitcode`/`jitcdde` are measured **truly cold** (fresh compile cache → the C
compiler runs); `jitcode (warm)` reloads a cached `.so`; `numba` recompiles per
fresh process.

| System (family) | **interp** | **jit** | jitcode (cold) | jitcode (warm) | jitcdde | numba | reference | scipy |
|---|---|---|---|---|---|---|---|---|
| Lorenz (ODE)            | **0.278** | **0.311** | 0.890 | 0.257 | — | — | 4.255 | 2.777 |
| Rössler (ODE)           | **0.307** | **0.273** | 0.852 | 0.292 | — | — | 2.677 | 1.870 |
| Oregonator (stiff ODE)  | **0.287** | **0.295** | 0.816 | 0.296 | — | — | 0.679 | 0.268 |
| Lorenz-96 N=128 (ODE)   | **0.443** | **0.410** | 0.553 | 0.157 | — | — | — | — |
| Mackey–Glass (DDE)      | **0.337** | **0.279** | — | — | 1.244 | — | — | — |
| Hénon (map)             | **0.249** | **0.219** | — | — | — | 0.423 | — | — |
| OU (SDE)                | **0.336** | **0.270** | — | — | — | — | 1.721 | — |

**Reading it (unchanged from the first edition — BDF/JIT changes are warm-only).**
The engine has **no external compiler in the loop**: `interp`/`jit` cold is a flat
~0.22–0.44 s set by Python-side tape lowering, not a compile, and `jit`'s Cranelift
penalty over `interp` is negligible. vs the v2 compile backends the engine wins
cold across the board — ODE **~2–3×** faster than a cold `jitcode` compile, **DDE
~4×** faster than `jitcdde` (1.24 s → 0.28 s), **map ~1.7×** faster than Numba's
first-call JIT. Honest caveat: a *warm-cached* JiTCODE `.so` (0.16–0.30 s) is
comparable to the engine for a fixed system+dim; the engine's structural win is
that **every** machine/CI run gets the cold number with no compiler and no
`~/.cache/tsdynamics` to manage. **Cold verdict: clear GO.**

---

## 3. Regime (b) — warm steady-state throughput

Warmup excluded; one long hot run, best-of-5. Lower is better. This is the regime
that previously held the only losses — both now resolved.

| System (family) | **interp** | **jit** | jitcode | jitcdde | numba | reference | scipy |
|---|---|---|---|---|---|---|---|
| Lorenz (ODE)            | 12.6 ms | **6.5 ms** | 206.4 ms | — | — | 3.86 s | 2.73 s |
| Rössler (ODE)           | 10.4 ms | **6.3 ms** | 238.5 ms | — | — | 2.36 s | 1.79 s |
| Oregonator (stiff, `bdf`)      | 9.5 ms | **6.9 ms** | 18.1 ms | — | — | 387 ms | 205 ms |
| ForcedVanDerPol (stiff, `bdf`) | 60.0 ms | **47.4 ms** | 84.7 ms | — | — | 516 ms | 381 ms |
| Lorenz-96 N=128 (ODE)   | 134.1 ms | **75.2 ms** | 86.5 ms | — | — | — | — |
| Mackey–Glass (DDE)      | **1.3 ms** | 1.9 ms | — | 2.6 ms | — | — | — |
| Hénon (map)             | 8.5 ms | 5.2 ms | — | — | **1.3 ms** | — | — |
| Logistic (map)          | 11.2 ms | **4.0 ms** | — | — | 24.8 ms | — | — |
| OU (SDE)                | 4.3 ms | **3.4 ms** | — | — | — | 1.55 s | — |
| GBM (SDE)               | 4.5 ms | **3.5 ms** | — | — | — | 1.84 s | — |

**Where the engine wins warm.**

- **Small/medium chaotic ODEs: 30–40×.** Lorenz `jit` 6.5 ms vs `jitcode` 206 ms
  (**32×**). The engine produces the whole dense-sampled trajectory in one
  GIL-released FFI call; JiTCODE pays Python overhead *per output sample*. (For
  final-state-only / coarse output the gap narrows.)
- **Stiff ODEs — now a WIN (was the headline loss).** With the variable-order
  `bdf` kernel (E-BDF), `jit` is **2.6×** faster than `jitcode`+LSODA on
  Oregonator (6.9 ms vs 18.1 ms) and **1.8×** on ForcedVanDerPol (47.4 ms vs
  84.7 ms). The old fixed-order Rosenbrock-W path was 3–6× *slower*; BDF closes
  and reverses it. The stiff regression that gated the migration **is gone**.
- **Large compute-bound ODE (Lorenz-96 N=128) — `jit` now BEATS gcc.** `jit`
  75 ms vs `jitcode` 86 ms. The interpreter trails (134 ms, 1.8× `jit`) — the
  tape interpreter's per-op dispatch is its worst case on a 128-dim RHS — so
  large/hot systems should route to `jit` (which is now the fastest backend
  measured on this case).
- **DDE ~2× JiTCDDE; SDE ~400×** the pure-Python reference (the *current* SDE
  default — the single biggest warm win).
- **Maps: `jit` is real now (E-JITOPT).** Logistic `jit` 4.0 ms is **6.2×
  faster** than Numba (24.8 ms). (Before E-JITOPT, map `jit` fell back to the
  interpreter, so this column did not exist.)

**The one remaining warm trail.**

- **Cheapest 2-D map (Hénon): Numba ~4×.** `jit` 5.2 ms vs Numba 1.3 ms for
  200 k iterations. A Hénon step is a few flops; Numba's fully-unrolled loop wins
  on the very cheapest bodies. It is **~5 ms in absolute terms**, it *improved*
  ~2× from the pre-E-JITOPT engine, the heavier Logistic map already beats Numba
  6×, and **map ensembles reverse it entirely** (§4). Not a blocker.

**Warm verdict: GO across every family.**

---

## 4. Regime (c) — ensemble / sweep throughput

Many trajectories, best-of-3, n = 512. The engine fans out over rayon; the v2
ensemble is a Python loop over the compiled stepper; `scipy-loop` is the floor.

| Workload | **interp** | **jit** | v2 loop | scipy loop | jit speedup vs v2 |
|---|---|---|---|---|---|
| Lorenz (n=512)            | 11.5 ms | **6.3 ms** | 2.676 s (jitcode-loop) | 19.60 s | **425×** |
| Lorenz-96 N=128 (n=512)   | 804.9 ms | **379.2 ms** | 5.644 s (jitcode-loop) | 40.85 s | **14.9×** |
| OU-SDE (n=512)            | 18.8 ms | **13.9 ms** | 58.40 s (reference)    | —      | **4200×** |
| Hénon (n=512, ×2000 it)¹  | 3.1 ms | **1.5 ms** | 18.6 ms (numba-loop) | —    | **13×** |

¹ Map ensemble, measured separately: even the cheapest map — where Numba wins a
*single* orbit — loses to the engine's rayon fan-out by ~13× in a batch, which is
how maps are actually used (basins, bifurcation sweeps, Lyapunov ensembles).

**Ensemble verdict: emphatic GO** — 1–3 orders of magnitude, every family.

---

## 5. Scorecard

| Family | Cold (TTFR) | Warm | Ensemble | Net |
|---|---|---|---|---|
| ODE small/med (Lorenz, Rössler) | ✅ ~2.5× | ✅ 30–40× | ✅ 425× | **GO** |
| ODE large (Lorenz-96 N=128)     | ✅ ~1.3× | ✅ `jit` beats gcc (interp 1.8× slower → use `jit`) | ✅ 15× | **GO** |
| ODE **stiff** (Oregonator, VdP) | ✅ ~2× | ✅ **~2× (was 3–6× slower; fixed by `bdf`)** | ✅ | **GO** |
| DDE (Mackey–Glass)              | ✅ ~4× | ✅ ~2× | n/a² | **GO** |
| Map (Hénon/Logistic)            | ✅ ~1.7× | ⚠️ Numba ~4× on the cheapest single orbit; engine 6× on Logistic | ✅ 13× | **GO** |
| SDE (OU, GBM)                   | ✅ ~5× | ✅ ~400× | ✅ 4200× | **GO** |

² DDEs integrate one trajectory at a time; sweeps loop in Python over the
per-trajectory engine call, still far ahead of recompiling JiTCDDE per value.

---

## 6. Remaining levers (none are blockers)

1. **Hénon-class cheap maps vs Numba (~4×, single orbit).** The engine map loop
   has per-step dispatch Numba unrolls away. A `jit`-specialised tight map loop
   would erase even the single-orbit gap; in practice maps run in ensembles where
   the engine already wins 13×, so this is polish, not a blocker.
2. **Interpreter on large RHS.** `interp` is 1.8× `jit` on Lorenz-96 N=128 — route
   large/hot systems to `jit` via the auto-promotion heuristic (ROADMAP §11);
   `jit` is already the fastest backend on that case.
3. **Cranelift headroom.** `opt_level` is now set (E-JITOPT) and measured roughly
   neutral on straight-line tape SSA; SIMD / tape auto-vectorisation remains a
   larger later lever for compute-bound ODEs.

---

## 7. What changed since the first edition (the conditions landed)

| Case | 1st report (PR #94) | Now (after #97 + #98) |
|---|---|---|
| Oregonator (stiff) warm | `jit` 87 ms — **3.9× slower** than LSODA | `jit`-`bdf` **6.9 ms — 2.6× faster** |
| ForcedVanDerPol (stiff) warm | `jit` 313 ms — **2.7× slower** | `jit`-`bdf` **47 ms — 1.8× faster** |
| Lorenz-96 N=128 warm | `jit` 102 ms — *ties* gcc (99 ms) | `jit` **75 ms — beats** gcc (86 ms) |
| Logistic (map) warm | `jit` = `interp` (no real JIT) ~14 ms | `jit` **4.0 ms — 6.2× faster** than Numba |
| Hénon (map) warm | `jit` 11 ms (was interp), 5× vs Numba | `jit` **5.2 ms**, ~4× vs Numba (improved 2×) |
| **Verdict** | GO-WITH-CONDITIONS (stiff BDF; jit-for-big) | **GO** — both conditions met |

**E-BDF (#98)** added the variable-order (1–5) BDF kernel, the auto-stiffness
default — it takes large steps through stiff transients like LSODA does, turning
the engine's worst regime into a win. **E-JITOPT (#97)** found maps never reached
the Cranelift JIT (the bridge hard-coded the interpreter) and wired them through,
plus set the JIT opt-level; the first report's map `jit` column was therefore
really the interpreter.

---

## 8. Recommendation — GO

**The Rust engine is fast enough to retire the v2 backends — yes, now
unconditionally on speed.** It wins **cold time-to-first-result** (flat ~0.3 s, no
compiler) and **ensemble throughput** (15–4000×) outright, and after E-BDF +
E-JITOPT it also wins **warm** on every family: small/large ODE, **stiff** (the
former blocker, now ~2× ahead via `bdf`), DDE, and SDE. Cranelift `jit` now
*beats* gcc-`O3` JiTCODE on the 128-dim adversarial case. Flip
`_default_backend` to the engine, defaulting to `interp` (zero warmup) and `jit`
for large/hot work.

The only residual is single-orbit iteration of the very cheapest maps (Hénon ~4×
vs Numba, ~5 ms absolute), which is polish and reverses in ensembles. The one
remaining gate is **correctness, not speed**: the flip still waits on I-XVAL's
Rust-vs-v2 numerical gate over the catalogue (the engine's own `interp == jit` is
already bit-for-bit). On performance grounds, M3 is clear.

---

## 9. Reproduce

```bash
# 1. Build the engine extension (NOT built by default)
cargo build --release --features extension-module \
    --manifest-path crates/tsdyn-core/Cargo.toml --locked
cp crates/tsdyn-core/target/release/lib_rust.so src/tsdynamics/_rust.abi3.so

# 2. Run the full three-regime harness (writes Markdown + JSON)
uv run python benches/bench_engine.py --out benches/results.json

# CI-sized smoke across all regimes/families:
uv run python benches/bench_engine.py --quick --regime all
```

The CI **Perf (I-BENCH)** workflow (`.github/workflows/perf.yml`) builds the
engine and runs `--quick --regime all` on every change to `benches/**` or the
engine/family paths, asserts the engine path runs, prints the time-to-first-result
table to the run summary, and uploads the raw JSON — a guardrail against
engine-path regressions, not a noisy wall-clock gate.
