---
description: The Rust engine — the Cranelift JIT, the SSA-tape interpreter, and the pure-Python reference oracle.
---

# Backends

Every family — ODE, DDE, SDE, and discrete maps — integrates on the **Rust
engine** (`tsdynamics._rust`). A system's symbolic `_equations` (or `_step`) is
lowered once to a flat list of SSA instructions — an *IR tape* — that the engine
evaluates with no Python callbacks and no C/LLVM toolchain. The `backend=`
argument of [`integrate`](../analysis/integration-and-methods.md) (and `iterate`
/ `lyapunov_spectrum`) picks *how* that tape is evaluated:

| `backend` | how the tape runs | needs the wheel? |
|---|---|---|
| `"jit"` (default) | a pure-Rust **Cranelift JIT** (no LLVM) — the tape compiled to native code | yes |
| `"interp"` | the SSA-tape **interpreter** — no compile at all | yes |
| `"reference"` | a dependency-light **pure-Python** oracle (SciPy on the lowered tape) | no |

`"jit"` and `"interp"` are numerically **identical by contract** (bit-for-bit) —
they drive the same solver kernels through the same `Evaluator` trait, so
switching between them can only change speed, never answers. `"reference"` is the
validation oracle: it reproduces the engine without the compiled extension, so it
runs anywhere (ODE and maps only; it raises for DDE/SDE, which have no
pure-Python integrator). **It is not meant for production runs** — it is the
independent cross-check the engine is validated against, and the wheel-free
fallback.

The compile is **once per distinct system**, not once per call: the engine
memoises the compiled evaluator on the tape's identity, so a repeat run, a
parameter sweep or an ensemble reuses it (`tsdynamics.engine.run.jit_cache_stats()`
reports hits/misses; `clear_jit_cache()` empties it, and
`TSDYNAMICS_NO_JIT_CACHE=1` disables it).

## Why `"jit"` is the default

Until v6 the default was `"interp"`, for a good reason at the time: the JIT
recompiled the whole tape on **every** call, which made it slower than the
interpreter for short runs. The v6 compiled-evaluator cache removed that per-call
compile, so the default was re-measured across all 136 catalogue ODE systems as
the catalogue stood then — it is 142 today, and the conclusion is unchanged
(cold caches per system):

- **First-call cost** of `"jit"` over `"interp"`: median **+0.65 ms**, p90
  +4.57 ms. The one large outlier is `GrayScott` (+639 ms), a 4608-dimensional
  method-of-lines field whose tape is enormous — and which then wins that back
  many times over on any real run.
- **Steady-state throughput**: median **1.55×** faster, p10 1.13×. The sweep
  flagged three tiny flows (`PehlivanWei`, `SprottB`, `SprottF`) as *slower* on
  the JIT, but a controlled re-check — same pinned initial condition, interleaved
  A/B, min-of-25 — puts all three at **1.19–1.55× faster**. Those systems declare
  no `default_ic`, so each timed call drew a *different* random start and did a
  different amount of adaptive work; the apparent slowdown was the initial
  condition, not the backend. No system is known to be genuinely slower on the
  JIT.

So: faster across the catalogue, for a one-off sub-millisecond median cost per
distinct system. If you want to avoid the compile entirely — a very short run, a
one-shot script, or profiling the interpreter itself — ask for
`backend="interp"`.

!!! note "Many distinct systems in one process"
    The compiled-evaluator cache holds 64 entries. A session that cycles
    round-robin through *more* than 64 distinct systems is the LRU worst case:
    every entry is evicted before it is reused, so each visit re-compiles. Memory
    stays bounded (RSS plateaus once the cache fills), and each miss costs the
    sub-millisecond median above, but if you are sweeping hundreds of systems in
    one process and the compile shows up in a profile, `backend="interp"` avoids
    it entirely.

```python
import tsdynamics as ts

ts.systems.Lorenz().run(final_time=100, dt=0.01)                       # jit (default)
ts.systems.Lorenz().run(final_time=100, dt=0.01, backend="interp")     # SSA interpreter
ts.systems.Lorenz().run(final_time=100, dt=0.01, backend="reference")  # pure-Python oracle
```

## Choosing a solver

`integrate(method=...)` selects the solver kernel; the name is resolved by the
solver registry (`tsdynamics.solvers`), which canonicalises spellings and
aliases:

| `method` | kernel |
|---|---|
| `rk45` (default) / `dopri5` | adaptive Dormand–Prince 5(4) |
| `dop853` | adaptive Dormand–Prince 8(5,3) |
| `tsit5` | Tsitouras 5(4) |
| `rk4` | fixed-step classic RK4 |
| `bdf` | variable-order (1–5) BDF — the stiff workhorse |
| `rosenbrock`, `trbdf2` | fixed-order implicit (Rosenbrock-W, TR-BDF2) |

**Stiff systems need an implicit solver.** An explicit method (the default
`rk45`) can fail outright on a stiff right-hand side. Systems known to need an
implicit solver declare it themselves — e.g. `Oregonator`, `KuramotoSivashinsky`
and several Sprott jerk flows set `_default_method = "bdf"` — so
`sys.run()` "just works" without the caller having to know. When you
define a stiff system of your own, set `_default_method = "bdf"` on the class.
The implicit kernels need the system's analytic Jacobian on the tape;
`run.integrate` builds it automatically when the resolved method is implicit, so
there is nothing extra to pass. (The legacy SciPy name `"LSODA"` is not an engine
kernel — use `"bdf"`.)

## The expression-tape VM

The engine's keystone is the **expression tape**: the symbolic `_equations` are
lowered to a flat list of SSA instructions over a small register file, which the
interpreter (or the Cranelift-compiled native function) evaluates directly.
Control-parameter values are read live from the system on every run, so a
parameter sweep never re-lowers; only a *structural* parameter or a DDE *delay*
value — which changes the tape shape — triggers a re-lowering.

Because the tape carries no Python state, **ensemble integration** — thousands of
independent trajectories from a grid of initial conditions — is embarrassingly
parallel via [rayon](https://github.com/rayon-rs/rayon), and is checked
bit-identical to a serial reference. That is the primitive that makes
basin-of-attraction and Monte-Carlo sweeps tractable. A diverging trajectory
raises (single integration) or becomes a `NaN` row (ensemble — so an escaped
initial condition is flagged, not faked).

The implicit kernels lower the system's symbolic Jacobian into the *same* tape
(abs/sign derivatives resolved a.e.), so they need no finite differences.

## Architecture

The engine is a small Cargo workspace of single-concern crates:

| crate | concern |
|---|---|
| `tsdyn-ir` | the instruction-tape contract |
| `tsdyn-vm` | the interpreter `Evaluator` |
| `tsdyn-jit` | the Cranelift `Evaluator` (same trait, no LLVM) |
| `tsdyn-solvers` | one solver kernel per module (explicit, implicit, SDE) |
| `tsdyn-engine` | the integrate loop, ensembles, seeded RNG |
| `tsdyn-core` | the PyO3 bindings → `tsdynamics._rust` |

## See also

- [Compilation pipeline](compilation.md) — symbolic `_equations` → IR tape
- [Packaging](packaging.md) — how the engine ships (one maturin wheel)
