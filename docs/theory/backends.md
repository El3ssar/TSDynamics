---
description: The Rust engine — the Cranelift JIT, the SSA-tape interpreter, and the pure-Python reference oracle.
---

# Backends

Every family — ODE, DDE, SDE, and discrete maps — integrates on the **Rust
engine** (`tsdynamics._rust`). A system's symbolic `_equations` (or `_step`) is
lowered once to a flat list of SSA instructions — an *IR tape* — that the engine
evaluates with no Python callbacks and no C/LLVM toolchain. The `backend=`
argument of [`run`](../analysis/integration-and-methods.md) and
`lyapunov_spectrum` selects how that tape is evaluated:

| `backend` | how the tape runs | needs the wheel? |
|---|---|---|
| `"jit"` (default) | a pure-Rust **Cranelift JIT** (no LLVM) — the tape compiled to native code | yes |
| `"interp"` | the SSA-tape **interpreter** — no compile at all | yes |
| `"reference"` | a dependency-light **pure-Python** oracle (SciPy on the lowered tape) | no |

`"jit"` and `"interp"` use the same solver kernels and expression tape, and are
tested for bitwise agreement on a given platform. Cross-platform floating-point
details can differ. `"reference"` provides independent Python implementations
for ODEs, SDEs and maps; ODE integration uses SciPy. DDE integration requires the
compiled extension. Use the reference backend to cross-check numerical results,
accounting for solver and roundoff differences.

The compile is **once per distinct system**, not once per call: the engine
memoises the compiled evaluator on the tape's identity, so a repeat run, a
parameter sweep or an ensemble reuses it (`tsdynamics._engine.run.jit_cache_stats()`
reports hits/misses; `clear_jit_cache()` empties it, and
`TSDYNAMICS_NO_JIT_CACHE=1` disables it).

## Why `"jit"` is the default

The JIT compiles a tape once and reuses the compiled evaluator for subsequent
runs. This usually benefits repeated integrations and ensembles. Compilation
cost depends on equation size, so `backend="interp"` can be useful for a short
one-off run or a large system that has not been compiled yet. Consult the
[benchmarks](../references/benchmarks.md) for timings on stated workloads;
backend choice is not a universal speed guarantee.

The compiled-evaluator cache holds 64 entries. Cycling through more distinct
tapes can evict entries before reuse. If compilation dominates your workload,
compare `jit` and `interp` with fixed initial conditions and equivalent settings.

```python
import tsdynamics as ts

ts.systems.Lorenz().run(final_time=100, dt=0.01)                       # jit (default)
ts.systems.Lorenz().run(final_time=100, dt=0.01, backend="interp")     # SSA interpreter
ts.systems.Lorenz().run(final_time=100, dt=0.01, backend="reference")  # pure-Python oracle
```

## Choosing a solver

`run(solver=...)` selects the solver kernel; the name is resolved by the
solver registry (`tsdynamics._solvers`), which canonicalises spellings and
aliases:

| `solver` | kernel |
|---|---|
| `rk45` (default) / `dopri5` | adaptive Dormand–Prince 5(4) |
| `dop853` | adaptive Dormand–Prince 8(5,3) |
| `tsit5` | Tsitouras 5(4) |
| `rk4` | fixed-step classic RK4 |
| `bdf` | variable-order (1–5) BDF — the stiff workhorse |
| `rosenbrock`, `trbdf2` | fixed-order implicit (Rosenbrock-W, TR-BDF2) |

**Stiff systems often benefit from an implicit solver.** An explicit method can
require very small steps or exhaust its step budget. Catalogue systems with a
known stiff default, such as `Oregonator` and `KuramotoSivashinsky`, declare
`_default_method = "bdf"`. You can do the same for a custom model or choose
`solver="bdf"` per run. The engine derives the Jacobian required by implicit
kernels automatically. Check convergence for your parameter regime; a declared
default is a starting choice, not a stability guarantee.

The SciPy name `"LSODA"` is not an engine kernel; use a supported name such as
`"bdf"`.

## The expression-tape VM

The **expression tape** represents the equations numerically: the symbolic `_equations` are
lowered to a flat list of SSA instructions over a small register file, which the
interpreter (or the Cranelift-compiled native function) evaluates directly.
Control-parameter values are read live from the system on every run, so a
parameter sweep never re-lowers; only a *structural* parameter or a DDE *delay*
value — which changes the tape shape — triggers a re-lowering.

Because the tape carries no Python state, **ensemble integration** — thousands of
independent trajectories from a grid of initial conditions — is embarrassingly
parallel via [rayon](https://github.com/rayon-rs/rayon), and is checked
bit-identical to a serial reference. That is the primitive that makes
basin-of-attraction and Monte-Carlo sweeps tractable. A detected divergence raises for a single integration or produces a `NaN`
trajectory in an ensemble. Check those failures before computing ensemble
statistics.

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
