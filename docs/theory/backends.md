---
description: The Rust engine — the SSA-tape interpreter, the Cranelift JIT, and the pure-Python reference oracle.
---

# Backends

Every family — ODE, DDE, SDE, and discrete maps — integrates on the **Rust
engine** (`tsdynamics._rust`). A system's symbolic `_equations` (or `_step`) is
lowered once to a flat list of SSA instructions — an *IR tape* — that the engine
evaluates with no Python callbacks, no runtime compiler, and **no warmup**. The
`backend=` argument of [`integrate`](../analysis/integrate.md) (and `iterate` /
`lyapunov_spectrum`) picks *how* that tape is evaluated:

| `backend` | how the tape runs | needs the wheel? |
|---|---|---|
| `"interp"` (default) | the SSA-tape **interpreter** — zero warmup | yes |
| `"jit"` | a pure-Rust **Cranelift JIT** (no LLVM) — native code | yes |
| `"reference"` | a dependency-light **pure-Python** oracle (SciPy on the lowered tape) | no |

`"interp"` and `"jit"` are numerically **identical by contract** (bit-for-bit);
`"jit"` trades a small one-time compile for faster steady-state throughput on
long or large runs. `"reference"` is the validation oracle — it reproduces the
engine without the compiled extension, so it runs anywhere (ODE and maps only;
it raises for DDE/SDE, which have no pure-Python integrator).

```python
import tsdynamics as ts

ts.Lorenz().integrate(final_time=100, dt=0.01)                      # interp (default)
ts.Lorenz().integrate(final_time=100, dt=0.01, backend="jit")      # Cranelift JIT
ts.Lorenz().integrate(final_time=100, dt=0.01, backend="reference")  # pure-Python oracle
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
`sys.integrate()` "just works" without the caller having to know. When you
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
