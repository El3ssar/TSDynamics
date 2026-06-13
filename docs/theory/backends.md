---
description: Choosing an integration backend — JiTCODE (C) vs the diffsol Rust/LLVM backend.
---

# Backends

ODE integration runs through one of two backends, selected per call with the
`backend=` argument of [`integrate`](../analysis/integrate.md).

| | `jitcode` (default) | `diffsol` | `auto` |
|---|---|---|---|
| Engine | symbolic → C → compiled `.so` | symbolic → DiffSL → LLVM JIT (Rust solvers) | picks `diffsol` if installed, else `jitcode` |
| Toolchain | needs a **C compiler** | **no compiler** — prebuilt wheels | — |
| Install | core | `pip install "tsdynamics[diffsol]"` | — |
| Cold start | seconds (C compile) | sub-second (LLVM JIT) | — |
| Families | ODE · DDE · maps | **ODE only** | ODE only |

```python
import tsdynamics as ts

ts.Lorenz().integrate(final_time=100, dt=0.01)                   # jitcode (default)
ts.Lorenz().integrate(final_time=100, dt=0.01, backend="diffsol")  # explicit Rust path
ts.Lorenz().integrate(final_time=100, dt=0.01, backend="auto")     # diffsol if installed
```

## The diffsol backend

`backend="diffsol"` translates the *same* symbolic `_equations` to
[DiffSL](https://martinjrobins.github.io/diffsol/) — a small DSL that the
Rust [diffsol](https://github.com/martinjrobins/diffsol) crate JIT-compiles
through LLVM — and integrates with Rust solver kernels. Initial conditions
and parameters are solve-time inputs, so one compiled module serves every IC
and every parameter value (the same caching economics as the JiTCODE path,
without the C-compiler dependency). SciPy-style method names map onto the
Rust solvers: `RK45`/`dopri5`/`DOP853` → `tsit45`; `LSODA`/`VODE` → `bdf`;
plus `tr_bdf2` and `esdirk34`.

**Coverage.** The translator handles **all 118 built-in ODE systems**
(guarded on every CI run by `tests/test_diffsol_translation.py`), and the
trajectories are cross-validated against JiTCODE — the curated sample on
every `slow` run, the full catalogue nightly.

**Speed** (Lorenz/Rössler/Chen, 1000 time units, `rtol=1e-9`, 100k output
points, warm caches; regenerate with `benches/bench_backends.py`):

| system | jitcode | diffsol | scipy | diffsol speedup |
|---|---|---|---|---|
| Lorenz | 1.20 s | **0.10 s** | 18.5 s | 11.9× |
| Rössler | 0.89 s | **0.03 s** | 5.6 s | 28.7× |
| Chen | 1.40 s | **0.15 s** | 28.8 s | 9.2× |

## Limitations

- **ODE only.** Delay systems and maps keep their own backends. (A Rust DDE
  solver and an SDE family are a future milestone — see the project roadmap.)
- The RHS must use functions DiffSL provides (`sin`/`cos`/`tan`/`exp`/`log`/
  `sqrt`/`abs`/`sign`/`tanh`/…); an unsupported construct raises a clear
  `DiffSLTranslationError` (none occur in the built-in catalogue).
- A couple of pathological built-ins integrate on the default JiTCODE backend
  but not on diffsol's solvers: **BlinkingRotlet** (near-discontinuous blinking
  protocol stalls every adaptive solver) and **WindmiReduced** (a
  `tanh(2200·…)` near-step plus fractional powers that go complex for negative
  arguments). They're excluded from the diffsol cross-validation sweep; use the
  default backend for them.
- Experimental: `jitcode` remains the default. Once the nightly full-catalogue
  cross-validation has a green track record, `auto` becomes the default so a
  plain `pip install "tsdynamics[diffsol]"` gives a zero-compiler install.

## See also

- [Compilation pipeline](compilation.md) — the JiTCODE path and the shared symbolic core
- API: `tsdynamics.backends.diffsol`
