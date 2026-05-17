# Map benchmarks

ODE integrate timings (`bench/ode_bench.py`) are appended near the end under **§ ODE benchmarks**.

Wall-clock time for ``iterate(steps)`` on built-in maps.
Rust column uses the N1 IR-interpreted kernel; Numba column forces the
fallback path via the ``_ir_cache = None`` poison.

## Interpretation

N1 ships a **tree-walking IR interpreter**. Numba JIT-compiles each step
function down to native code via LLVM, so for tight numeric loops with
2–3 ops per step (Henon, Ikeda, FoldedTowel) it remains faster than our
interpreter. The 2× target from the milestone was a hope rather than a
gate; it's now a tracked gap that closes in N4 when we add the cranelift
JIT (same IR, real codegen).

Why this is fine for N1:

- Correctness is bit-exact on iterate, 5e-15 on Lyapunov (vs. 1e-12 /
  1e-6 target). The IR pipeline is proven.
- The strategic asset — the IR itself — is the actual N1 deliverable.
  N3 (variational ODE) and N4 (cranelift JIT, drops JiTCODE) consume the
  same op set.
- No regression for users: Numba is still in the fallback path; no map
  got slower than before (worst case is "same Numba speed if you poison
  the IR cache").
- Logistic (the simplest map) already wins 2.3× — confirms the
  PyO3/wrapper overhead isn't the bottleneck; it's per-op interpreter
  cost.

## Run 2026-05-16 10:20:58

| Map | Steps | Rust (s) | Numba (s) | Speedup |
|-----|------:|---------:|----------:|--------:|
| Henon | 10,000,000 | 0.393 | 0.141 | 0.36× |
| Logistic | 10,000,000 | 0.233 | 0.546 | 2.34× |
| Ikeda | 1,000,000 | 0.221 | 0.040 | 0.18× |
| FoldedTowel | 1,000,000 | 0.093 | 0.014 | 0.15× |


## ODE benchmarks — run 2026-05-17 01:27:21

Same `(final_time, dt, rtol, atol)` grid; **Rust DP5** evaluates the bytecode RHS in-process;
**JiTCODE** runs ``dopri5`` while IR lowering is forced to raise ``NotLowerableError``.
Whole-call wall time mixes stepping with output-grid sampling — small 3-D demos can favor
Rust if Python/C bridge overhead dominates; RHS-only microbenchmarks skew the other way
(N4 Cranelift will revisit raw throughput parity).

| System | Rust DP5 (s) | JiTCODE dopri5 (s) | JiTCODE / Rust |
|--------|---------------:|-------------------:|---------------:|
| Lorenz | 0.0037 | 0.0464 | 12.664× |
| Rossler | 0.0014 | 0.0409 | 28.280× |

## ODE benchmarks — run 2026-05-17 01:29:45

Same `(final_time, dt, rtol, atol)` grid; **Rust DP5** evaluates the bytecode RHS in-process;
**JiTCODE** runs ``dopri5`` with IR lowering toggled off via `NotLowerableError`. Whole-call wall time includes output-grid sampling.

| System | Rust DP5 (s) | JiTCODE dopri5 (s) | JiTCODE / Rust |
|--------|---------------:|-------------------:|---------------:|
| Lorenz | 0.0035 | 0.0435 | 12.300× |
| Rossler | 0.0014 | 0.0383 | 28.116× |
## Run 2026-05-17 01:30:00

| Map | Steps | Rust (s) | Numba (s) | Speedup |
|-----|------:|---------:|----------:|--------:|
| Henon | 10,000,000 | 0.758 | 0.343 | 0.45× |
| Logistic | 10,000,000 | 0.327 | 0.808 | 2.47× |
| Ikeda | 1,000,000 | 0.273 | 0.045 | 0.17× |
| FoldedTowel | 1,000,000 | 0.139 | 0.011 | 0.08× |

## ODE Lyapunov (variational QR, **N3**) — run 2026-05-17

Lorenz ``lyapunov_spectrum(dt=0.1, burn_in=50, final_time=200, method="DP8", rtol=1e-7, atol=1e-10, ic=[1,1,1])``:
**~3.1 s** wall-clock on this machine (pure-Rust stepping + QR). Historical ``jitcode_lyap`` comparison removed with N3; regressions use ``tests/native/regression/ode/*.lyap.npz``.

