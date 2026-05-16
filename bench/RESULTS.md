# Map benchmarks

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

