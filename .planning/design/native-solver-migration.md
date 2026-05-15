# Native solver migration (Track E)

The strategic track. The goal is to replace JiTCODE, JiTCDDE, and Numba with a
pure-Rust compute layer under our own control. The migration is **phased**, runs
**side-by-side internally** until parity, and is **invisible to users** at every
step. The public API never grows a backend flag.

## Why

- DynamicalSystems.jl's real edge is DifferentialEquations.jl — native, tuned
  solvers under their control. To supersede them we need solvers under our control.
- JiTCODE / JiTCDDE require a C compiler on the user's machine at runtime. That's
  a real adoption friction, especially on Windows.
- Both are single-maintainer projects. Bus factor is unacceptable for a reference
  library.
- Three compile pipelines (JiTCODE C, JiTCDDE C, Numba) means three sets of failure
  modes. One Rust pipeline means one.
- FFI overhead from Rust analysis kernels into Python-stepped RHS is a permanent
  ceiling. All-Rust eliminates it.

## Why phased and not big-bang

A production-grade adaptive ODE/DDE suite — DP5, DP8, V9, Rosenbrock, stiffness
switching, dense output, event detection, variational equations, continuous
extension for DDE, breakpoint propagation — is **months** of careful work. JiTCODE
and JiTCDDE bundle most of this maturely. We migrate piece by piece, never
shipping a half-working solver as the only option.

## Phasing

### N1 — Rust map stepper

Easiest target. Maps don't need adaptive stepping; they're just tight loops over
`X_{n+1} = f(X_n)`.

- SymEngine RHS → small bytecode-or-cranelift IR → Rust closure.
- `DiscreteMap.iterate` loop moves to Rust.
- QR-based Lyapunov loop moves to Rust (uses `nalgebra` for QR).
- Numba is removed from the runtime dispatch of `DiscreteMap`. It stays a transitive
  dev-only dep until N4 (since `@staticjit` is still used as a decorator in user
  systems — N4 makes it a no-op shim).

End state of N1: ~25% of the system catalogue (maps) has zero Numba in its hot
path.

### N2 — Rust ODE stepper, JiTCODE-RHS edition

The biggest single milestone in raw code volume but architecturally low-risk: we
swap *only the stepper*, not the symbolic pipeline.

- Steppers: Dormand-Prince 5(4), Dormand-Prince 8(5,3), Verner 9(8) for high
  accuracy, Rosenbrock-23 + Rosenbrock-34 for stiff.
- Adaptive step control: I-controller (FSAL-aware) by default, optional
  PI/PID for stiff problems.
- Dense output: native to each method (5th- and 8th-order interpolants).
- Event detection: root-bracketed with bisection / Brent on the dense output.
- RHS: JiTCODE-compiled C function pointer, called via Rust cffi.

`ContinuousSystem.integrate` stops calling `jitcode.set_integrator(...)` and
starts calling the new Rust dispatcher. JiTCODE is still imported, but only for
symbolic→C compilation.

### N3 — Variational ODE for Lyapunov in Rust

- Auto-derive the variational system `Δ̇ = J(x) Δ` from the symbolic Jacobian
  (JiTCODE / SymEngine already provides this).
- Integrate the augmented system `(x, Δ_1, …, Δ_k)` with the N2 stepper.
- Reorthogonalize with `nalgebra` QR every `reortho_interval` steps.
- Drops `jitcode_lyap` entirely.

### N4 — Symbolic → cranelift JIT (drops JiTCODE)

The architecturally hard milestone. Multi-chat.

- Define an IR (`tsdyn_core::ir::Expr`) that mirrors a subset of SymEngine ops:
  arithmetic, transcendentals, `pow`, `abs`, `min`/`max`, `where` (for branching).
- Lower SymEngine expressions → IR in Python (small Python module that walks
  `symengine.Basic` trees).
- IR → cranelift in Rust. Emit a `fn(state: *const f64, t: f64, params: *const f64,
  dy: *mut f64)` function.
- Cache compiled functions by IR hash (much cheaper than recompiling the whole
  Python module).

End state of N4: JiTCODE is removed from runtime dependencies. No C compiler is
needed on the user's machine. ODE integration is end-to-end Rust.

### N5 — Rust DDE solver suite (drops JiTCDDE)

Hardest milestone. Multi-chat.

- Continuous extension: cubic Hermite interpolation over a sliding history buffer.
- Breakpoint propagation: solver detects discontinuities at `t = t_0 + k*tau` for
  each delay; refines step size around them.
- State-dependent delays: re-evaluate at every step.
- Lyapunov: variational DDE with augmented history buffers per tangent vector.
- Reference: Shampine & Thompson's DDE_SOLVER, JiTCDDE's source.

End state of N5: JiTCDDE is removed. The compute layer is pure Rust.

### N6 — Compatibility validation & default flip

This milestone exists to make N1–N5 safe to land in production.

- `tests/native/regression/` stores reference trajectories (golden files) for
  every built-in system at fixed (IC, params, t_final, dt). Generated once,
  ahead of N1, against the current JiTCODE/JiTCDDE/Numba paths.
- After N5, every Rust path is validated against its golden file with
  `np.allclose(rtol=1e-8, atol=1e-10)` (looser for chaotic systems beyond Lyapunov
  time — there we compare statistical observables: mean, var, Lyapunov spectrum,
  attractor histogram).
- When green across the entire catalogue, the Rust path becomes the
  implementation. No user-visible change.

### N7 — Deprecate JiTCODE / JiTCDDE / Numba

- Remove from runtime `[project.dependencies]`.
- Keep in `[dependency-groups.dev]` as the reference implementations used by the
  regression tests.
- Release a major version (1.0?) with a clear note: install size shrinks, no C
  compiler required.

## Architecture decisions

### Dispatch

Inside `ContinuousSystem.integrate`:

```python
def integrate(self, ...):
    handle = self._get_problem_handle()       # opaque, cached per system
    return _native.integrate_ode(handle, t0, t1, dt, ic, method, rtol, atol)
```

`_get_problem_handle` returns either a JiTCODE-cffi-backed `ProblemHandle` or a
cranelift-JIT-backed `ProblemHandle`. The native side doesn't care which. There
is **no** `backend=` argument to `integrate`. The implementation makes the choice
based on what's available; once N4 lands, only cranelift is available.

### Variational system auto-derivation

After N3, every system gets an automatically-derived variational RHS. The
existing `_jacobian` method (when provided) is used as the per-state Jacobian; if
absent, SymEngine differentiation derives it once per system at first use.

### Caching

- Compiled functions cached on disk by IR hash (in `~/.cache/tsdynamics/`,
  unchanged path; the contents are different).
- Cache version-stamped so cranelift codegen changes invalidate old entries
  automatically.

### `staticjit` decorator

Today `@staticjit` is `@staticmethod` + `@njit`. After N4 it becomes
`@staticmethod` only — the JIT happens via the SymEngine pipeline, not Numba.
Users with custom `DiscreteMap` subclasses see no change; their `_step` methods
keep working but get JIT'd by us instead of by Numba.

## Validation strategy (the entire reason this can be safely shipped)

1. **Golden files first.** Before any N-milestone code lands, generate
   `tests/native/regression/<SystemName>.npz` from the current
   JiTCODE/JiTCDDE/Numba paths.
2. **Per-milestone:** the milestone's tests assert agreement with the golden
   files on the systems it covers.
3. **N6 acceptance:** the full catalogue agrees with golden files.
4. **Bonus:** every milestone with a DynamicalSystems.jl equivalent records a
   benchmark in `bench/RESULTS.md` — wall-clock and solution-norm vs the
   reference.

## Risk register

| Risk | Mitigation |
|---|---|
| Rust ODE stepper slower than JiTCODE | N2 must hit "within 10% of JiTCODE wall-clock on standard suite" before merge. Profile-guided tuning required. |
| Cranelift codegen quality worse than tcc | Compare emitted assembly on a few benchmarks before committing to cranelift. If quality is poor, fall back to LLVM via `inkwell`. |
| DDE breakpoint tracking is buggy | N5 is multi-chat with a long validation phase. Twin-DDE benchmark suite recorded against JiTCDDE on at least 20 systems before any default flip. |
| Stiff Rosenbrock harder than expected | Defer Rosenbrock to a sub-milestone N2.5 if it's blocking N2. Most built-ins are non-stiff. |
| Variational DDE numerics | Defer to N5+, gated by N5's stability story. Keep `jitcdde_lyap` as the implementation for DDE Lyapunov until then. |

## Open questions (revisit when N2 starts)

- LLVM (inkwell) vs cranelift? Cranelift is simpler, lower compile latency, and
  pure Rust. LLVM is faster runtime in some cases. Recommended: cranelift first,
  switch if a specific benchmark demands it.
- Should the IR include vectorisable annotations (SIMD intrinsics)? Probably yes
  for variational systems with many tangent vectors.
- How do we handle user-defined Python helpers inside `_equations` that aren't
  pure SymEngine? Currently JiTCODE rejects these silently in some cases. We
  should detect and either lower to IR or raise a clear error.
