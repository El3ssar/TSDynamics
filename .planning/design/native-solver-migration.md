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

### N2 — Pure-Rust ODE stepper suite

**The biggest architectural milestone of Track E.** ODE integration runs
end-to-end in Rust: the RHS *and* the stepper. JiTCODE stops being on the
hot path; it stays imported during a transition window only because it's
the cleanest way to get a symbolic Jacobian out of `_equations` until N3
makes that an internal SymEngine pass.

The lesson from N1 is "do not reinvent the wheel": the same IR + tracer
machinery that lowered map `_step` to bytecode is the right substrate
here, extended with the handful of additional ops that ODE RHSes use
(trig, exp, log, conditional via `Where`, rational powers). The Rust
stepper interprets that IR at every RHS call.

For the actual time-stepping algorithms we lean on the existing Rust
ODE ecosystem rather than rolling our own from scratch:

- **`ode_solvers`** ([github.com/srenevey/ode_solvers](https://github.com/srenevey/ode_solvers))
  gives DP5, DP8, Verner 6/7/8/9, RK4. Well-tested, `nalgebra`-based,
  matches our state-vector representation. Use as-is where it fits.
- **`peroxide`** is a broader scientific Rust crate that includes its
  own ODE solvers; useful for benchmarking against `ode_solvers` and as
  a fallback if `ode_solvers` doesn't expose a hook we need.
- **`nalgebra`** for state vectors, QR (variational track), and the LU
  factorizations that stiff Rosenbrock-Wanner methods need.

What the existing crates **don't** cover well enough to use off-the-shelf
is the stiff side (Rosenbrock-Wanner, Rodas, Radau): for those we
implement against `nalgebra` LU with the Hairer-Wanner coefficient
tables. Defer the more exotic implicit methods (BDF, Radau, NDF) to a
follow-up milestone if they're not needed by any built-in system before
M-level work demands them.

**Method catalogue (the DynamicalSystems.jl-equivalent ambition)**:

| Method        | Order | Adaptive | Stiff?      | Source |
|---------------|-------|----------|-------------|--------|
| DP5 / DOPRI5  | 5(4)  | yes      | non-stiff   | `ode_solvers` |
| DP8 / DOP853  | 8(5,3)| yes      | non-stiff   | `ode_solvers` |
| Tsit5         | 5(4)  | yes      | non-stiff   | implement (coefs published) |
| Vern6/7/8/9   | 6–9   | yes      | non-stiff   | `ode_solvers` |
| RK4           | 4     | no       | non-stiff   | implement (trivial) |
| BS3           | 3(2)  | yes      | non-stiff   | implement |
| Rosenbrock23  | 2(3)  | yes      | stiff       | implement on `nalgebra` LU |
| Rosenbrock34  | 3(4)  | yes      | stiff       | implement on `nalgebra` LU |
| Rodas4        | 4(3)  | yes      | very stiff  | implement on `nalgebra` LU |

The point isn't to ship all of these in one chat — it's that the
**stepper trait** must be designed so adding a new method is a single
file containing the Butcher tableau (or Rosenbrock coefficient table)
and an `impl Stepper`. Reference design: `ode_solvers`' `System` trait
and DiffEq.jl's `OrdinaryDiffEqAlgorithm`.

**RHS evaluation, the architectural payload**:

The Python side extends `tsdynamics.base._lowering` with an
`ode_lowering` pass: takes a `ContinuousSystem`, runs `_equations` once
with `symengine` symbols (JiTCODE already does this internally; we tap
the same expressions), lowers the resulting expression trees to the
existing IR. The IR opcode space gains whichever ops weren't needed by
N1's map catalogue but are needed by the built-in ODE catalogue.

The Rust side reuses N1's interpreter unchanged — bytecode is bytecode.
The stepper calls `eval_rhs(state, t, params, out_dy)` which dispatches
through `tsdyn-core::ir::eval`.

**Adaptive control / dense output / events**:

- Step controller: I-controller default, PI-controller available
  (`ode_solvers` exposes these; reuse).
- Dense output: per-method continuous extensions (DP5's 4th-order
  interpolant, DP8's 7th-order, Vern's per-order interpolants). This
  is needed by M2's `detect_events` post-N2 so refinement uses the
  integrator's native interpolant rather than the current cubic Hermite
  on sampled state.
- Events: in-flight bracket detection during the step loop, refined via
  Brent on the dense-output interpolant. The Python-side `EventCondition`
  protocol stays unchanged — it's reused as the bracketed-root specifier;
  the difference is *where* the integration happens.

**Dispatch**:

```python
def integrate(self, ...):
    handle = self._compile_ode_ir()              # cached per system
    return _native.integrate_ode(
        handle, t_span, ic, params,
        method=method, rtol=rtol, atol=atol,
        events=events,
    )
```

No `backend=` flag. No JiTCODE fallback path exposed to users. If IR
lowering fails on a user's `_equations` we raise a clear error pointing
at the offending node — the way out is to file an issue requesting the
op be added to the IR, not to flip a backend switch.

### Extensibility — growing the method catalogue (DiffEq-style pool)

The long-term product shape matches **DynamicalSystems.jl’s reliance on a deep
solver ecosystem**: scientists pick an algorithm for their stiffness / accuracy /
density-output needs; the library stays cohesive because **RHS/Jacobian access,
adaptivity hooks, and output grids share one IR-backed evaluation layer**.

Design rules so new methods remain cheap to maintain:

1. **Single evaluation contract.** Every solver consumes the same IR-driven
   `eval_rhs` / `eval_jacobian` paths (`tsdyn-core`). Adding Radau or another
   implicit scheme later must **not** fork RHS semantics.

2. **Localized implementations.** Each algorithm is a submodule tree under ``crates/tsdyn-ode/src/methods/``
   (Butcher tableau, embedded explicit runners, Rosenbrock/LU bundles, Verner 6–9 subtrees, …). Register new work in
   ``crate::driver::integrate_ode`` and Python `_rust_integrator_name` / `_RUST_NATIVE_METHODS` only — never fork IR eval.

3. **Stable Python surface.** New Rust methods appear under documented `method=`
   strings (aliases optional). User-facing docs carry a **methods table** listing
   stiffness class, order, and caveats — mirror OrdinaryDiffEq’s algorithm docs
   in miniature.

4. **Regression per addition.** Every new integrator gets at least one focused test
   (short horizon + golden or reference trajectory) so the pool stays trustworthy
   as it grows.

5. **N2.d housekeeping.** Optional **behaviour-neutral** module splits and clearer
   dispatch grouping are encouraged **only** when they directly support (1)–(4).
   Large architectural gambits belong to later milestones (e.g. N4 JIT), not N2.d.

6. **Shared sampling.** The **`tsdyn-solver-base`** crate owns `uniform_time_grid` plus future grids/schedulers usable by
   both ODE (**`tsdyn-ode`**) and DDE (**`tsdyn-dde`**, milestone N5) without circular crate dependencies.

| Crate | Role |
|-------|------|
| `tsdyn-core` | IR bytecode interpreters (`CompiledOde`, discrete maps). |
| `tsdyn-solver-base` | Output grids / schedulers unbiased toward ODE vs DDE. |
| `tsdyn-ode` | ODE catalogue (`methods/`, `driver`; **N3** variational augmentation lands here next). |
| `tsdyn-dde` *(N5 planned)* | Parallel crate for histories + breakpoints + its own catalogue. |

**End state of N2**:

- All built-in ODE systems integrate end-to-end in Rust.
- JiTCODE is *still imported* (used only as a SymEngine-expression
  source for lowering) but no longer on the hot path.
- M2 event detection optionally consumes the integrator's dense output.

### N3 — Variational ODE Lyapunov in Rust

Builds directly on N2. The IR already knows how to evaluate the
Jacobian (the map case in N1 did this). N3 adds:

- Auto-derive the variational system `Δ̇ = J(x) Δ` symbolically once per
  system. SymEngine differentiation is already used by JiTCODE's
  `jitcode_lyap`; reuse the same trees. Lower J(x) to the same IR
  vocabulary, no new opcodes.
- Augmented stepper: state vector grows to `(x, Δ_1, …, Δ_k)`; the
  block-diagonal-ish Jacobian structure stays implicit (we just call
  `eval_rhs` and `eval_jacobian` on the IR).
- Reorthogonalize with `nalgebra` QR every `reortho_interval` steps.
- Time-weight-averaged local exponents collected on the Rust side.

End state: `ContinuousSystem.lyapunov_spectrum` drops `jitcode_lyap`
entirely. JiTCODE is no longer needed for *any* ODE compute path — it
remains only as a SymEngine convenience.

### N4 — Cranelift JIT for the IR

Pure performance milestone. Architecturally low-risk: the IR is the
contract; N4 swaps the interpreter for a cranelift-codegen'd native
function with the same `fn(state, t, params, out_dy)` shape.

- Add `cranelift-codegen`, `cranelift-frontend`, `cranelift-module`,
  `cranelift-jit` as dependencies of a new `crates/tsdyn-jit/` crate.
- Walk the IR, emit cranelift IR, JIT-compile to a function pointer.
- Cache compiled functions on disk by IR hash (extends N1's
  process-local cache to disk-backed).
- The stepper does not know whether it's calling the interpreter or
  the JIT; the dispatch is per-system at first integrate.

End state of N4: JiTCODE is removed from runtime dependencies entirely.
No C compiler needed on the user's machine. Symbolic→IR is the only
preprocessing step before integration runs.

### N5 — Pure-Rust DDE solver suite

The hardest milestone. The good news: the IR, stepper trait, dense
output, and event detection from N2/N4 all transfer. The hard parts
are DDE-specific.

- **History buffer**: ring buffer of past `(t, y, slope_for_Hermite)`
  triples. Sized from the max delay × headroom. Random-access lookup
  via binary search + Hermite interpolation.
- **Breakpoint propagation**: at every integration start, queue the
  discontinuity surface set `{t_0 + k * τ_i : k ∈ ℕ}` for each delay.
  Step controller refines step size when crossing one to preserve
  order.
- **State-dependent delays**: re-evaluate τ = τ(x(t)) at every step;
  the breakpoint set is recomputed lazily.
- **Lyapunov**: variational DDE with augmented history buffers per
  tangent vector. Drop `jitcdde_lyap`.

References to read carefully *before* writing Rust:

- Shampine & Thompson, "Solving DDEs in MATLAB" (the dde23 paper).
- Bellen & Zennaro, *Numerical Methods for Delay Differential Equations*.
- JiTCDDE's source — it's a clean implementation of the Bellen-Zennaro
  approach atop CVODE; the discontinuity-tracking machinery is the
  reference.

Existing Rust DDE crates are sparse; expect to implement this layer
in-house. Verify the assumption at milestone-start time.

End state of N5: JiTCDDE removed. The compute layer is pure Rust.

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
    handle = self._compile_ode_ir()           # opaque IR handle, cached per system
    return _native.integrate_ode(handle, t_span, ic, params, method, rtol, atol)
```

`_compile_ode_ir` builds (and caches per class) an IR representation of
the RHS — same machinery as N1's `_lowering`, extended for ODE ops. The
native side sees only the IR; it does not know or care whether the
interpreter (N2) or the cranelift JIT (N4) is evaluating each op. There
is **no** `backend=` argument to `integrate`. Once N7 lands, IR is the
only supported representation.

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
| IR interpreter slower than JiTCODE-compiled C | Acceptable for N2 (correctness > speed); closed by N4's cranelift JIT. Record the gap in `bench/RESULTS.md`. |
| `ode_solvers`' stepper trait too rigid for what we need (e.g. event detection in-flight, dense output access) | Wrap or fork as needed. Worst case, port the Butcher tableaux and write our own stepper trait — the per-method code is small once the trait is right. |
| Cranelift codegen quality worse than tcc/gcc | Compare emitted assembly on Lorenz / Rössler before committing. Fall back to LLVM via `inkwell` if needed (bigger build, but used by `numba` so dependency precedent exists). |
| Stiff Rosenbrock harder than expected | Spin off as N2.5 if blocking. Most built-ins are non-stiff; ship N2 with non-stiff methods + a clear "stiff is coming" note. |
| DDE breakpoint tracking is buggy | N5 has a long validation phase. Twin-DDE benchmark suite recorded against JiTCDDE on ≥ 20 systems before any default flip. |
| User `_equations` uses Python control flow / NumPy / non-SymEngine helpers | Detect at lowering time and raise a clear `NotLowerableError` pointing at the offending construct + the workaround (use `symengine.Piecewise`, etc.). Same machinery N1 uses. |
| `ode_solvers` is single-maintainer (like JiTCODE) | We're not betting the architecture on it — only the explicit RK methods. The stiff side, dense output, and event detection are in-house. Worst case we vendor the explicit-method code. |

## Open questions (revisit when N2 starts)

- LLVM (`inkwell`) vs cranelift for N4? Cranelift is simpler, lower compile
  latency, pure Rust. LLVM is faster runtime in some cases. Recommended:
  cranelift first, switch if a specific benchmark demands it.
- Should the IR include vectorisable annotations (SIMD intrinsics)? Probably
  yes for variational systems with many tangent vectors. Investigate during
  N3.
- For N2's stepper trait: use `ode_solvers`' `System` trait as-is, or define
  our own (so we can plug both `ode_solvers`-provided methods and in-house
  Rosenbrock cleanly)? Recommended: our own, keeping it shaped close enough
  to `ode_solvers`' that adapting their methods is mechanical.
- Tsit5 / Vern coefficients: pull from `OrdinaryDiffEq.jl`'s tableau modules
  (MIT-licensed, well-maintained), record source in code comments. Don't
  retype from papers.
- Dense output for events: N2 must expose it through PyO3 so M2's
  `detect_events` can be retrofitted to use the integrator's native
  interpolant. The current cubic-Hermite-on-sampled-state path stays as the
  raw-arrays fallback.
