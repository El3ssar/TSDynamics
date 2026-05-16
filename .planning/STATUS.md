# Status — updated 2026-05-16 (N2.a landed; N2.b is next)

Current milestone: **N2 — Pure-Rust ODE stepper suite (WIP)**.  Sub-stage
N2.a closed; next chat picks up at N2.b.

Phase: implement.  Track E (Rust solver migration) is now in progress;
Tracks A (M5/M6/M7/M10/M11), B (V1), and C (R2) remain unblocked and can
run in parallel chats.

Last-touched files (N2.a):

- `src/tsdynamics/base/_ir.py` — added `OP_TIME`, `OP_TANH`, `OP_SINH`,
  `OP_COSH`, `OP_POW2`, `OP_POWF`; `Time` and `PowF` nodes; `CompiledOde`
  dataclass + `serialize_ode` payload (RHS + optional Jacobian +
  `param_names`).
- `src/tsdynamics/base/_ode_lowering.py` (new) — SymEngine-tree walker
  that lowers `cls._equations` to IR and best-effort auto-diffs the
  Jacobian (skips silently to `has_jacobian=False` when SymEngine yields
  unevaluated `Derivative` nodes on `Abs`/`sign`).
- `crates/tsdyn-core/src/ir.rs` — added the matching Rust opcodes,
  `Expr::{Time, Tanh, Sinh, Cosh, PowF, Pow2}`, a `CompiledOde::from_bytes`
  decoder, and a `CompiledOde::eval_rhs(t, y, params)` evaluator.  Map
  evaluator now takes `t: f64` (maps pass `0.0`).
- `crates/tsdyn-native/src/ode.rs` (new) — PyO3 bindings:
  `eval_ode_rhs(bytecode, t, y, params)` (single) and
  `eval_ode_rhs_batch(bytecode, ts, ys, params)` (batched).
- `crates/tsdyn-native/src/lib.rs`, `crates/tsdyn-maps/src/lib.rs` —
  wiring + the `t: 0.0` pass-through for maps.
- `src/tsdynamics/_native/__init__.py` — re-export the two new ODE
  evaluators.
- `scripts/generate_ode_goldens.py` (new) — runs JiTCODE on every
  built-in continuous system with a per-system 45 s wallclock timeout
  (`SIGALRM`) and writes `tests/native/regression/ode/<System>.npz` with
  `t, y, ic, params, param_names, structural_params, structural_values,
  final_time, dt, rtol, atol, method, seed`.  110 of 115 systems
  succeed; 5 (`Duffing`, `SprottD`, `SprottI`, `ExcitableCell`,
  `BlinkingRotlet`) skip because random ICs don't land in the attractor
  basin or the system is too stiff for any of the explicit-RK / lsoda /
  vode methods JiTCODE exposes — N2.b will hand-pick ICs for them.
- `tests/native/regression/ode/` — 110 `.npz` files committed alongside.
- `tests/test_native_ode.py` (new) — three parametrised tests per system
  (115 × 3 = 345 cases): `test_lower_ode_succeeds`,
  `test_ode_rhs_matches_symengine` (IR vs SymEngine `Lambdify` on 16
  random `(t, y)` samples, rtol=1e-9 / atol=1e-12; loose atol=1e-6 for
  `JerkCircuit` whose `exp(y/y0)` reaches ~5e7), and
  `test_golden_ode_loadable` (skips the 5 by design).  340 pass, 5 skip.
- `CHANGELOG.md` — entry under Unreleased.
- `.planning/STATUS.md`, `.planning/ROADMAP.md` — N2 marked WIP.

## What's done

- **M0**: planning framework bootstrapped.
- **R1**: Rust toolchain + maturin + CI wheels — landed 2026-05-16.
- **N1**: Rust map stepper — landed 2026-05-16.
- **M1**: Trajectory enrichment — landed 2026-05-16.
- **M2**: Event & section detection — landed 2026-05-16.
- **N2.a**: ODE IR extension + SymEngine lowering + PyO3 RHS evaluator
  + golden trajectory snapshot — landed 2026-05-16.
  - Full IR opcode set is now ODE-complete: hyperbolic functions,
    fractional and symbolic powers, explicit-time variable.
  - 115 built-in continuous systems all lower to IR without error
    (`NotLowerableError` doesn't fire for anyone in the catalogue).
  - IR evaluation matches SymEngine's `Lambdify` within rtol=1e-9 on
    every system; the NaN pattern (e.g. `WindmiReduced`'s
    `v ** Rational(1, 2)` on negative samples) matches bit-exactly
    between IR and SymEngine.
  - Jacobian auto-diff lowers successfully for most systems; 6 with
    `Abs`/`sign` keep `has_jacobian=False` until N2.c needs J for the
    Rosenbrock family (then `_jacobian` can be supplied explicitly).
  - `CompiledOde` carries `param_names` so the Rust evaluator can
    consume Python's `ParamSet` insertion order verbatim.
  - 110/115 systems also have golden trajectory snapshots; the 5
    skipped ones get hand-picked ICs in N2.b.
  - Tests: 1192 passed / 61 skipped across the full Python suite in
    148 s; `cargo test --workspace --release` green (5 + 2 = 7 Rust
    unit tests, including a new `eval_tanh_sinh_cosh_powf` over the
    extended opcode set).  `ruff check` + `ruff format --check` clean.

## What's in progress

- **N2** is WIP at sub-stage N2.b.  N2.a is closed; N2.b starts next.

## Next action

Pick up **N2.b — Explicit RK family** from
[`milestones/N2-rust-ode-stepper.md`](milestones/N2-rust-ode-stepper.md).
Concretely:

1. Create `crates/tsdyn-ode/` (workspace member) with the `Stepper` /
   `Rhs` traits, an `IrInterpreterRhs` wrapping `CompiledOde`, an
   `I-controller` (PI is N2.c's concern), and concrete steppers for the
   explicit-RK family: **DP5, DP8, Tsit5, Vern9, RK4, BS3**.
2. Vendor Butcher tableaux from `ode_solvers` (MIT) for DP5/DP8/RK4 and
   from `OrdinaryDiffEq.jl` (MIT) for Tsit5/Vern9 — file-header
   attribution.  Do not retype from papers.
3. Native dense output per method (4th-order Hermite for DP5, the DP8
   7th-order extension, per-order Vern interpolants, BS3's natural
   3rd-order extension, linear for RK4 — sample exactly on
   `np.arange(t0, final_time + dt/2, dt)`).
4. PyO3 wire: `_native.integrate_ode(handle, t_span, ic, params,
   method, dt_output, rtol, atol) -> (t, y)`.
5. `ContinuousSystem.integrate` dispatches to `_native.integrate_ode`
   for the methods the new crate ships; falls back to the JiTCODE
   path for the deprecated `"LSODA"`/`"VODE"` names *and* for any
   system that raises `NotLowerableError` (none today, defensive only).
6. Method-name compat: `"RK45"`/`"dopri5"` → `"DP5"`, `"DOP853"` →
   `"DP8"`.  Stiff family + alias deprecations are N2.c.
7. Tests: turn the 110 goldens into a full regression
   (rtol=1e-8, atol=1e-10) against DP8 in `tests/test_native_ode.py`,
   add `tests/test_ode_methods.py` (every explicit-RK method
   integrates Lorenz on `[0, 10]` and agrees with DP8 at rtol=1e-12).
8. Make `lyapunov_spectrum` keep using JiTCODE (that's N3) but
   sanity-check the public surface still works once `integrate`
   dispatches through Rust.

Per the milestone, **the user answered the six open questions during
N2.a's planning** — those answers stick:

- Pure-Rust steppers (no `ode_solvers` Cargo dep; vendor tables only).
- Always include `t_final` in the output grid
  (`np.arange(t0, final_time + dt/2, dt)`).
- Per-method controller default (I for explicit-RK; PI in N2.c for
  Rosenbrock).
- Silent fallback to JiTCODE on `NotLowerableError` for N2's lifetime.
- Cache key `(class, dim, hash(structural_params))`.
- Default method stays `"DP5"`.

Alternative milestones still open if a different chat picks a different
track:

- **R2** (Rust parameter-sweep kernel): rayon-backed.  Unlocks M3 → M4.
- **V1** (Viz skeleton).  M2 left three `to_dataspec` placeholder dicts.
- **M5** (Equilibria), **M6** (Embedding utils), **M7** (Spectral
  toolkit) — all independent of N2.

## R1 follow-ups (still parked)

- SCM-driven versioning via `update_version.py`.
- Collapse `publish.yml` + `release.yml` + the wheel-emitting half of
  `wheels.yml` into a single tag-triggered publish workflow.
- Windows wheels: add to `wheels.yml` matrix after N2 proves no blockers.

## N1 follow-ups (parked)

- **Performance**: constant-folding pass on the IR.  Optional — N4
  cranelift makes it moot.
- **Disk-backed IR cache**: process-local for N1; revisit in N4.
- **DynamicalSystems.jl comparison**: not yet recorded.  Add when P1's
  benchmark harness lands.

## M1 / M2 follow-ups (parked)

(unchanged — see git history of this file for the full lists)

## N2.a follow-ups (parked for N2.b/c)

- **Jacobian for `Abs`/`sign` systems**: today the lowering silently
  sets `has_jacobian=False`.  N2.c needs J for Rosenbrock.  Either
  hand-supply `_jacobian` on the six affected systems
  (MultiChua, AnishchenkoAstakhov, StickSlipOscillator,
  CellularNeuralNetwork, Colpitts, FluidTrampoline) or add a
  smooth-approximation `Abs` pass.
- **5 missing goldens**: `Duffing`, `SprottD`, `SprottI`,
  `ExcitableCell`, `BlinkingRotlet`.  Pick basin-anchored ICs in N2.b
  (e.g. Duffing on its strange attractor needs `[1, 1, 0]`-ish; the
  Sprotts need careful selection from Sprott's original tables).
- **`JerkCircuit` numeric-precision quirk**: the `exp(y/y0)` term blows
  past ~5e7 at random samples; current test uses `atol=1e-6` for it.
  N2.b should switch to a smaller IC range for that one system in the
  regression test (the `tests/test_native_ode.py` `RHS_SAMPLE_RANGE`
  knob is the right place).
- **PyO3 batched evaluator perf**: today every batch call re-decodes the
  bytecode.  N2.b can hoist the decode out of the loop by passing a
  pre-allocated `OdeProblemHandle` (the milestone already plans this).

## How to resume

A future chat should:

1. Read this file.
2. Pick N2.b (or a parallel milestone from the alternatives above).
3. Open `.planning/milestones/N2-rust-ode-stepper.md` and follow it.
4. After landing the sub-milestone: tick N2.b in this STATUS.md (or
   close N2 entirely if all four sub-stages land in one chat), update
   ROADMAP.md if the status changes, commit `.planning/` with the code.
