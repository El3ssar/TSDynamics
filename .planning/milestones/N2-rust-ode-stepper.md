# Milestone N2 — Pure-Rust ODE stepper suite

Status: **COMPLETE — functional path** (Rust `integrate` catalogue shipped). Stretch items listed in § Acceptance backlog remain unchecked in the boxed list on purpose until they land.
Depends on: R1 (toolchain), N1 (IR + tracer + interpreter)
Estimated scope: **multi-chat** (probably 3–5 sessions if cleanly split)
Design doc: [design/native-solver-migration.md](../design/native-solver-migration.md)

## Read before writing code

1. This file.
2. [`design/native-solver-migration.md`](../design/native-solver-migration.md)
   — especially the **N2** section and the **Risk register**.
3. [`milestones/N1-rust-map-stepper.md`](N1-rust-map-stepper.md) — the IR /
   tracer / lowering machinery N2 extends is described there.
4. The current state:
   - `src/tsdynamics/base/ode_base.py` — the JiTCODE-backed
     `ContinuousSystem.integrate` / `lyapunov_spectrum`.  N2 replaces the
     `jitcode.set_integrator(...)` call site, not the public API.
   - `src/tsdynamics/base/_ir.py`, `_lowering.py`, `_tracer.py` — N1's IR
     plumbing.  N2 extends `_ir.py`'s opcode space and adds an
     `_ode_lowering.py` analogous to `_lowering.py`.
   - `crates/tsdyn-core/src/ir.rs` — the Rust IR interpreter.  N2 adds
     opcodes here; nothing about its structure changes.
5. The Rust crates we'll lean on:
   - **`ode_solvers`** (https://github.com/srenevey/ode_solvers,
     MIT-licensed) — explicit RK methods, Butcher tableau format, dense
     output for some methods.  Inspect their `System` trait *before*
     designing ours; we want the trait shape to be either identical or
     near-identical so adapting their methods is mechanical.
   - **`nalgebra`** — already a transitive dep via N1 (`Matrix3`, QR).
     Use for state vectors / LU factorisations / variational QR.
   - **`peroxide`** — only as a benchmarking reference for the explicit
     methods.  Not a build dependency.

## Mission

ODE integration runs **end-to-end in Rust**.  The RHS is evaluated by a
Rust interpreter against an IR derived from the user's `_equations`
(same machinery as N1, extended); the stepper, adaptive control, dense
output, and event refinement are all Rust.  JiTCODE stays as a transitive
import for one more milestone (N3) only because it's the cleanest way to
get a symbolic Jacobian out of `_equations` — N3 replaces that with a
SymEngine-only pass.

**Multiple solver methods.**  The point of N2 is not just "swap one
stepper for another" but to give scientists the DynamicalSystems.jl-grade
choice between high-order non-stiff methods, stiff-aware Rosenbrock-type
methods, and fixed-step methods.  See the catalogue below.

**Architecturally, N2 = N1's machinery extended.**  N1's IR, tracer,
lowering, and Rust interpreter all stay.  N2 adds:

1. Whichever opcodes the ODE catalogue needs that N1's map catalogue
   didn't (trig was there; full `Pow(f64)`, `exp/log/sinh/cosh/tanh/atan2`,
   `Piecewise/Where` we'll verify).
2. A new lowering module `_ode_lowering.py` that takes a
   `ContinuousSystem` and emits `CompiledOde`.
3. A new Rust crate `crates/tsdyn-ode/` holding the stepper trait + the
   concrete methods.
4. PyO3 bindings in `crates/tsdyn-pyo3/` (or wherever N1 put them) that
   expose `integrate_ode(handle, t_span, ic, params, method, rtol, atol)`.
5. A dispatch change in `ContinuousSystem.integrate` to use the new Rust
   path.

## Public API

The user-facing API is *unchanged*:

```python
lor = ts.Lorenz()
traj = lor.integrate(final_time=100.0, dt=0.01, method="dop853",
                     rtol=1e-8, atol=1e-10)
exps = lor.lyapunov_spectrum(dt=0.1, burn_in=50.0, final_time=300.0)
```

The `method=` kwarg already exists.  N2 expands its accepted values; the
old names (`"RK45"`, `"dopri5"`, `"DOP853"`, `"LSODA"`, `"VODE"`) keep
working with explicit deprecation paths where appropriate:

| Old name (JiTCODE) | New name (Rust) | Notes |
|--------------------|-----------------|-------|
| `"RK45"` / `"dopri5"` | `"DP5"`          | Old names alias to `"DP5"`. |
| `"DOP853"`            | `"DP8"`          | Old name aliases. |
| `"LSODA"`             | (deprecated)     | Was the legacy stiff switcher; new code should pick `"Rosenbrock23"` explicitly. Print a deprecation warning suggesting an alternative. |
| `"VODE"`              | (deprecated)     | Same as LSODA. |

New method names (the N2 catalogue):

- Non-stiff explicit RK: `"DP5"`, `"DP8"`, `"Tsit5"`, `"Vern6"`,
  `"Vern7"`, `"Vern8"`, `"Vern9"`, `"BS3"`, `"RK4"` (fixed-step).
- Stiff Rosenbrock-Wanner: `"Rosenbrock23"`, `"Rosenbrock34"`, `"Rodas4"`.

Defer the rest (`Radau`, `BDF`, `NDF`) to a follow-up; flag in
"Out of scope" below.

## Internal API (Python ↔ Rust boundary)

### New: `tsdynamics.base._ode_lowering`

```python
def lower_ode_to_ir(
    equations_fn: Callable,
    jacobian_fn: Callable | None,
    params: ParamSet,
    dim: int,
) -> CompiledOde:
    """
    Evaluate ``equations_fn(y_sym, t_sym, **params)`` with SymEngine
    symbols, lower each component to IR.  If ``jacobian_fn`` is None,
    differentiate symbolically with SymEngine to get J(x).  Raise
    :class:`NotLowerableError` (re-using N1's exception) on unsupported
    expression nodes — caller decides whether to fall back or error.

    Returns a dataclass with:

    - ``rhs: list[Expr]``       — length = dim
    - ``jacobian: list[list[Expr]]`` — dim × dim
    - ``dim: int``
    - ``n_params: int``
    - ``param_names: tuple[str, ...]``
    """
```

`CompiledOde` is a sibling of N1's `CompiledMap` in `_ir.py`.

### Extended: `tsdynamics._native`

```python
def integrate_ode(
    handle: OdeProblemHandle,
    t_span: tuple[float, float],
    ic: np.ndarray,
    params: np.ndarray,
    *,
    method: str,
    dt_output: float,            # only — internal stepping is adaptive
    rtol: float,
    atol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (t, y) sampled on the uniform output grid."""
```

The dense output is what gets sampled onto `np.arange(t0, t1, dt_output)`.
This matches `ContinuousSystem.integrate`'s current contract.

### Modified: `ContinuousSystem`

```python
class ContinuousSystem(SystemBase):
    def _compile_ode_ir(self) -> OdeProblemHandle:
        # Cache per class on first integrate.  Cache key:
        #   (class, dim, hash(structural_params)).
        # Disk cache lives at the same path as JiTCODE's today.
        ...

    def integrate(self, final_time, dt, *, t0=0.0, ic=None,
                  method="DP5", rtol=1e-6, atol=1e-9, **kw) -> Trajectory:
        handle = self._compile_ode_ir()
        params = np.asarray(self.params.as_tuple(), dtype=float)
        ic = self.resolve_ic(ic)
        t, y = _native.integrate_ode(
            handle, (t0, final_time), ic, params,
            method=method, dt_output=dt, rtol=rtol, atol=atol,
        )
        return Trajectory(t, y, system=self)
```

The fallback path (catch `NotLowerableError` → call legacy JiTCODE) lives
for one milestone, then goes away in N3.

## Rust side — `crates/tsdyn-ode/`

### Stepper trait

Designed to be a near-superset of `ode_solvers`' `System` + `Stepper`
contracts so adapting their methods is mechanical.  Concretely:

```rust
pub trait Stepper {
    /// Single adaptive step from t -> t + dt.
    /// Updates state in place; returns (dt_used, dt_next_proposal).
    fn step(
        &mut self,
        rhs: &dyn Rhs,
        t: f64,
        state: &mut [f64],
        dt_attempted: f64,
        rtol: f64,
        atol: f64,
    ) -> StepResult;

    /// Sample the dense-output interpolant at t* ∈ [t_prev, t_curr].
    fn dense_at(&self, theta: f64, out: &mut [f64]);

    fn order(&self) -> u32;
    fn name(&self) -> &'static str;
}

pub trait Rhs {
    fn eval(&self, t: f64, y: &[f64], params: &[f64], out_dy: &mut [f64]);
    fn dim(&self) -> usize;
    /// Stiff methods consult the Jacobian; non-stiff ignore.
    fn jacobian(&self, t: f64, y: &[f64], params: &[f64], out_j: &mut [f64]);
}
```

`Rhs` is implemented by `IrInterpreterRhs` (N2) and later by
`CraneliftJittedRhs` (N4).  The stepper does not know which it has.

### Concrete steppers (one file each)

- `crates/tsdyn-ode/src/methods/dp5.rs`
- `crates/tsdyn-ode/src/methods/dp8.rs`
- `crates/tsdyn-ode/src/methods/tsit5.rs`
- `crates/tsdyn-ode/src/methods/vern.rs` (parameterised over order
  6/7/8/9; one Butcher table per order)
- `crates/tsdyn-ode/src/methods/bs3.rs`
- `crates/tsdyn-ode/src/methods/rk4.rs`
- `crates/tsdyn-ode/src/methods/rosenbrock.rs` (Rosenbrock23/34/Rodas4)

Each file is: Butcher / Rosenbrock coefficient table + `impl Stepper`.
Where `ode_solvers` already has a coefficient table, vendor it (with
file-header attribution and a link).  Where `OrdinaryDiffEq.jl` is the
canonical source (Tsit5, Vern, Rodas), pull from their tableau modules
(also MIT-licensed) with attribution.  Do **not** retype from papers.

### Adaptive control

`crates/tsdyn-ode/src/controller.rs`:

- I-controller default (`dt_next = dt * (tol / err)^(1 / (order + 1))`).
- PI-controller available (`gustafsson`-style), opt-in via stepper config.
- Step-rejection backoff: 0.9 safety factor, min 0.2, max 10 per step.

### Dense output

Native to each method.  At minimum: 4th-order interpolant for DP5,
7th-order for DP8 (using DP8's three additional stages), per-order
interpolants for Vern, linear for RK4 (no real dense output for fixed-step
— sample at exactly the output grid by adjusting `dt`).

The Python side gets a sampled trajectory on `np.arange(t0, t1, dt)`,
*not* the integrator's internal step points.  Internal step points stay
inside Rust; we don't surface them yet (consider it as a follow-up if
asked).

### Event detection (placeholder for M2 retrofit)

N2 does **not** retrofit M2's `detect_events` to use the dense output
yet — keep the M2 implementation as-is (cubic Hermite on the sampled
state).  Add a `crates/tsdyn-ode/src/events.rs` skeleton that wraps the
in-flight event API with a TODO comment; the M2 retrofit is a follow-up
once N2 is green.

This is deliberate scope control — N2 is already large.

### Variational equations

**Not in N2.**  That's N3.  But: design the stepper trait so adding
"augmented state" is `n_aug = dim * (n_exp + 1)` and an
`IrInterpreterRhs` that evaluates `(f(x), J(x) Δ_1, …, J(x) Δ_k)` is a
plug-in — no stepper code changes.

## Files to create / modify

Create:

- `crates/tsdyn-ode/Cargo.toml`
- `crates/tsdyn-ode/src/lib.rs`
- `crates/tsdyn-ode/src/stepper.rs` — the `Stepper` + `Rhs` traits + `StepResult`.
- `crates/tsdyn-ode/src/controller.rs`
- `crates/tsdyn-ode/src/methods/{dp5,dp8,tsit5,vern,bs3,rk4,rosenbrock}.rs`
- `crates/tsdyn-ode/src/dense.rs` — per-method continuous extensions.
- `crates/tsdyn-ode/src/events.rs` — skeleton, TODO for M2 retrofit.
- `crates/tsdyn-ode/src/integrate.rs` — the top-level driver:
  `integrate_ode(handle, t_span, ic, params, method, dt_output, rtol, atol)`.
- `src/tsdynamics/base/_ode_lowering.py`
- `src/tsdynamics/_native/_ode.pyi`
- `tests/native/regression/ode/<system>.npz` — golden trajectories
  generated from the *current* JiTCODE path on a fixed seed before any
  N2 code lands.  One per built-in continuous system.
- `tests/test_native_ode.py` — regression vs goldens (rtol 1e-8 / atol
  1e-10) for non-chaotic systems; statistical comparisons (mean, var,
  Lyapunov spectrum, attractor histogram) for chaotic ones beyond
  Lyapunov time.
- `tests/test_ode_methods.py` — for each method in the catalogue, integrate
  Lorenz on `[0, 10]` and assert agreement with DP8 at `rtol=1e-12`.
- `bench/ode_bench.py` — wall-clock + accuracy vs JiTCODE on the
  standard suite.
- `bench/RESULTS.md` — record numbers (extends the N1 table).

Modify:

- `crates/Cargo.toml` — add `tsdyn-ode` to workspace.
- `crates/tsdyn-core/src/ir.rs` — extend `Expr` / opcode space for any
  ODE-only ops (verify against the catalogue first; trig/exp/log already
  exist).  Bump opcode version stamp.
- `src/tsdynamics/base/_ir.py` — mirror the new opcodes.
- `pyproject.toml` — add `tsdyn-ode` to maturin's module list.
- `src/tsdynamics/base/ode_base.py` — `integrate` dispatches to
  `_native.integrate_ode`; `lyapunov_spectrum` stays on JiTCODE until N3.
- `src/tsdynamics/_native/__init__.py` — re-export `integrate_ode`.

Do **not** touch:

- `src/tsdynamics/base/dde_base.py` — N5.
- `src/tsdynamics/base/map_base.py` — N1 owns it; don't refactor.
- M2 event detection — see "Event detection" note above.

## Acceptance criteria

**Progress.** The verbatim checklist preserves the ideal bar; unchecked rows are deliberate **stretch / N2.x** backlog (not regressions blocking N3 unless otherwise noted).

**Done in-tree today (high level):**

- Rust `DP5`/`DP8`/`Tsit5`/`VERN6`/`VERN7`/`VERN8`/`VERN9`/`BS3`/`RK4` + Rosenbrock family wired through `integrate`; JiTCODE silent fallback persists.
- `tests/test_native_ode.py`, `bench/RESULTS.md` + `bench/ode_bench.py`, tightened Lorenz-vs-DP8 tolerances (`tests/test_ode_methods.py` with symmetric high-precision tolerances), `tests/test_ode_ir_cache.py` instrumenting **`lower_ode_to_ir` call count**, `events.rs` placeholder for eventual M2 dense-output retrofit.

**N2.x backlog (not yet matching the original rubric literally):**

- [x] **Vern6 / Vern7 / Vern8** timesteppers (Rust + Python catalogue; coefficients from SciML via `scripts/gen_verner_ode_coeffs.py`).
- [ ] **JiTCODE-only genesis** golden rule for *every* NPZ (several systems use Rust-regenerated basins; a few catalogue goldens remain intentionally absent).
- [ ] **Chaotic-system statistics** vs JiTCODE (5σ bands / histograms across many ICs — needs a calibrated harness; naive long-horizon JIT vs Rust pointwise stats disagree because chaos).
- [ ] **Bench gate** “≤1.5× JiTCODE RHS walltime on full suite” remains an aspirational KPI (interpreter-bound until N4).

**Original checklist retained for audit history** (note: many bullets are now satisfied in practice even though the typography still shows `[ ]` — see “Done in-tree” bullets above for the live source of truth):

- [ ] Golden files exist under `tests/native/regression/ode/` for every
      built-in continuous system, generated from the *current* JiTCODE
      path before any other N2 code is committed.
- [ ] All built-in continuous systems lower cleanly to IR
      (`lower_ode_to_ir` doesn't raise on any of them).
- [ ] For each non-chaotic built-in system: Rust trajectory matches its
      golden file with `rtol=1e-8`, `atol=1e-10`.
- [ ] For each chaotic built-in system: statistical observables (mean,
      variance, leading Lyapunov exponent, 2-D histogram on the first
      two state components) match the JiTCODE reference within their
      empirical 5σ band over 10 random ICs.
- [ ] Every method in the catalogue (DP5, DP8, Tsit5, Vern6/7/8/9, BS3,
      RK4, Rosenbrock23/34, Rodas4) integrates Lorenz on `[0, 10]` and
      matches DP8 with `rtol=1e-12` (or, for the lower-order methods,
      within their nominal accuracy band).
- [ ] `_compile_ode_ir` caches per class; second integrate on the same
      class triggers zero IR work (verifiable via instrumentation
      counter in tests).
- [ ] `ContinuousSystem.integrate(...)` API unchanged.  Existing tests in
      `tests/test_ode_systems.py` pass without modification.
- [ ] Wall-clock benchmark recorded in `bench/RESULTS.md`.  Target: not
      slower than 1.5× JiTCODE on the standard suite (interpreter is the
      bottleneck; N4 closes the rest).
- [ ] Method-name backward compat: `"RK45"`, `"dopri5"`, `"DOP853"` alias
      to `"DP5"` / `"DP8"`.  `"LSODA"` / `"VODE"` print a deprecation
      warning suggesting `"Rosenbrock23"`.
- [ ] `uv run pytest --no-cov` passes (full suite, including slow).
- [ ] `uv run ruff check src/ tests/` clean.
- [ ] CHANGELOG entry, STATUS.md update, ROADMAP.md tick.

## Out of scope

- **Variational equations / Lyapunov in Rust** — that's N3.
  `ContinuousSystem.lyapunov_spectrum` continues to call `jitcode_lyap`
  during N2's lifetime.
- **Cranelift JIT** — that's N4.  N2's RHS evaluator is the IR
  interpreter; the per-step interpretation overhead is the known
  performance cost.
- **DDE** — N5.
- **Event detection retrofit** — keep M2 on its current cubic-Hermite
  path; the Rust dense output is exposed but not consumed by M2 in N2.
  Follow-up milestone.
- **Implicit Runge-Kutta** (Radau, RadauIIA5), **BDF / NDF multistep**,
  **exponential integrators** — none are needed by built-in systems.
  Add when a user case demands.
- **State-dependent tolerance** (per-component `atol`).  Scalar tolerances
  for N2.

## Open questions for the user (ask before writing code)

1. **Method default**: today the default is `"RK45"` (i.e. DP5).  Keep
   that, or switch to `"DP8"` (which is the modern recommendation for
   smooth non-stiff problems)?  My recommendation: keep `"DP5"` to
   minimise surprise — but flag the recommendation in `integrate`'s
   docstring.

2. **`_compile_ode_ir` cache invalidation**: structural params drive
   loop length in variable-dim systems (Lorenz-96, KS, MultiChua).  N1
   already handles this for maps via `_structural_params`; we mirror
   it.  Confirm the cache key shape:
   `(class, dim, hash(structural_params))`.

3. **Fallback during transition**: should `_compile_ode_ir` catch
   `NotLowerableError` and fall back to JiTCODE silently, or raise an
   explicit error with a "report an unsupported op" pointer?  My
   recommendation: silent fallback for N2's lifetime (so any user's
   exotic `_equations` keeps working), explicit error after N3 ties off
   the JiTCODE deprecation.

4. **`ode_solvers` integration depth**: do we vendor the *coefficient
   tables* from `ode_solvers` (with attribution) and write our own
   `Stepper impl`, or do we depend on the crate at the Cargo level and
   wrap its `System` trait with an adapter?  My recommendation: vendor
   the tables.  Less coupling, easier to evolve our trait shape.

5. **PI controller default**: I-controller is fine for non-stiff
   problems but PI is materially better for stiff ones.  Tie the choice
   to the method (Rosenbrock methods default to PI; explicit RKs default
   to I)?  My recommendation: yes — method picks its own controller,
   user can override via an integrator kwarg.

6. **Output sampling near `t_final`**: JiTCODE-stepped trajectories
   currently include `t_final` exactly.  Our integrator's last internal
   step usually overshoots; we sample down via dense output.  Confirm
   we want `np.arange(t0, final_time + dt/2, dt)` (include `t_final`)
   rather than `np.arange(t0, final_time, dt)` (may exclude it).
   Recommendation: include `t_final` exactly.

## How to split this milestone across chats

Suggested split if N2 is too big for one session (likely):

- **N2.a — IR extension + lowering**: extend `_ir.py` / `_ode_lowering.py`,
  generate golden files, write `tests/test_native_ode.py` skeleton.  No
  Rust stepper yet — RHS is callable from Python via PyO3 so we can unit
  test the IR end-to-end.
- **N2.b — Explicit RK family**: ship `tsdyn-ode` crate with DP5, DP8,
  Tsit5, Vern9, RK4, BS3.  Adaptive control + dense output.  Lorenz /
  Rössler / Van der Pol pass goldens; non-stiff catalogue regression.
- **N2.c — Stiff Rosenbrock family**: Rosenbrock23, Rosenbrock34, Rodas4.
  Stiff catalogue (Van der Pol at large μ, Brusselator, others)
  regression.  Methods table + deprecation aliases.
- **N2.d — Polish + scalability groundwork**: benchmarks; docstrings; CHANGELOG;
  STATUS.md rewrite; **default-flip readiness**.  Optionally include **bounded,
  behaviour-neutral Rust organisation** in `crates/tsdyn-ode/` (module boundaries,
  dispatch/registrations grouped so **adding another solver stays a localized
  change**: tableau/step routine + enum/`match` arm + tests + docs row — see
  [`design/native-solver-migration.md`](../design/native-solver-migration.md)
  § Extensibility).  Avoid turning N2.d into an open-ended rewrite; cap scope in
  STATUS.md before starting.

Each sub-chat updates STATUS.md before closing so the next one can pick
up cleanly.  N2.a is the only one with a hard prerequisite.

## After N2 lands

- N3 (variational ODE Lyapunov) becomes unblocked.
- M2's `detect_events` can be retrofitted to use the Rust dense output —
  parked as a "M2 follow-up" today; promote to a small milestone.
- N4 (cranelift JIT) becomes interesting — the IR interpreter is the
  measured bottleneck and N4 swaps it for native code.
