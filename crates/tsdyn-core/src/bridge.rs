//! The pure-Rust engine bridge — everything `tsdynamics._rust` does, expressed
//! over plain Rust types so it is unit-testable with `cargo test` (no Python).
//!
//! The thin PyO3 layer in [`crate`] (`lib.rs`) only marshals NumPy arrays to/from
//! slices, releases the GIL, and forwards to the functions here. Keeping the
//! numerics Python-free means a transcription error in the integrate dispatch or
//! the solver/tolerance resolution fails a fast Rust unit test rather than only a
//! slower end-to-end pytest.
//!
//! What lives here:
//!
//! - [`VmEvaluator`] — adapts E1's [`Interpreter`] to the engine's object-safe
//!   [`Evaluator`] trait. The orphan rule forbids `impl Evaluator for Interpreter`
//!   in this crate (both are foreign), so this newtype bridges them — exactly the
//!   pattern `tsdyn-engine`'s own test kit uses.
//! - [`build_tape`] — the FFI ingestion: raw wire arrays → a validated
//!   [`Tape`](tsdyn_ir::Tape).
//! - [`resolve_solver`] / [`build_solver`] — turn a `method=` string into a boxed
//!   [`Solver`], honouring the user's `rtol`/`atol` for the built-in adaptive
//!   kernels and falling back to the registry's default-tolerance factory for the
//!   rest (so out-of-tree plugin kernels stay selectable).
//! - [`eval_rhs`] / [`eval_jac`] / [`iterate_map`] / [`integrate_dense`] /
//!   [`ensemble_final`] — the five engine entry points the Python
//!   `tsdynamics.engine.run` seam calls.
//!
//! Errors surface as [`EngineError`], a Python-free enum the binding layer maps
//! onto the right Python exception type.

use tsdyn_engine::{
    ensemble_final as engine_ensemble, integrate_dde_grid, integrate_grid, iterate_dense,
    DelaySlot, IntegrateConfig, IntegrateError, MapError,
};
use tsdyn_ir::{Evaluator, Tape};
use tsdyn_solvers::explicit::{Dop853, Rk45, Tsit5};
use tsdyn_solvers::implicit::{RosenbrockW, TrBdf2};
use tsdyn_solvers::{Solver, SolverKind};
use tsdyn_vm::Interpreter;

/// Why an engine call could not be served.
///
/// Python-free so the numeric core stays testable without pyo3; the binding
/// layer ([`crate::to_py_err`]) maps each variant to the matching Python
/// exception (`ValueError` / `NotImplementedError` / `RuntimeError`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EngineError {
    /// The wire arrays do not form a well-formed tape (the [`tsdyn_ir::IrError`]
    /// message). → `ValueError`.
    BadTape(String),
    /// A buffer length disagrees with the tape (state/param width, ODE shape). →
    /// `ValueError`.
    BadShape(String),
    /// No solver kernel is registered under the requested `method=` name. →
    /// `ValueError`.
    UnknownMethod(String),
    /// A capability that is wired in the Python seam but not yet built in the
    /// engine (the Cranelift JIT; DDE/SDE engine integration). →
    /// `NotImplementedError`.
    Unsupported(String),
    /// The trajectory diverged or the step collapsed before the target time (the
    /// "diverge loudly" contract). → `RuntimeError`.
    Diverged(String),
}

impl core::fmt::Display for EngineError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EngineError::BadTape(m)
            | EngineError::BadShape(m)
            | EngineError::UnknownMethod(m)
            | EngineError::Unsupported(m)
            | EngineError::Diverged(m) => f.write_str(m),
        }
    }
}

// ---------------------------------------------------------------------------
// Evaluator adapter
// ---------------------------------------------------------------------------

/// Adapts the interpreter ([`Interpreter`]) to the engine's [`Evaluator`] trait.
///
/// `Interpreter` exposes the right inherent methods but does not itself
/// `impl Evaluator` (the orphan rule puts that forward in `tsdyn-vm` or
/// `tsdyn-ir`, not here); this newtype bridges it so the integrate loop can drive
/// it as `&dyn Evaluator`. It is `Sync` because `Interpreter` is, which is what
/// lets one evaluator be shared across the ensemble's rayon workers.
pub struct VmEvaluator {
    interp: Interpreter,
}

impl VmEvaluator {
    /// Build an evaluator over a validated tape.
    pub fn new(tape: Tape) -> Self {
        VmEvaluator {
            interp: Interpreter::new(tape),
        }
    }
}

impl Evaluator for VmEvaluator {
    fn dim(&self) -> usize {
        self.interp.dim()
    }
    fn n_param(&self) -> usize {
        self.interp.n_param()
    }
    fn n_scratch(&self) -> usize {
        self.interp.n_scratch()
    }
    fn has_jacobian(&self) -> bool {
        self.interp.has_jacobian()
    }
    fn eval(&self, u: &[f64], p: &[f64], t: f64, scratch: &mut [f64], deriv: &mut [f64]) {
        self.interp.eval(u, p, t, scratch, deriv);
    }
    fn eval_jac(
        &self,
        u: &[f64],
        p: &[f64],
        t: f64,
        scratch: &mut [f64],
        deriv: &mut [f64],
        jac: &mut [f64],
    ) {
        self.interp.eval_jac(u, p, t, scratch, deriv, jac);
    }
}

// ---------------------------------------------------------------------------
// Tape ingestion
// ---------------------------------------------------------------------------

/// Build a validated [`Tape`] from the raw wire arrays (the FFI ingestion path).
///
/// Mirrors the argument order of the Python `Tape.to_arrays()`:
/// `(ops, a, b, imm, outputs, jac_outputs, n_state, n_param)`. A malformed tape
/// is rejected here, at the boundary, as [`EngineError::BadTape`].
#[allow(clippy::too_many_arguments)] // the eight arrays are the tape contract.
pub fn build_tape(
    ops: &[i32],
    a: &[i32],
    b: &[i32],
    imm: &[f64],
    outputs: &[i32],
    jac_outputs: &[i32],
    n_state: usize,
    n_param: usize,
) -> Result<Tape, EngineError> {
    Tape::from_arrays(ops, a, b, imm, outputs, jac_outputs, n_state, n_param)
        .map_err(|e| EngineError::BadTape(e.to_string()))
}

// ---------------------------------------------------------------------------
// Solver resolution
// ---------------------------------------------------------------------------

/// Resolve a `method=` string to a registered solver name, case-insensitively.
///
/// Tries the exact name first, then its lowercase form (so the SciPy-style
/// `"RK45"`/`"DOP853"` spellings reach the `"rk45"`/`"dop853"` kernels). Richer
/// alias resolution is the Python `solvers` layer's job (stream C-SOLV); this is
/// the minimal normalisation that makes the common names work over the FFI.
pub fn resolve_solver(method: &str) -> Result<&'static str, EngineError> {
    if let Some(reg) = tsdyn_solvers::find(method) {
        return Ok(reg.name);
    }
    let lower = method.to_lowercase();
    if let Some(reg) = tsdyn_solvers::find(&lower) {
        return Ok(reg.name);
    }
    Err(EngineError::UnknownMethod(format!(
        "unknown method {method:?}; available: {:?}",
        tsdyn_solvers::available()
    )))
}

/// Build a fresh boxed solver for a **registered** name, applying the user's
/// tolerances where the kernel supports them.
///
/// `name` must already be a registry name (from [`resolve_solver`]). Every
/// built-in *adaptive* kernel — the explicit family (`rk45`/`tsit5`/`dop853`) and
/// the implicit family (`rosenbrock`/`trbdf2`) — owns its `rtol`/`atol` (the
/// frozen `Solver::step` carries none), so each is constructed through its
/// `with_tolerances` constructor here to honour the requested accuracy. The
/// fixed-step `rk4` (no tolerances) and any out-of-tree plugin kernel build
/// through the registry's default-tolerance factory; threading tolerances onto a
/// plugin generically is the `solvers` layer's job (stream C-SOLV). This keeps the
/// open registry intact (a new kernel is still selectable) while no built-in
/// adaptive method silently ignores the user's tolerances.
pub fn build_solver(name: &'static str, rtol: f64, atol: f64) -> Box<dyn Solver> {
    match name {
        // Every built-in *adaptive* kernel — explicit and implicit — owns its
        // error tolerances, so build it through `with_tolerances` to honour the
        // user's request.
        "rk45" => Box::new(Rk45::with_tolerances(rtol, atol)),
        "tsit5" => Box::new(Tsit5::with_tolerances(rtol, atol)),
        "dop853" => Box::new(Dop853::with_tolerances(rtol, atol)),
        "rosenbrock" => Box::new(RosenbrockW::with_tolerances(rtol, atol)),
        "trbdf2" => Box::new(TrBdf2::with_tolerances(rtol, atol)),
        // The fixed-step `rk4` (no tolerances) and any out-of-tree plugin kernel
        // build through the registry factory. `name` came from the registry
        // (via `resolve_solver`), so `make` is guaranteed `Some`.
        other => tsdyn_solvers::make(other)
            .expect("a name returned by resolve_solver is always registered"),
    }
}

// ---------------------------------------------------------------------------
// Pointwise evaluation
// ---------------------------------------------------------------------------

/// Evaluate `du/dt` (or the next-state, for a map tape) once at `(u, p, t)`.
pub fn eval_rhs(tape: Tape, u: &[f64], p: &[f64], t: f64) -> Result<Vec<f64>, EngineError> {
    check_inputs(&tape, u, p)?;
    Ok(Interpreter::new(tape).eval_alloc(u, p, t))
}

/// Evaluate `(du/dt, Jacobian)` once; the Jacobian is row-major `dim * dim`.
pub fn eval_jac(
    tape: Tape,
    u: &[f64],
    p: &[f64],
    t: f64,
) -> Result<(Vec<f64>, Vec<f64>), EngineError> {
    if !tape.has_jacobian() {
        return Err(EngineError::BadShape(
            "eval_jac requires a tape compiled with a Jacobian (with_jacobian=True)".to_string(),
        ));
    }
    check_inputs(&tape, u, p)?;
    Ok(Interpreter::new(tape).eval_jac_alloc(u, p, t))
}

/// Validate that the state/param slices match the tape's declared widths.
fn check_inputs(tape: &Tape, u: &[f64], p: &[f64]) -> Result<(), EngineError> {
    if u.len() < tape.n_state() {
        return Err(EngineError::BadShape(format!(
            "state vector has length {}, need n_state = {}",
            u.len(),
            tape.n_state()
        )));
    }
    if p.len() < tape.n_param() {
        return Err(EngineError::BadShape(format!(
            "parameter vector has length {}, need n_param = {}",
            p.len(),
            tape.n_param()
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Map iteration
// ---------------------------------------------------------------------------

/// Iterate a discrete-map tape for `steps` steps from `ic`, returning the state
/// after each step as a flat row-major `(steps, dim)` buffer.
///
/// Row `i` is the state after `i + 1` applications of the map; the initial
/// condition itself is **not** included (matching the Python reference map loop).
/// Map parameters are folded into the tape as constants (`n_param == 0`), so no
/// parameter vector is threaded. A blown-up orbit raises
/// [`EngineError::Diverged`] at the first non-finite iterate — the engine's
/// "diverge loudly" contract — rather than returning a silently poisoned buffer.
///
/// The iteration itself is the shared engine map loop
/// ([`tsdyn_engine::iterate_dense`]); this wrapper only adds the boundary shape
/// checks the FFI seam needs, since the engine loop validates shapes with
/// `debug_assert!` (compiled out in release).
pub fn iterate_map(tape: Tape, ic: &[f64], steps: usize) -> Result<Vec<f64>, EngineError> {
    let dim = tape.dim();
    // A map's next-state dimension equals its state dimension, and the lowering
    // folds parameters into constants, so the iteration feeds a `dim`-length state
    // straight back in with no parameter vector. Reject any tape that disagrees
    // (reachable only via a hand-built tape, not the Python seam) so a shape
    // mismatch is a clean error rather than a release-mode out-of-bounds panic
    // inside the eval loop — matching the validation the other entry points do.
    if tape.n_param() != 0 {
        return Err(EngineError::BadShape(format!(
            "map iteration expects a tape with n_param == 0 (map parameters fold to \
             constants); got n_param = {}",
            tape.n_param()
        )));
    }
    if tape.n_state() != dim {
        return Err(EngineError::BadShape(format!(
            "map iteration expects n_state == dim (state and next-state share a \
             dimension); got n_state = {} != dim = {dim}",
            tape.n_state()
        )));
    }
    if ic.len() < dim {
        return Err(EngineError::BadShape(format!(
            "initial state has length {}, need dim = {dim}",
            ic.len()
        )));
    }
    // Delegate to the shared engine map loop: one iteration implementation, one
    // divergence contract. Parameters are folded into the tape (`n_param == 0`),
    // so the parameter slice is empty; the guards above stand in for the engine
    // loop's release-stripped `debug_assert!`s.
    let ev = VmEvaluator::new(tape);
    iterate_dense(&ev, &ic[..dim], &[], steps)
        .map_err(|e| EngineError::Diverged(map_diverge_msg(&e)))
}

// ---------------------------------------------------------------------------
// Integration
// ---------------------------------------------------------------------------

/// Reject the families the engine cannot integrate yet, and the JIT backend
/// until stream E2 lands. A DDE tape carries delay-slot inputs beyond the real
/// state (`n_state > dim`); only ODE-shaped tapes (`n_state == dim`) integrate
/// here today.
fn guard_continuous(tape: &Tape, jit: bool) -> Result<(), EngineError> {
    if jit {
        return Err(EngineError::Unsupported(
            "the JIT evaluator (tsdyn-jit, stream E2) is not built yet; use backend='interp'"
                .to_string(),
        ));
    }
    if tape.n_state() != tape.dim() {
        return Err(EngineError::Unsupported(format!(
            "engine integration currently supports ODE-shaped tapes (n_state == dim); got \
             n_state = {} > dim = {} (a DDE/SDE tape). DDE/SDE integration lands with streams \
             E-DDE / E-SDE.",
            tape.n_state(),
            tape.dim()
        )));
    }
    Ok(())
}

/// Reject a method that needs an analytic Jacobian when the tape carries none.
///
/// The implicit kernels (`rosenbrock`/`trbdf2`) freeze `∂f/∂u` each step and
/// solve `(I − c·h·J)`. Driving them over a tape compiled *without* a Jacobian
/// (`with_jacobian=False`) leaves `eval_jac`'s `jac` buffer untouched — i.e. the
/// all-zeros it was initialised to — so the iteration matrix collapses to `I` and
/// the L-stable step silently degrades to an *unstable forward-Euler* one: a
/// plausible-but-wrong trajectory with no error raised. Reject it loudly here, at
/// the boundary, mirroring the DDE path's explicit-method guard. The implicit
/// kernels also self-guard (defence in depth), but this gives the actionable
/// message (recompile with a Jacobian, or pick an explicit method).
fn require_jacobian_if_needed(tape: &Tape, name: &'static str) -> Result<(), EngineError> {
    let reg = tsdyn_solvers::find(name).expect("resolve_solver returns a registered name");
    if reg.caps.needs_jacobian && !tape.has_jacobian() {
        return Err(EngineError::BadShape(format!(
            "method {name:?} needs an analytic Jacobian, but the tape was compiled without one \
             (with_jacobian=False). Recompile the problem with a Jacobian, or choose an explicit \
             method such as 'rk45', 'tsit5', 'dop853', or 'rk4'."
        )));
    }
    Ok(())
}

/// First trial step from an output grid: the first positive gap, else a small
/// default. For an adaptive kernel this is only the first attempt (it adapts);
/// for the fixed-step `rk4` it is the step itself.
fn first_step_from_grid(t_eval: &[f64]) -> f64 {
    t_eval
        .windows(2)
        .map(|w| w[1] - w[0])
        .find(|&d| d > 0.0)
        .unwrap_or(1e-3)
}

/// Validate an output grid: every time finite and the grid non-decreasing.
///
/// The engine enforces monotonicity only with a `debug_assert!` (stripped from
/// the shipped release wheel) and treats a non-finite first step as a
/// hard-`assert!` caller error. Without this check a descending grid would
/// silently return stale rows in release, and a non-finite time (e.g. an `inf`
/// grid gap) would trip the engine assert into a `PanicException`. Validating
/// here turns both into a clean `EngineError::BadShape` (→ `ValueError`).
fn validate_grid(t_eval: &[f64]) -> Result<(), EngineError> {
    if let Some(&t) = t_eval.iter().find(|t| !t.is_finite()) {
        return Err(EngineError::BadShape(format!(
            "t_eval must be all finite; found {t}"
        )));
    }
    for (i, w) in t_eval.windows(2).enumerate() {
        if w[1] < w[0] {
            return Err(EngineError::BadShape(format!(
                "t_eval must be non-decreasing; t_eval[{}] = {} < t_eval[{i}] = {}",
                i + 1,
                w[1],
                w[0]
            )));
        }
    }
    Ok(())
}

/// Integrate a single trajectory, sampling at every `t_eval` point, into a flat
/// row-major `(t_eval.len(), dim)` buffer (the first row is `ic`).
///
/// Divergence raises ([`EngineError::Diverged`]) rather than returning plausible
/// numbers — the single-trajectory analogue of the ensemble's NaN row.
#[allow(clippy::too_many_arguments)]
pub fn integrate_dense(
    tape: Tape,
    ic: &[f64],
    p: &[f64],
    t_eval: &[f64],
    method: &str,
    rtol: f64,
    atol: f64,
    jit: bool,
) -> Result<Vec<f64>, EngineError> {
    guard_continuous(&tape, jit)?;
    check_inputs(&tape, ic, p)?;
    if tape.dim() == 0 {
        return Err(EngineError::BadShape(
            "system dimension is zero (the tape has no outputs)".to_string(),
        ));
    }
    validate_grid(t_eval)?;
    let name = resolve_solver(method)?;
    require_jacobian_if_needed(&tape, name)?;
    let ev = VmEvaluator::new(tape);
    let mut solver = build_solver(name, rtol, atol);
    let cfg = IntegrateConfig::new(first_step_from_grid(t_eval));
    integrate_grid(&ev, &mut *solver, &ic[..ev.dim()], p, t_eval, &cfg)
        .map_err(|e| EngineError::Diverged(diverge_msg(&e)))
}

/// Integrate a delay differential equation through the output grid `t_eval`,
/// returning a flat row-major `(t_eval.len(), dim)` buffer (the first row is `ic`).
///
/// `tape` is the lowered DDE right-hand side over `dim + n_slots` inputs — the
/// extra inputs are the delay slots, and delay-system parameters are folded into
/// the tape as constants, so `n_param == 0`. `slot_components` / `slot_delays`
/// (each length `n_slots = n_state − dim`) describe slot `k`: the true-state
/// component it samples and its positive, constant delay. `past_t` / `past_y` are
/// the user-supplied past on `[t0 − max_delay, t0]` — ascending times and a flat
/// `(n_past, dim)` value buffer; a single sample is a constant past. Only
/// **explicit** kernels are supported (the method of steps treats each delayed
/// term as known history, so the step is an explicit ODE step; an implicit kernel
/// would need a delayed Jacobian). Divergence raises ([`EngineError::Diverged`]).
#[allow(clippy::too_many_arguments)]
pub fn integrate_dde_dense(
    tape: Tape,
    slot_components: &[i32],
    slot_delays: &[f64],
    ic: &[f64],
    past_t: &[f64],
    past_y: &[f64],
    t_eval: &[f64],
    method: &str,
    rtol: f64,
    atol: f64,
    jit: bool,
) -> Result<Vec<f64>, EngineError> {
    if jit {
        return Err(EngineError::Unsupported(
            "the JIT evaluator (tsdyn-jit, stream E2) is not built yet; use backend='interp'"
                .to_string(),
        ));
    }
    let dim = tape.dim();
    let n_slots = slot_components.len();
    // The DDE lowering's shape contract: n_state = dim + n_slots, parameters folded.
    if slot_delays.len() != n_slots {
        return Err(EngineError::BadShape(format!(
            "slot_components ({n_slots}) and slot_delays ({}) must have equal length",
            slot_delays.len()
        )));
    }
    if tape.n_param() != 0 {
        return Err(EngineError::BadShape(format!(
            "DDE integration expects a tape with n_param == 0 (delay-system parameters fold to \
             constants); got n_param = {}",
            tape.n_param()
        )));
    }
    if tape.n_state() != dim + n_slots {
        return Err(EngineError::BadShape(format!(
            "DDE tape has n_state = {} but dim = {dim} and {n_slots} delay slots (expected \
             n_state = dim + n_slots = {})",
            tape.n_state(),
            dim + n_slots
        )));
    }
    if ic.len() < dim {
        return Err(EngineError::BadShape(format!(
            "initial state has length {}, need dim = {dim}",
            ic.len()
        )));
    }
    // The past buffer: at least one sample, finite ascending times, (n_past, dim).
    let n_past = past_t.len();
    if n_past == 0 {
        return Err(EngineError::BadShape(
            "past history must have at least one sample (the constant-past case)".to_string(),
        ));
    }
    if past_y.len() != n_past * dim {
        return Err(EngineError::BadShape(format!(
            "past_y length {} != n_past ({n_past}) * dim ({dim})",
            past_y.len()
        )));
    }
    if past_t.iter().any(|t| !t.is_finite()) {
        return Err(EngineError::BadShape(
            "past_t must be all finite".to_string(),
        ));
    }
    for w in past_t.windows(2) {
        if w[1] < w[0] {
            return Err(EngineError::BadShape(
                "past_t must be non-decreasing".to_string(),
            ));
        }
    }
    // Build the delay slots, validating each component index and delay magnitude.
    let mut slots = Vec::with_capacity(n_slots);
    for (k, (&c, &d)) in slot_components.iter().zip(slot_delays.iter()).enumerate() {
        if c < 0 || (c as usize) >= dim {
            return Err(EngineError::BadShape(format!(
                "delay slot {k} references component {c}, outside 0..{dim}"
            )));
        }
        if d <= 0.0 || !d.is_finite() {
            return Err(EngineError::BadShape(format!(
                "delay slot {k} has a non-positive or non-finite delay {d}"
            )));
        }
        slots.push(DelaySlot {
            component: c as usize,
            delay: d,
        });
    }
    validate_grid(t_eval)?;
    // The method of steps drives explicit kernels only (no delayed Jacobian here).
    let name = resolve_solver(method)?;
    let reg = tsdyn_solvers::find(name).expect("resolve_solver returns a registered name");
    if reg.caps.kind == SolverKind::Implicit || reg.caps.needs_jacobian {
        return Err(EngineError::Unsupported(format!(
            "DDE integration supports explicit methods only; {name:?} is implicit. Use an \
             explicit method such as 'rk45', 'tsit5', 'dop853', or 'rk4'."
        )));
    }
    let ev = VmEvaluator::new(tape);
    let mut solver = build_solver(name, rtol, atol);
    let cfg = IntegrateConfig::new(first_step_from_grid(t_eval));
    integrate_dde_grid(
        &ev,
        &mut *solver,
        dim,
        &slots,
        &ic[..dim],
        past_t,
        past_y,
        t_eval,
        &cfg,
    )
    .map_err(|e| EngineError::Diverged(diverge_msg(&e)))
}

/// Integrate a batch of initial conditions to `t1` in parallel, returning each
/// final state as a flat row-major `(n_ic, dim)` buffer.
///
/// `ics` is a row-major `(n_ic, dim)` buffer. A diverging trajectory yields a
/// `NaN` row rather than aborting the batch (the engine's ensemble contract); the
/// run is seeded/scheduling-independent, so results are bit-for-bit reproducible.
#[allow(clippy::too_many_arguments)]
pub fn ensemble_final(
    tape: Tape,
    ics: &[f64],
    p: &[f64],
    t0: f64,
    t1: f64,
    method: &str,
    rtol: f64,
    atol: f64,
    jit: bool,
) -> Result<Vec<f64>, EngineError> {
    guard_continuous(&tape, jit)?;
    if p.len() < tape.n_param() {
        return Err(EngineError::BadShape(format!(
            "parameter vector has length {}, need n_param = {}",
            p.len(),
            tape.n_param()
        )));
    }
    // Finite times keep the derived first step finite — the engine treats a
    // non-finite first step as a hard-`assert!` caller error, which would surface
    // as a `PanicException` rather than a clean exception.
    if !(t0.is_finite() && t1.is_finite()) {
        return Err(EngineError::BadShape(format!(
            "integration times must be finite; got t0 = {t0}, t1 = {t1}"
        )));
    }
    // Backward integration is not supported yet: the engine loop runs while
    // `t < t1`, so a request with `t1 < t0` would never step and silently return
    // the unchanged initial conditions as a successful batch. Reject it loudly
    // until a backward-time path lands, rather than return stale ICs as `Ok`.
    if t1 < t0 {
        return Err(EngineError::BadShape(format!(
            "backward integration is not supported (t1 = {t1} < t0 = {t0}); the engine integrates \
             forward in time. Request t1 >= t0."
        )));
    }
    let dim = tape.dim();
    // The engine's ensemble path hard-asserts a positive dimension; guard it here
    // so a zero-output tape is a clean error, not a panic.
    if dim == 0 {
        return Err(EngineError::BadShape(
            "system dimension is zero (the tape has no outputs)".to_string(),
        ));
    }
    if !ics.len().is_multiple_of(dim) {
        return Err(EngineError::BadShape(format!(
            "ensemble initial-condition buffer length {} is not a multiple of dim = {dim}",
            ics.len()
        )));
    }
    // Validate the method up front (one clear error before the rayon fan-out).
    let name = resolve_solver(method)?;
    require_jacobian_if_needed(&tape, name)?;
    let span = t1 - t0; // non-negative: the backward case is rejected above
    let first_step = if span > 0.0 { span / 100.0 } else { 1.0 };
    let cfg = IntegrateConfig::new(first_step);
    let ev = VmEvaluator::new(tape);
    let result = engine_ensemble(
        &ev,
        |_i| build_solver(name, rtol, atol),
        ics,
        p,
        t0,
        t1,
        &cfg,
    );
    Ok(result.states)
}

/// Prefix the engine's diverge message so the Python `RuntimeError` reads clearly.
fn diverge_msg(e: &IntegrateError) -> String {
    format!("integration diverged before reaching the final time: {e}")
}

/// Prefix the engine's map-divergence message, mirroring [`diverge_msg`].
fn map_diverge_msg(e: &MapError) -> String {
    format!("map diverged before completing all iterations: {e}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tsdyn_ir::TapeBuilder;

    /// dx/dt = -k x ⇒ x(t) = x0 e^{-k t}; one parameter, one state. No Jacobian
    /// (jac_outputs empty), so `has_jacobian()` is false — the case an implicit
    /// kernel must refuse.
    fn decay_tape() -> Tape {
        let mut b = TapeBuilder::new();
        let k = b.param(0);
        let x = b.state(0);
        let kx = b.mul(k, x);
        let dx = b.neg(kx);
        b.finish(&[dx], &[], 1, 1).unwrap()
    }

    /// The same decay system carrying its analytic Jacobian ∂(dx)/∂x = −k, so the
    /// implicit kernels have the `∂f/∂u` they require.
    fn decay_tape_jac() -> Tape {
        let mut b = TapeBuilder::new();
        let k = b.param(0);
        let x = b.state(0);
        let kx = b.mul(k, x);
        let dx = b.neg(kx);
        let neg_k = b.neg(k);
        b.finish(&[dx], &[neg_k], 1, 1).unwrap()
    }

    /// Undamped oscillator dx=v, dv=-x ⇒ (cos t, -sin t) from (1, 0).
    fn oscillator_tape() -> Tape {
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let v = b.state(1);
        let dv = b.neg(x);
        b.finish(&[v, dv], &[], 2, 0).unwrap()
    }

    /// dx/dt = x² ⇒ finite-time blow-up at t = 1 from x0 = 1.
    fn blowup_tape() -> Tape {
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let dx = b.mul(x, x);
        b.finish(&[dx], &[], 1, 0).unwrap()
    }

    /// Lorenz with its analytic Jacobian (for eval_jac).
    fn lorenz_tape() -> Tape {
        let mut b = TapeBuilder::new();
        let (sg, rho, bt) = (b.param(0), b.param(1), b.param(2));
        let (x, y, z) = (b.state(0), b.state(1), b.state(2));
        let ymx = b.sub(y, x);
        let dx = b.mul(sg, ymx);
        let rmz = b.sub(rho, z);
        let xrmz = b.mul(x, rmz);
        let dy = b.sub(xrmz, y);
        let xy = b.mul(x, y);
        let bz = b.mul(bt, z);
        let dz = b.sub(xy, bz);
        let neg_sg = b.neg(sg);
        let zero = b.constant(0.0);
        let neg_one = b.constant(-1.0);
        let neg_x = b.neg(x);
        let neg_bt = b.neg(bt);
        b.finish(
            &[dx, dy, dz],
            &[neg_sg, sg, zero, rmz, neg_one, neg_x, y, x, neg_bt],
            3,
            3,
        )
        .unwrap()
    }

    /// Hénon map next-state: x' = 1 - a x² + y, y' = b x, with a=1.4, b=0.3
    /// folded in as constants (n_param = 0, the map lowering convention).
    fn henon_tape() -> Tape {
        let mut bld = TapeBuilder::new();
        let x = bld.state(0);
        let y = bld.state(1);
        let one = bld.constant(1.0);
        let a = bld.constant(1.4);
        let bb = bld.constant(0.3);
        let x2 = bld.mul(x, x);
        let ax2 = bld.mul(a, x2);
        let one_m = bld.sub(one, ax2);
        let nx = bld.add(one_m, y);
        let ny = bld.mul(bb, x);
        bld.finish(&[nx, ny], &[], 2, 0).unwrap()
    }

    #[test]
    fn eval_rhs_matches_hand_value() {
        let got = eval_rhs(decay_tape(), &[2.0], &[3.0], 0.0).unwrap();
        assert!((got[0] - (-6.0)).abs() < 1e-15);
    }

    #[test]
    fn eval_rhs_rejects_short_state() {
        let err = eval_rhs(lorenz_tape(), &[1.0, 2.0], &[10.0, 28.0, 2.0], 0.0).unwrap_err();
        assert!(matches!(err, EngineError::BadShape(_)));
    }

    #[test]
    fn eval_jac_returns_rhs_and_jacobian() {
        let p = [10.0, 28.0, 8.0 / 3.0];
        let (d, j) = eval_jac(lorenz_tape(), &[1.0, 2.0, 3.0], &p, 0.0).unwrap();
        assert_eq!(d.len(), 3);
        assert_eq!(j.len(), 9);
        // Row 0 of J is [-sigma, sigma, 0].
        assert!((j[0] - (-10.0)).abs() < 1e-15);
        assert!((j[1] - 10.0).abs() < 1e-15);
        assert!((j[2]).abs() < 1e-15);
    }

    #[test]
    fn eval_jac_without_jacobian_is_an_error() {
        let err = eval_jac(decay_tape(), &[1.0], &[1.0], 0.0).unwrap_err();
        assert!(matches!(err, EngineError::BadShape(_)));
    }

    #[test]
    fn resolve_solver_is_case_insensitive() {
        assert_eq!(resolve_solver("rk45").unwrap(), "rk45");
        assert_eq!(resolve_solver("RK45").unwrap(), "rk45");
        assert_eq!(resolve_solver("DOP853").unwrap(), "dop853");
        assert!(matches!(
            resolve_solver("nope").unwrap_err(),
            EngineError::UnknownMethod(_)
        ));
    }

    #[test]
    fn build_solver_honours_tolerances_and_names() {
        // Smoke: every documented name builds a kernel reporting that name.
        for name in ["rk4", "rk45", "tsit5", "dop853", "rosenbrock", "trbdf2"] {
            let resolved = resolve_solver(name).unwrap();
            let s = build_solver(resolved, 1e-9, 1e-12);
            assert_eq!(s.name(), name);
        }
    }

    #[test]
    fn integrate_dense_matches_closed_form_decay() {
        let t_eval: Vec<f64> = (0..=10).map(|i| i as f64 * 0.3).collect();
        let y = integrate_dense(
            decay_tape(),
            &[1.0],
            &[2.0],
            &t_eval,
            "rk45",
            1e-10,
            1e-12,
            false,
        )
        .unwrap();
        assert_eq!(y.len(), t_eval.len());
        assert_eq!(y[0], 1.0); // first row is the IC
        for (i, &t) in t_eval.iter().enumerate() {
            let want = (-2.0 * t).exp();
            assert!((y[i] - want).abs() < 1e-7, "row {i}: {} vs {want}", y[i]);
        }
    }

    #[test]
    fn integrate_dense_tracks_oscillator() {
        let t_eval: Vec<f64> = (0..=8)
            .map(|i| i as f64 * core::f64::consts::FRAC_PI_4)
            .collect();
        let y = integrate_dense(
            oscillator_tape(),
            &[1.0, 0.0],
            &[],
            &t_eval,
            "rk45",
            1e-11,
            1e-13,
            false,
        )
        .unwrap();
        for (k, &t) in t_eval.iter().enumerate() {
            let (x, v) = (y[2 * k], y[2 * k + 1]);
            assert!((x - t.cos()).abs() < 1e-7, "x at t={t}: {x}");
            assert!((v + t.sin()).abs() < 1e-7, "v at t={t}: {v}");
        }
    }

    #[test]
    fn integrate_dense_reports_divergence() {
        let t_eval = [0.0, 0.5, 1.5, 2.0];
        let err = integrate_dense(
            blowup_tape(),
            &[1.0],
            &[],
            &t_eval,
            "rk45",
            1e-8,
            1e-10,
            false,
        )
        .unwrap_err();
        assert!(matches!(err, EngineError::Diverged(_)), "got {err:?}");
    }

    #[test]
    fn integrate_dense_rejects_jit_and_unknown_method() {
        let t_eval = [0.0, 1.0];
        assert!(matches!(
            integrate_dense(
                decay_tape(),
                &[1.0],
                &[1.0],
                &t_eval,
                "rk45",
                1e-6,
                1e-9,
                true
            )
            .unwrap_err(),
            EngineError::Unsupported(_)
        ));
        assert!(matches!(
            integrate_dense(
                decay_tape(),
                &[1.0],
                &[1.0],
                &t_eval,
                "no-such",
                1e-6,
                1e-9,
                false
            )
            .unwrap_err(),
            EngineError::UnknownMethod(_)
        ));
    }

    #[test]
    fn ensemble_matches_a_serial_dense_loop() {
        let ics: Vec<f64> = (0..16).map(|i| 0.5 + i as f64).collect();
        let states = ensemble_final(
            decay_tape(),
            &ics,
            &[2.0],
            0.0,
            1.5,
            "rk45",
            1e-10,
            1e-12,
            false,
        )
        .unwrap();
        assert_eq!(states.len(), ics.len());
        let factor = (-2.0_f64 * 1.5).exp();
        for (i, &x0) in ics.iter().enumerate() {
            assert!(
                (states[i] - x0 * factor).abs() < 1e-7,
                "traj {i}: {} vs {}",
                states[i],
                x0 * factor
            );
        }
    }

    #[test]
    fn ensemble_isolates_a_diverging_trajectory() {
        // x0 = 1 blows up before t = 2; x0 = -1 decays and stays finite.
        let states = ensemble_final(
            blowup_tape(),
            &[1.0, -1.0],
            &[],
            0.0,
            2.0,
            "rk45",
            1e-8,
            1e-10,
            false,
        )
        .unwrap();
        assert!(states[0].is_nan(), "diverged row should be NaN");
        assert!(states[1].is_finite());
    }

    #[test]
    fn ensemble_rejects_ragged_batch() {
        // 5 is not a multiple of dim = 2.
        let err = ensemble_final(
            oscillator_tape(),
            &[1.0, 0.0, 2.0, 0.0, 3.0],
            &[],
            0.0,
            1.0,
            "rk45",
            1e-6,
            1e-9,
            false,
        )
        .unwrap_err();
        assert!(matches!(err, EngineError::BadShape(_)));
    }

    #[test]
    fn iterate_map_reproduces_henon_orbit() {
        let out = iterate_map(henon_tape(), &[0.1, 0.1], 3).unwrap();
        assert_eq!(out.len(), 6);
        // Hand-roll three Hénon steps (a=1.4, b=0.3).
        let (mut x, mut y) = (0.1_f64, 0.1_f64);
        for i in 0..3 {
            let nx = 1.0 - 1.4 * x * x + y;
            let ny = 0.3 * x;
            x = nx;
            y = ny;
            assert!((out[2 * i] - x).abs() < 1e-14, "step {i} x");
            assert!((out[2 * i + 1] - y).abs() < 1e-14, "step {i} y");
        }
    }

    #[test]
    fn iterate_map_diverges_loudly() {
        // The Hénon map from a far-outside initial condition escapes to infinity;
        // delegation to the engine loop must raise at the first non-finite iterate
        // instead of returning a buffer of inf/NaN.
        let err = iterate_map(henon_tape(), &[10.0, 10.0], 200).unwrap_err();
        assert!(matches!(err, EngineError::Diverged(_)), "got {err:?}");
    }

    #[test]
    fn build_tape_rejects_unknown_opcode() {
        let err = build_tape(
            &[1, 2, 99],
            &[0, 0, 0],
            &[0, 0, 1],
            &[0.0; 3],
            &[2],
            &[],
            1,
            1,
        )
        .unwrap_err();
        assert!(matches!(err, EngineError::BadTape(_)));
    }

    #[test]
    fn implicit_kernels_honour_requested_tolerances() {
        // The build_solver fix: rosenbrock/trbdf2 must integrate at the requested
        // tolerance, not the kernel default — a tight run on the smooth decay must
        // hit the closed form to well under the loose 1e-3/1e-6 default accuracy.
        let t_eval: Vec<f64> = (0..=6).map(|i| i as f64 * 0.4).collect();
        for method in ["rosenbrock", "trbdf2"] {
            let y = integrate_dense(
                decay_tape_jac(),
                &[1.0],
                &[2.0],
                &t_eval,
                method,
                1e-10,
                1e-12,
                false,
            )
            .unwrap_or_else(|e| panic!("{method} failed: {e}"));
            for (i, &t) in t_eval.iter().enumerate() {
                let want = (-2.0 * t).exp();
                assert!(
                    (y[i] - want).abs() < 1e-7,
                    "{method} row {i}: {} vs {want}",
                    y[i]
                );
            }
        }
    }

    #[test]
    fn integrate_dense_rejects_non_finite_or_descending_grid() {
        // Non-finite grid time → clean BadShape (not a PanicException via the
        // engine's hard first-step assert).
        let bad_inf = integrate_dense(
            decay_tape(),
            &[1.0],
            &[2.0],
            &[0.0, f64::INFINITY],
            "rk45",
            1e-6,
            1e-9,
            false,
        )
        .unwrap_err();
        assert!(
            matches!(bad_inf, EngineError::BadShape(_)),
            "got {bad_inf:?}"
        );
        // Descending grid → clean BadShape (not a silently-stale row in release).
        let bad_desc = integrate_dense(
            decay_tape(),
            &[1.0],
            &[2.0],
            &[0.0, 1.0, 0.5],
            "rk45",
            1e-6,
            1e-9,
            false,
        )
        .unwrap_err();
        assert!(
            matches!(bad_desc, EngineError::BadShape(_)),
            "got {bad_desc:?}"
        );
    }

    #[test]
    fn ensemble_rejects_non_finite_times() {
        let err = ensemble_final(
            decay_tape(),
            &[1.0, 2.0],
            &[2.0],
            0.0,
            f64::INFINITY,
            "rk45",
            1e-6,
            1e-9,
            false,
        )
        .unwrap_err();
        assert!(matches!(err, EngineError::BadShape(_)), "got {err:?}");
    }

    #[test]
    fn iterate_map_rejects_mismatched_tape() {
        // A tape with a Param op (n_param > 0) is not a valid map tape; iteration
        // must reject it cleanly rather than index an empty parameter slice.
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let k = b.param(0);
        let nx = b.add(x, k);
        let tape = b.finish(&[nx], &[], 1, 1).unwrap();
        let err = iterate_map(tape, &[0.1], 3).unwrap_err();
        assert!(matches!(err, EngineError::BadShape(_)), "got {err:?}");
    }

    /// The lowered DDE tape for `y'(t) = −y(t − 1)`: one output `−u1` over two
    /// inputs (`u0 = y(t)`, `u1 = y(t − 1)`, the single delay slot), no params.
    fn neg_delay_dde_tape() -> Tape {
        let mut b = TapeBuilder::new();
        let _y = b.state(0);
        let y_tau = b.state(1);
        let dy = b.neg(y_tau);
        b.finish(&[dy], &[], 2, 0).unwrap()
    }

    #[test]
    fn integrate_dde_matches_method_of_steps_closed_form() {
        // Constant past 1; analytic y(1) = 0, y(2) = −0.5 (method of steps).
        let t_eval: Vec<f64> = (0..=20).map(|i| i as f64 * 0.1).collect();
        let y = integrate_dde_dense(
            neg_delay_dde_tape(),
            &[0],   // slot 0 → component 0
            &[1.0], // delay 1
            &[1.0], // ic
            &[0.0], // past times (single sample → constant past)
            &[1.0], // past values
            &t_eval,
            "rk45",
            1e-9,
            1e-11,
            false,
        )
        .unwrap();
        assert_eq!(y.len(), t_eval.len());
        assert_eq!(y[0], 1.0); // first row is the IC
        assert!((y[10] - 0.0).abs() < 1e-6, "y(1) = {}", y[10]);
        assert!((y[20] + 0.5).abs() < 1e-6, "y(2) = {}", y[20]);
    }

    #[test]
    fn integrate_dde_rejects_implicit_method() {
        let err = integrate_dde_dense(
            neg_delay_dde_tape(),
            &[0],
            &[1.0],
            &[1.0],
            &[0.0],
            &[1.0],
            &[0.0, 1.0],
            "rosenbrock",
            1e-6,
            1e-9,
            false,
        )
        .unwrap_err();
        assert!(matches!(err, EngineError::Unsupported(_)), "got {err:?}");
    }

    #[test]
    fn integrate_dde_rejects_shape_mismatches() {
        // n_state (2) != dim (1) + slots (2): the tape has only one delay input.
        let bad_slots = integrate_dde_dense(
            neg_delay_dde_tape(),
            &[0, 0],
            &[1.0, 2.0],
            &[1.0],
            &[0.0],
            &[1.0],
            &[0.0, 1.0],
            "rk45",
            1e-6,
            1e-9,
            false,
        )
        .unwrap_err();
        assert!(
            matches!(bad_slots, EngineError::BadShape(_)),
            "got {bad_slots:?}"
        );

        // A delay slot referencing a non-existent component.
        let bad_comp = integrate_dde_dense(
            neg_delay_dde_tape(),
            &[5],
            &[1.0],
            &[1.0],
            &[0.0],
            &[1.0],
            &[0.0, 1.0],
            "rk45",
            1e-6,
            1e-9,
            false,
        )
        .unwrap_err();
        assert!(
            matches!(bad_comp, EngineError::BadShape(_)),
            "got {bad_comp:?}"
        );

        // A non-positive delay.
        let bad_delay = integrate_dde_dense(
            neg_delay_dde_tape(),
            &[0],
            &[0.0],
            &[1.0],
            &[0.0],
            &[1.0],
            &[0.0, 1.0],
            "rk45",
            1e-6,
            1e-9,
            false,
        )
        .unwrap_err();
        assert!(
            matches!(bad_delay, EngineError::BadShape(_)),
            "got {bad_delay:?}"
        );
    }

    #[test]
    fn implicit_methods_refuse_a_tape_without_a_jacobian() {
        // The critical guard: rosenbrock/trbdf2 freeze ∂f/∂u each step. On a tape
        // compiled without a Jacobian the iteration matrix would collapse to I and
        // the L-stable step would silently degrade to forward Euler. The engine
        // must reject this loudly (BadShape → ValueError), not integrate.
        let t_eval = [0.0, 0.5, 1.0];
        for method in ["rosenbrock", "trbdf2"] {
            let err =
                integrate_dense(decay_tape(), &[1.0], &[2.0], &t_eval, method, 1e-6, 1e-9, false)
                    .unwrap_err();
            assert!(
                matches!(err, EngineError::BadShape(_)),
                "{method} (no Jacobian): got {err:?}"
            );
            // The ensemble path guards identically.
            let err = ensemble_final(
                decay_tape(),
                &[1.0, 2.0],
                &[2.0],
                0.0,
                1.0,
                method,
                1e-6,
                1e-9,
                false,
            )
            .unwrap_err();
            assert!(
                matches!(err, EngineError::BadShape(_)),
                "{method} ensemble (no Jacobian): got {err:?}"
            );
        }
        // With the Jacobian present, the same implicit method integrates fine.
        let y = integrate_dense(
            decay_tape_jac(),
            &[1.0],
            &[2.0],
            &t_eval,
            "rosenbrock",
            1e-9,
            1e-11,
            false,
        );
        assert!(y.is_ok(), "rosenbrock with a Jacobian should integrate: {y:?}");
        // Explicit methods never need a Jacobian, so the no-Jacobian tape is fine.
        assert!(
            integrate_dense(decay_tape(), &[1.0], &[2.0], &t_eval, "rk45", 1e-6, 1e-9, false)
                .is_ok()
        );
    }

    #[test]
    fn ensemble_rejects_backward_integration() {
        // t1 < t0 would never enter the forward step loop and silently return the
        // unchanged ICs as a successful batch; reject it instead.
        let err = ensemble_final(
            decay_tape(),
            &[1.0, 2.0],
            &[2.0],
            1.0,
            0.0,
            "rk45",
            1e-6,
            1e-9,
            false,
        )
        .unwrap_err();
        assert!(matches!(err, EngineError::BadShape(_)), "got {err:?}");
    }

    #[test]
    fn integrate_dde_rejects_jit() {
        let err = integrate_dde_dense(
            neg_delay_dde_tape(),
            &[0],
            &[1.0],
            &[1.0],
            &[0.0],
            &[1.0],
            &[0.0, 1.0],
            "rk45",
            1e-6,
            1e-9,
            true,
        )
        .unwrap_err();
        assert!(matches!(err, EngineError::Unsupported(_)), "got {err:?}");
    }
}
