//! Shared engine-bridge plumbing: the Python-free error enum, the evaluator
//! adapter/builders, tape ingestion, solver resolution, the input/grid/shape
//! guards, and the divergence-message helpers.
//!
//! Every family entry point ([`super::ode`] / [`super::dde`] / [`super::sde`] /
//! [`super::map`] / [`super::events`] / [`super::stepper`]) is built on these —
//! the one place the FFI boundary marshalling and validation lives. Moved here
//! verbatim from the original `bridge.rs` (the bridge-split reorg); behaviour is
//! unchanged.

use tsdyn_engine::{EventDirection, IntegrateError, MapError, SdeError};
use tsdyn_ir::{Evaluator, Tape};
use tsdyn_jit::JitEvaluator;
use tsdyn_solvers::explicit::{Bs3, CashKarp, Dop853, HeunEuler, Rk45, Rkf45, Tsit5};
use tsdyn_solvers::implicit::{
    BackwardEuler, Bdf, ImplicitMidpoint, RosenbrockW, Sdirk2, TrBdf2, Trapezoid,
};
use tsdyn_solvers::Solver;
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
    /// engine (e.g. state-dependent DDE delays). → `NotImplementedError`.
    Unsupported(String),
    /// The Cranelift JIT (`tsdyn-jit`) could not compile the tape — a host-ISA
    /// or codegen failure, distinct from a malformed tape. → `RuntimeError`.
    JitCompile(String),
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
            | EngineError::JitCompile(m)
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

/// Build the [`Evaluator`] a run drives: the Cranelift JIT ([`JitEvaluator`])
/// when `jit`, else the zero-warmup interpreter ([`VmEvaluator`]).
///
/// This is the seam decision D2 hangs on — both back the *same* [`Evaluator`]
/// trait, so every integrate / iterate / ensemble loop below drives either as
/// `&dyn Evaluator` with no other change, and the two are numerically identical
/// (bit-for-bit on `eval`; the JIT's contract). Returning a boxed trait object
/// rather than a concrete type is what lets one call site choose the backend.
/// `dyn Evaluator: Sync` (the trait's supertrait), so the boxed evaluator shares
/// freely across the rayon ensemble workers. A Cranelift build failure (host ISA
/// / codegen) surfaces as [`EngineError::JitCompile`]; the interpreter never
/// fails to build.
pub(super) fn build_evaluator(tape: Tape, jit: bool) -> Result<Box<dyn Evaluator>, EngineError> {
    if jit {
        let ev = JitEvaluator::new(&tape).map_err(|e| EngineError::JitCompile(e.to_string()))?;
        Ok(Box::new(ev))
    } else {
        Ok(Box::new(VmEvaluator::new(tape)))
    }
}

/// Build an evaluator as a `Box<dyn Evaluator + Send>` — the variant the durable
/// [`super::stepper::OdeStepper`] handle owns so it (and an `&mut` to it) can
/// cross the `Python::detach` GIL-release boundary (which requires `Send`).
///
/// Both concrete evaluators are `Send` (the interpreter is plain data; the JIT's
/// `JITModule` and `fn` pointers are `Send` — see `tsdyn_jit`'s safety note), so
/// erasing them to `dyn Evaluator + Send` is sound. The batch entry points keep
/// the `Sync`-only [`build_evaluator`] because they only ever *share* `&dyn
/// Evaluator` across rayon workers (which needs `Sync`, not `Send`).
pub(super) fn build_evaluator_send(
    tape: Tape,
    jit: bool,
) -> Result<Box<dyn Evaluator + Send>, EngineError> {
    if jit {
        let ev = JitEvaluator::new(&tape).map_err(|e| EngineError::JitCompile(e.to_string()))?;
        Ok(Box::new(ev))
    } else {
        Ok(Box::new(VmEvaluator::new(tape)))
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
/// the implicit family (`rosenbrock`/`trbdf2`/`bdf`) — owns its `rtol`/`atol` (the
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
        "heun_euler" => Box::new(HeunEuler::with_tolerances(rtol, atol)),
        "bs3" => Box::new(Bs3::with_tolerances(rtol, atol)),
        "rkf45" => Box::new(Rkf45::with_tolerances(rtol, atol)),
        "cashkarp" => Box::new(CashKarp::with_tolerances(rtol, atol)),
        "rosenbrock" => Box::new(RosenbrockW::with_tolerances(rtol, atol)),
        "trbdf2" => Box::new(TrBdf2::with_tolerances(rtol, atol)),
        "bdf" => Box::new(Bdf::with_tolerances(rtol, atol)),
        "backward_euler" => Box::new(BackwardEuler::with_tolerances(rtol, atol)),
        "implicit_midpoint" => Box::new(ImplicitMidpoint::with_tolerances(rtol, atol)),
        "trapezoid" => Box::new(Trapezoid::with_tolerances(rtol, atol)),
        "sdirk2" => Box::new(Sdirk2::with_tolerances(rtol, atol)),
        // The fixed-step kernels (`euler`/`midpoint`/`heun`/`ralston`/`rk4`/
        // `rk4_38`/`ssprk3`), the Adams multistep kernels (`ab3`/`ab4`/`abm4`,
        // which take no tolerances), and any out-of-tree plugin kernel build
        // through the registry factory. `name` came from the registry
        // (via `resolve_solver`), so `make` is guaranteed `Some`.
        other => tsdyn_solvers::make(other)
            .expect("a name returned by resolve_solver is always registered"),
    }
}

// ---------------------------------------------------------------------------
// Input / grid / shape guards
// ---------------------------------------------------------------------------

/// Validate that the state/param slices match the tape's declared widths.
pub(super) fn check_inputs(tape: &Tape, u: &[f64], p: &[f64]) -> Result<(), EngineError> {
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

/// Reject a tape that is not ODE-shaped on the single/ensemble ODE path. A DDE
/// tape carries delay-slot inputs beyond the real state (`n_state > dim`) and an
/// SDE is a tape *pair*; both have their own entry points
/// ([`super::dde::integrate_dde_dense`] / [`super::sde::sde_integrate_dense`]), so
/// the bare ODE driver only accepts `n_state == dim`. (The JIT is no longer
/// rejected here — both evaluators are built; see [`build_evaluator`].)
pub(super) fn guard_continuous(tape: &Tape) -> Result<(), EngineError> {
    if tape.n_state() != tape.dim() {
        return Err(EngineError::Unsupported(format!(
            "engine ODE integration supports ODE-shaped tapes (n_state == dim); got \
             n_state = {} > dim = {} (a DDE/SDE tape). Use the DDE entry point \
             (integrate_dde_dense) or the SDE entry points (integrate_sde_*).",
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
pub(super) fn require_jacobian_if_needed(
    tape: &Tape,
    name: &'static str,
) -> Result<(), EngineError> {
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
pub(super) fn first_step_from_grid(t_eval: &[f64]) -> f64 {
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
pub(super) fn validate_grid(t_eval: &[f64]) -> Result<(), EngineError> {
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

/// Map a Python-side direction code (`+1` / `-1` / `0`) to an [`EventDirection`].
pub(super) fn event_direction(direction: i32) -> Result<EventDirection, EngineError> {
    match direction {
        1 => Ok(EventDirection::Rising),
        -1 => Ok(EventDirection::Falling),
        0 => Ok(EventDirection::Either),
        other => Err(EngineError::BadShape(format!(
            "event direction must be +1 (rising), -1 (falling) or 0 (either), got {other}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Divergence messages
// ---------------------------------------------------------------------------

/// Prefix the engine's diverge message so the Python `RuntimeError` reads clearly.
pub(super) fn diverge_msg(e: &IntegrateError) -> String {
    format!("integration diverged before reaching the final time: {e}")
}

/// Prefix the engine's map-divergence message, mirroring [`diverge_msg`].
pub(super) fn map_diverge_msg(e: &MapError) -> String {
    format!("map diverged before completing all iterations: {e}")
}

/// Prefix the engine's SDE-divergence message, mirroring [`diverge_msg`].
pub(super) fn sde_diverge_msg(e: &SdeError) -> String {
    format!("SDE integration diverged before reaching the final time: {e}")
}
