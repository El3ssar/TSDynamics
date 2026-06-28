//! ODE engine entry points: pointwise RHS/Jacobian evaluation, the dense
//! single-trajectory integrator, and the parallel ensemble.
//!
//! Moved verbatim from the original `bridge.rs` (the bridge-split reorg); the
//! numerics and validation are unchanged. The shared plumbing
//! (evaluator/solver builders, input/grid guards, divergence messages) lives in
//! [`super::marshal`].

use tsdyn_engine::{ensemble_final as engine_ensemble, integrate_grid, IntegrateConfig};
use tsdyn_ir::Tape;
use tsdyn_vm::Interpreter;

use super::marshal::{
    build_evaluator, build_solver, check_inputs, diverge_msg, first_step_from_grid,
    guard_continuous, require_jacobian_if_needed, resolve_solver, validate_grid, EngineError,
};

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

// ---------------------------------------------------------------------------
// Integration
// ---------------------------------------------------------------------------

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
    guard_continuous(&tape)?;
    check_inputs(&tape, ic, p)?;
    if tape.dim() == 0 {
        return Err(EngineError::BadShape(
            "system dimension is zero (the tape has no outputs)".to_string(),
        ));
    }
    validate_grid(t_eval)?;
    let name = resolve_solver(method)?;
    require_jacobian_if_needed(&tape, name)?;
    let ev = build_evaluator(tape, jit)?;
    let mut solver = build_solver(name, rtol, atol);
    let cfg = IntegrateConfig::new(first_step_from_grid(t_eval));
    integrate_grid(&*ev, &mut *solver, &ic[..ev.dim()], p, t_eval, &cfg)
        .map_err(|e| EngineError::Diverged(diverge_msg(&e)))
}

/// Integrate a batch of initial conditions to `t1` in parallel, returning each
/// final state as a flat row-major `(n_ic, dim)` buffer.
///
/// `ics` is a row-major `(n_ic, dim)` buffer. A diverging trajectory yields a
/// `NaN` row rather than aborting the batch (the engine's ensemble contract); the
/// run is seeded/scheduling-independent, so results are bit-for-bit reproducible.
///
/// `first_step` is the integration cadence the caller threads in (the user's
/// `dt`). For an adaptive kernel it is only the first *trial* step — the
/// controller adapts away from it — but for the fixed-step `rk4` it *is* the step
/// for the whole run. The single-trajectory [`integrate_dense`] derives its
/// cadence from the output grid ([`first_step_from_grid`]); the ensemble has no
/// grid, so it must be handed the same cadence rather than inventing a
/// `span`-relative guess. Without this knob a fixed-step ensemble and the dense
/// path disagree numerically — the bug this argument fixes. The SDE ensemble
/// ([`super::sde::sde_ensemble_final`]) already carries an explicit `dt` for the
/// same reason.
#[allow(clippy::too_many_arguments)]
pub fn ensemble_final(
    tape: Tape,
    ics: &[f64],
    p: &[f64],
    t0: f64,
    t1: f64,
    first_step: f64,
    method: &str,
    rtol: f64,
    atol: f64,
    jit: bool,
) -> Result<Vec<f64>, EngineError> {
    guard_continuous(&tape)?;
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
    // The cadence is the caller's (the user's `dt`), no longer a `span/100` guess:
    // for an adaptive kernel it seeds the controller, but for the fixed-step `rk4`
    // it *is* the step, so it must match the dense path's grid-derived step or the
    // two entry points disagree (the bug this fixes). The engine treats a
    // non-finite or non-positive first step as a hard-`assert!` caller error
    // (it would spin to the step limit or step the wrong way), so reject it here
    // as a clean `BadShape`, mirroring the SDE path's `check_sde_dt`.
    if !(first_step.is_finite() && first_step > 0.0) {
        return Err(EngineError::BadShape(format!(
            "integration cadence (first_step) must be finite and positive; got {first_step}"
        )));
    }
    let cfg = IntegrateConfig::new(first_step);
    let ev = build_evaluator(tape, jit)?;
    let result = engine_ensemble(
        &*ev,
        |_i| build_solver(name, rtol, atol),
        ics,
        p,
        t0,
        t1,
        &cfg,
    );
    Ok(result.states)
}
