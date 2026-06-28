//! Event-detection engine entry point (two tapes: right-hand side + event
//! function `g`): integrate a span and return the refined crossings of
//! `g(u, t) = 0`.
//!
//! Moved verbatim from the original `bridge.rs` (the bridge-split reorg); the
//! event march, Hermite refinement and validation are unchanged. Shared plumbing
//! lives in [`super::marshal`].

use tsdyn_engine::{integrate_events, EventSpec, IntegrateConfig};
use tsdyn_ir::Tape;

use super::marshal::{
    build_evaluator, build_solver, check_inputs, diverge_msg, event_direction, guard_continuous,
    require_jacobian_if_needed, resolve_solver, EngineError,
};

/// Integrate `[t0, t1]` and return every crossing of the event function
/// `g(u, t) = 0` in `direction`, refined with the engine's O(h⁴) cubic-Hermite
/// dense output (stream WS-CROSSKERNEL — wiring the previously dead
/// [`tsdyn_engine::integrate_events`]).
///
/// `rhs` is the system right-hand side; `g` a single-output tape evaluated over
/// the full state (`g.dim() == 1`, `g.n_param() <= rhs.n_param()`). `first_step`
/// seeds the solver; with the fixed-step `rk4` it *is* the march step, so a caller
/// that wants the Python `PoincareMap`'s dt-grid detection (no skipped crossings,
/// answer-identical refinement) passes `method = "rk4"`, `first_step = dt`.
///
/// Returns `(times, states_flat, t_final, u_final, terminated)` — the `K` crossing
/// times, the row-major `(K, dim)` crossing states, the time/state the run stopped
/// at (so a caller can resume the next span exactly), and whether a terminal event
/// stopped it. Divergence raises ([`EngineError::Diverged`]), preserving the
/// "diverge loudly" contract of the per-`dt` Python path.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn integrate_events_dense(
    rhs: Tape,
    g: Tape,
    ic: &[f64],
    p: &[f64],
    t0: f64,
    t1: f64,
    first_step: f64,
    direction: i32,
    terminal: bool,
    method: &str,
    rtol: f64,
    atol: f64,
    jit: bool,
) -> Result<(Vec<f64>, Vec<f64>, f64, Vec<f64>, bool), EngineError> {
    guard_continuous(&rhs)?;
    check_inputs(&rhs, ic, p)?;
    if let Some(&bad) = ic.iter().find(|x| !x.is_finite()) {
        return Err(EngineError::BadShape(format!(
            "initial state must be finite, found {bad}"
        )));
    }
    let dim = rhs.dim();
    if dim == 0 {
        return Err(EngineError::BadShape(
            "system dimension is zero (the tape has no outputs)".to_string(),
        ));
    }
    // The event function is a single-output tape over the full state; it shares
    // the system parameter vector (it may declare fewer and read the leading
    // slice). `eval_event` in the engine asserts both, so reject violations here
    // as a clean `ValueError` rather than a release-build panic.
    if g.dim() != 1 {
        return Err(EngineError::BadShape(format!(
            "event function must have a single output, got {}",
            g.dim()
        )));
    }
    if g.n_state() > dim {
        return Err(EngineError::BadShape(format!(
            "event function reads {} state inputs but the system has dim = {dim}",
            g.n_state()
        )));
    }
    if g.n_param() > p.len() {
        return Err(EngineError::BadShape(format!(
            "event function declares {} parameters but the system has {}",
            g.n_param(),
            p.len()
        )));
    }
    if !(t0.is_finite() && t1.is_finite()) {
        return Err(EngineError::BadShape(format!(
            "event integration span must be finite, got t0 = {t0}, t1 = {t1}"
        )));
    }
    if t1 < t0 {
        return Err(EngineError::BadShape(format!(
            "integrate_events is forward only: need t1 >= t0, got t0 = {t0}, t1 = {t1}"
        )));
    }
    // The engine asserts a finite, positive first step (a hard `assert!`, not a
    // debug one) — turn a bad value into a clean error at the boundary.
    if !(first_step.is_finite() && first_step > 0.0) {
        return Err(EngineError::BadShape(format!(
            "first step (the detection dt) must be finite and positive, got {first_step}"
        )));
    }
    let dir = event_direction(direction)?;
    let name = resolve_solver(method)?;
    require_jacobian_if_needed(&rhs, name)?;

    let rhs_ev = build_evaluator(rhs, jit)?;
    let g_ev = build_evaluator(g, jit)?;
    let mut solver = build_solver(name, rtol, atol);
    let cfg = IntegrateConfig::new(first_step);

    let spec = if terminal {
        EventSpec::terminal(&*g_ev, dir)
    } else {
        EventSpec::new(&*g_ev, dir)
    };
    let outcome = integrate_events(&*rhs_ev, &mut *solver, &ic[..dim], p, t0, t1, &[spec], &cfg)
        .map_err(|e| EngineError::Diverged(diverge_msg(&e)))?;

    let mut times = Vec::with_capacity(outcome.hits.len());
    let mut states = Vec::with_capacity(outcome.hits.len() * dim);
    for hit in &outcome.hits {
        times.push(hit.t);
        states.extend_from_slice(&hit.u);
    }
    Ok((
        times,
        states,
        outcome.t_final,
        outcome.u_final,
        outcome.terminated,
    ))
}
