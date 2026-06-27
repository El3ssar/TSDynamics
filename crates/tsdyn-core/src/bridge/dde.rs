//! DDE engine entry point: the dense method-of-steps integrator for delay
//! differential equations.
//!
//! Moved verbatim from the original `bridge.rs` (the bridge-split reorg); the
//! numerics and validation are unchanged. Shared plumbing lives in
//! [`super::marshal`].

use tsdyn_engine::{integrate_dde_grid, DelaySlot, IntegrateConfig};
use tsdyn_ir::Tape;
use tsdyn_solvers::SolverKind;

use super::marshal::{
    build_evaluator, build_solver, diverge_msg, first_step_from_grid, resolve_solver,
    validate_grid, EngineError,
};

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
    let ev = build_evaluator(tape, jit)?;
    let mut solver = build_solver(name, rtol, atol);
    let cfg = IntegrateConfig::new(first_step_from_grid(t_eval));
    integrate_dde_grid(
        &*ev,
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
