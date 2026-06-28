//! Discrete-map engine entry points: the single-orbit iterator and the parallel
//! map ensemble.
//!
//! Moved verbatim from the original `bridge.rs` (the bridge-split reorg); the
//! iteration loop, shape contract and divergence behaviour are unchanged. Shared
//! plumbing lives in [`super::marshal`].

use tsdyn_engine::{iterate_dense, iterate_ensemble_final as engine_iterate_ensemble};
use tsdyn_ir::Tape;

use super::marshal::{build_evaluator, map_diverge_msg, EngineError};

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
/// `jit` selects the evaluator the loop drives: the Cranelift native-code
/// [`tsdyn_jit::JitEvaluator`] when `true` (each step is one native `f(cur)` call
/// instead of a tape walk — faster for the cheapest single-orbit maps), else the
/// zero-warmup interpreter. Both back the same `Evaluator` trait and are
/// bit-for-bit identical on `eval`, so the iterate is numerically identical
/// either way.
///
/// The iteration itself is the shared engine map loop
/// ([`tsdyn_engine::iterate_dense`]); this wrapper only adds the boundary shape
/// checks the FFI seam needs, since the engine loop validates shapes with
/// `debug_assert!` (compiled out in release).
pub fn iterate_map(
    tape: Tape,
    ic: &[f64],
    steps: usize,
    jit: bool,
) -> Result<Vec<f64>, EngineError> {
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
    let ev = build_evaluator(tape, jit)?;
    iterate_dense(&*ev, &ic[..dim], &[], steps)
        .map_err(|e| EngineError::Diverged(map_diverge_msg(&e)))
}

/// Iterate a batch of map initial conditions to their `f^{steps}` in parallel,
/// returning each final state as a flat row-major `(n_ic, dim)` buffer.
///
/// `ics` is a row-major `(n_ic, dim)` buffer; map parameters fold into the tape
/// (`n_param == 0`). Mirrors the ODE [`super::ode::ensemble_final`] contract: a
/// diverging trajectory yields a `NaN` row rather than aborting the batch, so this
/// returns the buffer directly (`NaN` rows mark the failures) instead of raising —
/// the per-trajectory analogue of the single map loop's [`EngineError::Diverged`].
/// Maps carry no randomness, so the result is independent of thread count.
///
/// `jit` selects the per-worker evaluator (Cranelift native code vs the
/// interpreter); both are bit-for-bit identical, and the one `Evaluator` is
/// shared across the rayon workers (`Evaluator: Sync`).
pub fn map_ensemble_final(
    tape: Tape,
    ics: &[f64],
    steps: usize,
    jit: bool,
) -> Result<Vec<f64>, EngineError> {
    let dim = tape.dim();
    // The same shape contract `iterate_map` enforces, applied here before the
    // rayon fan-out so a malformed tape is a clean error, not a release-mode
    // out-of-bounds inside a worker (the engine loop guards with debug_assert!).
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
    // Maps fold their parameters into the tape, so the parameter slice is empty.
    let ev = build_evaluator(tape, jit)?;
    let result = engine_iterate_ensemble(&*ev, ics, &[], steps);
    Ok(result.states)
}
