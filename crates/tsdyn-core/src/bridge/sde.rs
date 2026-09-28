//! SDE engine entry points (diagonal-Itô, two tapes: drift + diffusion): kernel
//! resolution, tape-pair / dt validation, the dense single-trajectory
//! integrator and the parallel ensemble.
//!
//! Moved verbatim from the original `bridge.rs` (the bridge-split reorg); the
//! numerics, the seeded RNG threading and the validation are unchanged. Shared
//! plumbing lives in [`super::marshal`].

use tsdyn_engine::interrupt::Poller;
use tsdyn_engine::rng::SplitMix64;
use tsdyn_engine::{sde_ensemble_final as engine_sde_ensemble, sde_integrate_grid, SdeConfig};
use tsdyn_ir::{Op, OpKind, Tape};
use tsdyn_solvers::sde::{self, SdeKernel};

use super::marshal::{build_evaluator, sde_failure, validate_grid, EngineError};

/// Resolve a `method=` string to an SDE-kernel factory (case-insensitively),
/// validating that the diffusion tape carries `∂g/∂u` when the scheme needs it.
///
/// The SDE kernels live in their *own* registry (`tsdyn_solvers::sde`), separate
/// from the `Solver` registry — an SDE step needs two evaluators and a Wiener
/// increment, not the frozen `Solver::step` signature. Milstein (order 1.0)
/// reads the diagonal diffusion Jacobian, so a Milstein run over a diffusion
/// tape lowered without one (`with_diffusion_jacobian=False`) is rejected here —
/// the SDE twin of [`super::marshal::require_jacobian_if_needed`]. Returns the
/// `fn` factory (a fresh kernel per call; `fn` pointers are `Sync`, so it threads
/// through the ensemble fan-out).
fn resolve_sde(method: &str, diffusion: &Tape) -> Result<fn() -> Box<dyn SdeKernel>, EngineError> {
    let reg = sde::find(method)
        .or_else(|| sde::find(&method.to_lowercase()))
        .ok_or_else(|| {
            EngineError::UnknownMethod(format!(
                "unknown SDE method {method:?}; available: {:?}",
                sde::available()
            ))
        })?;
    if reg.name == "milstein" {
        require_componentwise_noise(diffusion)?;
    }
    if reg.caps.needs_jacobian && !diffusion.has_jacobian() {
        return Err(EngineError::BadShape(format!(
            "SDE method {:?} needs the diffusion Jacobian ∂g/∂u, but the diffusion tape was \
             lowered without one (with_diffusion_jacobian=False). Lower the diffusion with its \
             Jacobian, or use 'euler_maruyama'.",
            reg.name
        )));
    }
    Ok(reg.make)
}

/// Establish the structural sufficient condition for the implemented correction.
///
/// For distinct coordinates i and j the omitted stochastic term has coefficient
/// g_j * partial_j(g_i). We accept an absent state dependency or a literal zero
/// driver g_j. Supplied Jacobian zeros, current parameter values, and numerical
/// probes are not proofs. In particular, diagonal coefficient storage alone
/// does not establish this condition. Unknown/cancelled dependencies may refuse.
///
/// Walk only primal-output ancestors, including all three Select operands. The
/// visited stamps and stack use O(registers) memory rather than a dense
/// registers-by-dimension dependency matrix. Tape construction validates every
/// index; marking before pushing keeps the stack within the reserved capacity.
fn require_componentwise_noise(tape: &Tape) -> Result<(), EngineError> {
    let dim = tape.dim();
    if tape.n_state() != dim {
        return Err(EngineError::BadShape(format!(
            "Milstein diffusion must have n_state == dim; got {} != {dim}",
            tape.n_state()
        )));
    }
    let mut seen = Vec::new();
    let mut stack = Vec::new();
    for buffer in [&mut seen, &mut stack] {
        buffer.try_reserve_exact(tape.n_reg()).map_err(|_| {
            EngineError::OutOfMemory(
                "cannot allocate the Milstein diffusion dependency check".to_string(),
            )
        })?;
    }
    seen.resize(tape.n_reg(), usize::MAX);
    let mut poll = Poller::new();
    for (component, &output) in tape.outputs().iter().enumerate() {
        let output = output as usize;
        seen[output] = component;
        stack.push(output);
        while let Some(register) = stack.pop() {
            if poll.tick() {
                return Err(EngineError::Interrupted);
            }
            let op = tape.ops()[register];
            if op == Op::State {
                let other = tape.a()[register] as usize;
                let driver = tape.outputs()[other] as usize;
                let zero_driver = tape.ops()[driver] == Op::Const && tape.imm()[driver] == 0.0;
                if other != component && !zero_driver {
                    return Err(EngineError::Unsupported(format!(
                        "Milstein's componentwise correction cannot establish zero cross-noise \
                         terms: diffusion component {component} reads state {other}, whose \
                         noise coefficient is not a literal zero. Use 'euler_maruyama' for \
                         this diffusion structure."
                    )));
                }
                continue;
            }
            let a = tape.a()[register] as usize;
            let (parents, count) = match op.kind() {
                OpKind::Leaf => continue,
                OpKind::Unary | OpKind::Powi => ([a, 0, 0], 1),
                OpKind::Binary => ([a, tape.b()[register] as usize, 0], 2),
                OpKind::Select => (
                    [
                        a,
                        tape.b()[register] as usize,
                        tape.imm()[register] as usize,
                    ],
                    3,
                ),
            };
            for &parent in &parents[..count] {
                if seen[parent] != component {
                    seen[parent] = component;
                    stack.push(parent);
                }
            }
        }
    }
    Ok(())
}

/// Validate the SDE tape pair and shared parameters, returning the dimension.
///
/// The drift and diffusion are each `dim → dim` over the *same* (state,
/// parameter) layout (the diagonal-Itô contract); both must be ODE-shaped
/// (`n_state == dim`) and agree on `dim` and `n_param`, and `p` must cover that
/// `n_param`. Reachable from a hand-built tape pair, not the Python seam, so a
/// mismatch is a clean error rather than a release-mode panic in the SDE loop's
/// `debug_assert!`s.
fn check_sde_tapes(drift: &Tape, diffusion: &Tape, p: &[f64]) -> Result<usize, EngineError> {
    let dim = drift.dim();
    if dim == 0 {
        return Err(EngineError::BadShape(
            "system dimension is zero (the drift tape has no outputs)".to_string(),
        ));
    }
    if drift.n_state() != dim {
        return Err(EngineError::BadShape(format!(
            "SDE drift tape must be ODE-shaped (n_state == dim); got n_state = {} != dim = {dim}",
            drift.n_state()
        )));
    }
    if diffusion.dim() != dim || diffusion.n_state() != dim {
        return Err(EngineError::BadShape(format!(
            "SDE diffusion tape must match the drift (dim = {dim}); got diffusion dim = {}, \
             n_state = {}",
            diffusion.dim(),
            diffusion.n_state()
        )));
    }
    if drift.n_param() != diffusion.n_param() {
        return Err(EngineError::BadShape(format!(
            "SDE drift and diffusion must share the parameter layout; got drift n_param = {}, \
             diffusion n_param = {}",
            drift.n_param(),
            diffusion.n_param()
        )));
    }
    if p.len() < drift.n_param() {
        return Err(EngineError::BadShape(format!(
            "parameter vector has length {}, need n_param = {}",
            p.len(),
            drift.n_param()
        )));
    }
    Ok(dim)
}

/// Reject a non-finite or non-positive SDE step. `dt` is the step *and* the
/// noise scale (`√dt`), and the engine hard-`assert!`s a positive finite step;
/// validate at the boundary so it is a clean error, not a `PanicException`.
fn check_sde_dt(dt: f64) -> Result<(), EngineError> {
    if !(dt.is_finite() && dt > 0.0) {
        return Err(EngineError::InvalidParameter(format!(
            "SDE step dt must be finite and positive; got {dt}"
        )));
    }
    Ok(())
}

/// Integrate one diagonal-Itô SDE trajectory through the output grid `t_eval`,
/// returning a flat row-major `(t_eval.len(), dim)` buffer (the first row is the
/// state at `t_eval[0]`).
///
/// `drift`/`diffusion` are the two lowered tapes; `method` selects the SDE
/// kernel by name (`euler_maruyama` / `milstein`); `dt` is the fixed step (and
/// the noise scale `√dt`); `seed` makes the sample path reproducible. A landing
/// step to a grid point that is not on the `dt` lattice is shortened with an
/// `N(0, h)` increment (a valid step on a non-uniform partition). Divergence
/// raises ([`EngineError::Diverged`]).
#[allow(clippy::too_many_arguments)] // drift + diffusion + the usual grid args + the SDE step/seed.
pub fn sde_integrate_dense(
    drift: Tape,
    diffusion: Tape,
    ic: &[f64],
    p: &[f64],
    t_eval: &[f64],
    method: &str,
    dt: f64,
    seed: u64,
    jit: bool,
) -> Result<Vec<f64>, EngineError> {
    let kernel_make = resolve_sde(method, &diffusion)?;
    let dim = check_sde_tapes(&drift, &diffusion, p)?;
    if ic.len() < dim {
        return Err(EngineError::BadShape(format!(
            "initial state has length {}, need dim = {dim}",
            ic.len()
        )));
    }
    check_sde_dt(dt)?;
    validate_grid(t_eval)?;
    let drift_ev = build_evaluator(drift, jit)?;
    let diff_ev = build_evaluator(diffusion, jit)?;
    let mut kernel = kernel_make();
    let mut rng = SplitMix64::new(seed);
    let cfg = SdeConfig::new(dt);
    sde_integrate_grid(
        &*drift_ev,
        &*diff_ev,
        &mut *kernel,
        &mut rng,
        &ic[..dim],
        p,
        t_eval,
        &cfg,
    )
    .map_err(sde_failure)
}

/// Integrate a batch of SDE initial conditions to `t1` in parallel, returning
/// each final state as a flat row-major `(n_ic, dim)` buffer.
///
/// `ics` is a row-major `(n_ic, dim)` buffer. Worker `i` draws its noise from a
/// stream seeded by `seed_for(seed, i)` — depending only on the index — so the
/// batch is **parallel == serial** bit-for-bit (the engine owns the per-index
/// seeding here, unlike the ODE ensemble). A diverging trajectory yields a `NaN`
/// row rather than aborting the batch.
#[allow(clippy::too_many_arguments)] // drift + diffusion + batch + the SDE step/seed.
pub fn sde_ensemble_final(
    drift: Tape,
    diffusion: Tape,
    ics: &[f64],
    p: &[f64],
    t0: f64,
    t1: f64,
    method: &str,
    dt: f64,
    seed: u64,
    jit: bool,
) -> Result<Vec<f64>, EngineError> {
    let kernel_make = resolve_sde(method, &diffusion)?;
    let dim = check_sde_tapes(&drift, &diffusion, p)?;
    check_sde_dt(dt)?;
    if !(t0.is_finite() && t1.is_finite()) {
        return Err(EngineError::InvalidParameter(format!(
            "integration times must be finite; got t0 = {t0}, t1 = {t1}"
        )));
    }
    // The fixed-step SDE loop runs while `t < t1`; a backward request would never
    // step and silently return the unchanged ICs. Reject it (mirroring the ODE
    // ensemble) rather than return stale ICs as a successful batch.
    if t1 < t0 {
        return Err(EngineError::InvalidParameter(format!(
            "backward integration is not supported (t1 = {t1} < t0 = {t0}); request t1 >= t0."
        )));
    }
    if !ics.len().is_multiple_of(dim) {
        return Err(EngineError::BadShape(format!(
            "ensemble initial-condition buffer length {} is not a multiple of dim = {dim}",
            ics.len()
        )));
    }
    let drift_ev = build_evaluator(drift, jit)?;
    let diff_ev = build_evaluator(diffusion, jit)?;
    let cfg = SdeConfig::new(dt);
    // `kernel_make` is a `fn` pointer (a fresh kernel per worker; `fn` is `Sync`).
    let result = engine_sde_ensemble(
        &*drift_ev,
        &*diff_ev,
        kernel_make,
        seed,
        ics,
        p,
        t0,
        t1,
        &cfg,
    );
    // A cancelled batch is an interrupt, not a batch of `NaN`s (see the ODE
    // ensemble): the signal has already been consumed by the poll that cancelled
    // it, so it has to be re-raised here.
    if result.interrupted {
        return Err(EngineError::Interrupted);
    }
    Ok(result.states)
}

#[cfg(test)]
mod componentwise_tests {
    use super::*;
    use tsdyn_ir::{Reg, TapeBuilder};

    fn finish(mut builder: TapeBuilder, outputs: &[Reg], n_param: usize) -> Tape {
        let zero = builder.constant(0.0);
        // Deliberately uninformative claimed derivatives: eligibility must read
        // the primal dependencies rather than infer independence from this matrix.
        builder
            .finish(outputs, &[zero; 4], outputs.len(), n_param)
            .unwrap()
    }

    fn cross_noise(zero_driver: bool) -> Tape {
        let mut builder = TapeBuilder::new();
        let x = builder.state(0);
        let driver = builder.constant(if zero_driver { -0.0 } else { 1.0 });
        finish(builder, &[driver, x], 0)
    }

    #[test]
    fn cross_noise_does_not_become_valid_from_a_zero_claimed_jacobian() {
        let diffusion = cross_noise(false);
        assert!(matches!(
            resolve_sde("milstein", &diffusion),
            Err(EngineError::Unsupported(_))
        ));
        assert!(resolve_sde("euler_maruyama", &diffusion).is_ok());
    }

    #[test]
    fn unsupported_structure_is_not_given_an_unusable_add_jacobian_remedy() {
        let mut builder = TapeBuilder::new();
        let one = builder.constant(1.0);
        let x = builder.state(0);
        let coupled = builder.finish(&[one, x], &[], 2, 0).unwrap();
        assert!(matches!(
            resolve_sde("milstein", &coupled),
            Err(EngineError::Unsupported(_))
        ));
        let mut builder = TapeBuilder::new();
        let one = builder.constant(1.0);
        let independent = builder.finish(&[one, one], &[], 2, 0).unwrap();
        assert!(matches!(
            resolve_sde("milstein", &independent),
            Err(EngineError::BadShape(_))
        ));
    }

    #[test]
    fn a_literal_zero_driver_allows_dependence_on_its_coordinate() {
        assert!(resolve_sde("milstein", &cross_noise(true)).is_ok());
    }

    #[test]
    fn a_control_driver_is_not_proved_zero_by_its_current_value() {
        let mut builder = TapeBuilder::new();
        let x = builder.state(0);
        let sigma = builder.param(0);
        let diffusion = finish(builder, &[sigma, x], 1);
        assert!(matches!(
            resolve_sde("milstein", &diffusion),
            Err(EngineError::Unsupported(_))
        ));
    }

    #[test]
    fn own_coordinate_time_and_control_dependencies_are_admitted() {
        let mut builder = TapeBuilder::new();
        let x = builder.state(0);
        let y = builder.state(1);
        let p = builder.param(0);
        let t = builder.time();
        let cubic = builder.powi(x, -3); // exponent is an immediate, not a register
        let g0 = builder.mul(p, cubic);
        let g1 = builder.add(y, t);
        let diffusion = finish(builder, &[g0, g1], 1);
        assert!(resolve_sde("milstein", &diffusion).is_ok());
    }

    #[test]
    fn every_selection_operand_contributes_dependencies() {
        for which in 0..3 {
            let mut builder = TapeBuilder::new();
            let x = builder.state(0);
            let one = builder.constant(1.0);
            let args = match which {
                0 => [x, one, one],
                1 => [one, x, one],
                _ => [one, one, x],
            };
            let selected = builder.select(args[0], args[1], args[2]);
            let diffusion = finish(builder, &[one, selected], 0);
            assert!(matches!(
                resolve_sde("milstein", &diffusion),
                Err(EngineError::Unsupported(_))
            ));
        }
    }

    #[test]
    fn unreachable_and_jacobian_only_dependencies_do_not_restrict_the_primal() {
        let mut builder = TapeBuilder::new();
        let x = builder.state(0);
        let y = builder.state(1);
        let zero = builder.constant(0.0);
        let _unused = builder.mul(x, y);
        let diffusion = builder.finish(&[x, y], &[x, y, zero, zero], 2, 0).unwrap();
        assert!(resolve_sde("milstein", &diffusion).is_ok());
    }

    #[test]
    fn dense_and_ensemble_manual_tapes_cannot_bypass_the_guard() {
        let mut builder = TapeBuilder::new();
        let zero = builder.constant(0.0);
        let drift = builder.finish(&[zero, zero], &[], 2, 0).unwrap();
        for jit in [false, true] {
            let dense = sde_integrate_dense(
                drift.clone(),
                cross_noise(false),
                &[0., 0.],
                &[],
                &[0., 0.1],
                "milstein",
                0.1,
                123,
                jit,
            );
            assert!(matches!(dense, Err(EngineError::Unsupported(_))));
            let ensemble = sde_ensemble_final(
                drift.clone(),
                cross_noise(false),
                &[0., 0.],
                &[],
                0.,
                0.1,
                "milstein",
                0.1,
                123,
                jit,
            );
            assert!(matches!(ensemble, Err(EngineError::Unsupported(_))));
        }
    }
}
