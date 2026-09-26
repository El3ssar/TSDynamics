//! The ODE Benettin Lyapunov-spectrum renormalisation loop, in Rust (stream
//! `perf/ode-lyapunov-engine`).
//!
//! [`lyapunov_spectrum_ode`] runs the **entire** burn-in + averaging Benettin
//! construction of the Python `TangentSystem` ODE path inside one engine call: it
//! integrates the *extended* variational ODE (base state ⊕ `k` tangent vectors)
//! one `dt` chunk at a time, QR-reorthonormalises the tangent block after every
//! chunk, and accumulates `Σ log|diag R|` over the averaging window — so a whole
//! Lyapunov run pays **one** Python→FFI round-trip and zero per-chunk Python /
//! NumPy QR, instead of the released loop's `(burn_in + final_time)/dt` round-trips
//! each followed by a NumPy `qr`.
//!
//! # What the kernel reproduces
//!
//! The released Python path
//! ([`tsdynamics.derived.tangent.TangentSystem._step_ode_engine`]) advances the
//! extended state one `dt` via a *fresh* two-node `integrate_grid([t, t+dt])` (the
//! adaptive controller re-seeds each chunk — exactly the [`crate::basin`] /
//! [`crate::bridge::stepper`] per-`dt` contract), then unpacks the `(dim, k)`
//! tangent block, QR-reorthonormalises it, and re-embeds the orthonormal frame.
//! This kernel does the same, chunk for chunk: the per-`dt` integration is
//! byte-for-byte the released numerics (so `interp == jit`), and the QR is a
//! hand-rolled **modified Gram–Schmidt** (the Lyapunov contributions `log|diag R|`
//! are the column norms after orthogonalisation — invariant to the QR algorithm to
//! floating-point tolerance, which is the documented match against the
//! NumPy-Householder Python path).
//!
//! [`tsdynamics.derived.tangent.TangentSystem._step_ode_engine`]: the released loop
//! this kernel folds into the engine.

use tsdyn_ir::Evaluator;
use tsdyn_solvers::Solver;

use crate::integrate::{integrate_grid_polled, IntegrateConfig, IntegrateError};
use crate::interrupt::Poller;

/// Why a Lyapunov run could not be set up or completed.
#[derive(Clone, Debug, PartialEq)]
pub enum LyapunovError {
    /// A buffer length / dimension invariant disagrees with the tape (a
    /// caller-side mistake the binding maps to `ValueError`).
    BadShape(String),
    /// The extended variational integration diverged or the step collapsed before
    /// the run completed (the "diverge loudly" contract).
    Diverged(String),
    /// The run exhausted a chunk's solver-step budget with a finite state — it
    /// stalled rather than blew up. Distinct from
    /// [`Diverged`](LyapunovError::Diverged) for the same reason the integrate
    /// loop distinguishes them: the remedy is a solver knob, not a fix to the
    /// equations.
    StepBudget(String),
    /// The embedder's interrupt hook stopped the run (a Ctrl-C at the Python
    /// prompt). This used to be *reported as a divergence*: the chunk loop
    /// funnelled every [`IntegrateError`] into
    /// [`Diverged`](LyapunovError::Diverged), so an interrupted spectrum came
    /// back as "your system blew up".
    Interrupted,
}

impl core::fmt::Display for LyapunovError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LyapunovError::BadShape(m)
            | LyapunovError::Diverged(m)
            | LyapunovError::StepBudget(m) => f.write_str(m),
            LyapunovError::Interrupted => f.write_str("interrupted"),
        }
    }
}

impl std::error::Error for LyapunovError {}

/// Failure to resolve the propagated tangent frame without inventing directions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TangentQrError {
    NonFinite,
    UnresolvedRank,
}

pub(crate) const UNRESOLVED_TANGENT_RANK: &str =
    "tangent frame has dependent or numerically unresolved nonzero columns; reduce \
     reortho_interval (maps) or dt (flows). General rank-deficient frames are \
     unsupported; exact zero map columns are retained as -inf growth.";

/// Scaled modified Gram--Schmidt, with logarithmic growth accumulated before
/// restoring each column's physical scale. Exact zero columns stay zero; an
/// arbitrary complement would fabricate surviving tangent directions. Nonzero
/// columns whose independence cannot be resolved are explicitly refused.
pub(crate) fn mgs_renormalise(
    w: &mut [f64],
    dim: usize,
    k: usize,
    growths: &mut [f64],
) -> Result<(), TangentQrError> {
    if !w.iter().all(|v| v.is_finite()) {
        return Err(TangentQrError::NonFinite);
    }
    // Relative floating-point rank resolution on columns scaled to max|w|=1.
    // This is not a physical growth floor and is invariant to column scaling.
    let rank_resolution = 8.0 * dim as f64 * f64::EPSILON;
    for i in 0..k {
        let scale = w[i * dim..(i + 1) * dim]
            .iter()
            .fold(0.0_f64, |a, v| a.max(v.abs()));
        if scale == 0.0 {
            growths[i] = f64::NEG_INFINITY;
            continue;
        }
        for r in 0..dim {
            w[i * dim + r] /= scale;
        }
        // A second pass controls accumulated projection roundoff. All operands
        // are now on unit scale, avoiding tiny/huge norm and dot-product errors.
        for _ in 0..2 {
            for j in 0..i {
                let mut dot = 0.0;
                for r in 0..dim {
                    dot += w[j * dim + r] * w[i * dim + r];
                }
                for r in 0..dim {
                    w[i * dim + r] -= dot * w[j * dim + r];
                }
            }
        }
        let norm = w[i * dim..(i + 1) * dim]
            .iter()
            .fold(0.0_f64, |a, v| a.hypot(*v));
        if !norm.is_finite() {
            return Err(TangentQrError::NonFinite);
        }
        if norm <= rank_resolution {
            return Err(TangentQrError::UnresolvedRank);
        }
        growths[i] = scale.ln() + norm.ln();
        for r in 0..dim {
            w[i * dim + r] /= norm;
        }
    }
    Ok(())
}

/// Advance the extended variational state `z` to one exact chunk endpoint,
/// re-seeding a fresh solver as in the Python per-chunk integration path.
fn advance_chunk<F>(
    ev: &dyn Evaluator,
    solver_factory: &F,
    z: &mut [f64],
    t: f64,
    tf: f64,
    p: &[f64],
    poll: &mut Poller,
) -> Result<(), IntegrateError>
where
    F: Fn() -> Box<dyn Solver>,
{
    let n = z.len();
    let t_eval = [t, tf];
    let mut solver = solver_factory();
    // First step is the grid-derived `tf - t` (NOT raw `dt`), matching
    // `OdeStepper::advance` / the basin march exactly.
    let cfg = IntegrateConfig::new(tf - t);
    // The caller's poller, not a fresh one: a chunk is a handful of solver
    // steps, so a per-chunk poller would reset long before a stride and a
    // multi-minute spectrum would never check for signals.
    let out = integrate_grid_polled(ev, &mut *solver, &z[..n], p, &t_eval, &cfg, poll)?;
    // `out` is the flat `(2, n)` buffer; the last row is the advanced state.
    z.copy_from_slice(&out[n..2 * n]);
    Ok(())
}

/// Floating-point spacing on the clock's own scale, including subnormal times.
fn time_spacing(value: f64) -> f64 {
    let magnitude = value.abs();
    let next = f64::from_bits(magnitude.to_bits() + 1);
    if next.is_finite() {
        next - magnitude
    } else {
        magnitude - f64::from_bits(magnitude.to_bits() - 1)
    }
}

/// Use origin-anchored chunk targets; do not accumulate step-rounding drift.
/// A near-endpoint target is snapped, never skipped: even a tiny positive
/// window must integrate once. The slack matches the shared Python grid rule.
fn chunk_target(
    start: f64,
    end: f64,
    current: f64,
    dt: f64,
    index: usize,
) -> Result<f64, LyapunovError> {
    let slack = (0.5 * dt).min(8.0 * time_spacing(start).max(time_spacing(end)));
    let nominal = start + index as f64 * dt;
    let target = if nominal >= end || end - nominal <= slack {
        end
    } else {
        nominal
    };
    if !target.is_finite() || target <= current {
        return Err(LyapunovError::BadShape(
            "dt must produce distinct increasing renormalisation times".to_string(),
        ));
    }
    Ok(target)
}

/// The result of a Lyapunov run: the spectrum plus the final extended state (so
/// the Python wrapper can record the end state / deviations exactly as the
/// released loop left them).
#[derive(Clone, Debug)]
pub struct LyapunovOutcome {
    /// The `k` Lyapunov exponents, in QR column order.
    pub spectrum: Vec<f64>,
    /// The final extended state `z` (length `dim*(k+1)`) — base state ⊕ the
    /// orthonormal tangent frame, after the last QR.
    pub final_state: Vec<f64>,
    /// The most recent per-step log-stretch contributions (the released
    /// `self._last_growths`).
    pub last_growths: Vec<f64>,
}

/// Run the full burn-in + averaging Benettin Lyapunov-spectrum estimate for an ODE
/// flow, in one engine call.
///
/// # Arguments
///
/// - `ev` — the built **extended** variational evaluator (`dim*(k+1)` inputs and
///   outputs: the base RHS stacked with the `k` tangent-vector RHS blocks).
/// - `solver_factory` — builds a fresh kernel per `dt` chunk (the binding resolves
///   the method and threads the tolerances).
/// - `p` — live control parameters (the extended tape carries the base system's
///   control parameters, read live each chunk).
/// - `dim` — the **base** system dimension.
/// - `k` — the number of tangent vectors (`1 ≤ k ≤ dim`).
/// - `z0` — the initial extended state (length `dim*(k+1)`): base IC ⊕ the seed
///   tangent frame (`I[:, :k]` column-major), exactly the Python `embed_extended`.
/// - `t0` — the start time.
/// - `dt` — the renormalisation interval (the Python `dt`).
/// - `burn_in` — discard this non-negative duration before accumulating.
/// - `final_time` — the averaging-window length after burn-in.
/// - `rtol`, `atol` — the tolerances used by `solver_factory`; growth at the
///   corresponding endpoint error scale is refused before taking it as evidence.
///
/// The chunking follows the shared Python output-grid convention: the burn-in steps a
/// (possibly short) last chunk so it lands on `t0 + burn_in`, then the averaging
/// window steps a (possibly short) last chunk so it lands on `t_burn +
/// final_time`; each chunk is one `dt` (or the residual), and the QR after every
/// chunk reorthonormalises the frame. The spectrum is `Σ growths / elapsed` over
/// the averaging window.
#[allow(clippy::too_many_arguments)]
pub fn lyapunov_spectrum_ode<F>(
    ev: &dyn Evaluator,
    solver_factory: F,
    p: &[f64],
    dim: usize,
    k: usize,
    z0: &[f64],
    t0: f64,
    dt: f64,
    burn_in: f64,
    final_time: f64,
    rtol: f64,
    atol: f64,
) -> Result<LyapunovOutcome, LyapunovError>
where
    F: Fn() -> Box<dyn Solver>,
{
    // --- validation (mirrors the Python guards) ---
    if !rtol.is_finite()
        || !atol.is_finite()
        || rtol < 0.0
        || atol < 0.0
        || (rtol == 0.0 && atol == 0.0)
    {
        return Err(LyapunovError::BadShape(
            "rtol and atol must be finite, non-negative and not both zero".to_string(),
        ));
    }
    if dim == 0 {
        return Err(LyapunovError::BadShape(
            "system dimension is zero".to_string(),
        ));
    }
    if !(1..=dim).contains(&k) {
        return Err(LyapunovError::BadShape(format!(
            "k must be in [1, {dim}], got {k}"
        )));
    }
    let n = dim * (k + 1);
    if ev.dim() != n {
        return Err(LyapunovError::BadShape(format!(
            "extended evaluator dimension {} != dim*(k+1) = {n}",
            ev.dim()
        )));
    }
    if z0.len() != n {
        return Err(LyapunovError::BadShape(format!(
            "extended initial state has length {}, need dim*(k+1) = {n}",
            z0.len()
        )));
    }
    if p.len() < ev.n_param() {
        return Err(LyapunovError::BadShape(format!(
            "parameter vector has length {}, need n_param = {}",
            p.len(),
            ev.n_param()
        )));
    }
    if !(dt.is_finite() && dt > 0.0) {
        return Err(LyapunovError::BadShape(format!(
            "dt must be finite and positive, got {dt}"
        )));
    }
    if !(final_time.is_finite() && final_time > 0.0) {
        return Err(LyapunovError::BadShape(format!(
            "final_time must be finite and positive, got {final_time}"
        )));
    }
    if !t0.is_finite() || !burn_in.is_finite() || burn_in < 0.0 {
        return Err(LyapunovError::BadShape(
            "t0 must be finite and burn_in must be finite and non-negative".to_string(),
        ));
    }
    let t_burn = t0 + burn_in;
    let t_end = t_burn + final_time;
    if !t_end.is_finite() || (burn_in > 0.0 && t_burn <= t0) || t_end <= t_burn {
        return Err(LyapunovError::BadShape(
            "burn-in and averaging windows must have representable finite endpoints".to_string(),
        ));
    }
    if t0 + dt == t0 || (t_end - dt == t_end && dt < final_time.max(burn_in)) {
        return Err(LyapunovError::BadShape(
            "dt must be large enough to advance the floating-point time axis".to_string(),
        ));
    }

    let mut z = z0.to_vec();
    let mut t = t0;
    let mut growths = vec![0.0; k];
    let mut last_growths = vec![0.0; k];
    let mut log_error_floor = vec![0.0; k];

    // ONE poller for the whole run — burn-in and averaging window alike — so
    // the polling cadence is one check per `POLL_STRIDE` *solver steps*, not
    // per chunk (see `advance_chunk`).
    let mut poll = Poller::new();

    // --- burn-in: advance + QR, no accumulation ---
    let mut chunk = 1;
    while t < t_burn {
        let target = chunk_target(t0, t_burn, t, dt, chunk)?;
        advance_chunk(ev, &solver_factory, &mut z, t, target, p, &mut poll).map_err(classify)?;
        if !renorm_step(
            &mut z,
            dim,
            k,
            &mut last_growths,
            rtol,
            atol,
            &mut log_error_floor,
        )? {
            return Err(LyapunovError::Diverged(
                "extended variational state went non-finite during burn-in".to_string(),
            ));
        }
        t = target;
        chunk += 1;
    }

    // --- averaging window: advance + QR + accumulate ---
    chunk = 1;
    while t < t_end {
        let target = chunk_target(t_burn, t_end, t, dt, chunk)?;
        advance_chunk(ev, &solver_factory, &mut z, t, target, p, &mut poll).map_err(classify)?;
        if !renorm_step(
            &mut z,
            dim,
            k,
            &mut last_growths,
            rtol,
            atol,
            &mut log_error_floor,
        )? {
            return Err(LyapunovError::Diverged(
                "extended variational state went non-finite during the averaging window"
                    .to_string(),
            ));
        }
        for i in 0..k {
            growths[i] += last_growths[i];
        }
        t = target;
        chunk += 1;
    }

    let elapsed = t_end - t_burn;
    let spectrum: Vec<f64> = growths.iter().map(|&g| g / elapsed).collect();

    Ok(LyapunovOutcome {
        spectrum,
        final_state: z,
        last_growths,
    })
}

/// Unpack the tangent block of the extended state `z`, QR-reorthonormalise it in
/// place, and re-embed the orthonormal frame — the per-chunk renormalisation.
///
/// `z` is `[base state (dim) | tangent block (dim*k)]`; the tangent block is
/// reorthonormalised by [`mgs_renormalise`] and written back. Returns `Ok(false)`
/// if the frame went non-finite (divergence).
fn renorm_step(
    z: &mut [f64],
    dim: usize,
    k: usize,
    growths: &mut [f64],
    rtol: f64,
    atol: f64,
    log_error_floor: &mut [f64],
) -> Result<bool, LyapunovError> {
    // The base state must be finite too — a diverged flow shows up here.
    if !z[..dim].iter().all(|x| x.is_finite()) {
        return Ok(false);
    }
    let error_pool = 0.5 * (z.len() as f64).ln();
    let frame = &mut z[dim..dim + dim * k];
    if frame
        .chunks(dim)
        .any(|column| column.iter().all(|v| *v == 0.0))
    {
        return Err(LyapunovError::BadShape(
            "a flow tangent column collapsed to zero during numerical propagation; \
             reduce dt and tighten rtol/atol before interpreting contraction rates"
                .to_string(),
        ));
    }
    // sum((error_i/scale_i)^2) <= extended_dim bounds any column's Euclidean
    // error by sqrt(extended_dim)*max(scale_i). Retain this scale before QR:
    // projection may leave a tiny residual even when that column is large.
    // This detects unresolved growth, not accumulated global integration error.
    for (column, floor) in frame.chunks(dim).zip(log_error_floor.iter_mut()) {
        let scale = column.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        let absolute = atol.ln();
        let relative = rtol.ln() + scale.ln();
        let largest = absolute.max(relative);
        // At least one tolerance is positive and this column is nonzero, so
        // largest is finite. The log-sum never over/underflows physical units.
        *floor =
            largest + ((absolute - largest).exp() + (relative - largest).exp()).ln() + error_pool;
    }
    match mgs_renormalise(frame, dim, k, growths) {
        Ok(()) => {
            if growths
                .iter()
                .zip(log_error_floor)
                .any(|(growth, floor)| *growth <= *floor)
            {
                return Err(LyapunovError::BadShape(
                    "flow tangent growth is unresolved at the integration error scale; \
                     reduce dt (reorthonormalize more often) or tighten atol/rtol before \
                     interpreting contraction rates"
                        .to_string(),
                ));
            }
            Ok(true)
        }
        Err(TangentQrError::NonFinite) => Ok(false),
        Err(TangentQrError::UnresolvedRank) => {
            Err(LyapunovError::BadShape(UNRESOLVED_TANGENT_RANK.to_string()))
        }
    }
}

/// Prefix an integrate-loop divergence the way the bridge does, so the Python
/// `ConvergenceError` reads clearly.
fn diverge_msg(e: &IntegrateError) -> String {
    format!("Lyapunov extended variational integration diverged: {e}")
}

/// Lift a chunk's [`IntegrateError`] to the Lyapunov error it actually means.
///
/// The chunk loop used to collapse *every* integrate failure into
/// [`LyapunovError::Diverged`]. That was already wrong for a step-budget stall
/// and became actively misleading once the loop could be interrupted: a Ctrl-C
/// would have been reported to the user as a numerical blow-up.
fn classify(e: IntegrateError) -> LyapunovError {
    match e {
        IntegrateError::Interrupted { .. } => LyapunovError::Interrupted,
        // `Stalled` is the same condition as `StepLimit`, diagnosed from the step
        // size instead of after the whole budget, so it lifts to the same error:
        // a chunk that cannot finish is a solver-settings problem either way.
        IntegrateError::StepLimit { .. } | IntegrateError::Stalled { .. } => {
            LyapunovError::StepBudget(format!(
                "Lyapunov extended variational integration did not reach the end of a \
                 renormalisation chunk: {e}"
            ))
        }
        other => LyapunovError::Diverged(diverge_msg(&other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::VmEval;
    use tsdyn_ir::TapeBuilder;
    use tsdyn_solvers::explicit::Rk45;
    use tsdyn_vm::Interpreter;

    /// Build the extended variational tape of the 2-D linear flow
    /// `dx = a x, dy = b y` with `k` tangents. The Jacobian is the constant
    /// diagonal `diag(a, b)`, so the variational block is `dw_i = J · w_i`.
    /// Lyapunov spectrum is exactly `[max(a,b), min(a,b)]`.
    fn linear_extended(a: f64, b: f64, k: usize) -> Interpreter {
        let dim = 2;
        let mut bld = TapeBuilder::new();
        // base state inputs.
        let x = bld.state(0);
        let y = bld.state(1);
        let ac = bld.constant(a);
        let bc = bld.constant(b);
        let dx = bld.mul(ac, x);
        let dy = bld.mul(bc, y);
        let mut outs = vec![dx, dy];
        // tangent blocks: w_i has components at inputs [dim + i*dim + c].
        for i in 0..k {
            let base = dim + i * dim;
            let w0 = bld.state(base);
            let w1 = bld.state(base + 1);
            // dw0 = a*w0, dw1 = b*w1 (J = diag(a,b)).
            let dw0 = bld.mul(ac, w0);
            let dw1 = bld.mul(bc, w1);
            outs.push(dw0);
            outs.push(dw1);
        }
        let n = dim * (k + 1);
        Interpreter::new(bld.finish(&outs, &[], n, 0).unwrap())
    }

    fn rk45_factory() -> Box<dyn Solver> {
        Box::new(Rk45::with_tolerances(1e-9, 1e-11))
    }

    #[test]
    fn mgs_matches_known_orthonormalisation() {
        // Two orthogonal columns of norms 2 and 3 → growths ln 2, ln 3; the frame
        // is orthonormal afterwards.
        let dim = 2;
        let k = 2;
        // column 0 = (2, 0), column 1 = (0, 3) — already orthogonal.
        let mut w = vec![2.0, 0.0, 0.0, 3.0];
        let mut g = vec![0.0; k];
        assert!(mgs_renormalise(&mut w, dim, k, &mut g).is_ok());
        assert!((g[0] - 2.0_f64.ln()).abs() < 1e-14);
        assert!((g[1] - 3.0_f64.ln()).abs() < 1e-14);
        // Orthonormal: column norms 1, columns orthogonal.
        assert!((w[0] * w[0] + w[1] * w[1] - 1.0).abs() < 1e-14);
        assert!((w[2] * w[2] + w[3] * w[3] - 1.0).abs() < 1e-14);
        assert!((w[0] * w[2] + w[1] * w[3]).abs() < 1e-14);
    }

    #[test]
    fn scaled_qr_preserves_norms_without_forming_overflowing_squares() {
        for scale in [1e-300, 1e-200, 1e200, 1e308] {
            let mut w = vec![scale, scale];
            let mut growths = vec![0.0];
            mgs_renormalise(&mut w, 2, 1, &mut growths).unwrap();
            let expected = scale.ln() + 0.5 * 2.0_f64.ln();
            assert!((growths[0] - expected).abs() < 1e-12);
            assert!((w[0] - 1.0 / 2.0_f64.sqrt()).abs() < 1e-14);
            assert!((w[1] - w[0]).abs() < 1e-14);
        }
    }

    #[test]
    fn flow_collapse_is_not_a_negative_infinite_exponent_claim() {
        let mut z = vec![0.0, 0.0];
        let mut growths = vec![0.0];
        let err = renorm_step(&mut z, 1, 1, &mut growths, 1e-9, 1e-11, &mut [0.0]).unwrap_err();
        assert!(matches!(err, LyapunovError::BadShape(_)));
        assert!(err.to_string().contains("flow tangent column collapsed"));
    }

    #[test]
    fn integration_floor_does_not_become_a_measured_contraction() {
        let ev = VmEval::new(linear_extended(-1000.0, -1.0, 2));
        let z0 = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let coarse = lyapunov_spectrum_ode(
            &ev,
            rk45_factory,
            &[],
            2,
            2,
            &z0,
            0.0,
            0.1,
            0.0,
            0.2,
            1e-9,
            1e-11,
        )
        .unwrap_err();
        assert!(coarse.to_string().contains("integration error scale"));
        let fine = lyapunov_spectrum_ode(
            &ev,
            rk45_factory,
            &[],
            2,
            2,
            &z0,
            0.0,
            0.001,
            0.0,
            0.2,
            1e-9,
            1e-11,
        )
        .unwrap();
        assert!((fine.spectrum[0] + 1000.0).abs() < 1e-3);
        assert!((fine.spectrum[1] + 1.0).abs() < 1e-7);
        let tight = || -> Box<dyn Solver> { Box::new(Rk45::with_tolerances(1e-9, 1e-60)) };
        let resolved =
            lyapunov_spectrum_ode(&ev, tight, &[], 2, 2, &z0, 0.0, 0.1, 0.0, 0.2, 1e-9, 1e-60)
                .unwrap();
        assert!((resolved.spectrum[0] + 1000.0).abs() < 1e-3);
    }

    #[test]
    fn error_scale_is_checked_after_projection_and_in_log_units() {
        // Both columns are large, but their independent residual is below the
        // integration scale. Machine-precision rank alone cannot validate it.
        let mut z = vec![0.0, 0.0, 0.5, 0.5, 0.5, 0.5 + 1e-10];
        let err = renorm_step(&mut z, 2, 2, &mut [0.0; 2], 1e-9, 1e-12, &mut [0.0; 2]).unwrap_err();
        assert!(err.to_string().contains("integration error scale"));
        // One-sided controls and products outside normal floating-point range
        // must not turn the resolution screen into NaN or a zero threshold.
        for (scale, rtol, atol, accepted) in [
            (1e-300, 1e-300, 0.0, true),
            (1e-300, 0.0, 1e-310, true),
            (1e300, 1e300, 0.0, false),
        ] {
            let mut z = vec![0.0, scale];
            let result = renorm_step(&mut z, 1, 1, &mut [0.0], rtol, atol, &mut [0.0]);
            assert_eq!(result.is_ok(), accepted);
        }
    }

    #[test]
    fn linear_flow_spectrum_matches_analytic() {
        // dx = 0.5 x, dy = -2 x → exact Lyapunov spectrum [0.5, -2.0].
        let (a, b, k) = (0.5, -2.0, 2);
        let ev = VmEval::new(linear_extended(a, b, k));
        // z0: base (1, 1) ⊕ identity tangent frame (column-major).
        let z0 = vec![1.0, 1.0, /*w0*/ 1.0, 0.0, /*w1*/ 0.0, 1.0];
        let out = lyapunov_spectrum_ode(
            &ev,
            rk45_factory,
            &[],
            2,
            k,
            &z0,
            0.0,
            0.1,
            5.0,
            50.0,
            1e-9,
            1e-11,
        )
        .unwrap();
        assert!(
            (out.spectrum[0] - 0.5).abs() < 1e-4,
            "lambda1 = {}",
            out.spectrum[0]
        );
        assert!(
            (out.spectrum[1] - (-2.0)).abs() < 1e-4,
            "lambda2 = {}",
            out.spectrum[1]
        );
        // Spectrum descends.
        assert!(out.spectrum[0] > out.spectrum[1]);
    }

    #[test]
    fn linear_flow_rates_are_invariant_under_time_unit_changes() {
        // The same unit amount of expansion/contraction, measured in different
        // time units. Tiny physical durations must not become empty windows.
        for rate in [1e-12, 1.0, 1e12] {
            let ev = VmEval::new(linear_extended(rate, -rate, 2));
            let z0 = vec![1.0, 1.0, 1.0, 0.0, 0.0, 1.0];
            let duration = 1.0 / rate;
            let out = lyapunov_spectrum_ode(
                &ev,
                rk45_factory,
                &[],
                2,
                2,
                &z0,
                0.0,
                duration / 5.0,
                duration,
                duration,
                1e-9,
                1e-11,
            )
            .unwrap();
            assert!((out.spectrum[0] / rate - 1.0).abs() < 1e-7);
            assert!((out.spectrum[1] / rate + 1.0).abs() < 1e-7);
            // Burn-in and production must both run to their exact endpoints.
            assert!((out.final_state[0] - 2.0_f64.exp()).abs() < 1e-7);
            assert!((out.final_state[1] - (-2.0_f64).exp()).abs() < 1e-7);
        }
    }

    #[test]
    fn chunk_schedule_preserves_short_windows_and_avoids_roundoff_residuals() {
        assert_eq!(chunk_target(0.0, 1e-15, 0.0, 0.1, 1).unwrap(), 1e-15);
        let mut current = 0.0;
        for i in 1..=10 {
            current = chunk_target(0.0, 1.0, current, 0.1, i).unwrap();
        }
        assert_eq!(current, 1.0);
        assert!(chunk_target(1e16, 1e16 + 4.0, 1e16, 0.1, 1).is_err());
    }

    #[test]
    fn rejects_unrepresentable_averaging_window_instead_of_zero_spectrum() {
        let ev = VmEval::new(linear_extended(1.0, -1.0, 2));
        let z0 = vec![1.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        assert!(matches!(
            lyapunov_spectrum_ode(
                &ev,
                rk45_factory,
                &[],
                2,
                2,
                &z0,
                1e16,
                1.0,
                0.0,
                1.0,
                1e-9,
                1e-11
            )
            .unwrap_err(),
            LyapunovError::BadShape(_)
        ));
    }

    #[test]
    fn partial_spectrum_k_less_than_dim() {
        // Only the leading exponent (k = 1) of the same flow → 0.5.
        let (a, b, k) = (0.5, -2.0, 1);
        let ev = VmEval::new(linear_extended(a, b, k));
        let z0 = vec![1.0, 1.0, 1.0, 0.0];
        let out = lyapunov_spectrum_ode(
            &ev,
            rk45_factory,
            &[],
            2,
            k,
            &z0,
            0.0,
            0.1,
            5.0,
            50.0,
            1e-9,
            1e-11,
        )
        .unwrap();
        assert_eq!(out.spectrum.len(), 1);
        assert!((out.spectrum[0] - 0.5).abs() < 1e-4, "{}", out.spectrum[0]);
    }

    #[test]
    fn rejects_bad_k_and_shapes() {
        let ev = VmEval::new(linear_extended(0.5, -2.0, 2));
        let z0 = vec![1.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        // k out of range.
        assert!(matches!(
            lyapunov_spectrum_ode(
                &ev,
                rk45_factory,
                &[],
                2,
                3,
                &z0,
                0.0,
                0.1,
                1.0,
                1.0,
                1e-9,
                1e-11
            )
            .unwrap_err(),
            LyapunovError::BadShape(_)
        ));
        // wrong z0 length.
        assert!(matches!(
            lyapunov_spectrum_ode(
                &ev,
                rk45_factory,
                &[],
                2,
                2,
                &[1.0, 1.0],
                0.0,
                0.1,
                1.0,
                1.0,
                1e-9,
                1e-11
            )
            .unwrap_err(),
            LyapunovError::BadShape(_)
        ));
        // non-positive dt.
        assert!(matches!(
            lyapunov_spectrum_ode(
                &ev,
                rk45_factory,
                &[],
                2,
                2,
                &z0,
                0.0,
                0.0,
                1.0,
                1.0,
                1e-9,
                1e-11
            )
            .unwrap_err(),
            LyapunovError::BadShape(_)
        ));
    }

    #[test]
    fn diverging_flow_raises() {
        // dx = x², dy = 0 with a trivial tangent (a finite-time blow-up).
        let mut bld = TapeBuilder::new();
        let x = bld.state(0);
        let _y = bld.state(1);
        let w0 = bld.state(2);
        let w1 = bld.state(3);
        let dx = bld.mul(x, x);
        let zero = bld.constant(0.0);
        let dy = bld.mul(zero, x);
        // tangent dynamics dw = 2x * w0 ; dw1 = 0 (not important — blow-up first).
        let two = bld.constant(2.0);
        let twox = bld.mul(two, x);
        let dw0 = bld.mul(twox, w0);
        let dw1 = bld.mul(zero, w1);
        let tape = bld.finish(&[dx, dy, dw0, dw1], &[], 4, 0).unwrap();
        let ev = VmEval::new(Interpreter::new(tape));
        let z0 = vec![1.0, 0.0, 1.0, 0.0];
        let err = lyapunov_spectrum_ode(
            &ev,
            rk45_factory,
            &[],
            2,
            1,
            &z0,
            0.0,
            0.05,
            0.0,
            10.0,
            1e-9,
            1e-11,
        )
        .unwrap_err();
        assert!(matches!(err, LyapunovError::Diverged(_)), "got {err:?}");
    }

    /// An armed interrupt stops the chunk loop — and is reported *as* an
    /// interrupt, not as the divergence this loop used to collapse every
    /// integrate failure into.
    ///
    /// The poller is threaded down into the per-chunk `integrate_grid_polled`,
    /// which is what makes this work at all: each chunk is a handful of solver
    /// steps, so a poller created per chunk would reset before ever reaching a
    /// stride and a multi-minute spectrum would never poll.
    #[test]
    fn an_armed_interrupt_stops_the_chunk_loop_and_is_not_called_a_divergence() {
        let _stop = crate::interrupt::testing::force_stop();
        let _armed = crate::interrupt::arm();

        // A stable 2-D linear flow (no divergence anywhere) integrated over a
        // long window in tiny chunks, so the run needs far more than one poll
        // stride of solver steps.
        let ev = VmEval::new(linear_extended(-0.5, -1.0, 1));
        let z0 = vec![1.0, 1.0, 1.0, 0.0];
        let err = lyapunov_spectrum_ode(
            &ev,
            rk45_factory,
            &[],
            2,
            1,
            &z0,
            0.0,
            1e-3,
            0.0,
            1e4,
            1e-9,
            1e-11,
        )
        .unwrap_err();
        assert_eq!(err, LyapunovError::Interrupted, "got {err:?}");
    }
}
