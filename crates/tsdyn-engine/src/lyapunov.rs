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

use crate::integrate::{integrate_grid, IntegrateConfig, IntegrateError};

/// Why a Lyapunov run could not be set up or completed.
#[derive(Clone, Debug, PartialEq)]
pub enum LyapunovError {
    /// A buffer length / dimension invariant disagrees with the tape (a
    /// caller-side mistake the binding maps to `ValueError`).
    BadShape(String),
    /// The extended variational integration diverged or the step collapsed before
    /// the run completed (the "diverge loudly" contract).
    Diverged(String),
}

impl core::fmt::Display for LyapunovError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LyapunovError::BadShape(m) | LyapunovError::Diverged(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for LyapunovError {}

/// Reorthonormalise the `(dim, k)` tangent block in place by **modified
/// Gram–Schmidt**, returning each column's `log` stretch factor.
///
/// The tangent vectors are stored column-major in `w` (`w[i*dim .. (i+1)*dim]` is
/// the `i`-th tangent vector, of length `dim`) — the same layout
/// [`crate::bridge`] marshals to/from the Python `embed_extended` /
/// `split_extended`. Modified Gram–Schmidt orthonormalises them left-to-right and
/// records `log(norm_i)` (the `i`-th diagonal of the implicit `R`), with the
/// `|·|`-free convention (the norm is non-negative by construction) and the same
/// `tiny`-floor on a vanishing norm the Python `_qr_growths` applies.  Returns
/// `Err` if a non-finite value appears (divergence surfaced through the QR rather
/// than as a plausible number).
fn mgs_renormalise(w: &mut [f64], dim: usize, k: usize, growths: &mut [f64]) -> bool {
    // The smallest positive normal f64 — the floor the Python `_qr_growths` uses
    // (`np.finfo(float).tiny`) so a vanishing stretch logs a large-negative
    // contribution instead of `-inf`.
    const TINY: f64 = f64::MIN_POSITIVE;

    for i in 0..k {
        // Orthogonalise column i against the already-normalised columns 0..i
        // (modified Gram–Schmidt: subtract each projection as soon as it is
        // computed, using the *updated* column — more stable than the classical
        // variant).
        for j in 0..i {
            // dot = <q_j, w_i>
            let mut dot = 0.0;
            for r in 0..dim {
                dot += w[j * dim + r] * w[i * dim + r];
            }
            for r in 0..dim {
                w[i * dim + r] -= dot * w[j * dim + r];
            }
        }
        // Norm of the orthogonalised column i → its stretch factor.
        let mut norm2 = 0.0;
        for r in 0..dim {
            let v = w[i * dim + r];
            norm2 += v * v;
        }
        let norm = norm2.sqrt();
        if !norm.is_finite() {
            return false;
        }
        let safe = if norm < TINY { TINY } else { norm };
        growths[i] = safe.ln();
        // Normalise the column (divide by the floored norm so we never divide by
        // zero; the contribution is already recorded).
        let inv = 1.0 / safe;
        for r in 0..dim {
            w[i * dim + r] *= inv;
            if !w[i * dim + r].is_finite() {
                return false;
            }
        }
    }
    true
}

/// Advance the extended variational state `z` (length `dim*(k+1)`) one `dt` chunk,
/// re-seeding a fresh solver — byte-for-byte the per-`dt` `integrate_grid([t,
/// t+dt])` the released Python `TangentSystem._step_ode_engine` runs.
fn advance_chunk<F>(
    ev: &dyn Evaluator,
    solver_factory: &F,
    z: &mut [f64],
    t: f64,
    dt: f64,
    p: &[f64],
) -> Result<(), IntegrateError>
where
    F: Fn() -> Box<dyn Solver>,
{
    let n = z.len();
    let tf = t + dt;
    let t_eval = [t, tf];
    let mut solver = solver_factory();
    // First step is the grid-derived `tf - t` (NOT raw `dt`), matching
    // `OdeStepper::advance` / the basin march exactly.
    let cfg = IntegrateConfig::new(tf - t);
    let out = integrate_grid(ev, &mut *solver, &z[..n], p, &t_eval, &cfg)?;
    // `out` is the flat `(2, n)` buffer; the last row is the advanced state.
    z.copy_from_slice(&out[n..2 * n]);
    Ok(())
}

/// The result of a Lyapunov run: the spectrum plus the final extended state (so
/// the Python wrapper can record the end state / deviations exactly as the
/// released loop left them).
#[derive(Clone, Debug)]
pub struct LyapunovOutcome {
    /// The `k` Lyapunov exponents, in QR (descending-by-construction) order.
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
/// - `burn_in` — discard this much time before accumulating (`max(0, burn_in)`).
/// - `final_time` — the averaging-window length after burn-in.
///
/// The chunking matches the released Python loop exactly: the burn-in steps a
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
) -> Result<LyapunovOutcome, LyapunovError>
where
    F: Fn() -> Box<dyn Solver>,
{
    // --- validation (mirrors the Python guards) ---
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

    let mut z = z0.to_vec();
    let mut t = t0;
    let mut growths = vec![0.0; k];
    let mut last_growths = vec![0.0; k];

    // The same end-of-window tolerance the Python loop uses (`while t < t_end -
    // 1e-12`), so the chunk count / residual-step structure is identical.
    const EPS: f64 = 1e-12;

    // --- burn-in: advance + QR, no accumulation ---
    let t_burn = t0 + burn_in.max(0.0);
    while t < t_burn - EPS {
        let h = dt.min(t_burn - t);
        advance_chunk(ev, &solver_factory, &mut z, t, h, p)
            .map_err(|e| LyapunovError::Diverged(diverge_msg(&e)))?;
        if !renorm_step(&mut z, dim, k, &mut last_growths)? {
            return Err(LyapunovError::Diverged(
                "extended variational state went non-finite during burn-in".to_string(),
            ));
        }
        t += h;
    }

    // --- averaging window: advance + QR + accumulate ---
    let mut elapsed = 0.0;
    let t_end = t + final_time;
    while t < t_end - EPS {
        let h = dt.min(t_end - t);
        advance_chunk(ev, &solver_factory, &mut z, t, h, p)
            .map_err(|e| LyapunovError::Diverged(diverge_msg(&e)))?;
        if !renorm_step(&mut z, dim, k, &mut last_growths)? {
            return Err(LyapunovError::Diverged(
                "extended variational state went non-finite during the averaging window"
                    .to_string(),
            ));
        }
        for i in 0..k {
            growths[i] += last_growths[i];
        }
        t += h;
        elapsed += h;
    }

    let spectrum: Vec<f64> = if elapsed == 0.0 {
        vec![0.0; k]
    } else {
        growths.iter().map(|&g| g / elapsed).collect()
    };

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
) -> Result<bool, LyapunovError> {
    // The base state must be finite too — a diverged flow shows up here.
    if !z[..dim].iter().all(|x| x.is_finite()) {
        return Ok(false);
    }
    let ok = mgs_renormalise(&mut z[dim..dim + dim * k], dim, k, growths);
    Ok(ok)
}

/// Prefix an integrate-loop divergence the way the bridge does, so the Python
/// `ConvergenceError` reads clearly.
fn diverge_msg(e: &IntegrateError) -> String {
    format!("Lyapunov extended variational integration diverged: {e}")
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
        assert!(mgs_renormalise(&mut w, dim, k, &mut g));
        assert!((g[0] - 2.0_f64.ln()).abs() < 1e-14);
        assert!((g[1] - 3.0_f64.ln()).abs() < 1e-14);
        // Orthonormal: column norms 1, columns orthogonal.
        assert!((w[0] * w[0] + w[1] * w[1] - 1.0).abs() < 1e-14);
        assert!((w[2] * w[2] + w[3] * w[3] - 1.0).abs() < 1e-14);
        assert!((w[0] * w[2] + w[1] * w[3]).abs() < 1e-14);
    }

    #[test]
    fn linear_flow_spectrum_matches_analytic() {
        // dx = 0.5 x, dy = -2 x → exact Lyapunov spectrum [0.5, -2.0].
        let (a, b, k) = (0.5, -2.0, 2);
        let ev = VmEval::new(linear_extended(a, b, k));
        // z0: base (1, 1) ⊕ identity tangent frame (column-major).
        let z0 = vec![1.0, 1.0, /*w0*/ 1.0, 0.0, /*w1*/ 0.0, 1.0];
        let out =
            lyapunov_spectrum_ode(&ev, rk45_factory, &[], 2, k, &z0, 0.0, 0.1, 5.0, 50.0).unwrap();
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
    fn partial_spectrum_k_less_than_dim() {
        // Only the leading exponent (k = 1) of the same flow → 0.5.
        let (a, b, k) = (0.5, -2.0, 1);
        let ev = VmEval::new(linear_extended(a, b, k));
        let z0 = vec![1.0, 1.0, 1.0, 0.0];
        let out =
            lyapunov_spectrum_ode(&ev, rk45_factory, &[], 2, k, &z0, 0.0, 0.1, 5.0, 50.0).unwrap();
        assert_eq!(out.spectrum.len(), 1);
        assert!((out.spectrum[0] - 0.5).abs() < 1e-4, "{}", out.spectrum[0]);
    }

    #[test]
    fn rejects_bad_k_and_shapes() {
        let ev = VmEval::new(linear_extended(0.5, -2.0, 2));
        let z0 = vec![1.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        // k out of range.
        assert!(matches!(
            lyapunov_spectrum_ode(&ev, rk45_factory, &[], 2, 3, &z0, 0.0, 0.1, 1.0, 1.0)
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
                1.0
            )
            .unwrap_err(),
            LyapunovError::BadShape(_)
        ));
        // non-positive dt.
        assert!(matches!(
            lyapunov_spectrum_ode(&ev, rk45_factory, &[], 2, 2, &z0, 0.0, 0.0, 1.0, 1.0)
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
        let err = lyapunov_spectrum_ode(&ev, rk45_factory, &[], 2, 1, &z0, 0.0, 0.05, 0.0, 10.0)
            .unwrap_err();
        assert!(matches!(err, LyapunovError::Diverged(_)), "got {err:?}");
    }
}
