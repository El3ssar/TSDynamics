//! Frozen robust QR/propagation baseline for a same-process Criterion comparison.
//!
//! This benchmark-only snapshot preserves the validated safety checks and work
//! of the kernel before the guarded fast norm and deferred zero diagnostics.
//! Both paths use the same evaluator, input, dimensions and interrupt poller.

use tsdyn_engine::{MapLyapunovError, MapLyapunovOutcome};
use tsdyn_ir::Evaluator;

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

/// Compute discrete-map tangent growth rates in frame-column order over the
/// lowered map RHS + Jacobian `ev`; a partial frame need not contain the maximum.
///
/// `ev` must be a lowered map evaluator carrying its Jacobian (`has_jacobian()`):
/// `eval` writes the next state `f(x)` (a map's RHS *is* the next state), and
/// `eval_jac` the next state plus the row-major `dim × dim` step Jacobian
/// `∂f_k/∂u_j` (the convention the lowered map tape uses). `p` is the parameter
/// vector (empty for a lowered map, whose parameters fold into the tape). `ic` is
/// the `dim`-length start state; `steps` the iterate budget; `k` the number of
/// exponents (`1 ≤ k ≤ dim`); `reortho_interval` the QR cadence (`≥ 1`).
///
/// Returns the `k` time-averaged exponents and the completed-interval count, or a
/// [`MapLyapunovError`] (a malformed call, a Jacobian-less tape, or a divergence
/// before the budget is exhausted — the "diverge loudly" contract).
pub fn map_lyapunov(
    ev: &dyn Evaluator,
    p: &[f64],
    ic: &[f64],
    steps: usize,
    k: usize,
    reortho_interval: usize,
) -> Result<MapLyapunovOutcome, MapLyapunovError> {
    let dim = ev.dim();
    if dim == 0 {
        return Err(MapLyapunovError::BadShape(
            "system dimension is zero".to_string(),
        ));
    }
    if !ev.has_jacobian() {
        return Err(MapLyapunovError::NoJacobian);
    }
    if k == 0 || k > dim {
        return Err(MapLyapunovError::BadShape(format!(
            "k (number of exponents) must satisfy 1 <= k <= dim = {dim}, got {k}"
        )));
    }
    if reortho_interval == 0 {
        return Err(MapLyapunovError::BadShape(
            "reortho_interval must be >= 1".to_string(),
        ));
    }
    if ic.len() < dim {
        return Err(MapLyapunovError::BadShape(format!(
            "initial state has length {}, need dim = {dim}",
            ic.len()
        )));
    }
    if p.len() < ev.n_param() {
        return Err(MapLyapunovError::BadShape(format!(
            "parameter vector has length {}, need n_param = {}",
            p.len(),
            ev.n_param()
        )));
    }

    // Live state and the next-state buffer (a map's RHS writes the next state).
    let mut x = ic[..dim].to_vec();
    let mut x_next = vec![0.0; dim];
    // The deviation frame, column-major dim × k, seeded to the leading k columns of
    // the identity (matching the released `np.eye(dim)[:, :k]`).
    let mut w = vec![0.0; dim * k];
    for j in 0..k {
        w[j * dim + j] = 1.0;
    }
    // The propagated frame W' = J · W (column-major dim × k), a scratch buffer.
    let mut w_prop = vec![0.0; dim * k];
    let mut jac = vec![0.0; dim * dim];
    let mut scratch = vec![0.0; ev.n_scratch()];
    let mut growths = vec![0.0; k];

    let mut sums = vec![0.0; k];
    let mut intervals = 0usize;
    // One iterate is a Jacobian evaluation plus a `dim × k` propagation — the
    // per-step scale the default stride is sized for.
    let mut poll = tsdyn_engine::interrupt::Poller::new();

    for i in 0..steps {
        if poll.tick() {
            return Err(MapLyapunovError::Interrupted { step: i });
        }
        // (1) Jacobian at the pre-image x_n, plus the next state in one pass.
        ev.eval_jac(&x, p, 0.0, &mut scratch, &mut x_next, &mut jac);
        // (2) advance the map; a non-finite iterate is divergence.
        if !x_next.iter().all(|v| v.is_finite()) {
            return Err(MapLyapunovError::Diverged(format!(
                "non-finite state at iterate {i}; cannot continue map iteration"
            )));
        }
        if !jac.iter().all(|v| v.is_finite()) {
            return Err(MapLyapunovError::BadShape(format!(
                "non-finite Jacobian at iterate {i}; inspect model derivatives before \
                 interpreting tangent growth"
            )));
        }
        x.copy_from_slice(&x_next[..dim]);

        // (3) propagate the deviation frame: W' = J · W (row-major J, column-major W).
        for j in 0..k {
            let wcol = &w[j * dim..(j + 1) * dim];
            let pcol = &mut w_prop[j * dim..(j + 1) * dim];
            let mut underflowed_term = false;
            let mut nonzero_term = false;
            for r in 0..dim {
                let jrow = &jac[r * dim..(r + 1) * dim];
                let mut acc = 0.0;
                for c in 0..dim {
                    let term = jrow[c] * wcol[c];
                    underflowed_term |= term == 0.0 && jrow[c] != 0.0 && wcol[c] != 0.0;
                    nonzero_term |= term != 0.0;
                    acc += term;
                }
                pcol[r] = acc;
            }
            if underflowed_term && pcol.iter().all(|v| *v == 0.0) {
                return Err(MapLyapunovError::BadShape(
                    "map tangent propagation underflowed to zero; reduce reortho_interval \
                     before interpreting contraction rates"
                        .to_string(),
                ));
            }
            if nonzero_term && pcol.iter().all(|v| *v == 0.0) {
                return Err(MapLyapunovError::BadShape(
                    "map tangent propagation vanished through cancellation; exact rank loss \
                     cannot be distinguished from roundoff. Reduce reortho_interval or \
                     rescale the model before interpreting contraction rates"
                        .to_string(),
                ));
            }
        }
        if !w_prop.iter().all(|v| v.is_finite()) {
            return Err(MapLyapunovError::BadShape(format!(
                "non-finite tangent propagation at iterate {i}; reduce reortho_interval \
                 or rescale the model before interpreting growth rates"
            )));
        }
        w.copy_from_slice(&w_prop);

        // (4) reorthonormalise every reortho_interval steps; accumulate log|R_ii|.
        if (i + 1) % reortho_interval == 0 {
            match mgs_renormalise(&mut w, dim, k, &mut growths) {
                Ok(()) => (),
                Err(TangentQrError::NonFinite) => {
                    return Err(MapLyapunovError::BadShape(format!(
                        "non-finite tangent reorthonormalisation at iterate {i}; reduce \
                         reortho_interval or rescale the model before interpreting growth rates"
                    )))
                }
                Err(TangentQrError::UnresolvedRank) => {
                    return Err(MapLyapunovError::BadShape(
                        UNRESOLVED_TANGENT_RANK.to_string(),
                    ))
                }
            }
            for j in 0..k {
                sums[j] += growths[j];
            }
            intervals += 1;
        }
    }

    if intervals == 0 {
        // No interval completed (steps < reortho_interval) — the released path's
        // `intervals == 0` soft failure.
        return Err(MapLyapunovError::Diverged(
            "no reorthonormalisation interval completed (steps < reortho_interval)".to_string(),
        ));
    }

    let elapsed = (intervals * reortho_interval) as f64;
    let exponents: Vec<f64> = sums.iter().map(|s| s / elapsed).collect();
    Ok(MapLyapunovOutcome {
        exponents,
        intervals,
    })
}
