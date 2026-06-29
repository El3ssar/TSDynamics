//! Discrete-map Lyapunov spectrum, in Rust (stream `perf/map-lyapunov-kernel`).
//!
//! [`map_lyapunov`] runs the **entire** QR tangent-map iteration of the Python
//! `TangentSystem._accumulate_map` inside one engine call: it iterates the lowered
//! map `f` while propagating `k` tangent vectors through the lowered Jacobian
//! `J(x_n)` (the **pre-image** convention — `J` evaluated at the state *before* the
//! step), reorthonormalises the deviation frame every `reortho_interval` steps with
//! a hand-rolled modified Gram–Schmidt, accumulates `Σ log|R_ii|`, and returns the
//! time-averaged spectrum (the maximal exponent is the leading entry). Maps are
//! ~6000× slower native than the per-step Python QR loop they replace, because that
//! loop pays a Python→FFI / NumPy round-trip *per iterate*; this kernel pays none.
//!
//! # Algorithm (line-for-line with the released Python loop)
//!
//! Mirroring `_accumulate_map` exactly so the spectrum is reproduced to tolerance:
//!
//! 1. evaluate `J = ∂f/∂u` at the current state `x_n` (the **pre-image**);
//! 2. step the map `x_{n+1} = f(x_n)` (a non-finite iterate is divergence);
//! 3. update the deviation frame `W ← J · W`;
//! 4. every `reortho_interval` steps, QR-reorthonormalise `W = Q·R`, keep `Q`,
//!    accumulate `log|R_ii|` (flushed to `tiny` to dodge `log 0`).
//!
//! So the accumulated product is `J(x_{N-1}) ··· J(x_0)` — the exact tangent map —
//! and `λ_i = (Σ log|R_ii|) / (#intervals · reortho_interval)`.
//!
//! # Why modified Gram–Schmidt (not Householder)
//!
//! The released path calls `numpy.linalg.qr` (Householder). The exponents depend
//! only on the **magnitudes** `|R_ii|` of the upper-triangular factor, which a
//! modified Gram–Schmidt reproduces (`R_ii = ‖ŵ_i‖` after orthogonalising column
//! `i` against the earlier orthonormal columns) — identical up to the column-sign
//! convention that `|·|` already absorbs. MGS over a small `dim × k` frame needs no
//! heavy linalg dependency and is allocation-free here.
//!
//! # Determinism & equivalence
//!
//! No RNG, no rayon: the iteration is deterministic and the per-step numerics are
//! the engine's. Driven over the *same* lowered IR tape, the interpreter and the
//! Cranelift JIT agree **bit-for-bit** (`eval`/`eval_jac` are bit-identical between
//! them). Against the released Python path the kernel differs only by the lowered
//! IR vs the pure-Python `_step`/`_jacobian` floating-point order (the WS-MAPITER
//! IR-vs-NumPy caveat) — the same attractor, the same spectrum to tolerance.

use tsdyn_ir::Evaluator;

/// Smallest positive normal `f64` — the floor the released path applies to a
/// diagonal `|R_ii|` before `log`, so a momentarily collapsed direction yields a
/// large finite negative contribution instead of `-inf` (mirrors
/// `numpy.finfo(float).tiny`).
const TINY: f64 = f64::MIN_POSITIVE;

/// Why a map Lyapunov run could not be set up or completed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MapLyapunovError {
    /// A buffer length / shape invariant disagrees with the tape, or `k` is out of
    /// range. → `ValueError`.
    BadShape(String),
    /// The tape carries no Jacobian (`with_jacobian=False`) — the tangent map needs
    /// `∂f/∂u`. → `ValueError`.
    NoJacobian,
    /// The iteration diverged: a non-finite iterate, Jacobian, deviation frame, or
    /// reorthonormalisation before completing all steps. → `RuntimeError`.
    Diverged(String),
}

impl core::fmt::Display for MapLyapunovError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MapLyapunovError::BadShape(m) => f.write_str(m),
            MapLyapunovError::NoJacobian => f.write_str(
                "map Lyapunov needs the step Jacobian, but the tape was compiled without one \
                 (with_jacobian=False)",
            ),
            MapLyapunovError::Diverged(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for MapLyapunovError {}

/// The accumulated spectrum a [`map_lyapunov`] run returns.
#[derive(Clone, Debug, PartialEq)]
pub struct MapLyapunovOutcome {
    /// The `k` Lyapunov exponents, largest first (QR order) — the time-averaged
    /// `log|R_ii|`.
    pub exponents: Vec<f64>,
    /// The number of reorthonormalisation intervals completed (the `_elapsed` the
    /// released path divides by is `intervals · reortho_interval`).
    pub intervals: usize,
}

/// In-place modified Gram–Schmidt over the column-major `dim × k` frame `w`.
///
/// Column `j` occupies `w[j*dim .. (j+1)*dim]`. Orthonormalises the columns left to
/// right, writing the orthonormal `Q` back into `w` and the diagonal stretch
/// factors `|R_jj|` into `rdiag` (`rdiag[j] = ‖ŵ_j‖` after removing the projections
/// onto columns `0..j`). Returns `false` if any norm or coefficient is non-finite
/// (the deviation frame blew up). A column whose norm underflows below [`TINY`] is
/// recorded as [`TINY`] and replaced with a zero vector — exactly the released
/// path's `diag = where(diag < tiny, tiny, diag)` guard, so a collapsed direction
/// contributes a large finite negative exponent rather than `NaN`.
#[inline]
fn mgs(w: &mut [f64], dim: usize, k: usize, rdiag: &mut [f64]) -> bool {
    for j in 0..k {
        // Orthogonalise column j against the already-orthonormal columns 0..j.
        for i in 0..j {
            // r_ij = q_i · w_j
            let mut dot = 0.0;
            for d in 0..dim {
                dot += w[i * dim + d] * w[j * dim + d];
            }
            if !dot.is_finite() {
                return false;
            }
            // w_j ← w_j − r_ij q_i
            for d in 0..dim {
                w[j * dim + d] -= dot * w[i * dim + d];
            }
        }
        // r_jj = ‖w_j‖
        let mut norm2 = 0.0;
        for d in 0..dim {
            let v = w[j * dim + d];
            norm2 += v * v;
        }
        if !norm2.is_finite() {
            return false;
        }
        let norm = norm2.sqrt();
        if norm < TINY {
            // Collapsed direction: record the floor and zero the column so the
            // subsequent columns orthogonalise against a clean (zero) vector — the
            // released `where(diag < tiny, tiny, diag)` outcome.
            rdiag[j] = TINY;
            for d in 0..dim {
                w[j * dim + d] = 0.0;
            }
        } else {
            rdiag[j] = norm;
            let inv = 1.0 / norm;
            for d in 0..dim {
                w[j * dim + d] *= inv;
            }
        }
    }
    true
}

/// Compute the discrete-map Lyapunov spectrum (the maximal exponent is the leading
/// entry) over the lowered map RHS + Jacobian `ev`.
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
    let mut rdiag = vec![0.0; k];

    let mut sums = vec![0.0; k];
    let mut intervals = 0usize;

    for i in 0..steps {
        // (1) Jacobian at the pre-image x_n, plus the next state in one pass.
        ev.eval_jac(&x, p, 0.0, &mut scratch, &mut x_next, &mut jac);
        if !jac.iter().all(|v| v.is_finite()) {
            return Err(MapLyapunovError::Diverged(format!(
                "non-finite Jacobian at iterate {i} (the map diverged)"
            )));
        }
        // (2) advance the map; a non-finite iterate is divergence.
        if !x_next.iter().all(|v| v.is_finite()) {
            return Err(MapLyapunovError::Diverged(format!(
                "non-finite state at iterate {i} (the map diverged)"
            )));
        }
        x.copy_from_slice(&x_next[..dim]);

        // (3) propagate the deviation frame: W' = J · W (row-major J, column-major W).
        for j in 0..k {
            let wcol = &w[j * dim..(j + 1) * dim];
            let pcol = &mut w_prop[j * dim..(j + 1) * dim];
            for r in 0..dim {
                let jrow = &jac[r * dim..(r + 1) * dim];
                let mut acc = 0.0;
                for c in 0..dim {
                    acc += jrow[c] * wcol[c];
                }
                pcol[r] = acc;
            }
        }
        if !w_prop.iter().all(|v| v.is_finite()) {
            return Err(MapLyapunovError::Diverged(format!(
                "non-finite deviation frame at iterate {i} (the map diverged)"
            )));
        }
        w.copy_from_slice(&w_prop);

        // (4) reorthonormalise every reortho_interval steps; accumulate log|R_ii|.
        if (i + 1) % reortho_interval == 0 {
            if !mgs(&mut w, dim, k, &mut rdiag) {
                return Err(MapLyapunovError::Diverged(format!(
                    "non-finite reorthonormalisation at iterate {i} (the map diverged)"
                )));
            }
            for j in 0..k {
                let d = if rdiag[j] < TINY { TINY } else { rdiag[j] };
                sums[j] += d.ln();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::VmEval;
    use tsdyn_ir::TapeBuilder;
    use tsdyn_vm::Interpreter;

    /// Hénon map `(x, y) ← (1 - a x² + y, b x)` with `a, b` folded in, **carrying
    /// its analytic Jacobian** `[[-2 a x, 1], [b, 0]]` (the lowered-map convention,
    /// n_param == 0).
    fn henon_jac(a: f64, bcoef: f64) -> Interpreter {
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let y = b.state(1);
        let one = b.constant(1.0);
        let ac = b.constant(a);
        let bc = b.constant(bcoef);
        let xx = b.mul(x, x);
        let axx = b.mul(ac, xx);
        let omaxx = b.sub(one, axx);
        let nx = b.add(omaxx, y);
        let ny = b.mul(bc, x);
        // Jacobian rows: d(nx)/dx = -2 a x, d(nx)/dy = 1 ; d(ny)/dx = b, d(ny)/dy = 0
        let two = b.constant(2.0);
        let twoa = b.mul(two, ac);
        let twoax = b.mul(twoa, x);
        let neg_twoax = b.neg(twoax);
        let zero = b.constant(0.0);
        Interpreter::new(
            b.finish(&[nx, ny], &[neg_twoax, one, bc, zero], 2, 0)
                .unwrap(),
        )
    }

    /// `x ← r x (1 - x)` logistic with `r` folded in, carrying `d/dx = r(1 - 2x)`.
    fn logistic_jac(r: f64) -> Interpreter {
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let rc = b.constant(r);
        let one = b.constant(1.0);
        let two = b.constant(2.0);
        let omx = b.sub(one, x);
        let rx = b.mul(rc, x);
        let nx = b.mul(rx, omx);
        let twox = b.mul(two, x);
        let om2x = b.sub(one, twox);
        let dnx = b.mul(rc, om2x);
        Interpreter::new(b.finish(&[nx], &[dnx], 1, 0).unwrap())
    }

    #[test]
    fn henon_spectrum_matches_literature() {
        // Hénon at (1.4, 0.3): λ ≈ [0.419, -1.623] (Sprott 2003).
        let ev = VmEval::new(henon_jac(1.4, 0.3));
        let out = map_lyapunov(&ev, &[], &[0.1, 0.1], 10_000, 2, 1).unwrap();
        assert_eq!(out.exponents.len(), 2);
        assert!(
            (out.exponents[0] - 0.419).abs() < 0.05,
            "λ1 = {} (want ≈ 0.419)",
            out.exponents[0]
        );
        assert!(
            (out.exponents[1] - (-1.623)).abs() < 0.05,
            "λ2 = {} (want ≈ -1.623)",
            out.exponents[1]
        );
        // Descending order (QR convention).
        assert!(out.exponents[0] > out.exponents[1]);
    }

    #[test]
    fn logistic_r4_top_exponent_is_ln2() {
        // The fully-chaotic logistic (r = 4) has λ = ln 2 ≈ 0.6931.
        let ev = VmEval::new(logistic_jac(4.0));
        let out = map_lyapunov(&ev, &[], &[0.1], 50_000, 1, 1).unwrap();
        assert!(
            (out.exponents[0] - std::f64::consts::LN_2).abs() < 0.02,
            "λ = {} (want ≈ ln 2 = 0.6931)",
            out.exponents[0]
        );
    }

    #[test]
    fn partial_spectrum_k_less_than_dim() {
        // Requesting k = 1 of a 2-D map returns only the maximal exponent.
        let ev = VmEval::new(henon_jac(1.4, 0.3));
        let full = map_lyapunov(&ev, &[], &[0.1, 0.1], 8000, 2, 1).unwrap();
        let top = map_lyapunov(&ev, &[], &[0.1, 0.1], 8000, 1, 1).unwrap();
        assert_eq!(top.exponents.len(), 1);
        // The leading exponent agrees (same orbit, same leading direction).
        assert!(
            (top.exponents[0] - full.exponents[0]).abs() < 1e-9,
            "top {} vs full[0] {}",
            top.exponents[0],
            full.exponents[0]
        );
    }

    #[test]
    fn reortho_interval_is_answer_preserving() {
        // Reorthonormalising every step vs every 5 steps gives the same spectrum to
        // tolerance (the variational dynamics is linear between reorthos).
        let ev = VmEval::new(henon_jac(1.4, 0.3));
        let every1 = map_lyapunov(&ev, &[], &[0.1, 0.1], 10_000, 2, 1).unwrap();
        let every5 = map_lyapunov(&ev, &[], &[0.1, 0.1], 10_000, 2, 5).unwrap();
        for (a, b) in every1.exponents.iter().zip(every5.exponents.iter()) {
            assert!((a - b).abs() < 1e-2, "{a} vs {b}");
        }
    }

    #[test]
    fn rejects_tape_without_jacobian() {
        // A map tape lowered without a Jacobian cannot drive the tangent map.
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let two = b.constant(2.0);
        let nx = b.mul(two, x);
        let ev = VmEval::new(Interpreter::new(b.finish(&[nx], &[], 1, 0).unwrap()));
        let err = map_lyapunov(&ev, &[], &[0.5], 100, 1, 1).unwrap_err();
        assert_eq!(err, MapLyapunovError::NoJacobian);
    }

    #[test]
    fn rejects_bad_k() {
        let ev = VmEval::new(henon_jac(1.4, 0.3));
        assert!(matches!(
            map_lyapunov(&ev, &[], &[0.1, 0.1], 100, 0, 1).unwrap_err(),
            MapLyapunovError::BadShape(_)
        ));
        assert!(matches!(
            map_lyapunov(&ev, &[], &[0.1, 0.1], 100, 3, 1).unwrap_err(),
            MapLyapunovError::BadShape(_)
        ));
    }

    #[test]
    fn diverges_loudly() {
        // x ← 2x doubling, but carrying a Jacobian (d/dx = 2): the orbit overflows,
        // and the run must raise rather than return a poisoned spectrum.
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let two = b.constant(2.0);
        let nx = b.mul(two, x);
        let ev = VmEval::new(Interpreter::new(b.finish(&[nx], &[two], 1, 0).unwrap()));
        let err = map_lyapunov(&ev, &[], &[1.0], 100_000, 1, 1).unwrap_err();
        assert!(matches!(err, MapLyapunovError::Diverged(_)), "got {err:?}");
    }
}
