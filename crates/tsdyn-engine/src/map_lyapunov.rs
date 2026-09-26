//! Discrete-map Lyapunov spectrum, in Rust (stream `perf/map-lyapunov-kernel`).
//!
//! [`map_lyapunov`] runs the **entire** QR tangent-map iteration of the Python
//! `TangentSystem._accumulate_map` inside one engine call: it iterates the lowered
//! map `f` while propagating `k` tangent vectors through the lowered Jacobian
//! `J(x_n)` (the **pre-image** convention — `J` evaluated at the state *before* the
//! step), reorthonormalises the deviation frame every `reortho_interval` steps with
//! a hand-rolled modified Gram–Schmidt, accumulates `Σ log|R_ii|`, and returns the
//! time-averaged rates in tangent-column order. The public analysis sorts them. Maps are
//! integrated without the Python→FFI / NumPy round-trip per iterate.
//!
//! # Algorithm (line-for-line with the released Python loop)
//!
//! Mirroring `_accumulate_map` exactly so the spectrum is reproduced to tolerance:
//!
//! 1. evaluate `J = ∂f/∂u` at the current state `x_n` (the **pre-image**);
//! 2. step the map `x_{n+1} = f(x_n)` (a non-finite iterate is divergence);
//! 3. update the deviation frame `W ← J · W`;
//! 4. every `reortho_interval` steps, QR-reorthonormalise `W = Q·R`, keep `Q`,
//!    accumulate logarithmic growth, retaining exact zero columns as `-inf`.
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

use crate::lyapunov::{mgs_renormalise, TangentQrError, UNRESOLVED_TANGENT_RANK};

/// Why a map Lyapunov run could not be set up or completed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MapLyapunovError {
    /// A buffer length / shape invariant disagrees with the tape, or `k` is out of
    /// range. → `ValueError`.
    BadShape(String),
    /// The tape carries no Jacobian (`with_jacobian=False`) — the tangent map needs
    /// `∂f/∂u`. → `ValueError`.
    NoJacobian,
    /// A non-finite state iterate before completing all steps. → `RuntimeError`.
    Diverged(String),
    /// The embedder's interrupt hook stopped the run (see [`crate::interrupt`])
    /// — normally a Ctrl-C at the Python prompt. A `steps`-heavy spectrum is one
    /// of the longest single calls the engine offers, so it is one of the calls
    /// that most needs to be escapable.
    Interrupted {
        /// The iterate index reached when the interrupt was observed.
        step: usize,
    },
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
            MapLyapunovError::Interrupted { step } => {
                write!(f, "interrupted at iterate {step}")
            }
        }
    }
}

impl std::error::Error for MapLyapunovError {}

/// The accumulated spectrum a [`map_lyapunov`] run returns.
#[derive(Clone, Debug, PartialEq)]
pub struct MapLyapunovOutcome {
    /// The `k` Lyapunov exponents in QR column order — the time-averaged
    /// `log|R_ii|`.
    pub exponents: Vec<f64>,
    /// The number of reorthonormalisation intervals completed (the `_elapsed` the
    /// released path divides by is `intervals · reortho_interval`).
    pub intervals: usize,
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
    let mut poll = crate::interrupt::Poller::new();

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

    fn linear_plane_jac(a: f64, b: f64, c: f64, d: f64) -> Interpreter {
        let mut tape = TapeBuilder::new();
        let x = tape.state(0);
        let y = tape.state(1);
        let ac = tape.constant(a);
        let bc = tape.constant(b);
        let cc = tape.constant(c);
        let dc = tape.constant(d);
        let ax = tape.mul(ac, x);
        let by = tape.mul(bc, y);
        let cx = tape.mul(cc, x);
        let dy = tape.mul(dc, y);
        let nx = tape.add(ax, by);
        let ny = tape.add(cx, dy);
        Interpreter::new(tape.finish(&[nx, ny], &[ac, bc, cc, dc], 2, 0).unwrap())
    }

    #[test]
    fn logarithmic_rates_preserve_tiny_huge_and_exact_zero_multipliers() {
        for factor in [0.0, 1e-320, 1e-300, 1e-200, 1e200, -1e200] {
            let ev = VmEval::new(linear_plane_jac(factor, 0.0, 0.0, 1.0));
            let out = map_lyapunov(&ev, &[], &[0.0, 0.0], 6, 2, 1).unwrap();
            if factor == 0.0 {
                assert_eq!(out.exponents[0], f64::NEG_INFINITY);
            } else {
                assert!((out.exponents[0] - factor.abs().ln()).abs() < 1e-10);
            }
            assert_eq!(out.exponents[1], 0.0);
        }
    }

    #[test]
    fn tangent_overflow_does_not_claim_that_a_zero_orbit_diverged() {
        let ev = VmEval::new(linear_plane_jac(1e200, 0.0, 0.0, 1.0));
        let err = map_lyapunov(&ev, &[], &[0.0, 0.0], 2, 2, 2).unwrap_err();
        assert!(matches!(err, MapLyapunovError::BadShape(_)));
        assert!(err.to_string().contains("non-finite tangent propagation"));
        assert!(err.to_string().contains("reortho_interval"));
    }

    #[test]
    fn zero_columns_do_not_hide_or_regenerate_surviving_directions() {
        let ev = VmEval::new(linear_plane_jac(0.0, 0.0, 0.0, 2.0));
        let out = map_lyapunov(&ev, &[], &[0.0, 0.0], 6, 2, 1).unwrap();
        assert_eq!(out.exponents[0], f64::NEG_INFINITY);
        assert!((out.exponents[1] - 2.0_f64.ln()).abs() < 1e-14);
        let nilpotent = VmEval::new(linear_plane_jac(0.0, 2.0, 0.0, 0.0));
        let one = map_lyapunov(&nilpotent, &[], &[0.0, 0.0], 1, 2, 1).unwrap();
        assert_eq!(one.exponents[0], f64::NEG_INFINITY);
        assert!((one.exponents[1] - 2.0_f64.ln()).abs() < 1e-14);
        let two = map_lyapunov(&nilpotent, &[], &[0.0, 0.0], 2, 2, 1).unwrap();
        assert_eq!(two.exponents, vec![f64::NEG_INFINITY; 2]);
    }

    #[test]
    fn dependent_nonzero_frames_and_underflow_are_explicitly_refused() {
        let dependent = VmEval::new(linear_plane_jac(1.0, 1.0, 0.0, 0.0));
        let err = map_lyapunov(&dependent, &[], &[0.0, 0.0], 2, 2, 1).unwrap_err();
        assert!(matches!(err, MapLyapunovError::BadShape(_)));
        assert!(err
            .to_string()
            .contains("dependent or numerically unresolved"));
        let small = VmEval::new(linear_plane_jac(1e-300, 0.0, 0.0, 1.0));
        let err = map_lyapunov(&small, &[], &[0.0, 0.0], 2, 2, 2).unwrap_err();
        assert!(matches!(err, MapLyapunovError::BadShape(_)));
        assert!(err.to_string().contains("underflowed to zero"));
    }

    #[test]
    fn cancellation_to_zero_is_not_certified_as_exact_collapse() {
        let a = f64::from_bits(1.0_f64.to_bits() + 1);
        // In exact arithmetic on these represented coefficients, A²=2^-104 I.
        let ev = VmEval::new(linear_plane_jac(a, 1.0, -(a * a), -a));
        let err = map_lyapunov(&ev, &[], &[0.0, 0.0], 2, 2, 2).unwrap_err();
        assert!(matches!(err, MapLyapunovError::BadShape(_)));
        assert!(err.to_string().contains("cancellation"));
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
        // This seed's single direction converges to the leading Hénon rate.
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

    /// An armed interrupt stops the QR iteration. A `steps`-heavy map spectrum
    /// is one of the longest single calls the engine offers, so it has to be
    /// escapable.
    #[test]
    fn an_armed_interrupt_stops_the_qr_iteration() {
        let _stop = crate::interrupt::testing::force_stop();
        let _armed = crate::interrupt::arm();

        let ev = VmEval::new(henon_jac(1.4, 0.3));
        let steps = 50 * crate::interrupt::POLL_STRIDE;
        let err = map_lyapunov(&ev, &[], &[0.1, 0.1], steps, 2, 1).unwrap_err();
        assert!(
            matches!(err, MapLyapunovError::Interrupted { .. }),
            "got {err:?}"
        );
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
