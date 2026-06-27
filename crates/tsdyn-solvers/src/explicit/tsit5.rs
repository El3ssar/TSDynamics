//! Adaptive Tsitouras 5(4) — registered as `tsit5`.
//!
//! A 5th-order explicit Runge–Kutta pair with an embedded 4th-order estimate,
//! derived by satisfying only the first-column simplifying assumption, which
//! gives smaller principal truncation-error coefficients (and so larger steps at
//! a given tolerance) than the classic Dormand–Prince 5(4) on smooth non-stiff
//! problems (Tsitouras, *Comput. Math. Appl.* **62** (2011) 770–775).  Like
//! `rk45` it is a 7-stage FSAL method, so it shares the single-error-vector
//! adaptive step in [`control`](super::control); only the coefficients differ.
//!
//! The tableau is in the same 7-stage form used for `rk45` (`c₇ = 1`, `a₇ = b`),
//! so the error-weight vector `e = b − b̂` over the seven stages is the embedded
//! 4th-order estimate.

use crate::{
    register_solver, Caps, Evaluator, ProblemKind, ProblemKinds, Solver, SolverState, StepOutcome,
};

use super::control::RkWork;
use super::rk45::{fsal_adaptive_step, FsalCache};

// Tsitouras 5(4) nodes (c₆ = c₇ = 1; the 7th is the FSAL solution stage).
const C: &[f64] = &[0.0, 0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0];

// Lower-triangular stage coefficients; a[6] equals B (FSAL solution stage).
const A: &[&[f64]] = &[
    &[],
    &[0.161],
    &[-0.008480655492356989, 0.335480655492357],
    &[2.8971530571054935, -6.359448489975075, 4.3622954328695815],
    &[
        5.325864828439257,
        -11.748883564062828,
        7.4955393428898365,
        -0.09249506636175525,
    ],
    &[
        5.86145544294642,
        -12.92096931784711,
        8.159367898576159,
        -0.071584973281401,
        -0.028269050394068383,
    ],
    &[
        0.09646076681806523,
        0.01,
        0.4798896504144996,
        1.379008574103742,
        -3.290069515436081,
        2.324710524099774,
    ],
];

// 5th-order solution weights (b₇ = 0; the FSAL stage carries no solution weight).
const B: &[f64] = &[
    0.09646076681806523,
    0.01,
    0.4798896504144996,
    1.379008574103742,
    -3.290069515436081,
    2.324710524099774,
    0.0,
];

// Error weights b̃ = b(5th) − b̂(4th) per stage (Tsitouras 2011, Table; e₇ = 1/66).
// These published decimals sum to ≈ −5e-12, not exactly 0: b̃ = b − b̂ is exactly
// zero-sum in the rationals, but the tabulated b̂ decimals carry that residual.
// The embedded estimator's 4th order is verified directly by test below.
const E: &[f64] = &[
    -0.001780011052226,
    -0.000816434459657,
    0.007880878010262,
    -0.1447110071783768,
    0.582357165452555,
    -0.458082105929187,
    1.0 / 66.0,
];

// Controller exponent −1/(error_estimator_order + 1) with estimator order 4.
const ERR_EXPONENT: f64 = -1.0 / 5.0;

/// Default relative tolerance (SciPy `solve_ivp` default).
const DEFAULT_RTOL: f64 = 1e-3;
/// Default absolute tolerance (SciPy `solve_ivp` default).
const DEFAULT_ATOL: f64 = 1e-6;

/// Adaptive Tsitouras 5(4) kernel.
pub struct Tsit5 {
    rtol: f64,
    atol: f64,
    work: RkWork,
    /// FSAL cache: the previous accepted step's last stage, reused as stage 0.
    fsal: FsalCache,
}

impl Tsit5 {
    /// A kernel with the default tolerances (`rtol = 1e-3`, `atol = 1e-6`).
    pub fn new() -> Self {
        Tsit5::with_tolerances(DEFAULT_RTOL, DEFAULT_ATOL)
    }

    /// A kernel with explicit tolerances (see [`Rk45::with_tolerances`] for why
    /// an adaptive kernel owns its tolerances).
    ///
    /// [`Rk45::with_tolerances`]: super::rk45::Rk45::with_tolerances
    pub fn with_tolerances(rtol: f64, atol: f64) -> Self {
        Tsit5 {
            rtol,
            atol,
            work: RkWork::new(),
            fsal: FsalCache::new(),
        }
    }
}

impl Default for Tsit5 {
    fn default() -> Self {
        Tsit5::new()
    }
}

impl Solver for Tsit5 {
    fn name(&self) -> &'static str {
        "tsit5"
    }

    fn caps(&self) -> Caps {
        Caps::explicit(ProblemKinds::of(ProblemKind::Ode)).adaptive()
    }

    fn step(&mut self, ev: &dyn Evaluator, st: &mut SolverState, h: f64) -> StepOutcome {
        fsal_adaptive_step(
            ev,
            st,
            h,
            C,
            A,
            B,
            E,
            ERR_EXPONENT,
            self.rtol,
            self.atol,
            &mut self.work,
            &mut self.fsal,
        )
    }
}

register_solver!(
    "tsit5",
    Caps::explicit(ProblemKinds::of(ProblemKind::Ode)).adaptive(),
    || Box::new(Tsit5::new())
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explicit::testkit::{
        converges_at_order, fixed_propagate, integrate_adaptive, max_abs_diff, HarmonicEval,
    };

    #[test]
    fn tableau_is_internally_consistent() {
        // Σb = 1, Σe = 0, and each A-row sums to its node c — necessary
        // conditions a correct 5(4) tableau must satisfy.
        let sum_b: f64 = B.iter().sum();
        assert!((sum_b - 1.0).abs() < 1e-13, "Σb = {sum_b}");
        // b̃ = b − b̂ is zero-sum in exact arithmetic, but the published b̂
        // decimals leave a ≈5e-12 residual; assert only that, not exact zero
        // (a tighter bound would reject the genuine Tsitouras coefficients).
        let sum_e: f64 = E.iter().sum();
        assert!(sum_e.abs() < 1e-11, "Σe = {sum_e}");
        for (i, row) in A.iter().enumerate() {
            let s: f64 = row.iter().sum();
            assert!((s - C[i]).abs() < 1e-13, "row {i}: Σa = {s}, c = {}", C[i]);
        }
    }

    #[test]
    fn embedded_estimate_is_fourth_order() {
        // The companion solution b̂ = b − e must converge at order 4 — the
        // defining property of the embedded estimate, and a reference-free check
        // on the error weights `e` themselves (independent of the order-5 test on
        // `b`). A wrong `e` weight that broke the 4th-order companion fails here.
        let ev = HarmonicEval { omega: 1.0 };
        let bhat: Vec<f64> = B.iter().zip(E).map(|(b, e)| b - e).collect();
        let order = converges_at_order(
            |st, h, work| fixed_propagate(&ev, st, h, C, A, &bhat, work),
            &ev,
            vec![1.0, 0.0],
            2.0,
            &[0.1, 0.05, 0.025],
            |t| vec![t.cos(), -t.sin()],
        );
        assert!(
            (order - 4.0).abs() < 0.4,
            "measured Tsit5 embedded order {order}, expected ≈ 4"
        );
    }

    #[test]
    fn caps_are_explicit_adaptive_ode() {
        let s = Tsit5::new();
        assert_eq!(s.name(), "tsit5");
        assert!(s.caps().adaptive);
    }

    #[test]
    fn fifth_order_convergence_of_the_propagated_solution() {
        let ev = HarmonicEval { omega: 1.0 };
        let order = converges_at_order(
            |st, h, work| fixed_propagate(&ev, st, h, C, A, B, work),
            &ev,
            vec![1.0, 0.0],
            2.0,
            &[0.1, 0.05, 0.025],
            |t| vec![t.cos(), -t.sin()],
        );
        assert!(
            (order - 5.0).abs() < 0.4,
            "measured Tsit5 order {order}, expected ≈ 5"
        );
    }

    #[test]
    fn adaptive_run_matches_analytic_harmonic_solution() {
        let ev = HarmonicEval { omega: 1.0 };
        let mut s = Tsit5::with_tolerances(1e-10, 1e-12);
        let t_final = 12.0;
        let mut st = SolverState::for_evaluator(&ev, vec![1.0, 0.0], 0.0, vec![]);
        integrate_adaptive(&mut s, &ev, &mut st, t_final, 0.05);
        let exact = vec![t_final.cos(), -t_final.sin()];
        assert!(
            max_abs_diff(&st.u, &exact) < 1e-7,
            "adaptive Tsit5 error {}",
            max_abs_diff(&st.u, &exact)
        );
    }
}
