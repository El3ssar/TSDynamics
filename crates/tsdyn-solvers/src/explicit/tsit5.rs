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

use super::control::{dense_poly, RkWork};
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
// The embedded estimator's 4th order is verified directly by test below, and each
// coefficient is pinned against the published table by `e_coefficients_match_table`.
//
// `E` drives ONLY the embedded error estimate (step-size control); the accepted
// 5th-order solution comes from `B` alone. So a perturbation here can only shift
// the step *sequence* near tolerance — never an accepted result — which is why the
// `interp == jit` / golden-fixture invariants are insensitive to it.
const E: &[f64] = &[
    -0.001780011052226,
    -0.000816434459657,
    0.007880878010262,
    -0.1447110071732629,
    0.582357165452555,
    -0.458082105929187,
    1.0 / 66.0,
];

// Tsitouras' free continuous extension — the dense-output coefficients `b*_i(θ)`,
// one row of four θ-powers per stage (lowest power first, no constant term),
// evaluated by `control::dense_poly` exactly as `rk45`'s `P` is:
// `u(t₀ + θh) = u₀ + h·Σ_i k_i·(P[i][0]θ + P[i][1]θ² + P[i][2]θ³ + P[i][3]θ⁴)`.
//
// Tsitouras (*Comput. Math. Appl.* **62** (2011) 770–775) derives the pair
// together with this interpolant; the decimals are the canonical published ones
// (the same values `OrdinaryDiffEq.jl`'s `Tsit5` carries as `r₁₁…r₇₄`).
//
// Two reference-free checks pin them, because a re-typed decimal table is exactly
// where a transcription slip hides: each row must sum to the solution weight
// `B[i]` — the θ = 1 identity, verified below to 4.5e-15, i.e. to the roundoff of
// the published decimals themselves — and the interpolant's measured order under
// h-halving must be 5 locally (`interpolate_is_a_fifth_order_local_extension`,
// measured 4.98–5.00). Neither could pass on guessed coefficients.
//
// Free: zero extra RHS evaluations, a pure linear combination of the seven stages
// the step already computed.
const P: &[&[f64]] = &[
    &[
        1.0,
        -2.763706197274826,
        2.9132554618219126,
        -1.0530884977290216,
    ],
    &[0.0, 0.13169999999999998, -0.2234, 0.1017],
    &[
        0.0,
        3.9302962368947516,
        -5.941033872131505,
        2.490627285651253,
    ],
    &[
        0.0,
        -12.411077166933676,
        30.33818863028232,
        -16.548102889244902,
    ],
    &[0.0, 37.50931341651104, -88.1789048947664, 47.37952196281928],
    &[
        0.0,
        -27.896526289197286,
        65.09189467479366,
        -34.8706578614966,
    ],
    &[0.0, 1.5, -4.0, 2.5],
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
        Caps::explicit(ProblemKinds::of(ProblemKind::Ode))
            .adaptive()
            .with_dense()
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

    fn interpolate(&self, u0: &[f64], h: f64, theta: f64, out: &mut [f64]) -> bool {
        // See `Rk45::interpolate`: valid only between an accepted `step` and the
        // next `step` call, with that step's own `u0`/`h`.
        #[cfg(debug_assertions)]
        debug_assert!(
            self.work.dense_valid,
            "interpolate() called outside the window of an accepted step"
        );
        if self.work.k.len() != P.len() || out.len() != u0.len() {
            return false;
        }
        dense_poly(u0, h, theta, &self.work.k, P, out);
        true
    }
}

register_solver!(
    "tsit5",
    Caps::explicit(ProblemKinds::of(ProblemKind::Ode))
        .adaptive()
        .with_dense(),
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
    fn e_coefficients_match_table() {
        // Pin EVERY embedded error-weight `e_i` against the canonical Tsitouras
        // (2011) table so an 11th-significant-digit transcription slip cannot drift
        // back in silently (the original `E[3]` carried a ~5.1e-12 typo:
        // -0.1447110071783768 instead of -0.1447110071732629). `E` drives only the
        // step controller, so such a slip never shows up in an accepted-value test
        // — only an exact-coefficient pin like this one catches it.
        const E_TABLE: &[f64] = &[
            -0.001780011052226,
            -0.000816434459657,
            0.007880878010262,
            -0.1447110071732629,
            0.582357165452555,
            -0.458082105929187,
            1.0 / 66.0,
        ];
        assert_eq!(E.len(), E_TABLE.len());
        for (i, (&got, &want)) in E.iter().zip(E_TABLE).enumerate() {
            assert_eq!(got.to_bits(), want.to_bits(), "E[{i}] = {got}, want {want}");
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

    /// **S1 (tsit5).** Pin the dense-output coefficients structurally.
    ///
    /// Unlike `rk45`'s `P`, which has an exact published rational form, the
    /// Tsitouras interpolant is only tabulated as decimals — so the pin here is
    /// the *identity* those decimals must satisfy rather than a re-typed copy of
    /// the same numbers (which would pin nothing): each row must sum to that
    /// stage's solution weight `B[i]`, which is what makes `theta = 1` reproduce
    /// the propagated solution. The published decimals satisfy it to ~4.5e-15,
    /// i.e. to their own printed precision; a single mistyped digit anywhere in
    /// the 28-entry table breaks it by orders of magnitude.
    ///
    /// The independent accuracy check is
    /// `interpolate_is_a_fifth_order_local_extension` below — an identity and an
    /// order measurement together cannot both pass on guessed coefficients.
    #[test]
    fn dense_p_matrix_row_sums_match_the_solution_weights() {
        assert_eq!(P.len(), C.len(), "one dense row per stage");
        for (i, (row, &b)) in P.iter().zip(B).enumerate() {
            assert_eq!(row.len(), 4, "row {i} must be a quartic in theta");
            let s: f64 = row.iter().sum();
            assert!(
                (s - b).abs() < 1e-14,
                "row {i} sums to {s}, but B[{i}] = {b} (delta {})",
                s - b
            );
        }
        // The first row is the only one with a constant-in-theta leading term of
        // 1: at theta -> 0 the extension must reduce to `u0 + h*theta*f(u0)`, so
        // stage 0 carries all the linear weight.
        assert_eq!(P[0][0], 1.0);
        for (i, row) in P.iter().enumerate().skip(1) {
            assert_eq!(row[0], 0.0, "row {i} must have no linear-in-theta term");
        }
    }

    /// **S2/S3 (tsit5).** Endpoint reproduction + measured interpolant order.
    #[test]
    fn interpolate_is_a_fifth_order_local_extension() {
        let ev = HarmonicEval { omega: 1.0 };

        // theta = 0 is u0 bit-for-bit; theta = 1 is the propagated solution.
        let mut s = Tsit5::with_tolerances(1e-4, 1e-6);
        let u0 = vec![0.7, -0.4];
        let mut st = SolverState::for_evaluator(&ev, u0.clone(), 0.0, vec![]);
        assert!(matches!(
            s.step(&ev, &mut st, 0.05),
            StepOutcome::Accepted { .. }
        ));
        let u_new = st.u.clone();
        let mut out = vec![0.0; 2];
        assert!(s.interpolate(&u0, 0.05, 0.0, &mut out));
        for (i, (&got, &want)) in out.iter().zip(&u0).enumerate() {
            assert_eq!(got.to_bits(), want.to_bits(), "theta=0, component {i}");
        }
        assert!(s.interpolate(&u0, 0.05, 1.0, &mut out));
        for (i, (&got, &want)) in out.iter().zip(&u_new).enumerate() {
            assert!(
                (got - want).abs() < 1e-14,
                "theta=1, component {i}: {got} vs {want}"
            );
        }

        // Order, by h-halving on the analytic solution.
        let hs = [0.4_f64, 0.1];
        let mut errs = Vec::new();
        for &h in &hs {
            let mut s = Tsit5::with_tolerances(1e18, 1e18); // accept any step
            let u0 = vec![1.0, 0.0];
            let mut st = SolverState::for_evaluator(&ev, u0.clone(), 0.0, vec![]);
            assert!(matches!(
                s.step(&ev, &mut st, h),
                StepOutcome::Accepted { .. }
            ));
            let mut e: f64 = 0.0;
            for j in 1..20 {
                let theta = j as f64 / 20.0;
                assert!(s.interpolate(&u0, h, theta, &mut out));
                let t = theta * h;
                e = e
                    .max((out[0] - t.cos()).abs())
                    .max((out[1] + t.sin()).abs());
            }
            errs.push(e);
        }
        let order = (errs[0] / errs[1]).ln() / (hs[0] / hs[1]).ln();
        assert!(
            order > 4.5,
            "measured Tsit5 dense-output order {order:.2} (errors {errs:?}), expected ~5"
        );
    }
}
