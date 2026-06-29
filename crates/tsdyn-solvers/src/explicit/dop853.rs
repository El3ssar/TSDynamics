//! Adaptive Dormand–Prince 8(5,3) — Hairer's `DOP853`, registered as `dop853`.
//!
//! An 8th-order explicit Runge–Kutta method whose step size is controlled by a
//! *blended* 5th- and 3rd-order error estimate, giving robust adaption at high
//! order — the method of choice when many digits of accuracy are wanted on a
//! smooth non-stiff problem (Hairer, Nørsett & Wanner, *Solving Ordinary
//! Differential Equations I*, 2nd ed., §II.5).  The coefficients are the standard
//! Hairer set; they match SciPy's `DOP853`, against which this is cross-validated.
//!
//! Unlike `rk45`/`tsit5` this does not reduce to a single error-weight vector, so
//! it does not use [`adaptive_step`](super::control::adaptive_step): it computes
//! the 12 stages with the shared machinery, then forms its own blended error
//! norm `|h|·‖e₅‖² / √((‖e₅‖² + 0.01·‖e₃‖²)·n)` — identical to SciPy's, where
//! `‖·‖` is the per-component-scaled Euclidean norm.
//!
//! There is **no propagation FSAL reuse** here: every accepted step recomputes
//! its 12 propagation stages from scratch (unlike `rk45`/`tsit5`/`bs3`, which
//! reuse the accepted step's last stage as the next step's first). The 13th
//! "FSAL" stage that SciPy carries is purely a *dense-output* stage — it has zero
//! weight in both error estimators and feeds no subsequent propagation step — so
//! it is not computed here; only the 12 propagation stages are needed.

// The Hairer coefficients are transcribed verbatim from the canonical published
// set (also SciPy's `dop853_coefficients.py`), at higher precision than f64 can
// hold. The compiler rounds each to the nearest representable f64 — identical to
// truncating by hand — so the extra digits change no value; keeping them makes
// every constant auditable digit-for-digit against the source.
#![allow(clippy::excessive_precision)]

use crate::{
    register_solver, Caps, Evaluator, ProblemKind, ProblemKinds, Solver, SolverState, StepOutcome,
};

use super::control::{combine, compute_stages, error_vector, step_factor, RkWork, MIN_FACTOR};

// Hairer 8(5,3) nodes (12 stages; C[0] = 0, C[11] = 1).
const C: &[f64] = &[
    0.0,
    0.526001519587677318785587544488e-1,
    0.789002279381515978178381316732e-1,
    0.118350341907227396726757197510,
    0.281649658092772603273242802490,
    0.333333333333333333333333333333,
    0.25,
    0.307692307692307692307692307692,
    0.651282051282051282051282051282,
    0.6,
    0.857142857142857142857142857142,
    1.0,
];

// Lower-triangular stage coefficients (unset Hairer entries are explicit zeros).
const A: &[&[f64]] = &[
    &[],
    &[5.26001519587677318785587544488e-2],
    &[
        1.97250569845378994544595329183e-2,
        5.91751709536136983633785987549e-2,
    ],
    &[
        2.95875854768068491816892993775e-2,
        0.0,
        8.87627564304205475450678981324e-2,
    ],
    &[
        2.41365134159266685502369798665e-1,
        0.0,
        -8.84549479328286085344864962717e-1,
        9.24834003261792003115737966543e-1,
    ],
    &[
        3.7037037037037037037037037037e-2,
        0.0,
        0.0,
        1.70828608729473871279604482173e-1,
        1.25467687566822425016691814123e-1,
    ],
    &[
        3.7109375e-2,
        0.0,
        0.0,
        1.70252211019544039314978060272e-1,
        6.02165389804559606850219397283e-2,
        -1.7578125e-2,
    ],
    &[
        3.70920001185047927108779319836e-2,
        0.0,
        0.0,
        1.70383925712239993810214054705e-1,
        1.07262030446373284651809199168e-1,
        -1.53194377486244017527936158236e-2,
        8.27378916381402288758473766002e-3,
    ],
    &[
        6.24110958716075717114429577812e-1,
        0.0,
        0.0,
        -3.36089262944694129406857109825,
        -8.68219346841726006818189891453e-1,
        2.75920996994467083049415600797e1,
        2.01540675504778934086186788979e1,
        -4.34898841810699588477366255144e1,
    ],
    &[
        4.77662536438264365890433908527e-1,
        0.0,
        0.0,
        -2.48811461997166764192642586468,
        -5.90290826836842996371446475743e-1,
        2.12300514481811942347288949897e1,
        1.52792336328824235832596922938e1,
        -3.32882109689848629194453265587e1,
        -2.03312017085086261358222928593e-2,
    ],
    &[
        -9.3714243008598732571704021658e-1,
        0.0,
        0.0,
        5.18637242884406370830023853209,
        1.09143734899672957818500254654,
        -8.14978701074692612513997267357,
        -1.85200656599969598641566180701e1,
        2.27394870993505042818970056734e1,
        2.49360555267965238987089396762,
        -3.0467644718982195003823669022,
    ],
    &[
        2.27331014751653820792359768449,
        0.0,
        0.0,
        -1.05344954667372501984066689879e1,
        -2.00087205822486249909675718444,
        -1.79589318631187989172765950534e1,
        2.79488845294199600508499808837e1,
        -2.85899827713502369474065508674,
        -8.87285693353062954433549289258,
        1.23605671757943030647266201528e1,
        6.43392746015763530355970484046e-1,
    ],
];

// 8th-order solution weights (Hairer's `b`; the SciPy `A[12, :12]` row).
const B: &[f64] = &[
    5.42937341165687622380535766363e-2,
    0.0,
    0.0,
    0.0,
    0.0,
    4.45031289275240888144113950566,
    1.89151789931450038304281599044,
    -5.8012039600105847814672114227,
    3.1116436695781989440891606237e-1,
    -1.52160949662516078556178806805e-1,
    2.01365400804030348374776537501e-1,
    4.47106157277725905176885569043e-2,
];

// 3rd-order error weights e₃ = b − b̂₃ (Hairer); zero weight on the 13th stage.
const E3: &[f64] = &[
    5.42937341165687622380535766363e-2 - 0.244094488188976377952755905512,
    0.0,
    0.0,
    0.0,
    0.0,
    4.45031289275240888144113950566,
    1.89151789931450038304281599044,
    -5.8012039600105847814672114227,
    3.1116436695781989440891606237e-1 - 0.733846688281611857341361741547,
    -1.52160949662516078556178806805e-1,
    2.01365400804030348374776537501e-1,
    4.47106157277725905176885569043e-2 - 0.220588235294117647058823529412e-1,
];

// 5th-order error weights e₅ (Hairer); zero weight on the 13th stage.
const E5: &[f64] = &[
    0.1312004499419488073250102996e-1,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.1225156446376204440720569753e1,
    -0.4957589496572501915214079952,
    0.1664377182454986536961530415e1,
    -0.3503288487499736816886487290,
    0.3341791187130174790297318841,
    0.8192320648511571246570742613e-1,
    -0.2235530786388629525884427845e-1,
];

// Controller exponent −1/(error_estimator_order + 1) with estimator order 7.
const ERR_EXPONENT: f64 = -1.0 / 8.0;

/// Default relative tolerance (SciPy `solve_ivp` default).
const DEFAULT_RTOL: f64 = 1e-3;
/// Default absolute tolerance (SciPy `solve_ivp` default).
const DEFAULT_ATOL: f64 = 1e-6;

/// Adaptive Dormand–Prince 8(5,3) kernel.
pub struct Dop853 {
    rtol: f64,
    atol: f64,
    work: RkWork,
    /// Σᵢ E5ᵢ·kᵢ (the 5th-order error estimate), length `dim`.
    err5: Vec<f64>,
    /// Σᵢ E3ᵢ·kᵢ (the 3rd-order error estimate), length `dim`.
    err3: Vec<f64>,
}

impl Dop853 {
    /// A kernel with the default tolerances (`rtol = 1e-3`, `atol = 1e-6`).
    pub fn new() -> Self {
        Dop853::with_tolerances(DEFAULT_RTOL, DEFAULT_ATOL)
    }

    /// A kernel with explicit tolerances (see [`Rk45::with_tolerances`] for why
    /// an adaptive kernel owns its tolerances).
    ///
    /// [`Rk45::with_tolerances`]: super::rk45::Rk45::with_tolerances
    pub fn with_tolerances(rtol: f64, atol: f64) -> Self {
        Dop853 {
            rtol,
            atol,
            work: RkWork::new(),
            err5: Vec::new(),
            err3: Vec::new(),
        }
    }
}

impl Default for Dop853 {
    fn default() -> Self {
        Dop853::new()
    }
}

impl Solver for Dop853 {
    fn name(&self) -> &'static str {
        "dop853"
    }

    fn caps(&self) -> Caps {
        Caps::explicit(ProblemKinds::of(ProblemKind::Ode)).adaptive()
    }

    fn step(&mut self, ev: &dyn Evaluator, st: &mut SolverState, h: f64) -> StepOutcome {
        let dim = st.u.len();
        self.work.ensure(C.len(), dim);
        if self.err5.len() != dim {
            self.err5 = vec![0.0; dim];
            self.err3 = vec![0.0; dim];
        }
        compute_stages(ev, st, h, C, A, &mut self.work);
        if self.work.k[0].iter().any(|x| !x.is_finite()) {
            return StepOutcome::Failed;
        }
        combine(&st.u, h, B, &self.work.k, &mut self.work.u_new);
        // Magnitude each error component is scaled against.
        for d in 0..dim {
            self.work.scale[d] = st.u[d].abs().max(self.work.u_new[d].abs());
        }
        // e₅ = Σ E5ᵢ kᵢ and e₃ = Σ E3ᵢ kᵢ (no h factor; it enters the norm below).
        error_vector(1.0, E5, &self.work.k, &mut self.err5);
        error_vector(1.0, E3, &self.work.k, &mut self.err3);
        // Hairer's blended, per-component-scaled error norm (matches SciPy DOP853).
        let mut err5_norm2 = 0.0;
        let mut err3_norm2 = 0.0;
        for d in 0..dim {
            let sc = self.atol + self.rtol * self.work.scale[d];
            let a = self.err5[d] / sc;
            let b = self.err3[d] / sc;
            err5_norm2 += a * a;
            err3_norm2 += b * b;
        }
        let err = if err5_norm2 == 0.0 && err3_norm2 == 0.0 {
            0.0
        } else {
            let denom = err5_norm2 + 0.01 * err3_norm2;
            h.abs() * err5_norm2 / (denom * dim as f64).sqrt()
        };
        if !self.work.u_new.iter().all(|x| x.is_finite()) || !err.is_finite() {
            return StepOutcome::Rejected {
                h_next: h * MIN_FACTOR,
            };
        }
        if err <= 1.0 {
            st.u.copy_from_slice(&self.work.u_new);
            st.t += h;
            StepOutcome::Accepted {
                h_next: h * step_factor(err, ERR_EXPONENT),
            }
        } else {
            StepOutcome::Rejected {
                h_next: h * step_factor(err, ERR_EXPONENT).min(1.0),
            }
        }
    }
}

register_solver!(
    "dop853",
    Caps::explicit(ProblemKinds::of(ProblemKind::Ode)).adaptive(),
    || Box::new(Dop853::new())
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explicit::testkit::{
        converges_at_order, fixed_propagate, integrate_adaptive, max_abs_diff, HarmonicEval,
    };

    #[test]
    fn tableau_is_internally_consistent() {
        // Σb = 1 and each A-row sums to its node c — necessary conditions a
        // correct 8th-order tableau must satisfy.
        let sum_b: f64 = B.iter().sum();
        assert!((sum_b - 1.0).abs() < 1e-12, "Σb = {sum_b}");
        for (i, row) in A.iter().enumerate() {
            let s: f64 = row.iter().sum();
            assert!((s - C[i]).abs() < 1e-12, "row {i}: Σa = {s}, c = {}", C[i]);
        }
    }

    #[test]
    fn caps_are_explicit_adaptive_ode() {
        let s = Dop853::new();
        assert_eq!(s.name(), "dop853");
        assert!(s.caps().adaptive);
    }

    #[test]
    fn eighth_order_convergence_of_the_propagated_solution() {
        // The b-weighted solution must converge at order 8 under h-refinement.
        // The steps are deliberately *large* (≈8–12 points/period): an 8th-order
        // method reaches the f64 round-off floor (≈1e-15) by h ≈ 0.1 on this
        // problem, so a finer grid would measure round-off noise, not order. Over
        // h ∈ [0.5, 0.25] the truncation error (≈5e-10 … 2e-12) is far above that
        // floor and falls by the expected ~2^8 per halving.
        let ev = HarmonicEval { omega: 1.0 };
        let order = converges_at_order(
            |st, h, work| fixed_propagate(&ev, st, h, C, A, B, work),
            &ev,
            vec![1.0, 0.0],
            2.0,
            &[0.5, 0.4, 0.25],
            |t| vec![t.cos(), -t.sin()],
        );
        assert!(
            (order - 8.0).abs() < 0.6,
            "measured DOP853 order {order}, expected ≈ 8"
        );
    }

    #[test]
    fn adaptive_run_matches_analytic_harmonic_solution() {
        let ev = HarmonicEval { omega: 1.0 };
        let mut s = Dop853::with_tolerances(1e-11, 1e-13);
        let t_final = 12.0;
        let mut st = SolverState::for_evaluator(&ev, vec![1.0, 0.0], 0.0, vec![]);
        integrate_adaptive(&mut s, &ev, &mut st, t_final, 0.1);
        let exact = vec![t_final.cos(), -t_final.sin()];
        assert!(
            max_abs_diff(&st.u, &exact) < 1e-8,
            "adaptive DOP853 error {}",
            max_abs_diff(&st.u, &exact)
        );
    }
}
