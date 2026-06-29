//! The explicit linear-multistep (Adams) family: Adams–Bashforth and the
//! Adams–Bashforth–Moulton predictor–corrector.
//!
//! Unlike a Runge–Kutta method, a multistep method reuses the right-hand-side
//! values it already computed at *previous* steps, so it advances with only **one**
//! (Adams–Bashforth) or **two** (predictor–corrector) new RHS evaluations per step
//! regardless of order. That makes them efficient on non-stiff problems with an
//! expensive `f` — e.g. the large ODE systems that arise from
//! method-of-lines/spectral discretisations of PDEs. They are fixed-step here and
//! self-start with RK4 until enough history has accumulated.
//!
//! | name   | method                                   | order | RHS/step |
//! |--------|------------------------------------------|-------|----------|
//! | `ab3`  | Adams–Bashforth 3-step                   | 3     | 1        |
//! | `ab4`  | Adams–Bashforth 4-step                   | 4     | 1        |
//! | `abm4` | Adams–Bashforth–Moulton (PECE)           | 4     | 2        |
//!
//! References: F. Bashforth & J. C. Adams, *An Attempt to Test the Theories of
//! Capillary Action…* (Cambridge Univ. Press, 1883) — the Adams–Bashforth and
//! Adams–Moulton formulas; E. Hairer, S. P. Nørsett & G. Wanner, *Solving
//! Ordinary Differential Equations I*, 2nd ed. (1993), §III.1 (Adams methods and
//! predictor–corrector pairs).
//!
//! # Why fixed step (and how variable `h` is handled)
//!
//! The classic Adams coefficients assume a *uniform* past mesh. In the engine's
//! grid integration each output segment is one step of the running `h`, so the
//! mesh is uniform; the kernel detects a change in `h` (the final landing step of
//! a non-commensurate grid, or a `step()` driven at a new `dt`) and transparently
//! **restarts** the history with an RK4 step until `k−1` uniform slopes have
//! re-accumulated. Variable-coefficient Adams methods are a possible future
//! refinement; they would slot in behind this same kernel without an interface
//! change.
//!
//! ## Commensurate-grid requirement (caller-facing invariant)
//!
//! Because every step-size *change* forces a full RK4 restart, these kernels keep
//! their advertised order **only on a grid whose spacing is a constant multiple of
//! the integration step** — i.e. the requested output `dt` must be commensurate
//! with the kernel step `h`. When the grid is commensurate the mesh is uniform end
//! to end (apart from at most one landing step), the multistep formula runs at full
//! order, and the result is bit-identical across runs. When it is *not* — an output
//! `dt` that is not an integer multiple of `h`, or a `step()` loop driven at an
//! ever-changing `dt` — almost every step is a fresh restart and the method
//! silently **degrades toward plain RK4** (its 4th-order bootstrap): still correct
//! and convergent, but with RK4's (not the multistep's) error constant and cost.
//! This is a deliberate scope choice: a true variable-coefficient Adams kernel is
//! out of scope here. The degradation is *quiet* (no warning, no failure) — pick a
//! commensurate grid to realise the multistep advantage.

use std::collections::VecDeque;

use crate::{
    register_solver, Caps, Evaluator, ProblemKind, ProblemKinds, Solver, SolverState, StepOutcome,
};

use super::control::{fixed_step, RkWork};

// Classic RK4 tableau for the self-start / history-restart bootstrap.
const RK4_C: &[f64] = &[0.0, 0.5, 0.5, 1.0];
const RK4_A: &[&[f64]] = &[&[], &[0.5], &[0.0, 0.5], &[0.0, 0.0, 1.0]];
const RK4_B: &[f64] = &[1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0];

/// Relative tolerance for the "is `h` the same as the previous step?" check, so
/// ULP-level grid roundoff does not spuriously restart the history every step.
const UNIFORM_REL_TOL: f64 = 1e-9;

/// Shared multistep state: the ring buffer of past slopes (most-recent first),
/// per-step scratch, and the RK4 bootstrap work. Grown to the system dimension on
/// first use and reused with no per-step allocation after warm-up.
#[derive(Default)]
struct AdamsState {
    /// Past slopes `f_{n-1}, f_{n-2}, …` (most recent at the front).
    hist: VecDeque<Vec<f64>>,
    /// `f` at the current point, recomputed each step.
    fc: Vec<f64>,
    /// `f` at the predicted point (predictor–corrector only).
    fp: Vec<f64>,
    /// Proposed next state.
    u_new: Vec<f64>,
    /// RK4 bootstrap buffers.
    rk: RkWork,
    /// The step size of the previous accepted step (`NaN` until the first).
    h_prev: f64,
}

impl AdamsState {
    fn new() -> Self {
        AdamsState {
            h_prev: f64::NAN,
            ..AdamsState::default()
        }
    }

    fn ensure(&mut self, n: usize) {
        if self.fc.len() != n {
            self.fc = vec![0.0; n];
            self.fp = vec![0.0; n];
            self.u_new = vec![0.0; n];
            self.hist.clear();
        }
    }

    fn uniform(&self, h: f64) -> bool {
        self.h_prev.is_finite() && (h - self.h_prev).abs() <= UNIFORM_REL_TOL * (1.0 + h.abs())
    }

    /// Push the current slope `fc` to the front of the history, capping the buffer
    /// at `keep` entries (reusing the evicted buffer to avoid per-step allocation).
    fn record(&mut self, keep: usize) {
        let mut buf = if self.hist.len() >= keep {
            self.hist
                .pop_back()
                .unwrap_or_else(|| vec![0.0; self.fc.len()])
        } else {
            vec![0.0; self.fc.len()]
        };
        buf.copy_from_slice(&self.fc);
        self.hist.push_front(buf);
        while self.hist.len() > keep {
            self.hist.pop_back();
        }
    }
}

/// One explicit-multistep step. `ab` are the Adams–Bashforth weights applied to
/// `[f_c, f_{n-1}, …]` (the predictor); `corrector`, when present, are the
/// Adams–Moulton weights applied to `[f_pred, f_c, f_{n-1}, …]` (the PECE
/// corrector). `keep = order − 1` past slopes are retained. The history restarts
/// with an RK4 step whenever it is too short or `h` changed.
fn adams_step(
    ev: &dyn Evaluator,
    st: &mut SolverState,
    h: f64,
    state: &mut AdamsState,
    keep: usize,
    ab: &[f64],
    corrector: Option<&[f64]>,
) -> StepOutcome {
    let n = st.u.len();
    state.ensure(n);

    // f at the current point.
    ev.eval(&st.u, &st.p, st.t, &mut st.scratch, &mut state.fc);
    if !state.fc.iter().all(|x| x.is_finite()) {
        return StepOutcome::Failed;
    }

    if state.hist.len() < keep || !state.uniform(h) {
        // A step-size *change* invalidates the uniform-mesh history (the classic
        // Adams coefficients assume a uniform past spacing): discard it so the
        // `hist.len() < keep` guard forces `keep` consecutive RK4 starter steps at
        // the new `h` before the Adams formula resumes — without this, the next
        // step would apply uniform-`h` weights to mixed-spacing slopes. A genuine
        // bootstrap (`h_prev` not yet finite) or an ongoing refill leaves the
        // (already new-`h`) history intact.
        if state.h_prev.is_finite() && !state.uniform(h) {
            state.hist.clear();
        }
        // Self-start / restart: one RK4 step. `fixed_step` advances st (and is the
        // exact RK4 used by the `rk4` kernel, so the bootstrap is 4th-order).
        match fixed_step(ev, st, h, RK4_C, RK4_A, RK4_B, &mut state.rk) {
            StepOutcome::Accepted { .. } => {}
            other => return other,
        }
        state.record(keep);
        state.h_prev = h;
        return StepOutcome::Accepted { h_next: h };
    }

    // Adams–Bashforth predictor: u_p = u + h·(ab[0]·f_c + Σ_j ab[1+j]·f_{n-1-j}).
    for d in 0..n {
        let mut acc = ab[0] * state.fc[d];
        for (j, hslot) in state.hist.iter().enumerate() {
            acc += ab[1 + j] * hslot[d];
        }
        state.u_new[d] = st.u[d] + h * acc;
    }

    if let Some(am) = corrector {
        // Evaluate at the predicted point, then correct (PECE).
        ev.eval(
            &state.u_new,
            &st.p,
            st.t + h,
            &mut st.scratch,
            &mut state.fp,
        );
        if !state.fp.iter().all(|x| x.is_finite()) {
            return StepOutcome::Rejected { h_next: 0.5 * h };
        }
        for d in 0..n {
            // u_{n+1} = u + h·(am[0]·f_p + am[1]·f_c + Σ_j am[2+j]·f_{n-1-j}).
            let mut acc = am[0] * state.fp[d] + am[1] * state.fc[d];
            for (j, hslot) in state.hist.iter().enumerate() {
                if 2 + j >= am.len() {
                    break;
                }
                acc += am[2 + j] * hslot[d];
            }
            state.u_new[d] = st.u[d] + h * acc;
        }
    }

    if !state.u_new.iter().all(|x| x.is_finite()) {
        return StepOutcome::Rejected { h_next: 0.5 * h };
    }
    st.u.copy_from_slice(&state.u_new);
    st.t += h;
    state.record(keep);
    state.h_prev = h;
    StepOutcome::Accepted { h_next: h }
}

// Adams–Bashforth weights [f_c, f_{n-1}, …].
const AB3: &[f64] = &[23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0];
const AB4: &[f64] = &[55.0 / 24.0, -59.0 / 24.0, 37.0 / 24.0, -9.0 / 24.0];
// Adams–Moulton-4 corrector weights [f_pred, f_c, f_{n-1}, f_{n-2}].
const AM4: &[f64] = &[9.0 / 24.0, 19.0 / 24.0, -5.0 / 24.0, 1.0 / 24.0];

/// Adams–Bashforth 3-step explicit multistep kernel (order 3, one RHS/step).
///
/// Holds order 3 on a commensurate (uniform-`h`) grid; an incommensurate output
/// `dt` restarts the history each step and degrades it toward RK4 (see the
/// module's *Commensurate-grid requirement*). Self-starts with RK4.
#[derive(Default)]
pub struct Ab3 {
    state: AdamsState,
}

impl Ab3 {
    /// A fresh kernel with empty history (self-starts with RK4).
    pub fn new() -> Self {
        Ab3 {
            state: AdamsState::new(),
        }
    }
}

impl Solver for Ab3 {
    fn name(&self) -> &'static str {
        "ab3"
    }

    fn caps(&self) -> Caps {
        Caps::explicit(ProblemKinds::of(ProblemKind::Ode))
    }

    fn step(&mut self, ev: &dyn Evaluator, st: &mut SolverState, h: f64) -> StepOutcome {
        adams_step(ev, st, h, &mut self.state, 2, AB3, None)
    }
}

register_solver!(
    "ab3",
    Caps::explicit(ProblemKinds::of(ProblemKind::Ode)),
    || Box::new(Ab3::new())
);

/// Adams–Bashforth 4-step explicit multistep kernel (order 4, one RHS/step).
///
/// Holds order 4 on a commensurate (uniform-`h`) grid; an incommensurate output
/// `dt` restarts the history each step and degrades it toward RK4 (see the
/// module's *Commensurate-grid requirement*). Self-starts with RK4.
#[derive(Default)]
pub struct Ab4 {
    state: AdamsState,
}

impl Ab4 {
    /// A fresh kernel with empty history (self-starts with RK4).
    pub fn new() -> Self {
        Ab4 {
            state: AdamsState::new(),
        }
    }
}

impl Solver for Ab4 {
    fn name(&self) -> &'static str {
        "ab4"
    }

    fn caps(&self) -> Caps {
        Caps::explicit(ProblemKinds::of(ProblemKind::Ode))
    }

    fn step(&mut self, ev: &dyn Evaluator, st: &mut SolverState, h: f64) -> StepOutcome {
        adams_step(ev, st, h, &mut self.state, 3, AB4, None)
    }
}

register_solver!(
    "ab4",
    Caps::explicit(ProblemKinds::of(ProblemKind::Ode)),
    || Box::new(Ab4::new())
);

/// Adams–Bashforth–Moulton predictor–corrector kernel (PECE, order 4).
///
/// Predicts with AB4 and corrects once with AM4 (two RHS/step). Holds order 4 on a
/// commensurate (uniform-`h`) grid; an incommensurate output `dt` restarts the
/// history each step and degrades it toward RK4 (see the module's
/// *Commensurate-grid requirement*). Self-starts with RK4.
#[derive(Default)]
pub struct Abm4 {
    state: AdamsState,
}

impl Abm4 {
    /// A fresh kernel with empty history (self-starts with RK4).
    pub fn new() -> Self {
        Abm4 {
            state: AdamsState::new(),
        }
    }
}

impl Solver for Abm4 {
    fn name(&self) -> &'static str {
        "abm4"
    }

    fn caps(&self) -> Caps {
        Caps::explicit(ProblemKinds::of(ProblemKind::Ode))
    }

    fn step(&mut self, ev: &dyn Evaluator, st: &mut SolverState, h: f64) -> StepOutcome {
        adams_step(ev, st, h, &mut self.state, 3, AB4, Some(AM4))
    }
}

register_solver!(
    "abm4",
    Caps::explicit(ProblemKinds::of(ProblemKind::Ode)),
    || Box::new(Abm4::new())
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explicit::testkit::{max_abs_diff, HarmonicEval};

    /// Drive a kernel through a uniform fixed-step loop to `t_final`.
    fn integrate_uniform(
        solver: &mut dyn Solver,
        ev: &dyn Evaluator,
        u0: Vec<f64>,
        t_final: f64,
        h: f64,
    ) -> Vec<f64> {
        let mut st = SolverState::for_evaluator(ev, u0, 0.0, vec![]);
        let n = (t_final / h).round() as usize;
        for _ in 0..n {
            match solver.step(ev, &mut st, h) {
                StepOutcome::Accepted { .. } => {}
                other => panic!("unexpected outcome {other:?}"),
            }
        }
        st.u
    }

    /// Empirical convergence order from two final-state errors under h-halving.
    fn measure_order(make: impl Fn() -> Box<dyn Solver>, hs: &[f64]) -> f64 {
        let ev = HarmonicEval { omega: 1.0 };
        let t_final = 5.0;
        let mut errs = Vec::new();
        for &h in hs {
            let got = integrate_uniform(&mut *make(), &ev, vec![1.0, 0.0], t_final, h);
            let exact = vec![t_final.cos(), -t_final.sin()];
            errs.push((h, max_abs_diff(&got, &exact)));
        }
        let (h1, e1) = errs[errs.len() - 2];
        let (h2, e2) = errs[errs.len() - 1];
        (e1 / e2).ln() / (h1 / h2).ln()
    }

    #[test]
    fn caps_are_explicit_non_adaptive_ode() {
        for s in [
            Box::new(Ab3::new()) as Box<dyn Solver>,
            Box::new(Ab4::new()),
            Box::new(Abm4::new()),
        ] {
            assert!(!s.caps().adaptive);
            assert!(!s.caps().needs_jacobian);
            assert!(s.caps().supports(ProblemKind::Ode));
        }
        assert_eq!(Ab3::new().name(), "ab3");
        assert_eq!(Ab4::new().name(), "ab4");
        assert_eq!(Abm4::new().name(), "abm4");
    }

    #[test]
    fn ab3_is_third_order() {
        let order = measure_order(|| Box::new(Ab3::new()), &[0.04, 0.02, 0.01]);
        assert!((order - 3.0).abs() < 0.4, "measured AB3 order {order}");
    }

    #[test]
    fn ab4_is_fourth_order() {
        let order = measure_order(|| Box::new(Ab4::new()), &[0.04, 0.02, 0.01]);
        assert!((order - 4.0).abs() < 0.4, "measured AB4 order {order}");
    }

    #[test]
    fn abm4_is_fourth_order() {
        let order = measure_order(|| Box::new(Abm4::new()), &[0.04, 0.02, 0.01]);
        assert!((order - 4.0).abs() < 0.4, "measured ABM4 order {order}");
    }

    #[test]
    fn ab4_matches_analytic_harmonic_solution() {
        // A wrong coefficient breaks accuracy against the exact cos/sin solution.
        let ev = HarmonicEval { omega: 1.0 };
        let got = integrate_uniform(&mut Ab4::new(), &ev, vec![1.0, 0.0], 6.0, 0.005);
        let exact = vec![(6.0_f64).cos(), -(6.0_f64).sin()];
        assert!(
            max_abs_diff(&got, &exact) < 1e-6,
            "AB4 error {}",
            max_abs_diff(&got, &exact)
        );
    }

    #[test]
    fn step_size_change_discards_stale_history() {
        // After an h-change the next `keep` steps must be pure RK4 restarts over a
        // freshly-rebuilt uniform mesh — NOT Adams steps over a contaminated
        // mixed-spacing history. We assert the post-change AB4 steps are
        // bit-for-bit identical to a standalone RK4 from the same state, which only
        // holds if the stale h1 history was discarded (regression for the restart
        // bug: record() alone never shrank the history below `keep`).
        use crate::explicit::Rk4;
        let ev = HarmonicEval { omega: 1.0 };
        let mut ab = Ab4::new();
        let mut st = SolverState::for_evaluator(&ev, vec![1.0, 0.0], 0.0, vec![]);
        for _ in 0..50 {
            ab.step(&ev, &mut st, 0.01); // fill a full uniform-h1 history
        }
        // Branch a standalone RK4 from the exact current state.
        let mut rk = Rk4::new();
        let mut st_rk = SolverState::for_evaluator(&ev, st.u.clone(), st.t, vec![]);
        let h2 = 0.017; // a genuine step-size change
        for _ in 0..3 {
            // keep = 3 for AB4
            ab.step(&ev, &mut st, h2);
            rk.step(&ev, &mut st_rk, h2);
        }
        assert_eq!(
            st.u, st_rk.u,
            "post-h-change AB4 steps must be exact RK4 restarts (stale history discarded)"
        );
    }

    #[test]
    fn restarts_cleanly_when_step_size_changes() {
        // Drive with one h, then a different h: the kernel must restart its history
        // (RK4) and still integrate accurately rather than corrupt the multistep
        // formula with a non-uniform mesh.
        let ev = HarmonicEval { omega: 1.0 };
        let mut s = Ab4::new();
        let mut st = SolverState::for_evaluator(&ev, vec![1.0, 0.0], 0.0, vec![]);
        for _ in 0..100 {
            assert!(matches!(
                s.step(&ev, &mut st, 0.01),
                StepOutcome::Accepted { .. }
            ));
        }
        for _ in 0..100 {
            assert!(matches!(
                s.step(&ev, &mut st, 0.013),
                StepOutcome::Accepted { .. }
            ));
        }
        let t = st.t;
        let exact = vec![t.cos(), -t.sin()];
        assert!(
            max_abs_diff(&st.u, &exact) < 1e-4,
            "post-restart error too large"
        );
    }
}
