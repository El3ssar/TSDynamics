//! Shared error control for the implicit kernels: step-doubling Richardson
//! extrapolation and the step-size controller, factored out so every stiff
//! method drives the *same* accept / reject / fail logic and only supplies its
//! own one-step formula.
//!
//! # Why step doubling
//!
//! The frozen [`Solver`] trait exposes a single `step`; the engine drives the
//! accept/retry loop off its [`StepOutcome`]. An adaptive kernel therefore needs
//! a local-error estimate *inside* one `step`. We get it the method-agnostic way
//! the v2 stiff kernel used and cross-validated: take one step of size `h` and
//! two of `h/2`, both with the same base method, and compare. The difference is
//! the leading local error; Richardson extrapolation then propagates the
//! higher-order combination ([local extrapolation]).
//!
//! This costs ~3× the base-method work per accepted step versus a native
//! embedded estimator, but it is robust, needs no per-method error coefficients
//! to get subtly wrong, and lets the Rosenbrock W-method and TR-BDF2 share one
//! controller. A native embedded pair is a later efficiency refinement, not a
//! correctness requirement — it would slot in behind this same [`BaseStep`] seam.
//!
//! [local extrapolation]: https://doi.org/10.1007/978-3-540-78862-1
//!
//! # Stability of the extrapolant
//!
//! For an L-stable base method with stability function `R(z) → 0` as
//! `z → −∞`, the step-doubled extrapolant `2·R(z/2)² − R(z)` (order-1 base) and
//! its higher-order analogues also tend to 0 at `z → −∞`, so the combination
//! stays L-stable at infinity — the property that lets these kernels take large
//! steps on stiff problems. The adaptive controller handles the rest.

use crate::solver::{SolverState, StepOutcome};
use tsdyn_ir::Evaluator;

/// Relative and absolute tolerances for the local-error test, owned by each
/// adaptive kernel (the frozen `step` signature carries no tolerance argument).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tolerances {
    /// Relative tolerance, weighting each component by its own magnitude.
    pub rtol: f64,
    /// Absolute tolerance, the floor that keeps near-zero components meaningful.
    pub atol: f64,
}

impl Tolerances {
    /// The kernels' default tolerances: accurate enough to cross-validate against
    /// reference stiff integrators, loose enough not to crawl. Override per
    /// instance with the kernel's `with_tolerances` builder.
    pub const DEFAULT: Tolerances = Tolerances {
        rtol: 1e-6,
        atol: 1e-9,
    };
}

impl Default for Tolerances {
    fn default() -> Self {
        Tolerances::DEFAULT
    }
}

// Controller constants — the classic Hairer–Wanner values, matching the v2
// stiff kernel so the migrated engine keeps the same step-size behavior.
const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 5.0;
/// A *rejected* step whose retry size falls below this fraction of `1 + |t|` is
/// treated as a genuine divergence (e.g. finite-time blow-up) and reported as
/// [`StepOutcome::Failed`], so the run fails loudly and fast instead of grinding
/// to the engine's per-segment step limit. Mirrors the v2 kernel's `h`-floor.
const STEP_FLOOR_REL: f64 = 1e-13;

/// The outcome of one base-method step (before error control).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BaseOutcome {
    /// Step computed; the candidate state was written to the `out` buffer.
    Ok,
    /// Recoverable failure — a singular iteration matrix or a Newton iteration
    /// that did not converge — almost always a too-large `h`. The controller
    /// retries with a smaller step.
    Recoverable,
    /// Unrecoverable: the right-hand side / Jacobian was non-finite at the step's
    /// starting point. Shrinking cannot help; the run aborts.
    Diverged,
}

/// One implicit integration formula, reduced to "advance one step of size `h`".
///
/// A kernel implements only this; [`DoublingWork::doubled_step`] turns it into an
/// adaptive [`Solver::step`](crate::Solver::step) via step doubling. Keeping the
/// base step this small is what lets the two stiff families share all of the
/// error-control machinery.
pub(crate) trait BaseStep {
    /// Advance one step of size `h` from `(t, u)`, writing the new state into
    /// `out`. `scratch` is the evaluator's working buffer (from the
    /// [`SolverState`]); `p` are the parameters. Must not assume anything about
    /// `out`'s prior contents and must leave `u` untouched (it is read-only).
    // The buffers are deliberately separate slices (caller-owned, disjoint), which
    // is what keeps the kernels allocation-free; bundling them into a struct would
    // only obscure the data flow.
    #[allow(clippy::too_many_arguments)]
    fn base_step(
        &mut self,
        ev: &dyn Evaluator,
        u: &[f64],
        t: f64,
        p: &[f64],
        h: f64,
        scratch: &mut [f64],
        out: &mut [f64],
    ) -> BaseOutcome;

    /// The classical order of the base step — sets the Richardson extrapolation
    /// weight `1/(2^p − 1)` and the controller exponent `−1/(p+1)`.
    fn order(&self) -> u32;
}

/// Per-kernel scratch for step doubling: the three candidate states (one big
/// step, the intermediate half step, two half steps) and the extrapolant. Grown
/// to the system dimension on first use and reused with no per-step allocation.
#[derive(Default)]
pub(crate) struct DoublingWork {
    u_big: Vec<f64>,
    u_half: Vec<f64>,
    u_two: Vec<f64>,
    u_new: Vec<f64>,
}

impl DoublingWork {
    pub(crate) fn new() -> Self {
        DoublingWork::default()
    }

    fn ensure(&mut self, dim: usize) {
        if self.u_new.len() != dim {
            self.u_big = vec![0.0; dim];
            self.u_half = vec![0.0; dim];
            self.u_two = vec![0.0; dim];
            self.u_new = vec![0.0; dim];
        }
    }

    /// Drive `base` for one adaptive step: one step of `h`, two of `h/2`,
    /// Richardson-extrapolate, and accept or reject against `tol`.
    ///
    /// On accept, commits the extrapolated state into `st.u`/`st.t`; on reject,
    /// leaves `st` exactly as found (the [`Solver`](crate::Solver) contract).
    // The extrapolation/error loop indexes several disjoint buffers (u, u_two,
    // u_big, u_new) together, where a range index reads more clearly than zipping
    // four iterators.
    #[allow(clippy::needless_range_loop)]
    pub(crate) fn doubled_step<B: BaseStep + ?Sized>(
        &mut self,
        base: &mut B,
        ev: &dyn Evaluator,
        st: &mut SolverState,
        h: f64,
        tol: Tolerances,
    ) -> StepOutcome {
        let n = st.u.len();
        self.ensure(n);
        let p = base.order();
        let SolverState {
            u,
            t,
            p: params,
            scratch,
        } = st;
        let t0 = *t;

        // One big step of size h.
        match base.base_step(ev, u, t0, params, h, scratch, &mut self.u_big) {
            BaseOutcome::Ok => {}
            BaseOutcome::Recoverable => return reject(h, MIN_FACTOR, t0),
            BaseOutcome::Diverged => return StepOutcome::Failed,
        }
        // Two half steps: u → u_half (over [t0, t0+h/2]) → u_two (over [t0+h/2, t0+h]).
        let hh = 0.5 * h;
        match base.base_step(ev, u, t0, params, hh, scratch, &mut self.u_half) {
            BaseOutcome::Ok => {}
            BaseOutcome::Recoverable => return reject(h, MIN_FACTOR, t0),
            BaseOutcome::Diverged => return StepOutcome::Failed,
        }
        match base.base_step(
            ev,
            &self.u_half,
            t0 + hh,
            params,
            hh,
            scratch,
            &mut self.u_two,
        ) {
            BaseOutcome::Ok => {}
            BaseOutcome::Recoverable => return reject(h, MIN_FACTOR, t0),
            BaseOutcome::Diverged => return StepOutcome::Failed,
        }

        // Richardson extrapolation to order p+1, and the step-doubling error
        // estimate. With `diff = u_two − u_big`, the Richardson local-error
        // estimate of the more accurate solution `u_two` is `diff / (2^p − 1)`,
        // and the extrapolant we propagate is `u_two + diff / (2^p − 1)`. (For the
        // order-1 W-method `2^p − 1 = 1`, so this is identical to the v2 kernel:
        // `u_new = 2·u_two − u_big`, error ∝ `diff`.)
        let denom = ((1u64 << p) - 1) as f64; // 2^p − 1
        let mut acc = 0.0;
        for i in 0..n {
            let err_i = (self.u_two[i] - self.u_big[i]) / denom;
            self.u_new[i] = self.u_two[i] + err_i;
            // Componentwise weight: relative to the larger of the old / new value.
            let scale = u[i].abs().max(self.u_two[i].abs());
            let e = err_i / (tol.atol + tol.rtol * scale);
            acc += e * e;
        }
        let err = (acc / n as f64).sqrt();

        if !err.is_finite() {
            // Non-finite estimate despite finite stages ⇒ the step was too large;
            // shrink hard (or fail if that collapses h).
            return reject(h, MIN_FACTOR, t0);
        }
        if err <= 1.0 {
            if !self.u_new.iter().all(|x| x.is_finite()) {
                return StepOutcome::Failed;
            }
            u.copy_from_slice(&self.u_new);
            *t = t0 + h;
            StepOutcome::Accepted {
                h_next: h * step_factor(err, p),
            }
        } else {
            reject(h, step_factor(err, p).min(1.0), t0)
        }
    }
}

/// The PI-free elementary step-size factor for a base method of order `p` from a
/// scaled error norm: `clamp(SAFETY · err^(−1/(p+1)), MIN, MAX)`.
fn step_factor(err: f64, order: u32) -> f64 {
    if err == 0.0 {
        MAX_FACTOR
    } else {
        let exponent = -1.0 / (order as f64 + 1.0);
        (SAFETY * err.powf(exponent)).clamp(MIN_FACTOR, MAX_FACTOR)
    }
}

/// Build a rejection outcome, downgrading to [`StepOutcome::Failed`] when the
/// retry step has collapsed (non-finite, non-positive, or below the relative
/// floor) — the "diverge loudly" contract.
fn reject(h: f64, factor: f64, t: f64) -> StepOutcome {
    let h_next = h * factor;
    if !(h_next.is_finite() && h_next > 0.0) || h_next < STEP_FLOOR_REL * (1.0 + t.abs()) {
        StepOutcome::Failed
    } else {
        StepOutcome::Rejected { h_next }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_factor_grows_when_accurate_and_shrinks_when_not() {
        // err well under 1 ⇒ grow (up to MAX); err over 1 ⇒ shrink (down to MIN).
        assert!(step_factor(1e-3, 1) > 1.0);
        assert!(step_factor(1e3, 1) < 1.0);
        assert_eq!(step_factor(0.0, 2), MAX_FACTOR);
        // Clamped to the configured envelope.
        assert!(step_factor(1e-30, 5) <= MAX_FACTOR);
        assert!(step_factor(1e30, 5) >= MIN_FACTOR);
    }

    #[test]
    fn higher_order_reacts_less_to_a_given_error() {
        // The exponent −1/(p+1) shrinks with p, so a high-order method changes
        // its step less aggressively for the same error ratio. Use err = 0.5, in
        // the unclamped growth region, so the difference is visible (a tiny err
        // would saturate both at MAX_FACTOR).
        let f1 = step_factor(0.5, 1);
        let f4 = step_factor(0.5, 4);
        assert!(f1 > f4, "order-1 factor {f1} should exceed order-4 {f4}");
        assert!(f1 < MAX_FACTOR && f4 > 1.0, "f1={f1}, f4={f4}");
    }

    #[test]
    fn reject_downgrades_a_collapsed_step_to_failed() {
        // A normal shrink is a Rejection…
        assert!(matches!(
            reject(1e-2, 0.5, 0.0),
            StepOutcome::Rejected { .. }
        ));
        // …but a step that has collapsed to ~0 is an unrecoverable Failure.
        assert_eq!(reject(1e-20, 0.2, 0.0), StepOutcome::Failed);
        assert_eq!(reject(f64::NAN, 0.5, 0.0), StepOutcome::Failed);
    }
}
