//! `trbdf2` — the TR-BDF2 one-step method: a trapezoidal substage followed by a
//! BDF2 substage, an L-stable, second-order, stiffly-accurate ESDIRK.
//!
//! One step of size `h` from `(t, u)` runs two implicit substages, both solved by
//! a modified-Newton iteration that reuses the analytic Jacobian the evaluator
//! provides:
//!
//! 1. **Trapezoidal** to `t + γh`:  solve `y₁ = u + (γh/2)·(f(t,u) + f(t+γh, y₁))`.
//! 2. **BDF2** to `t + h`:  solve `y₂ = c₁·y₁ + c₀·u + w·h·f(t+h, y₂)`.
//!
//! with `γ = 2 − √2` chosen so both substages share the same iteration-matrix
//! shape `I − a·h·J` and the method is L-stable. Being stiffly accurate (the last
//! stage *is* the step), it damps stiff transients cleanly — the property that
//! makes it a good general stiff workhorse alongside the cheaper Rosenbrock
//! W-method. Cross-checking the two (different mechanism: full Newton vs. a single
//! linear solve) is a strong correctness signal.
//!
//! The base step is order 2; [`DoublingWork`](super::control::DoublingWork)
//! Richardson-extrapolates and estimates the local error, exactly as for the
//! Rosenbrock kernel. A native embedded estimator would be cheaper and is a
//! possible refinement, but step doubling avoids hand-tuned error coefficients.
//!
//! References: R. E. Bank et al., "Transient simulation of silicon devices and
//! circuits", IEEE Trans. CAD 4(4), 1985; M. E. Hosea & L. F. Shampine,
//! "Analysis and implementation of TR-BDF2", Appl. Numer. Math. 20, 1996.

use super::control::{BaseOutcome, BaseStep, DoublingWork, Tolerances};
use super::newton::{solve_substage_reuse, NewtonWork};
use crate::caps::{Caps, ProblemKind, ProblemKinds};
use crate::register_solver;
use crate::solver::{Solver, SolverState, StepOutcome};
use tsdyn_ir::Evaluator;

/// Stage scratch for TR-BDF2: the start derivative, both stage bases, the first
/// stage's solution, and the shared Newton work (RHS, Jacobian, iteration matrix,
/// pivots, increment, plus the cached-factorization shift). Grown to the system
/// dimension on first use.
///
/// Both substages share the diagonal coefficient (`w = γ/2 = d`), so they share
/// the iteration-matrix shift; the [`NewtonWork`] cache lets the BDF2 substage
/// reuse the trapezoidal substage's frozen Jacobian / LU when the two shifts are
/// bit-equal, and lets the next step's first substage reuse it while `h` holds.
#[derive(Default)]
struct TrBdf2Base {
    f0: Vec<f64>,
    base_buf: Vec<f64>,
    y1: Vec<f64>,
    newton: NewtonWork,
    tol: Tolerances,
}

impl TrBdf2Base {
    fn ensure(&mut self, n: usize) {
        if self.f0.len() != n {
            self.f0 = vec![0.0; n];
            self.base_buf = vec![0.0; n];
            self.y1 = vec![0.0; n];
        }
        // NewtonWork grows and (re)initialises its own cache state.
        self.newton.ensure(n);
    }
}

impl BaseStep for TrBdf2Base {
    // The stage-base / predictor loops combine `u`, `f0`, `y1` and `out`
    // componentwise; a range index reads more clearly than zipping several
    // iterators across disjoint buffers.
    #[allow(clippy::needless_range_loop)]
    fn base_step(
        &mut self,
        ev: &dyn Evaluator,
        u: &[f64],
        t: f64,
        p: &[f64],
        h: f64,
        scratch: &mut [f64],
        out: &mut [f64],
    ) -> BaseOutcome {
        let n = u.len();
        self.ensure(n);

        // Defence in depth: both substages freeze the analytic Jacobian, so an
        // evaluator that cannot supply one would leave the iteration matrix at I
        // and silently degrade the step to forward Euler. The engine boundary
        // rejects this first (`bridge::require_jacobian_if_needed`); guard here too
        // for direct callers.
        if !ev.has_jacobian() {
            return BaseOutcome::Diverged;
        }

        // Method coefficients (γ = 2 − √2). The two substages share I − a·h·J.
        let g = 2.0 - std::f64::consts::SQRT_2;
        let d = 0.5 * g; // trapezoidal substage coefficient
        let denom = g * (2.0 - g);
        let c1 = 1.0 / denom;
        let c0 = -(1.0 - g) * (1.0 - g) / denom; // c1 + c0 == 1

        // BDF2 substage coefficient. (1-g)/(2-g) == γ/2 == d (both = 1 - √2/2),
        // so bind it to `d`: a *bit*-equal shift lets the BDF2 substage reuse the
        // trapezoidal substage's cached LU factorization (no re-factor).
        let w = d;

        // f(t, u).
        ev.eval(u, p, t, scratch, &mut self.f0);
        if !self.f0.iter().all(|x| x.is_finite()) {
            return BaseOutcome::Diverged;
        }

        // --- Substage 1: trapezoidal to t + γh ---
        // y1 = base1 + d·h·f(t+γh, y1), base1 = u + d·h·f0. The reuse path freezes
        // and factors I − d·h·J here (and offers it to substage 2 / the next step).
        for i in 0..n {
            self.base_buf[i] = u[i] + d * h * self.f0[i];
            self.y1[i] = u[i] + g * h * self.f0[i]; // explicit-Euler predictor over γh
        }
        // Borrow the disjoint buffer fields alongside the shared Newton work.
        let TrBdf2Base {
            base_buf,
            y1,
            newton,
            tol,
            ..
        } = self;
        match solve_substage_reuse(
            ev,
            t + g * h,
            p,
            d,
            h,
            base_buf,
            y1,
            newton,
            scratch,
            tol.rtol,
            tol.atol,
        ) {
            BaseOutcome::Ok => {}
            other => return other,
        }

        // --- Substage 2: BDF2 to t + h ---
        // y2 = base2 + w·h·f(t+h, y2), base2 = c1·y1 + c0·u; solution written to out.
        // w == d (stiffly accurate), so this shares the shift w·h = d·h with
        // substage 1 and reuses its cached LU (no Jacobian eval / re-factor) while
        // the quasi-Newton converges; it refreshes only if convergence degrades.
        for i in 0..n {
            self.base_buf[i] = c1 * self.y1[i] + c0 * u[i];
            out[i] = self.y1[i]; // predictor: the trapezoidal stage value
        }
        let TrBdf2Base {
            base_buf,
            newton,
            tol,
            ..
        } = self;
        match solve_substage_reuse(
            ev,
            t + h,
            p,
            w,
            h,
            base_buf,
            out,
            newton,
            scratch,
            tol.rtol,
            tol.atol,
        ) {
            BaseOutcome::Ok => {}
            other => return other,
        }

        if out.iter().all(|x| x.is_finite()) {
            BaseOutcome::Ok
        } else {
            BaseOutcome::Recoverable
        }
    }

    fn order(&self) -> u32 {
        2
    }
}

/// The `trbdf2` solver: an adaptive, L-stable, Jacobian-using stiff kernel. See
/// the [module docs](self).
#[derive(Default)]
pub struct TrBdf2 {
    tol: Tolerances,
    work: DoublingWork,
    base: TrBdf2Base,
}

impl TrBdf2 {
    /// A kernel with the default tolerances ([`Tolerances::DEFAULT`]).
    pub fn new() -> Self {
        TrBdf2 {
            tol: Tolerances::DEFAULT,
            work: DoublingWork::new(),
            base: TrBdf2Base {
                tol: Tolerances::DEFAULT,
                ..TrBdf2Base::default()
            },
        }
    }

    /// A kernel with explicit relative / absolute tolerances.
    pub fn with_tolerances(rtol: f64, atol: f64) -> Self {
        let tol = Tolerances { rtol, atol };
        TrBdf2 {
            tol,
            work: DoublingWork::new(),
            base: TrBdf2Base {
                tol,
                ..TrBdf2Base::default()
            },
        }
    }
}

impl Solver for TrBdf2 {
    fn name(&self) -> &'static str {
        "trbdf2"
    }

    fn caps(&self) -> Caps {
        Caps::implicit(ProblemKinds::of(ProblemKind::Ode)).adaptive()
    }

    fn step(&mut self, ev: &dyn Evaluator, st: &mut SolverState, h: f64) -> StepOutcome {
        self.work.doubled_step(&mut self.base, ev, st, h, self.tol)
    }
}

register_solver!(
    "trbdf2",
    Caps::implicit(ProblemKinds::of(ProblemKind::Ode)).adaptive(),
    || Box::new(TrBdf2::new())
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::SolverKind;

    #[test]
    fn metadata_is_consistent() {
        let s = TrBdf2::new();
        assert_eq!(s.name(), "trbdf2");
        let c = s.caps();
        assert_eq!(c.kind, SolverKind::Implicit);
        assert!(c.adaptive);
        assert!(c.needs_jacobian);
        assert!(c.supports(ProblemKind::Ode));
        assert_eq!(s.base.order(), 2);
    }

    #[test]
    fn coefficients_are_consistent() {
        // c1 + c0 == 1 (so the BDF2 substage reproduces a fixed point), and all
        // coefficients are the canonical γ = 2 − √2 values.
        let g = 2.0 - std::f64::consts::SQRT_2;
        let denom = g * (2.0 - g);
        let c1 = 1.0 / denom;
        let c0 = -(1.0 - g) * (1.0 - g) / denom;
        assert!((c1 + c0 - 1.0).abs() < 1e-15);
        let w = (1.0 - g) / (2.0 - g);
        assert!(w > 0.0 && w < 1.0);
        // Stiffly accurate: the BDF2 weight equals the diagonal d = γ/2.
        assert!((w - 0.5 * g).abs() < 1e-15);
    }

    #[test]
    fn tolerances_propagate_to_the_base() {
        let s = TrBdf2::with_tolerances(1e-7, 1e-10);
        assert_eq!(s.tol.rtol, 1e-7);
        assert_eq!(s.base.tol.atol, 1e-10);
    }
}
