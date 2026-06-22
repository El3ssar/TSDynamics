//! `backward_euler` — the implicit (backward) Euler method, order 1, L-stable.
//!
//! Each step solves the fully-implicit relation `u_{n+1} = u_n + h·f(t+h,
//! u_{n+1})` for the new state by a modified-Newton iteration on the analytic
//! Jacobian (the shared [`newton`](super::newton) machinery). It is the canonical
//! stiff integrator: unconditionally stable and strongly damping (L-stable), so
//! it never amplifies a stiff transient, at the cost of only first-order
//! accuracy. The base step is order 1; [`DoublingWork`](super::control::DoublingWork)
//! Richardson-extrapolates a big step and two half steps to an order-2 result and
//! estimates the local error.
//!
//! Reference: E. Hairer & G. Wanner, *Solving Ordinary Differential Equations II:
//! Stiff and Differential-Algebraic Problems*, 2nd ed. (1996), §IV.3.

use super::control::{BaseOutcome, BaseStep, DoublingWork, Tolerances};
use super::newton::{solve_substage, NewtonWork};
use crate::caps::{Caps, ProblemKind, ProblemKinds};
use crate::register_solver;
use crate::solver::{Solver, SolverState, StepOutcome};
use tsdyn_ir::Evaluator;

/// The backward-Euler base step and its scratch (the start slope used as an
/// explicit-Euler predictor, plus the shared Newton work).
#[derive(Default)]
struct BackwardEulerBase {
    f0: Vec<f64>,
    newton: NewtonWork,
    tol: Tolerances,
}

impl BaseStep for BackwardEulerBase {
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
        if self.f0.len() != n {
            self.f0 = vec![0.0; n];
        }
        // Defence in depth: an implicit kernel needs the analytic Jacobian (the
        // engine boundary rejects a Jacobian-free tape first).
        if !ev.has_jacobian() {
            return BaseOutcome::Diverged;
        }

        // Explicit-Euler predictor for the Newton initial guess.
        ev.eval(u, p, t, scratch, &mut self.f0);
        if !self.f0.iter().all(|x| x.is_finite()) {
            return BaseOutcome::Diverged;
        }
        for i in 0..n {
            out[i] = u[i] + h * self.f0[i];
        }

        // Solve out = u + h·f(t+h, out) by Newton (coef = 1, base = u).
        solve_substage(
            ev,
            t + h,
            p,
            1.0,
            h,
            u,
            out,
            &mut self.newton,
            scratch,
            self.tol.rtol,
            self.tol.atol,
        )
    }

    fn order(&self) -> u32 {
        1
    }
}

/// The `backward_euler` solver: an adaptive, L-stable, Jacobian-using stiff
/// kernel. See the [module docs](self).
#[derive(Default)]
pub struct BackwardEuler {
    tol: Tolerances,
    work: DoublingWork,
    base: BackwardEulerBase,
}

impl BackwardEuler {
    /// A kernel with the default tolerances ([`Tolerances::DEFAULT`]).
    pub fn new() -> Self {
        BackwardEuler {
            tol: Tolerances::DEFAULT,
            work: DoublingWork::new(),
            base: BackwardEulerBase {
                tol: Tolerances::DEFAULT,
                ..BackwardEulerBase::default()
            },
        }
    }

    /// A kernel with explicit relative / absolute tolerances.
    pub fn with_tolerances(rtol: f64, atol: f64) -> Self {
        let tol = Tolerances { rtol, atol };
        BackwardEuler {
            tol,
            work: DoublingWork::new(),
            base: BackwardEulerBase {
                tol,
                ..BackwardEulerBase::default()
            },
        }
    }
}

impl Solver for BackwardEuler {
    fn name(&self) -> &'static str {
        "backward_euler"
    }

    fn caps(&self) -> Caps {
        Caps::implicit(ProblemKinds::of(ProblemKind::Ode)).adaptive()
    }

    fn step(&mut self, ev: &dyn Evaluator, st: &mut SolverState, h: f64) -> StepOutcome {
        self.work.doubled_step(&mut self.base, ev, st, h, self.tol)
    }
}

register_solver!(
    "backward_euler",
    Caps::implicit(ProblemKinds::of(ProblemKind::Ode)).adaptive(),
    || Box::new(BackwardEuler::new())
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::SolverKind;

    #[test]
    fn metadata_is_consistent() {
        let s = BackwardEuler::new();
        assert_eq!(s.name(), "backward_euler");
        let c = s.caps();
        assert_eq!(c.kind, SolverKind::Implicit);
        assert!(c.adaptive);
        assert!(c.needs_jacobian);
        assert!(c.supports(ProblemKind::Ode));
        assert_eq!(s.base.order(), 1);
    }

    #[test]
    fn tolerances_propagate_to_the_base() {
        let s = BackwardEuler::with_tolerances(1e-7, 1e-10);
        assert_eq!(s.tol.rtol, 1e-7);
        assert_eq!(s.base.tol.atol, 1e-10);
    }
}
