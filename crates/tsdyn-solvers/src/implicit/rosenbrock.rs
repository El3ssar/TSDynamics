//! `rosenbrock` — an L-stable linearly-implicit Euler step (a one-stage
//! Rosenbrock / W-method) with step-doubling error control.
//!
//! Each step freezes the analytic Jacobian `J = ∂f/∂u` the evaluator provides at
//! the step start, forms the iteration matrix `I − h·J`, and solves **one** linear
//! system for the increment — no Newton iteration. That single linear solve is
//! what makes the method cheap per step while staying L-stable: it integrates
//! stiff systems on which the explicit kernels would need vanishingly small steps.
//! Because the Jacobian comes from the symbolic tape (stream E6), it is exact —
//! there is no finite-difference noise.
//!
//! The base step is order 1; [`DoublingWork`](super::control::DoublingWork)
//! Richardson-extrapolates a big step and two half steps to an order-2 result and
//! estimates the local error. `∂f/∂t` is neglected (the "W" in W-method): for
//! autonomous systems — the whole built-in catalogue — this is exact, and for a
//! non-autonomous RHS it perturbs only the order-1 base method, which the
//! extrapolation and adaptive controller absorb.
//!
//! Reference: Wanner & Hairer, *Solving Ordinary Differential Equations II:
//! Stiff and Differential-Algebraic Problems*, 2nd ed., §IV.7 (Rosenbrock and
//! W-methods).

use super::control::{BaseOutcome, BaseStep, DoublingWork, Tolerances};
use super::linalg::{build_shifted, lu_factor, lu_solve};
use crate::caps::{Caps, ProblemKind, ProblemKinds};
use crate::register_solver;
use crate::solver::{Solver, SolverState, StepOutcome};
use tsdyn_ir::Evaluator;

/// The linearly-implicit Euler base step and its per-step scratch (RHS, Jacobian,
/// iteration matrix, pivots, increment), grown to the system dimension on first
/// use and reused with no per-step allocation.
#[derive(Default)]
struct RosBase {
    f0: Vec<f64>,
    jac: Vec<f64>,
    mat: Vec<f64>,
    piv: Vec<usize>,
    rhs: Vec<f64>,
}

impl RosBase {
    fn ensure(&mut self, n: usize) {
        if self.f0.len() != n {
            self.f0 = vec![0.0; n];
            self.jac = vec![0.0; n * n];
            self.mat = vec![0.0; n * n];
            self.piv = vec![0; n];
            self.rhs = vec![0.0; n];
        }
    }
}

impl BaseStep for RosBase {
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

        // f(t,u) and J(t,u) in one tape pass.
        ev.eval_jac(u, p, t, scratch, &mut self.f0, &mut self.jac);
        if !self.f0.iter().all(|x| x.is_finite()) || !self.jac.iter().all(|x| x.is_finite()) {
            return BaseOutcome::Diverged;
        }

        // Solve (I − hJ) k = h f0; the new state is u + k.
        build_shifted(&mut self.mat, &self.jac, n, h);
        if !lu_factor(&mut self.mat, n, &mut self.piv) {
            return BaseOutcome::Recoverable; // singular shift ⇒ shrink h and retry
        }
        for i in 0..n {
            self.rhs[i] = h * self.f0[i];
        }
        lu_solve(&self.mat, n, &self.piv, &mut self.rhs);
        for i in 0..n {
            out[i] = u[i] + self.rhs[i];
        }
        if out.iter().all(|x| x.is_finite()) {
            BaseOutcome::Ok
        } else {
            BaseOutcome::Recoverable
        }
    }

    fn order(&self) -> u32 {
        1
    }
}

/// The `rosenbrock` solver: an adaptive, L-stable, Jacobian-using stiff kernel.
/// See the [module docs](self).
#[derive(Default)]
pub struct RosenbrockW {
    tol: Tolerances,
    work: DoublingWork,
    base: RosBase,
}

impl RosenbrockW {
    /// A kernel with the default tolerances ([`Tolerances::DEFAULT`]).
    pub fn new() -> Self {
        RosenbrockW {
            tol: Tolerances::DEFAULT,
            work: DoublingWork::new(),
            base: RosBase::default(),
        }
    }

    /// A kernel with explicit relative / absolute tolerances.
    pub fn with_tolerances(rtol: f64, atol: f64) -> Self {
        RosenbrockW {
            tol: Tolerances { rtol, atol },
            ..RosenbrockW::new()
        }
    }
}

impl Solver for RosenbrockW {
    fn name(&self) -> &'static str {
        "rosenbrock"
    }

    fn caps(&self) -> Caps {
        Caps::implicit(ProblemKinds::of(ProblemKind::Ode)).adaptive()
    }

    fn step(&mut self, ev: &dyn Evaluator, st: &mut SolverState, h: f64) -> StepOutcome {
        self.work.doubled_step(&mut self.base, ev, st, h, self.tol)
    }
}

register_solver!(
    "rosenbrock",
    Caps::implicit(ProblemKinds::of(ProblemKind::Ode)).adaptive(),
    || Box::new(RosenbrockW::new())
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::SolverKind;

    #[test]
    fn metadata_is_consistent() {
        let s = RosenbrockW::new();
        assert_eq!(s.name(), "rosenbrock");
        let c = s.caps();
        assert_eq!(c.kind, SolverKind::Implicit);
        assert!(c.adaptive);
        assert!(c.needs_jacobian);
        assert!(c.supports(ProblemKind::Ode));
        assert!(!c.supports(ProblemKind::Sde));
        assert_eq!(s.base.order(), 1);
    }

    #[test]
    fn tolerances_builder_overrides_defaults() {
        let s = RosenbrockW::with_tolerances(1e-8, 1e-11);
        assert_eq!(s.tol.rtol, 1e-8);
        assert_eq!(s.tol.atol, 1e-11);
    }
}
