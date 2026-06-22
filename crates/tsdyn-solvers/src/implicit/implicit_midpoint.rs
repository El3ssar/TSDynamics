//! `implicit_midpoint` — the implicit midpoint rule (the one-stage Gauss–Legendre
//! collocation method), order 2, A-stable.
//!
//! Each step solves `u_{n+1} = u_n + h·f(t + h/2, (u_n + u_{n+1})/2)`. Equivalent
//! to computing the midpoint stage `Y = u_n + (h/2)·f(t+h/2, Y)` by Newton and
//! advancing `u_{n+1} = 2Y − u_n`. It is second-order, A-stable (but, like the
//! trapezoidal rule, not L-stable).
//!
//! **On symplecticity.** The *base* implicit-midpoint step is symplectic and
//! time-symmetric (it conserves quadratic invariants of a Hamiltonian system),
//! but the adaptive integrator delivered here drives that base step through the
//! shared step-doubling + Richardson [`DoublingWork`](super::control::DoublingWork)
//! controller, and Richardson extrapolation of a symplectic method is **not**
//! symplectic (Hairer, Lubich & Wanner, *Geometric Numerical Integration*, 2nd
//! ed. (2006), §V.3 / §VIII.5). So this kernel does *not* provide the long-time
//! energy-conservation guarantee of a fixed-step symplectic integrator; it is an
//! accurate A-stable order-2 implicit method. A dedicated non-extrapolated
//! symplectic path is a possible future addition (it would commit the bare
//! `2Y − u` state instead of the extrapolant).
//!
//! Reference: E. Hairer, C. Lubich & G. Wanner, *Geometric Numerical
//! Integration*, 2nd ed. (2006), §II.1 (the implicit midpoint rule as a Gauss
//! collocation method).

use super::control::{BaseOutcome, BaseStep, DoublingWork, Tolerances};
use super::newton::{solve_substage, NewtonWork};
use crate::caps::{Caps, ProblemKind, ProblemKinds};
use crate::register_solver;
use crate::solver::{Solver, SolverState, StepOutcome};
use tsdyn_ir::Evaluator;

/// The implicit-midpoint base step and its scratch.
#[derive(Default)]
struct ImplicitMidpointBase {
    f0: Vec<f64>,
    newton: NewtonWork,
    tol: Tolerances,
}

impl BaseStep for ImplicitMidpointBase {
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
        if !ev.has_jacobian() {
            return BaseOutcome::Diverged;
        }

        // Predictor for the midpoint stage Y ≈ u + (h/2)·f(t, u).
        ev.eval(u, p, t, scratch, &mut self.f0);
        if !self.f0.iter().all(|x| x.is_finite()) {
            return BaseOutcome::Diverged;
        }
        for i in 0..n {
            out[i] = u[i] + 0.5 * h * self.f0[i];
        }
        // Solve the midpoint stage Y = u + (h/2)·f(t+h/2, Y) (coef = 1/2, base = u);
        // `out` carries Y on return.
        match solve_substage(
            ev,
            t + 0.5 * h,
            p,
            0.5,
            h,
            u,
            out,
            &mut self.newton,
            scratch,
            self.tol.rtol,
            self.tol.atol,
        ) {
            BaseOutcome::Ok => {}
            other => return other,
        }
        // u_{n+1} = 2·Y − u.
        for i in 0..n {
            out[i] = 2.0 * out[i] - u[i];
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

/// The `implicit_midpoint` solver: an adaptive, A-stable, symplectic stiff
/// kernel. See the [module docs](self).
#[derive(Default)]
pub struct ImplicitMidpoint {
    tol: Tolerances,
    work: DoublingWork,
    base: ImplicitMidpointBase,
}

impl ImplicitMidpoint {
    /// A kernel with the default tolerances ([`Tolerances::DEFAULT`]).
    pub fn new() -> Self {
        ImplicitMidpoint {
            tol: Tolerances::DEFAULT,
            work: DoublingWork::new(),
            base: ImplicitMidpointBase {
                tol: Tolerances::DEFAULT,
                ..ImplicitMidpointBase::default()
            },
        }
    }

    /// A kernel with explicit relative / absolute tolerances.
    pub fn with_tolerances(rtol: f64, atol: f64) -> Self {
        let tol = Tolerances { rtol, atol };
        ImplicitMidpoint {
            tol,
            work: DoublingWork::new(),
            base: ImplicitMidpointBase {
                tol,
                ..ImplicitMidpointBase::default()
            },
        }
    }
}

impl Solver for ImplicitMidpoint {
    fn name(&self) -> &'static str {
        "implicit_midpoint"
    }

    fn caps(&self) -> Caps {
        Caps::implicit(ProblemKinds::of(ProblemKind::Ode)).adaptive()
    }

    fn step(&mut self, ev: &dyn Evaluator, st: &mut SolverState, h: f64) -> StepOutcome {
        self.work.doubled_step(&mut self.base, ev, st, h, self.tol)
    }
}

register_solver!(
    "implicit_midpoint",
    Caps::implicit(ProblemKinds::of(ProblemKind::Ode)).adaptive(),
    || Box::new(ImplicitMidpoint::new())
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::SolverKind;

    #[test]
    fn metadata_is_consistent() {
        let s = ImplicitMidpoint::new();
        assert_eq!(s.name(), "implicit_midpoint");
        let c = s.caps();
        assert_eq!(c.kind, SolverKind::Implicit);
        assert!(c.adaptive);
        assert!(c.needs_jacobian);
        assert_eq!(s.base.order(), 2);
    }

    #[test]
    fn tolerances_propagate_to_the_base() {
        let s = ImplicitMidpoint::with_tolerances(1e-7, 1e-10);
        assert_eq!(s.tol.rtol, 1e-7);
        assert_eq!(s.base.tol.atol, 1e-10);
    }
}
