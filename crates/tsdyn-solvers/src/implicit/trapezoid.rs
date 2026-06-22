//! `trapezoid` — the implicit trapezoidal rule (Crank–Nicolson), order 2,
//! A-stable.
//!
//! Each step solves `u_{n+1} = u_n + (h/2)·(f(t,u_n) + f(t+h, u_{n+1}))` by a
//! modified-Newton iteration. The trapezoidal rule is second-order and A-stable
//! (so it is unconditionally stable on stiff problems), but it is *not* L-stable:
//! its stability function tends to −1 at infinity, so very stiff transients are
//! reflected rather than damped and can ring. It is the right choice for mildly
//! stiff, oscillatory problems where its symmetry preserves the phase, and it
//! cross-validates the strongly-damping L-stable kernels.
//!
//! Reference: J. Crank & P. Nicolson, "A practical method for numerical
//! evaluation of solutions of partial differential equations of the
//! heat-conduction type", *Proc. Camb. Phil. Soc.* **43** (1947) 50–67.

use super::control::{BaseOutcome, BaseStep, DoublingWork, Tolerances};
use super::newton::{solve_substage, NewtonWork};
use crate::caps::{Caps, ProblemKind, ProblemKinds};
use crate::register_solver;
use crate::solver::{Solver, SolverState, StepOutcome};
use tsdyn_ir::Evaluator;

/// The trapezoidal base step and its scratch.
#[derive(Default)]
struct TrapezoidBase {
    f0: Vec<f64>,
    base_buf: Vec<f64>,
    newton: NewtonWork,
    tol: Tolerances,
}

impl BaseStep for TrapezoidBase {
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
            self.base_buf = vec![0.0; n];
        }
        if !ev.has_jacobian() {
            return BaseOutcome::Diverged;
        }

        // f(t, u): the explicit half of the trapezoid and the predictor slope.
        ev.eval(u, p, t, scratch, &mut self.f0);
        if !self.f0.iter().all(|x| x.is_finite()) {
            return BaseOutcome::Diverged;
        }
        // base = u + (h/2)·f(t,u); solve out = base + (h/2)·f(t+h, out).
        for i in 0..n {
            self.base_buf[i] = u[i] + 0.5 * h * self.f0[i];
            out[i] = u[i] + h * self.f0[i]; // explicit-Euler predictor
        }
        solve_substage(
            ev,
            t + h,
            p,
            0.5,
            h,
            &self.base_buf,
            out,
            &mut self.newton,
            scratch,
            self.tol.rtol,
            self.tol.atol,
        )
    }

    fn order(&self) -> u32 {
        2
    }
}

/// The `trapezoid` solver: an adaptive, A-stable, Jacobian-using stiff kernel.
/// See the [module docs](self).
#[derive(Default)]
pub struct Trapezoid {
    tol: Tolerances,
    work: DoublingWork,
    base: TrapezoidBase,
}

impl Trapezoid {
    /// A kernel with the default tolerances ([`Tolerances::DEFAULT`]).
    pub fn new() -> Self {
        Trapezoid {
            tol: Tolerances::DEFAULT,
            work: DoublingWork::new(),
            base: TrapezoidBase {
                tol: Tolerances::DEFAULT,
                ..TrapezoidBase::default()
            },
        }
    }

    /// A kernel with explicit relative / absolute tolerances.
    pub fn with_tolerances(rtol: f64, atol: f64) -> Self {
        let tol = Tolerances { rtol, atol };
        Trapezoid {
            tol,
            work: DoublingWork::new(),
            base: TrapezoidBase {
                tol,
                ..TrapezoidBase::default()
            },
        }
    }
}

impl Solver for Trapezoid {
    fn name(&self) -> &'static str {
        "trapezoid"
    }

    fn caps(&self) -> Caps {
        Caps::implicit(ProblemKinds::of(ProblemKind::Ode)).adaptive()
    }

    fn step(&mut self, ev: &dyn Evaluator, st: &mut SolverState, h: f64) -> StepOutcome {
        self.work.doubled_step(&mut self.base, ev, st, h, self.tol)
    }
}

register_solver!(
    "trapezoid",
    Caps::implicit(ProblemKinds::of(ProblemKind::Ode)).adaptive(),
    || Box::new(Trapezoid::new())
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::SolverKind;

    #[test]
    fn metadata_is_consistent() {
        let s = Trapezoid::new();
        assert_eq!(s.name(), "trapezoid");
        let c = s.caps();
        assert_eq!(c.kind, SolverKind::Implicit);
        assert!(c.adaptive);
        assert!(c.needs_jacobian);
        assert_eq!(s.base.order(), 2);
    }

    #[test]
    fn tolerances_propagate_to_the_base() {
        let s = Trapezoid::with_tolerances(1e-7, 1e-10);
        assert_eq!(s.tol.rtol, 1e-7);
        assert_eq!(s.base.tol.atol, 1e-10);
    }
}
