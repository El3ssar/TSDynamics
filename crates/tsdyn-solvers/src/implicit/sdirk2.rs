//! `sdirk2` — a two-stage, second-order, L-stable singly-diagonally-implicit
//! Runge–Kutta method (Alexander's SDIRK).
//!
//! Both stages share the same diagonal coefficient `γ = 1 − √2/2`, so they reuse
//! one iteration-matrix shape `I − γ·h·J`, and the method is stiffly accurate
//! (the second stage *is* the step), hence L-stable. With γ chosen as the smaller
//! root of `γ² − 2γ + ½ = 0` the method is second-order and L-stable — a robust,
//! strongly-damping stiff workhorse with a genuine multi-stage Runge–Kutta
//! structure (distinct from the Rosenbrock W-method and TR-BDF2, which it
//! cross-validates).
//!
//! ```text
//!   γ | γ      0          (Y1 stage)
//!   1 | 1−γ    γ          (Y2 stage = step, stiffly accurate)
//!  ---+----------
//!     | 1−γ    γ
//! ```
//!
//! The base step is order 2; [`DoublingWork`](super::control::DoublingWork)
//! Richardson-extrapolates and estimates the local error.
//!
//! Reference: R. Alexander, "Diagonally implicit Runge–Kutta methods for stiff
//! ODEs", *SIAM J. Numer. Anal.* **14**(6) (1977) 1006–1021.

use super::control::{BaseOutcome, BaseStep, DoublingWork, Tolerances};
use super::newton::{solve_substage_reuse, NewtonWork};
use crate::caps::{Caps, ProblemKind, ProblemKinds};
use crate::register_solver;
use crate::solver::{Solver, SolverState, StepOutcome};
use tsdyn_ir::Evaluator;

/// The SDIRK2 base step: two Newton substages and their scratch.
#[derive(Default)]
struct Sdirk2Base {
    f0: Vec<f64>,
    y1: Vec<f64>,
    k1: Vec<f64>,
    base2: Vec<f64>,
    newton: NewtonWork,
    tol: Tolerances,
}

impl Sdirk2Base {
    fn ensure(&mut self, n: usize) {
        if self.f0.len() != n {
            self.f0 = vec![0.0; n];
            self.y1 = vec![0.0; n];
            self.k1 = vec![0.0; n];
            self.base2 = vec![0.0; n];
        }
    }
}

impl BaseStep for Sdirk2Base {
    // The stage predictor / combination loops index several disjoint buffers
    // (u, f0, y1, k1, base2, out) together; a range index reads more clearly than
    // zipping many iterators.
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
        if !ev.has_jacobian() {
            return BaseOutcome::Diverged;
        }
        let gamma = 1.0 - 0.5 * std::f64::consts::SQRT_2;

        // f(t, u) for the stage predictors.
        ev.eval(u, p, t, scratch, &mut self.f0);
        if !self.f0.iter().all(|x| x.is_finite()) {
            return BaseOutcome::Diverged;
        }

        // --- Stage 1: Y1 = u + γ·h·f(t+γh, Y1) ---
        // Both SDIRK substages share the diagonal γ and the step h, so the Newton
        // iteration matrix I − γ·h·J is *identical* for stage 1 and stage 2 (and,
        // while the controller holds h, for the next step's stage 1). The reuse
        // path freezes/factors it here and stage 2 reuses that LU; the shift key
        // (γ·h) makes the step-doubling half steps (γ·h/2) re-form automatically.
        for i in 0..n {
            self.y1[i] = u[i] + gamma * h * self.f0[i]; // explicit predictor
        }
        match solve_substage_reuse(
            ev,
            t + gamma * h,
            p,
            gamma,
            h,
            u,
            &mut self.y1,
            &mut self.newton,
            scratch,
            self.tol.rtol,
            self.tol.atol,
        ) {
            BaseOutcome::Ok => {}
            other => return other,
        }
        // k1 = f(t+γh, Y1).
        ev.eval(&self.y1, p, t + gamma * h, scratch, &mut self.k1);
        if !self.k1.iter().all(|x| x.is_finite()) {
            return BaseOutcome::Recoverable;
        }

        // --- Stage 2: Y2 = base2 + γ·h·f(t+h, Y2), base2 = u + (1−γ)·h·k1 ---
        for i in 0..n {
            self.base2[i] = u[i] + (1.0 - gamma) * h * self.k1[i];
            out[i] = self.base2[i] + gamma * h * self.k1[i]; // explicit predictor
        }
        // Stage 2 shares the shift γ·h with stage 1, so this reuses the cached LU
        // (no Jacobian evaluation, no re-factorization) while the quasi-Newton
        // converges; it refreshes only if convergence degrades.
        match solve_substage_reuse(
            ev,
            t + h,
            p,
            gamma,
            h,
            &self.base2,
            out,
            &mut self.newton,
            scratch,
            self.tol.rtol,
            self.tol.atol,
        ) {
            BaseOutcome::Ok => {}
            other => return other,
        }
        // u_{n+1} = Y2 (stiffly accurate), already in `out`.
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

/// The `sdirk2` solver: an adaptive, L-stable, Jacobian-using stiff kernel. See
/// the [module docs](self).
#[derive(Default)]
pub struct Sdirk2 {
    tol: Tolerances,
    work: DoublingWork,
    base: Sdirk2Base,
}

impl Sdirk2 {
    /// A kernel with the default tolerances ([`Tolerances::DEFAULT`]).
    pub fn new() -> Self {
        Sdirk2 {
            tol: Tolerances::DEFAULT,
            work: DoublingWork::new(),
            base: Sdirk2Base {
                tol: Tolerances::DEFAULT,
                ..Sdirk2Base::default()
            },
        }
    }

    /// A kernel with explicit relative / absolute tolerances.
    pub fn with_tolerances(rtol: f64, atol: f64) -> Self {
        let tol = Tolerances { rtol, atol };
        Sdirk2 {
            tol,
            work: DoublingWork::new(),
            base: Sdirk2Base {
                tol,
                ..Sdirk2Base::default()
            },
        }
    }
}

impl Solver for Sdirk2 {
    fn name(&self) -> &'static str {
        "sdirk2"
    }

    fn caps(&self) -> Caps {
        Caps::implicit(ProblemKinds::of(ProblemKind::Ode)).adaptive()
    }

    fn step(&mut self, ev: &dyn Evaluator, st: &mut SolverState, h: f64) -> StepOutcome {
        self.work.doubled_step(&mut self.base, ev, st, h, self.tol)
    }
}

register_solver!(
    "sdirk2",
    Caps::implicit(ProblemKinds::of(ProblemKind::Ode)).adaptive(),
    || Box::new(Sdirk2::new())
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::SolverKind;

    #[test]
    fn metadata_is_consistent() {
        let s = Sdirk2::new();
        assert_eq!(s.name(), "sdirk2");
        let c = s.caps();
        assert_eq!(c.kind, SolverKind::Implicit);
        assert!(c.adaptive);
        assert!(c.needs_jacobian);
        assert_eq!(s.base.order(), 2);
    }

    #[test]
    fn gamma_satisfies_the_order_two_condition() {
        // γ is the smaller root of γ² − 2γ + ½ = 0 (the L-stable, 2nd-order choice).
        let gamma = 1.0 - 0.5 * std::f64::consts::SQRT_2;
        assert!((gamma * gamma - 2.0 * gamma + 0.5).abs() < 1e-15);
        assert!(gamma > 0.25, "γ must be ≥ 1/4 for A-stability");
    }

    #[test]
    fn tolerances_propagate_to_the_base() {
        let s = Sdirk2::with_tolerances(1e-7, 1e-10);
        assert_eq!(s.tol.rtol, 1e-7);
        assert_eq!(s.base.tol.atol, 1e-10);
    }
}
