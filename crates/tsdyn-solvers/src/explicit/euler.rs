//! Forward (explicit) Euler — the order-1 fixed-step method, registered as
//! `euler`.
//!
//! The simplest Runge–Kutta method: one right-hand-side evaluation per step,
//! `u_{n+1} = u_n + h·f(t_n, u_n)`. It is first-order accurate and only
//! conditionally stable, so it is rarely the right production choice — but it is
//! the canonical teaching/reference integrator and a useful baseline against
//! which higher-order kernels are compared.
//!
//! Reference: L. Euler, *Institutionum calculi integralis* (1768), vol. I.
//!
//! Butcher tableau:
//! ```text
//!   0 |
//!  ---+---
//!     | 1
//! ```

use crate::{
    register_solver, Caps, Evaluator, ProblemKind, ProblemKinds, Solver, SolverState, StepOutcome,
};

use super::control::{fixed_step, RkWork};

const C: &[f64] = &[0.0];
const A: &[&[f64]] = &[&[]];
const B: &[f64] = &[1.0];

/// The forward-Euler fixed-step kernel.
#[derive(Default)]
pub struct Euler {
    work: RkWork,
}

impl Euler {
    /// A fresh kernel with unallocated buffers (sized on the first step).
    pub fn new() -> Self {
        Euler {
            work: RkWork::new(),
        }
    }
}

impl Solver for Euler {
    fn name(&self) -> &'static str {
        "euler"
    }

    fn caps(&self) -> Caps {
        Caps::explicit(ProblemKinds::of(ProblemKind::Ode))
    }

    fn step(&mut self, ev: &dyn Evaluator, st: &mut SolverState, h: f64) -> StepOutcome {
        fixed_step(ev, st, h, C, A, B, &mut self.work)
    }
}

register_solver!(
    "euler",
    Caps::explicit(ProblemKinds::of(ProblemKind::Ode)),
    || Box::new(Euler::new())
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explicit::testkit::{converges_at_order, fixed_propagate, DecayEval};

    #[test]
    fn caps_are_explicit_non_adaptive_ode() {
        let s = Euler::new();
        assert_eq!(s.name(), "euler");
        let c = s.caps();
        assert!(!c.adaptive);
        assert!(!c.needs_jacobian);
        assert!(c.supports(ProblemKind::Ode));
    }

    #[test]
    fn tableau_is_internally_consistent() {
        let sum_b: f64 = B.iter().sum();
        assert!((sum_b - 1.0).abs() < 1e-15, "Σb = {sum_b}");
    }

    #[test]
    fn first_order_convergence_on_decay() {
        // Global error of forward Euler must fall by ~2 when the step is halved.
        let ev = DecayEval { dim: 1 };
        let order = converges_at_order(
            |st, h, work| fixed_propagate(&ev, st, h, C, A, B, work),
            &ev,
            vec![1.0],
            1.0,
            &[0.01, 0.005, 0.0025],
            |t| vec![(-t).exp()],
        );
        assert!((order - 1.0).abs() < 0.2, "measured Euler order {order}");
    }
}
