//! Third-order Strong-Stability-Preserving Runge–Kutta (SSPRK3) — the
//! Shu–Osher method, registered as `ssprk3`.
//!
//! A three-stage, third-order explicit method built from convex combinations of
//! forward-Euler substeps, so it preserves any strong-stability (e.g.
//! total-variation-diminishing) bound the Euler step satisfies under a CFL
//! condition. This makes it the standard time integrator for hyperbolic
//! conservation laws and method-of-lines discretisations of PDEs, where ordinary
//! RK methods can introduce spurious oscillations.
//!
//! Reference: C.-W. Shu & S. Osher, "Efficient implementation of essentially
//! non-oscillatory shock-capturing schemes", *J. Comput. Phys.* **77** (1988)
//! 439–471.
//!
//! Butcher tableau:
//! ```text
//!   0  |
//!   1  |  1
//!  1/2 | 1/4  1/4
//!  ----+--------------
//!      | 1/6  1/6  2/3
//! ```

use crate::{
    register_solver, Caps, Evaluator, ProblemKind, ProblemKinds, Solver, SolverState, StepOutcome,
};

use super::control::{fixed_step, RkWork};

const C: &[f64] = &[0.0, 1.0, 0.5];
const A: &[&[f64]] = &[&[], &[1.0], &[1.0 / 4.0, 1.0 / 4.0]];
const B: &[f64] = &[1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0];

/// The SSPRK3 order-3 fixed-step kernel.
#[derive(Default)]
pub struct SspRk3 {
    work: RkWork,
}

impl SspRk3 {
    /// A fresh kernel with unallocated buffers (sized on the first step).
    pub fn new() -> Self {
        SspRk3 {
            work: RkWork::new(),
        }
    }
}

impl Solver for SspRk3 {
    fn name(&self) -> &'static str {
        "ssprk3"
    }

    fn caps(&self) -> Caps {
        Caps::explicit(ProblemKinds::of(ProblemKind::Ode))
    }

    fn step(&mut self, ev: &dyn Evaluator, st: &mut SolverState, h: f64) -> StepOutcome {
        fixed_step(ev, st, h, C, A, B, &mut self.work)
    }
}

register_solver!(
    "ssprk3",
    Caps::explicit(ProblemKinds::of(ProblemKind::Ode)),
    || Box::new(SspRk3::new())
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explicit::testkit::{converges_at_order, fixed_propagate, HarmonicEval};

    #[test]
    fn caps_are_explicit_non_adaptive_ode() {
        let s = SspRk3::new();
        assert_eq!(s.name(), "ssprk3");
        assert!(!s.caps().adaptive);
    }

    #[test]
    fn tableau_is_internally_consistent() {
        let sum_b: f64 = B.iter().sum();
        assert!((sum_b - 1.0).abs() < 1e-15, "Σb = {sum_b}");
        for (i, row) in A.iter().enumerate() {
            let s: f64 = row.iter().sum();
            assert!((s - C[i]).abs() < 1e-15, "row {i}: Σa = {s}, c = {}", C[i]);
        }
    }

    #[test]
    fn third_order_convergence_on_harmonic() {
        let ev = HarmonicEval { omega: 1.0 };
        let order = converges_at_order(
            |st, h, work| fixed_propagate(&ev, st, h, C, A, B, work),
            &ev,
            vec![1.0, 0.0],
            2.0,
            &[0.1, 0.05, 0.025],
            |t| vec![t.cos(), -t.sin()],
        );
        assert!((order - 3.0).abs() < 0.35, "measured SSPRK3 order {order}");
    }
}
