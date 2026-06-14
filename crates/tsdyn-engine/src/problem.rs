//! Problem definitions — the bundle of *what* is being integrated, separate
//! from *how* ([`Solver`]) and *with what evaluator* ([`Evaluator`]).
//!
//! [`OdeProblem`] is the ODE family's bundle: a shared evaluator plus its
//! parameter vector. It is a thin, ergonomic facade over the free functions in
//! [`crate::integrate`] and [`crate::ensemble`] — it holds the evaluator and
//! parameters once so callers (the Python binding in stream E7, derived systems,
//! analyses) don't re-thread them through every call. The DDE/SDE/map problem
//! types are added by their own streams (E-DDE/E-SDE/E-MAP) alongside this one.

use tsdyn_ir::Evaluator;
use tsdyn_solvers::Solver;

use crate::ensemble::{ensemble_final, EnsembleFinal};
use crate::integrate::{integrate_final, integrate_grid, IntegrateConfig, IntegrateError};

/// An ODE initial-value problem: a compiled right-hand side and its parameters.
///
/// Borrows the [`Evaluator`] (built once, shared — including across an
/// ensemble's rayon workers, since `Evaluator: Sync`) and owns the parameter
/// vector. Construct one, then call [`integrate_final`](OdeProblem::integrate_final),
/// [`integrate_grid`](OdeProblem::integrate_grid) or
/// [`ensemble_final`](OdeProblem::ensemble_final).
pub struct OdeProblem<'e> {
    ev: &'e dyn Evaluator,
    p: Vec<f64>,
}

impl<'e> OdeProblem<'e> {
    /// Bundle an evaluator with its parameter vector (`p.len()` must equal
    /// `ev.n_param()`).
    pub fn new(ev: &'e dyn Evaluator, p: Vec<f64>) -> Self {
        debug_assert_eq!(
            p.len(),
            ev.n_param(),
            "parameter vector length must equal the evaluator's n_param"
        );
        OdeProblem { ev, p }
    }

    /// The system dimension.
    pub fn dim(&self) -> usize {
        self.ev.dim()
    }

    /// The borrowed evaluator.
    pub fn evaluator(&self) -> &dyn Evaluator {
        self.ev
    }

    /// The parameter vector.
    pub fn params(&self) -> &[f64] {
        &self.p
    }

    /// Integrate from `t0` to `t1`, returning the final state. See
    /// [`crate::integrate::integrate_final`].
    pub fn integrate_final(
        &self,
        solver: &mut dyn Solver,
        u0: &[f64],
        t0: f64,
        t1: f64,
        cfg: &IntegrateConfig,
    ) -> Result<Vec<f64>, IntegrateError> {
        integrate_final(self.ev, solver, u0, &self.p, t0, t1, cfg)
    }

    /// Integrate through `t_eval`, returning a flat `(t_eval.len(), dim)` buffer.
    /// See [`crate::integrate::integrate_grid`].
    pub fn integrate_grid(
        &self,
        solver: &mut dyn Solver,
        u0: &[f64],
        t_eval: &[f64],
        cfg: &IntegrateConfig,
    ) -> Result<Vec<f64>, IntegrateError> {
        integrate_grid(self.ev, solver, u0, &self.p, t_eval, cfg)
    }

    /// Integrate a batch of initial conditions to `t1` in parallel. See
    /// [`crate::ensemble::ensemble_final`].
    pub fn ensemble_final<F>(
        &self,
        make_solver: F,
        u0_batch: &[f64],
        t0: f64,
        t1: f64,
        cfg: &IntegrateConfig,
    ) -> EnsembleFinal
    where
        F: Fn(usize) -> Box<dyn Solver> + Sync,
    {
        ensemble_final(self.ev, make_solver, u0_batch, &self.p, t0, t1, cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{Rk4, VmEval};
    use tsdyn_ir::TapeBuilder;
    use tsdyn_vm::Interpreter;

    /// dx/dt = -k x with one parameter k.
    fn decay() -> Interpreter {
        let mut b = TapeBuilder::new();
        let k = b.param(0);
        let x = b.state(0);
        let kx = b.mul(k, x);
        let dx = b.neg(kx);
        Interpreter::new(b.finish(&[dx], &[], 1, 1).unwrap())
    }

    #[test]
    fn problem_carries_params_through_the_facade() {
        let ev = VmEval::new(decay());
        let prob = OdeProblem::new(&ev, vec![3.0]);
        assert_eq!(prob.dim(), 1);
        assert_eq!(prob.params(), &[3.0]);

        let mut s = Rk4::new();
        let cfg = IntegrateConfig::new(0.001);
        let got = prob
            .integrate_final(&mut s, &[1.0], 0.0, 1.0, &cfg)
            .unwrap();
        let want = (-3.0_f64).exp();
        assert!((got[0] - want).abs() < 1e-9, "got {}, want {want}", got[0]);
    }

    #[test]
    fn problem_ensemble_uses_the_bound_params() {
        let ev = VmEval::new(decay());
        let prob = OdeProblem::new(&ev, vec![2.0]);
        let cfg = IntegrateConfig::new(0.001);
        let u0 = [1.0, 2.0, 3.0];
        let ens = prob.ensemble_final(|_| Box::new(Rk4::new()), &u0, 0.0, 1.0, &cfg);
        let decay_factor = (-2.0_f64).exp();
        for (i, &x0) in u0.iter().enumerate() {
            assert!((ens.row(i)[0] - x0 * decay_factor).abs() < 1e-9);
        }
    }
}
