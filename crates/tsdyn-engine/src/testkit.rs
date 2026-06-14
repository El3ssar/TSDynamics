//! Shared test fixtures for the engine's own unit tests (compiled only under
//! `cfg(test)`).
//!
//! The engine drives the *frozen* `Evaluator` and `Solver` seams by dynamic
//! dispatch, so to exercise it end-to-end it needs concrete implementations.
//! The real ones live in other streams (the interpreter is E1's `tsdyn-vm`; the
//! solver families are E3/E4/E-SDE), so here we keep deliberately minimal
//! stand-ins:
//!
//! - [`VmEval`] — wraps E1's [`Interpreter`] as an [`Evaluator`]. The interpreter
//!   exposes the right inherent methods but does not itself `impl Evaluator`
//!   (the orphan rule puts that one-line forward in `tsdyn-vm` or `tsdyn-ir`,
//!   not here); this newtype bridges it for our tests without touching another
//!   stream's crate.
//! - [`ConstantField`] — an evaluator with a constant right-hand side, for exact
//!   closed-form checks (e.g. that grid points land on their exact times).
//! - [`Rk4`] — a textbook fixed-step Runge–Kutta kernel, enough to integrate the
//!   smooth test systems. The production explicit family is stream E3.
//! - [`AlwaysReject`] — a kernel that never accepts, to drive the step-collapse
//!   path.
//! - [`NoisyEuler`] — a diagonal Euler–Maruyama kernel that owns a seeded
//!   [`SplitMix64`], to prove the ensemble's seeded determinism (parallel ==
//!   serial). The production SDE family is stream E-SDE.

use crate::rng::{fill_wiener, SplitMix64};
use tsdyn_ir::Evaluator;
use tsdyn_solvers::{Caps, ProblemKind, ProblemKinds, Solver, SolverState, StepOutcome};
use tsdyn_vm::Interpreter;

/// Adapts E1's [`Interpreter`] to the [`Evaluator`] trait (see module docs).
pub struct VmEval {
    interp: Interpreter,
}

impl VmEval {
    pub fn new(interp: Interpreter) -> Self {
        VmEval { interp }
    }
}

impl Evaluator for VmEval {
    fn dim(&self) -> usize {
        self.interp.dim()
    }
    fn n_param(&self) -> usize {
        self.interp.n_param()
    }
    fn n_scratch(&self) -> usize {
        self.interp.n_scratch()
    }
    fn has_jacobian(&self) -> bool {
        self.interp.has_jacobian()
    }
    fn eval(&self, u: &[f64], p: &[f64], t: f64, scratch: &mut [f64], deriv: &mut [f64]) {
        self.interp.eval(u, p, t, scratch, deriv);
    }
    fn eval_jac(
        &self,
        u: &[f64],
        p: &[f64],
        t: f64,
        scratch: &mut [f64],
        deriv: &mut [f64],
        jac: &mut [f64],
    ) {
        self.interp.eval_jac(u, p, t, scratch, deriv, jac);
    }
}

/// An evaluator with a constant right-hand side `du/dt = c`, so `u(t) = u0 + c t`
/// is exact — handy for checking the integrator's bookkeeping (time landing,
/// grid rows) without any discretization error.
pub struct ConstantField {
    c: Vec<f64>,
}

impl ConstantField {
    pub fn new(c: Vec<f64>) -> Self {
        ConstantField { c }
    }
}

impl Evaluator for ConstantField {
    fn dim(&self) -> usize {
        self.c.len()
    }
    fn n_param(&self) -> usize {
        0
    }
    fn n_scratch(&self) -> usize {
        0
    }
    fn has_jacobian(&self) -> bool {
        false
    }
    fn eval(&self, _u: &[f64], _p: &[f64], _t: f64, _scratch: &mut [f64], deriv: &mut [f64]) {
        deriv.copy_from_slice(&self.c);
    }
    fn eval_jac(
        &self,
        _u: &[f64],
        _p: &[f64],
        _t: f64,
        _scratch: &mut [f64],
        _deriv: &mut [f64],
        _jac: &mut [f64],
    ) {
        unreachable!("ConstantField carries no Jacobian");
    }
}

/// Classic fixed-step 4th-order Runge–Kutta. Stage buffers grow to the system
/// dimension on first use and are reused thereafter (no per-step allocation).
#[derive(Default)]
pub struct Rk4 {
    k1: Vec<f64>,
    k2: Vec<f64>,
    k3: Vec<f64>,
    k4: Vec<f64>,
    tmp: Vec<f64>,
}

impl Rk4 {
    pub fn new() -> Self {
        Rk4 {
            k1: Vec::new(),
            k2: Vec::new(),
            k3: Vec::new(),
            k4: Vec::new(),
            tmp: Vec::new(),
        }
    }

    fn ensure(&mut self, dim: usize) {
        if self.tmp.len() != dim {
            self.k1 = vec![0.0; dim];
            self.k2 = vec![0.0; dim];
            self.k3 = vec![0.0; dim];
            self.k4 = vec![0.0; dim];
            self.tmp = vec![0.0; dim];
        }
    }
}

impl Solver for Rk4 {
    fn name(&self) -> &'static str {
        "testkit-rk4"
    }
    fn caps(&self) -> Caps {
        Caps::explicit(ProblemKinds::of(ProblemKind::Ode))
    }
    fn step(&mut self, ev: &dyn Evaluator, st: &mut SolverState, h: f64) -> StepOutcome {
        let dim = st.u.len();
        self.ensure(dim);
        let SolverState { u, t, p, scratch } = st;

        ev.eval(u, p, *t, scratch, &mut self.k1);
        for ((tmp, &ui), &ki) in self.tmp.iter_mut().zip(u.iter()).zip(self.k1.iter()) {
            *tmp = ui + 0.5 * h * ki;
        }
        ev.eval(&self.tmp, p, *t + 0.5 * h, scratch, &mut self.k2);
        for ((tmp, &ui), &ki) in self.tmp.iter_mut().zip(u.iter()).zip(self.k2.iter()) {
            *tmp = ui + 0.5 * h * ki;
        }
        ev.eval(&self.tmp, p, *t + 0.5 * h, scratch, &mut self.k3);
        for ((tmp, &ui), &ki) in self.tmp.iter_mut().zip(u.iter()).zip(self.k3.iter()) {
            *tmp = ui + h * ki;
        }
        ev.eval(&self.tmp, p, *t + h, scratch, &mut self.k4);

        let sixth = h / 6.0;
        for ((((ui, &k1), &k2), &k3), &k4) in u
            .iter_mut()
            .zip(self.k1.iter())
            .zip(self.k2.iter())
            .zip(self.k3.iter())
            .zip(self.k4.iter())
        {
            *ui += sixth * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        }
        *t += h;
        StepOutcome::Accepted { h_next: h }
    }
}

/// A kernel that always rejects, halving the step each call without ever
/// touching the state — drives [`crate::integrate::IntegrateError::StepCollapsed`].
#[derive(Default)]
pub struct AlwaysReject;

impl AlwaysReject {
    pub fn new() -> Self {
        AlwaysReject
    }
}

impl Solver for AlwaysReject {
    fn name(&self) -> &'static str {
        "testkit-always-reject"
    }
    fn caps(&self) -> Caps {
        Caps::explicit(ProblemKinds::of(ProblemKind::Ode)).adaptive()
    }
    fn step(&mut self, _ev: &dyn Evaluator, _st: &mut SolverState, h: f64) -> StepOutcome {
        StepOutcome::Rejected { h_next: h * 0.5 }
    }
}

/// Diagonal Euler–Maruyama with additive noise: `u ← u + h·f(u) + σ·dW`, where
/// `dW ~ N(0, h)` per component from an owned, seeded [`SplitMix64`]. The seed is
/// fixed at construction, so building one per trajectory index (via
/// [`crate::rng::seed_for`]) gives a stream that depends only on the index — the
/// exact mechanism a real SDE kernel (stream E-SDE) will use, and what the
/// ensemble's parallel-equals-serial guarantee rests on.
pub struct NoisyEuler {
    sigma: f64,
    rng: SplitMix64,
    deriv: Vec<f64>,
    noise: Vec<f64>,
}

impl NoisyEuler {
    pub fn new(sigma: f64, seed: u64) -> Self {
        NoisyEuler {
            sigma,
            rng: SplitMix64::new(seed),
            deriv: Vec::new(),
            noise: Vec::new(),
        }
    }

    fn ensure(&mut self, dim: usize) {
        if self.deriv.len() != dim {
            self.deriv = vec![0.0; dim];
            self.noise = vec![0.0; dim];
        }
    }
}

impl Solver for NoisyEuler {
    fn name(&self) -> &'static str {
        "testkit-noisy-euler"
    }
    fn caps(&self) -> Caps {
        Caps::explicit(ProblemKinds::of(ProblemKind::Sde))
    }
    fn step(&mut self, ev: &dyn Evaluator, st: &mut SolverState, h: f64) -> StepOutcome {
        let dim = st.u.len();
        self.ensure(dim);
        let SolverState { u, t, p, scratch } = st;
        ev.eval(u, p, *t, scratch, &mut self.deriv);
        fill_wiener(&mut self.rng, h, &mut self.noise);
        for ((ui, &di), &ni) in u.iter_mut().zip(self.deriv.iter()).zip(self.noise.iter()) {
            *ui += h * di + self.sigma * ni;
        }
        *t += h;
        StepOutcome::Accepted { h_next: h }
    }
}
