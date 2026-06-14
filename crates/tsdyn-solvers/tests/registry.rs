//! End-to-end proof of the F2 seam from a *consumer* crate's point of view —
//! exactly how E3/E4/E-SDE (and out-of-tree kernel crates) will use it.
//!
//! An integration test is a separate crate that depends only on `tsdyn-solvers`.
//! It defines a kernel, registers it with [`register_solver!`], and then:
//!
//! 1. discovers it **by name** through the public registry ([`find`]/[`make`]/
//!    [`available`]) — the acceptance criterion "a dummy solver registers and is
//!    discoverable by name";
//! 2. drives it through the [`Solver`]/[`Evaluator`] traits against a stub
//!    evaluator, proving the whole pluggability contract actually *composes and
//!    runs*, not merely compiles.

use tsdyn_solvers::{
    available, duplicates, find, make, register_solver, Caps, Evaluator, ProblemKind, ProblemKinds,
    Solver, SolverKind, SolverState, StepOutcome,
};

/// A stub evaluator for `du/dt = -u` (exponential decay), so a known closed-form
/// solution `u(t) = u0 · e^{-t}` lets us check the kernel actually integrates.
/// It needs no register file, so `n_scratch() == 0` — the JIT's case.
struct DecayEval {
    dim: usize,
}

impl Evaluator for DecayEval {
    fn dim(&self) -> usize {
        self.dim
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
    fn eval(&self, u: &[f64], _p: &[f64], _t: f64, _scratch: &mut [f64], deriv: &mut [f64]) {
        for (d, &ui) in deriv.iter_mut().zip(u) {
            *d = -ui;
        }
    }
    fn eval_jac(
        &self,
        _u: &[f64],
        _p: &[f64],
        _t: f64,
        _scratch: &mut [f64],
        deriv: &mut [f64],
        jac: &mut [f64],
    ) {
        // Not exercised (has_jacobian() is false); filled for completeness so the
        // impl is total. ∂(-u)/∂u = -I.
        for d in deriv.iter_mut() {
            *d = 0.0;
        }
        let n = self.dim;
        for j in jac.iter_mut() {
            *j = 0.0;
        }
        for i in 0..n {
            jac[i * n + i] = -1.0;
        }
    }
}

/// A minimal explicit forward-Euler kernel — the F2 seam demonstrator. The real
/// explicit family ships in stream E3; this lives only in the test, so the
/// production crate registers no kernels of its own and never clashes with E3's
/// names.
struct DummyEuler {
    /// Stage buffer (the RHS value). Owned by the kernel, not in `SolverState`,
    /// mirroring how real kernels keep their `k_i`. Sized lazily on first step.
    k: Vec<f64>,
}

impl DummyEuler {
    fn new() -> Self {
        DummyEuler { k: Vec::new() }
    }
}

impl Solver for DummyEuler {
    fn name(&self) -> &'static str {
        "dummy"
    }

    fn caps(&self) -> Caps {
        Caps::explicit(ProblemKinds::of(ProblemKind::Ode))
    }

    fn step(&mut self, ev: &dyn Evaluator, st: &mut SolverState, h: f64) -> StepOutcome {
        if self.k.len() != st.dim() {
            self.k = vec![0.0; st.dim()];
        }
        ev.eval(&st.u, &st.p, st.t, &mut st.scratch, &mut self.k);
        for (ui, &ki) in st.u.iter_mut().zip(&self.k) {
            *ui += h * ki;
        }
        st.t += h;
        if st.u.iter().all(|x| x.is_finite()) {
            StepOutcome::Accepted { h_next: h }
        } else {
            StepOutcome::Failed
        }
    }
}

register_solver!(
    "dummy",
    Caps::explicit(ProblemKinds::of(ProblemKind::Ode)),
    || Box::new(DummyEuler::new())
);

#[test]
fn dummy_solver_is_discoverable_by_name() {
    let reg = find("dummy").expect("dummy kernel should be registered");
    assert_eq!(reg.name, "dummy");
    assert_eq!(reg.caps.kind, SolverKind::Explicit);
    assert!(reg.caps.supports(ProblemKind::Ode));
    assert!(!reg.caps.supports(ProblemKind::Sde));
    assert!(available().contains(&"dummy"));
    assert!(find("does-not-exist").is_none());
    assert!(make("does-not-exist").is_none());
}

#[test]
fn boxed_solver_is_send_for_ensembles() {
    // The engine's rayon ensemble (E5) moves a `Box<dyn Solver>` into each worker;
    // freeze the `Send` bound here so it can never silently regress.
    fn assert_send<T: Send>() {}
    assert_send::<Box<dyn Solver>>();
    let solver = make("dummy").expect("dummy kernel should build");
    std::thread::spawn(move || {
        // Owning the boxed kernel on another thread must compile (proves Send).
        let _ = solver.name();
    })
    .join()
    .unwrap();
}

#[test]
fn registered_names_are_unique() {
    // inventory cannot reject a clashing registration, so this is the tripwire
    // for a name collision between the parallel solver streams (E3/E4): it sees
    // every kernel linked into the test binary, so a duplicate fails CI here.
    let dups = duplicates();
    assert!(
        dups.is_empty(),
        "duplicate solver names registered: {dups:?}"
    );
}

#[test]
fn discovered_kernel_integrates_against_the_evaluator_seam() {
    // Build the kernel purely from its registry name — the engine's path.
    let mut solver = make("dummy").expect("dummy kernel should build");
    assert_eq!(solver.name(), "dummy");

    let ev = DecayEval { dim: 2 };
    let mut st = SolverState::for_evaluator(&ev, vec![1.0, 2.0], 0.0, vec![]);
    assert_eq!(st.scratch.len(), 0); // DecayEval needs no register file

    // Integrate du/dt = -u from t=0 to t=1 with small fixed steps. `&ev` coerces
    // to `&dyn Evaluator` — the shared-evaluator shape an ensemble worker uses.
    let h = 1e-4;
    let n = 10_000;
    for _ in 0..n {
        match solver.step(&ev, &mut st, h) {
            StepOutcome::Accepted { h_next } => assert_eq!(h_next, h),
            other => panic!("fixed-step kernel should accept, got {other:?}"),
        }
    }
    assert!((st.t - 1.0).abs() < 1e-9);

    // Forward Euler is O(h); the analytic solution is u0·e^{-1}.
    let decay = (-1.0_f64).exp();
    assert!((st.u[0] - decay).abs() < 1e-3, "u0 = {}", st.u[0]);
    assert!((st.u[1] - 2.0 * decay).abs() < 1e-3, "u1 = {}", st.u[1]);
}
