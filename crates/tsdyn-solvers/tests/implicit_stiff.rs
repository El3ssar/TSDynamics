//! Numerical acceptance tests for the implicit/stiff family (stream E4).
//!
//! These drive the registered `rosenbrock` and `trbdf2` kernels through the
//! frozen `Solver`/`Evaluator` seams against the canonical stiff benchmarks —
//! a stiff *linear* system with a closed-form matrix-exponential solution, Van
//! der Pol, Robertson, and the Oregonator — built as real instruction tapes with
//! **analytic Jacobians**, the exact production path (a symbolic tape carrying
//! `∂f/∂u`). The validation strategy, in the absence of an in-crate reference
//! integrator:
//!
//! 1. **Absolute truth** — the stiff linear system has an exact solution; both
//!    kernels must hit it, *and* must do so while taking steps far larger than an
//!    explicit method's stability limit (the point of an L-stable kernel).
//! 2. **Independent cross-validation** — `rosenbrock` (one linear solve per step)
//!    and `trbdf2` (Newton substages) are mechanically different methods; on the
//!    nonlinear benchmarks they must agree, which a shared bug could not fake.
//! 3. **Structural invariants** — Robertson conserves total mass; the kernels
//!    preserve that linear invariant to rounding.
//! 4. **Jacobian integrity** — every hand-built analytic Jacobian tape is checked
//!    against a finite difference of its own RHS, so a slip in the Jacobian can't
//!    silently pass.

use tsdyn_ir::{Evaluator, TapeBuilder};
use tsdyn_solvers::{make, Solver, SolverState, StepOutcome};
use tsdyn_vm::Interpreter;

// ---------------------------------------------------------------------------
// Harness: an Evaluator wrapper for the interpreter and a small integrate loop.
// ---------------------------------------------------------------------------

/// Adapts the interpreter (stream E1) to the `Evaluator` trait — the interpreter
/// exposes the right inherent methods but does not itself `impl Evaluator` (that
/// orphan impl lands in another stream), so consumers bridge it locally.
struct VmEval {
    interp: Interpreter,
}

impl VmEval {
    fn new(interp: Interpreter) -> Self {
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

/// A minimal step/accept/retry/fail driver — the same shape as the engine's
/// `advance_to` (stream E5), reproduced here so these tests depend only on the
/// solver crate and the interpreter, not on the engine (which would pull in
/// Cranelift). Integrates from `t0` to `t1`, returning the final state.
fn integrate_to(
    ev: &dyn Evaluator,
    solver: &mut dyn Solver,
    u0: &[f64],
    p: &[f64],
    t0: f64,
    t1: f64,
    first_step: f64,
) -> Result<Vec<f64>, String> {
    let mut st = SolverState::for_evaluator(ev, u0.to_vec(), t0, p.to_vec());
    let mut h = first_step;
    let mut steps = 0u64;
    while st.t < t1 {
        steps += 1;
        if steps > 5_000_000 {
            return Err(format!("step limit at t = {}", st.t));
        }
        let remaining = t1 - st.t;
        let landing = h >= remaining;
        let h_try = if landing { remaining } else { h };
        match solver.step(ev, &mut st, h_try) {
            StepOutcome::Accepted { h_next } => {
                if !st.u.iter().all(|x| x.is_finite()) {
                    return Err(format!("non-finite state at t = {}", st.t));
                }
                if landing {
                    st.t = t1;
                } else if h_next.is_finite() && h_next > 0.0 {
                    h = h_next;
                }
            }
            StepOutcome::Rejected { h_next } => {
                if !(h_next.is_finite() && h_next > 0.0) {
                    return Err(format!("step collapsed at t = {}", st.t));
                }
                h = h_next;
            }
            StepOutcome::Failed => return Err(format!("solver failed at t = {}", st.t)),
        }
    }
    Ok(st.u)
}

/// Verify a tape's analytic Jacobian against a central finite difference of its
/// own RHS at `(u, p, t)`, so a mistake in a hand-built Jacobian is caught here
/// rather than silently feeding the solver.
fn assert_jac_matches_fd(ev: &VmEval, u: &[f64], p: &[f64], t: f64, tol: f64) {
    let n = ev.dim();
    let mut scratch = vec![0.0; ev.n_scratch()];
    let mut f = vec![0.0; n];
    let mut jac = vec![0.0; n * n];
    ev.eval_jac(u, p, t, &mut scratch, &mut f, &mut jac);
    for j in 0..n {
        let hj = 1e-6 * (1.0 + u[j].abs());
        let mut up = u.to_vec();
        let mut um = u.to_vec();
        up[j] += hj;
        um[j] -= hj;
        let mut fp = vec![0.0; n];
        let mut fm = vec![0.0; n];
        ev.eval(&up, p, t, &mut scratch, &mut fp);
        ev.eval(&um, p, t, &mut scratch, &mut fm);
        for i in 0..n {
            let fd = (fp[i] - fm[i]) / (2.0 * hj);
            let an = jac[i * n + j];
            let scale = 1.0 + an.abs().max(fd.abs());
            assert!(
                (fd - an).abs() <= tol * scale,
                "jac[{i}][{j}]: analytic {an} vs finite-diff {fd}"
            );
        }
    }
}

/// Max-abs difference between two state vectors.
fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

// ---------------------------------------------------------------------------
// Benchmark tapes (each with an analytic Jacobian).
// ---------------------------------------------------------------------------

/// Stiff linear system `u' = A u` with `A = [[a, b], [b, a]]`, eigenvalues
/// `a ± b`. Choosing `a = -500.5, b = 499.5` gives eigenvalues `-1` and `-1000`
/// (stiffness ratio 1000) with eigenvectors `[1,1]` and `[1,-1]`, so for
/// `u0 = [1, 0]` the exact solution is
/// `u(t) = [ (e^{-t}+e^{-1000t})/2, (e^{-t}-e^{-1000t})/2 ]`.
/// Params: `[a, b]`. The Jacobian is the constant matrix `A`.
fn stiff_linear() -> VmEval {
    let mut b = TapeBuilder::new();
    let a = b.param(0);
    let off = b.param(1);
    let x0 = b.state(0);
    let x1 = b.state(1);
    // dx0 = a x0 + b x1 ; dx1 = b x0 + a x1
    let a_x0 = b.mul(a, x0);
    let b_x1 = b.mul(off, x1);
    let dx0 = b.add(a_x0, b_x1);
    let b_x0 = b.mul(off, x0);
    let a_x1 = b.mul(a, x1);
    let dx1 = b.add(b_x0, a_x1);
    // Constant Jacobian [[a, b], [b, a]] — reuse the parameter leaf registers.
    let jac = [a, off, off, a];
    VmEval::new(Interpreter::new(b.finish(&[dx0, dx1], &jac, 2, 2).unwrap()))
}

/// Exact solution of [`stiff_linear`] from `u0 = [1, 0]` at time `t`.
fn stiff_linear_exact(t: f64) -> [f64; 2] {
    let fast = (-1000.0 * t).exp();
    let slow = (-t).exp();
    [0.5 * (slow + fast), 0.5 * (slow - fast)]
}

/// Van der Pol oscillator `x' = y, y' = mu·((1 − x²) y − x)`. Param `[mu]`.
fn van_der_pol() -> VmEval {
    let mut b = TapeBuilder::new();
    let mu = b.param(0);
    let x = b.state(0);
    let y = b.state(1);
    let one = b.constant(1.0);
    let two = b.constant(2.0);
    let zero = b.constant(0.0);
    // dx = y
    let dx = y;
    // dy = mu * ((1 - x^2) y - x)
    let x2 = b.mul(x, x);
    let one_m_x2 = b.sub(one, x2);
    let term = b.mul(one_m_x2, y);
    let term_m_x = b.sub(term, x);
    let dy = b.mul(mu, term_m_x);
    // Jacobian: [[0, 1], [mu(-2xy - 1), mu(1 - x^2)]]
    let xy = b.mul(x, y);
    let two_xy = b.mul(two, xy);
    let neg_two_xy = b.neg(two_xy);
    let j10_inner = b.sub(neg_two_xy, one);
    let j10 = b.mul(mu, j10_inner);
    let j11 = b.mul(mu, one_m_x2);
    let jac = [zero, one, j10, j11];
    VmEval::new(Interpreter::new(b.finish(&[dx, dy], &jac, 2, 1).unwrap()))
}

/// Robertson's stiff chemical kinetics. Param `[k1, k2, k3]` (canonical
/// `0.04, 3e7, 1e4`). Conserves `x0 + x1 + x2` (the RHS sums to zero).
fn robertson() -> VmEval {
    let mut b = TapeBuilder::new();
    let k1 = b.param(0);
    let k2 = b.param(1);
    let k3 = b.param(2);
    let x0 = b.state(0);
    let x1 = b.state(1);
    let x2 = b.state(2);
    let two = b.constant(2.0);
    let zero = b.constant(0.0);
    let x1x2 = b.mul(x1, x2);
    let k3_x1x2 = b.mul(k3, x1x2);
    let x1sq = b.mul(x1, x1);
    let k2_x1sq = b.mul(k2, x1sq);
    let k1_x0 = b.mul(k1, x0);
    // dx0 = -k1 x0 + k3 x1 x2
    let neg_k1x0 = b.neg(k1_x0);
    let dx0 = b.add(neg_k1x0, k3_x1x2);
    // dx1 = k1 x0 - k3 x1 x2 - k2 x1^2
    let t1 = b.sub(k1_x0, k3_x1x2);
    let dx1 = b.sub(t1, k2_x1sq);
    // dx2 = k2 x1^2
    let dx2 = k2_x1sq;
    // Jacobian.
    let neg_k1 = b.neg(k1);
    let k3_x2 = b.mul(k3, x2);
    let k3_x1 = b.mul(k3, x1);
    let two_k2 = b.mul(two, k2);
    let two_k2_x1 = b.mul(two_k2, x1);
    let neg_k3x2 = b.neg(k3_x2);
    let j11 = b.sub(neg_k3x2, two_k2_x1);
    let neg_k3x1 = b.neg(k3_x1);
    let jac = [
        neg_k1, k3_x2, k3_x1, // row 0
        k1, j11, neg_k3x1, // row 1
        zero, two_k2_x1, zero, // row 2
    ];
    VmEval::new(Interpreter::new(
        b.finish(&[dx0, dx1, dx2], &jac, 3, 3).unwrap(),
    ))
}

/// The Oregonator (Field–Noyes) stiff chemical oscillator. Param `[s, q, w]`
/// (canonical `77.27, 8.375e-6, 0.161`).
fn oregonator() -> VmEval {
    let mut b = TapeBuilder::new();
    let s = b.param(0);
    let q = b.param(1);
    let w = b.param(2);
    let x0 = b.state(0);
    let x1 = b.state(1);
    let x2 = b.state(2);
    let one = b.constant(1.0);
    let two = b.constant(2.0);
    let zero = b.constant(0.0);
    // dx0 = s*(x1 + x0*(1 - q x0 - x1))
    let q_x0 = b.mul(q, x0);
    let one_m_qx0 = b.sub(one, q_x0);
    let inner0 = b.sub(one_m_qx0, x1);
    let x0_inner = b.mul(x0, inner0);
    let sum0 = b.add(x1, x0_inner);
    let dx0 = b.mul(s, sum0);
    // dx1 = (1/s)*(x2 - (1 + x0) x1)
    let s_recip = b.recip(s);
    let one_p_x0 = b.add(one, x0);
    let onepx0_x1 = b.mul(one_p_x0, x1);
    let diff1 = b.sub(x2, onepx0_x1);
    let dx1 = b.mul(s_recip, diff1);
    // dx2 = w*(x0 - x2)
    let x0_m_x2 = b.sub(x0, x2);
    let dx2 = b.mul(w, x0_m_x2);
    // Jacobian.
    let two_q = b.mul(two, q);
    let two_q_x0 = b.mul(two_q, x0);
    let j00_inner = b.sub(one, two_q_x0);
    let j00_inner2 = b.sub(j00_inner, x1);
    let j00 = b.mul(s, j00_inner2);
    let one_m_x0 = b.sub(one, x0);
    let j01 = b.mul(s, one_m_x0);
    let neg_x1 = b.neg(x1);
    let j10 = b.mul(s_recip, neg_x1);
    let neg_one_p_x0 = b.neg(one_p_x0);
    let j11 = b.mul(s_recip, neg_one_p_x0);
    let j12 = s_recip;
    let neg_w = b.neg(w);
    let jac = [
        j00, j01, zero, // row 0
        j10, j11, j12, // row 1
        w, zero, neg_w, // row 2
    ];
    VmEval::new(Interpreter::new(
        b.finish(&[dx0, dx1, dx2], &jac, 3, 3).unwrap(),
    ))
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

/// Both kernels register and resolve by name, with implicit + adaptive caps that
/// require the Jacobian.
#[test]
fn kernels_are_registered_with_implicit_caps() {
    for name in ["rosenbrock", "trbdf2"] {
        let solver = make(name).unwrap_or_else(|| panic!("{name} should be registered"));
        assert_eq!(solver.name(), name);
        let caps = solver.caps();
        assert!(caps.needs_jacobian, "{name} should need the Jacobian");
        assert!(caps.adaptive, "{name} should be adaptive");
        use tsdyn_solvers::{ProblemKind, SolverKind};
        assert_eq!(caps.kind, SolverKind::Implicit);
        assert!(caps.supports(ProblemKind::Ode));
    }
}

/// Every benchmark tape's analytic Jacobian matches a finite difference of its
/// RHS — the guard that the hand-built Jacobians are correct.
#[test]
fn analytic_jacobians_match_finite_difference() {
    assert_jac_matches_fd(&stiff_linear(), &[1.0, 0.3], &[-500.5, 499.5], 0.0, 1e-7);
    assert_jac_matches_fd(&van_der_pol(), &[2.0, -0.5], &[5.0], 0.0, 1e-6);
    assert_jac_matches_fd(
        &robertson(),
        &[0.8, 0.01, 0.19],
        &[0.04, 3.0e7, 1.0e4],
        0.0,
        1e-5,
    );
    assert_jac_matches_fd(
        &oregonator(),
        &[1.0, 2.0, 3.0],
        &[77.27, 8.375e-6, 0.161],
        0.0,
        1e-6,
    );
}

/// Both kernels reproduce the exact solution of the stiff linear system — and do
/// it starting from a step (`0.05`) 25× the explicit stability limit (`2/1000`),
/// which an explicit method could not survive. This is the L-stability proof.
#[test]
fn stiff_linear_matches_exact_solution() {
    let ev = stiff_linear();
    let p = [-500.5, 499.5];
    let u0 = [1.0, 0.0];
    let t1 = 0.1;
    let exact = stiff_linear_exact(t1);

    for name in ["rosenbrock", "trbdf2"] {
        let mut solver = make(name).unwrap();
        // First step deliberately far above the explicit stability limit.
        let got = integrate_to(&ev, solver.as_mut(), &u0, &p, 0.0, t1, 0.05)
            .unwrap_or_else(|e| panic!("{name} failed: {e}"));
        let err = max_diff(&got, &exact);
        assert!(
            err < 1e-4,
            "{name}: got {got:?}, exact {exact:?}, err {err:e}"
        );
    }
}

/// Tightening the tolerance reduces the error against the exact solution — the
/// error controller actually controls error (checked on `rosenbrock`).
#[test]
fn tighter_tolerance_reduces_error() {
    use tsdyn_solvers::implicit::RosenbrockW;
    let ev = stiff_linear();
    let p = [-500.5, 499.5];
    let u0 = [1.0, 0.0];
    let t1 = 0.1;
    let exact = stiff_linear_exact(t1);

    let mut loose = RosenbrockW::with_tolerances(1e-4, 1e-7);
    let mut tight = RosenbrockW::with_tolerances(1e-8, 1e-11);
    let e_loose = max_diff(
        &integrate_to(&ev, &mut loose, &u0, &p, 0.0, t1, 0.01).unwrap(),
        &exact,
    );
    let e_tight = max_diff(
        &integrate_to(&ev, &mut tight, &u0, &p, 0.0, t1, 0.01).unwrap(),
        &exact,
    );
    assert!(
        e_tight < e_loose,
        "tighter tol should reduce error: loose {e_loose:e}, tight {e_tight:e}"
    );
    assert!(e_tight < 1e-6, "tight error {e_tight:e} unexpectedly large");
}

/// On Van der Pol the two independent methods agree — a cross-validation a shared
/// bug could not produce. Run at a moderately stiff `mu = 5` over several periods.
#[test]
fn van_der_pol_methods_agree() {
    let ev = van_der_pol();
    let p = [5.0];
    let u0 = [2.0, 0.0];
    let t1 = 10.0;

    let mut ros = make("rosenbrock").unwrap();
    let mut trb = make("trbdf2").unwrap();
    let a = integrate_to(&ev, ros.as_mut(), &u0, &p, 0.0, t1, 1e-3).unwrap();
    let b = integrate_to(&ev, trb.as_mut(), &u0, &p, 0.0, t1, 1e-3).unwrap();
    let d = max_diff(&a, &b);
    assert!(d < 1e-3, "rosenbrock {a:?} vs trbdf2 {b:?}, diff {d:e}");
}

/// Robertson: both kernels integrate the very stiff (`k2 = 3e7`) kinetics,
/// conserve total mass to rounding, and agree with each other.
#[test]
fn robertson_conserves_mass_and_methods_agree() {
    let ev = robertson();
    let p = [0.04, 3.0e7, 1.0e4];
    let u0 = [1.0, 0.0, 0.0];
    let t1 = 4.0;

    let mut ros = make("rosenbrock").unwrap();
    let mut trb = make("trbdf2").unwrap();
    let a = integrate_to(&ev, ros.as_mut(), &u0, &p, 0.0, t1, 1e-6)
        .unwrap_or_else(|e| panic!("rosenbrock failed: {e}"));
    let b = integrate_to(&ev, trb.as_mut(), &u0, &p, 0.0, t1, 1e-6)
        .unwrap_or_else(|e| panic!("trbdf2 failed: {e}"));

    for (name, sol) in [("rosenbrock", &a), ("trbdf2", &b)] {
        let mass: f64 = sol.iter().sum();
        assert!(
            (mass - 1.0).abs() < 1e-7,
            "{name}: mass {mass} not conserved (state {sol:?})"
        );
        assert!(sol.iter().all(|&x| x.is_finite()), "{name}: non-finite");
    }
    // Both methods land on the same point (relative agreement on each component).
    for i in 0..3 {
        let scale = 1e-8 + a[i].abs().max(b[i].abs());
        assert!(
            (a[i] - b[i]).abs() < 1e-4 * scale,
            "component {i}: rosenbrock {} vs trbdf2 {}",
            a[i],
            b[i]
        );
    }
}

/// The Oregonator: both kernels integrate the stiff oscillator over an initial
/// span and agree, staying positive (the chemistry is non-negative).
#[test]
fn oregonator_methods_agree() {
    let ev = oregonator();
    let p = [77.27, 8.375e-6, 0.161];
    let u0 = [1.0, 2.0, 3.0];
    let t1 = 5.0;

    let mut ros = make("rosenbrock").unwrap();
    let mut trb = make("trbdf2").unwrap();
    let a = integrate_to(&ev, ros.as_mut(), &u0, &p, 0.0, t1, 1e-5)
        .unwrap_or_else(|e| panic!("rosenbrock failed: {e}"));
    let b = integrate_to(&ev, trb.as_mut(), &u0, &p, 0.0, t1, 1e-5)
        .unwrap_or_else(|e| panic!("trbdf2 failed: {e}"));

    assert!(
        a.iter().all(|&x| x > 0.0),
        "rosenbrock left positivity: {a:?}"
    );
    assert!(b.iter().all(|&x| x > 0.0), "trbdf2 left positivity: {b:?}");
    for i in 0..3 {
        let scale = 1e-6 + a[i].abs().max(b[i].abs());
        assert!(
            (a[i] - b[i]).abs() < 1e-3 * scale,
            "component {i}: rosenbrock {} vs trbdf2 {}",
            a[i],
            b[i]
        );
    }
}

/// A diverging RHS is reported, never silently returned (the engine contract,
/// exercised through the kernels' `Failed`/collapsed-step path). `x' = x²` blows
/// up at `t = 1`; integrating past it must error.
#[test]
fn divergence_is_reported() {
    // x' = x^2, Jacobian 2x.
    let mut b = TapeBuilder::new();
    let x = b.state(0);
    let two = b.constant(2.0);
    let dx = b.mul(x, x);
    let jac = b.mul(two, x);
    let ev = VmEval::new(Interpreter::new(b.finish(&[dx], &[jac], 1, 0).unwrap()));

    let mut solver = make("rosenbrock").unwrap();
    let res = integrate_to(&ev, solver.as_mut(), &[1.0], &[], 0.0, 2.0, 1e-3);
    assert!(res.is_err(), "integrating past the blow-up should error");
}
