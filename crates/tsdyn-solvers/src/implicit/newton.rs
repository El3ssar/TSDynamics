//! Shared modified-Newton substage solver for the implicit family.
//!
//! Every fully-implicit kernel here (backward Euler, the implicit midpoint and
//! trapezoidal rules, the SDIRK stages, and TR-BDF2) ultimately solves one or
//! more algebraic substages of the *same* shape
//!
//! ```text
//!     y = base + coef·h·f(t, y)
//! ```
//!
//! by a modified-Newton iteration: freeze the analytic Jacobian `J = ∂f/∂u` at
//! the initial guess, factor the iteration matrix `I − coef·h·J` once, and reuse
//! that factorization across iterations (the standard stiff-solver economy). The
//! routine lives here, written once, so every kernel that needs a Newton stage
//! shares one audited implementation rather than copying the loop.
//!
//! The base method's *defining* coefficients (`coef`, the stage time `t`, the
//! `base` vector) are the caller's; this module owns only the generic algebraic
//! solve and its convergence policy.
//!
//! # Factorization reuse across substages and steps
//!
//! A *singly*-diagonal kernel (SDIRK2, TR-BDF2) has every implicit substage use
//! the **same** shift `coef·h` — the SDIRK stages share one `γ`, and TR-BDF2's
//! trapezoidal and BDF2 substages are tuned so their diagonal coefficients are
//! equal (`w = γ/2 = d`). The iteration matrix `I − coef·h·J` is therefore
//! *identical* across the substages of one step, and (while `h` is held) across
//! consecutive steps. [`solve_substage_reuse`] exploits this: it keeps the frozen
//! `J` and its LU factorization in the kernel's [`NewtonWork`] and *reuses* them
//! whenever the shift matches, skipping the (expensive) Jacobian evaluation,
//! matrix assembly and LU factorization. This is still an ordinary modified
//! Newton — the iteration matrix is held fixed during the solve and only the
//! residual is re-evaluated — so it converges to the *same* root to
//! [`NEWTON_TOL`]; the accepted stage value is preserved to the solver's
//! iteration tolerance.
//!
//! Robustness is unchanged by a **refactor-on-degradation** trigger: if a reused
//! factorization fails to converge in the budget, the routine re-forms `J` at the
//! current iterate, re-factorizes at the *current* shift, and retries from where
//! it stalled. A genuine non-convergence (a too-large `h`) still returns
//! [`BaseOutcome::Recoverable`], so a stale Jacobian can never silently cause a
//! failed step — it costs at most one extra fresh solve before the controller
//! shrinks `h`. The reuse decision keys on the *exact* (bit-equal) shift, so the
//! big step and the two half steps of the step-doubling controller (sizes `h` and
//! `h/2`) never share a factorization across the size boundary.

use super::control::BaseOutcome;
use tsdyn_ir::Evaluator;

/// Maximum modified-Newton iterations per substage before the step is declared
/// non-convergent (recoverable — the controller retries with a smaller `h`, which
/// improves both the Newton conditioning and the initial guess).
pub(crate) const NEWTON_MAX_ITERS: usize = 20;

/// Newton convergence threshold on the weighted-RMS norm of the increment.
///
/// This is a **fixed** `1e-3`, deliberately *not* derived from `rtol` the way the
/// BDF corrector's `newton_tol` is ([`super::bdf`]). The norm it gates,
/// [`wrms`], is already tolerance-weighted — each increment component is divided by
/// `atol + rtol·|y_i|` before the RMS — so the test "`wrms(Δ) ≤ 1e-3`" means the
/// Newton increment has shrunk to a thousandth of the per-component integration
/// tolerance, *independently* of the absolute size of `rtol`. The residual Newton
/// error is therefore held ~3 orders of magnitude below the truncation error the
/// step-doubling estimate measures across the whole tolerance range, which is the
/// property a derived `newton_tol` buys on the BDF side (where the norm is scaled
/// by `atol + rtol·|y|` but the threshold is an *absolute* `f64`, so the threshold
/// must move with `rtol`). A fixed value here is both adequate and lower-risk.
pub(crate) const NEWTON_TOL: f64 = 1e-3;

/// Weighted-RMS norm `sqrt(mean( (v_i / (atol + rtol·|ref_i|))² ))` — the scaled
/// size of an increment relative to the solution magnitude.
pub(crate) fn wrms(v: &[f64], reference: &[f64], rtol: f64, atol: f64) -> f64 {
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    let mut acc = 0.0;
    for i in 0..n {
        let sc = atol + rtol * reference[i].abs();
        let e = v[i] / sc;
        acc += e * e;
    }
    (acc / n as f64).sqrt()
}

/// Reusable per-step scratch for a Newton substage: the residual / RHS buffer, the
/// frozen Jacobian, the iteration matrix, its pivots, and the increment. Grown to
/// the system dimension on first use and reused with no per-step allocation.
///
/// When driven through [`solve_substage_reuse`], `mat`/`piv` additionally cache a
/// *factored* iteration matrix across substages and steps; [`cached_shift`]
/// records the `coef·h` that factorization was built for (`None` ⇒ no valid
/// cache), the freshness key the reuse path matches against.
///
/// [`cached_shift`]: NewtonWork::cached_shift
#[derive(Default)]
pub(crate) struct NewtonWork {
    /// RHS / residual buffer (`f(t, y)` then the Newton residual), length `dim`.
    pub fy: Vec<f64>,
    /// Frozen analytic Jacobian `∂f/∂u`, row-major `dim × dim`.
    pub jac: Vec<f64>,
    /// Iteration matrix `I − coef·h·J`, row-major `dim × dim`.
    pub mat: Vec<f64>,
    /// LU pivots for `mat`, length `dim`.
    pub piv: Vec<usize>,
    /// Newton increment `Δ`, length `dim`.
    pub delta: Vec<f64>,
    /// The shift `coef·h` the cached factorization in `mat`/`piv` was built for,
    /// or `None` when there is no reusable factorization (the reuse path then
    /// always forms fresh). Set only by [`solve_substage_reuse`].
    pub cached_shift: Option<f64>,
}

impl NewtonWork {
    pub(crate) fn ensure(&mut self, n: usize) {
        if self.fy.len() != n {
            self.fy = vec![0.0; n];
            self.jac = vec![0.0; n * n];
            self.mat = vec![0.0; n * n];
            self.piv = vec![0; n];
            self.delta = vec![0.0; n];
            // A re-sized buffer set has no factorization to reuse.
            self.cached_shift = None;
        }
    }
}

/// Solve one implicit substage `y = base + coef·h·f(t, y)` by modified Newton.
///
/// `y` carries the initial guess in and the solution out. The iteration matrix
/// `I − coef·h·J` is factored once at the guess and reused across iterations.
/// All buffers are caller-owned (disjoint slices), so this is a free function to
/// keep the borrows simple.
///
/// Returns [`BaseOutcome::Ok`] on convergence, [`BaseOutcome::Recoverable`] when
/// the Jacobian/RHS is non-finite, the iteration matrix is singular, or Newton
/// does not converge in the budget (all "shrink `h` and retry" cases).
#[allow(clippy::too_many_arguments)]
pub(crate) fn newton_substage(
    ev: &dyn Evaluator,
    t: f64,
    p: &[f64],
    coef: f64,
    h: f64,
    base: &[f64],
    y: &mut [f64],
    fy: &mut [f64],
    jac: &mut [f64],
    mat: &mut [f64],
    piv: &mut [usize],
    delta: &mut [f64],
    scratch: &mut [f64],
    rtol: f64,
    atol: f64,
) -> BaseOutcome {
    let n = y.len();

    // Freeze J at the initial guess and factor (I − coef·h·J) once.
    match form_factorization(ev, t, p, coef, h, y, fy, jac, mat, piv, scratch, n) {
        BaseOutcome::Ok => {}
        other => return other,
    }
    newton_iterate(
        ev, t, p, coef, h, base, y, fy, mat, piv, delta, scratch, n, rtol, atol,
    )
}

/// Freeze the analytic Jacobian at `y` and factor the iteration matrix
/// `I − coef·h·J` into `mat`/`piv` (the once-per-Newton-solve setup, shared by the
/// re-form and reuse-refresh paths). `fy` receives `f(t, y)` as a side effect.
///
/// Returns [`BaseOutcome::Ok`] on a finite, non-singular factorization, else
/// [`BaseOutcome::Recoverable`] (a non-finite Jacobian or a singular shift — both
/// "shrink `h` and retry").
#[allow(clippy::too_many_arguments)]
fn form_factorization(
    ev: &dyn Evaluator,
    t: f64,
    p: &[f64],
    coef: f64,
    h: f64,
    y: &[f64],
    fy: &mut [f64],
    jac: &mut [f64],
    mat: &mut [f64],
    piv: &mut [usize],
    scratch: &mut [f64],
    n: usize,
) -> BaseOutcome {
    use super::linalg::{build_shifted, lu_factor};
    ev.eval_jac(y, p, t, scratch, fy, jac);
    if !jac.iter().all(|x| x.is_finite()) {
        return BaseOutcome::Recoverable;
    }
    build_shifted(mat, jac, n, coef * h);
    if !lu_factor(mat, n, piv) {
        return BaseOutcome::Recoverable; // singular ⇒ shrink h and retry
    }
    BaseOutcome::Ok
}

/// Run the modified-Newton loop against an *already factored* iteration matrix
/// `mat`/`piv` (the standard stiff economy: the matrix is held fixed and only the
/// residual is re-evaluated each iteration). `y` carries the current iterate in
/// and the converged value out.
///
/// Each iteration forms the residual `R(y) = y − base − coef·h·f(t, y)`, solves
/// `(I − coef·h·J) Δ = R` against the frozen factorization, applies `y ← y − Δ`,
/// and accepts when the weighted-RMS increment falls under [`NEWTON_TOL`].
///
/// # Divergence safeguard (contraction-rate early-out)
///
/// A flat increment-norm budget would burn all [`NEWTON_MAX_ITERS`] RHS
/// evaluations on a step where the (frozen) Newton iteration is plainly
/// *diverging* before returning [`BaseOutcome::Recoverable`]. This routine instead
/// mirrors the BDF corrector ([`super::bdf`]): it estimates the linear
/// contraction rate `rate = ‖Δ‖ / ‖Δ_prev‖` and abandons the solve early when
///
/// - `rate ≥ 1` — the increment is not shrinking, so the (modified) Newton is
///   diverging and more iterations cannot recover it; or
/// - the projected remaining error `rate^m/(1 − rate)·‖Δ‖` (with `m` the
///   iterations still in the budget) cannot reach [`NEWTON_TOL`] — convergence is
///   too slow to land within the budget.
///
/// The rate test fires **only after** the increment has been applied and only when
/// a previous increment exists, and it is a pure *early return of the same
/// [`BaseOutcome::Recoverable`]* the budget exhaustion would have produced. On a
/// **converging** step the increment shrinks monotonically (`rate < 1`) and the
/// convergence test trips first, so the accepted iterate `y` is **bit-identical**
/// to the pre-safeguard behaviour — the controller and step result are unchanged.
/// The only difference is that a diverging step is abandoned after one extra
/// iteration instead of the full budget, after which the caller refactors-and-
/// retries or shrinks `h` exactly as before.
///
/// Returns [`BaseOutcome::Ok`] when the increment falls under [`NEWTON_TOL`],
/// [`BaseOutcome::Recoverable`] on a non-finite residual / iterate, on a detected
/// divergence, or on exhausting the iteration budget (the caller may then refactor
/// and retry, or shrink `h`).
#[allow(clippy::too_many_arguments)]
fn newton_iterate(
    ev: &dyn Evaluator,
    t: f64,
    p: &[f64],
    coef: f64,
    h: f64,
    base: &[f64],
    y: &mut [f64],
    fy: &mut [f64],
    mat: &[f64],
    piv: &[usize],
    delta: &mut [f64],
    scratch: &mut [f64],
    n: usize,
    rtol: f64,
    atol: f64,
) -> BaseOutcome {
    use super::linalg::lu_solve;
    let mut dnorm_old: Option<f64> = None;
    for k in 0..NEWTON_MAX_ITERS {
        ev.eval(y, p, t, scratch, fy);
        if !fy.iter().all(|x| x.is_finite()) {
            return BaseOutcome::Recoverable;
        }
        // Residual R(y) = y − base − coef·h·f(t,y); Newton step (I − coef·h·J) Δ = R.
        for i in 0..n {
            delta[i] = y[i] - base[i] - coef * h * fy[i];
        }
        lu_solve(mat, n, piv, delta);
        for i in 0..n {
            y[i] -= delta[i];
        }
        if !y.iter().all(|x| x.is_finite()) {
            return BaseOutcome::Recoverable;
        }
        let dnorm = wrms(delta, y, rtol, atol);
        // Accept first: a converging step trips here before the divergence guard
        // below can fire, so its accepted iterate is bit-identical to the flat-budget
        // behaviour.
        if dnorm <= NEWTON_TOL {
            return BaseOutcome::Ok;
        }
        // Contraction-rate early-out (mirrors the BDF corrector): abandon a clearly
        // diverging / too-slowly-converging solve rather than burning the full budget.
        // `rate < 1` on every iteration of a converging step, so this never aborts one.
        if let Some(old) = dnorm_old {
            if old > 0.0 {
                let rate = dnorm / old;
                let remaining = (NEWTON_MAX_ITERS - 1 - k) as i32;
                if rate >= 1.0 || rate.powi(remaining) / (1.0 - rate) * dnorm > NEWTON_TOL {
                    return BaseOutcome::Recoverable;
                }
            }
        }
        dnorm_old = Some(dnorm);
    }
    BaseOutcome::Recoverable // did not converge in the iteration budget
}

/// A convenience wrapper that solves a substage using a [`NewtonWork`] buffer set
/// and the evaluator's own scratch from a [`SolverState`].
///
/// This is the **always-re-form** path: it freezes a fresh Jacobian and factors
/// the iteration matrix on every call, never consulting the [`NewtonWork`] cache.
/// The order-1/2 single-stage kernels (backward Euler, implicit midpoint) call it
/// — they have nothing to reuse a factorization across — and its numerics are
/// byte-for-byte the pre-reuse behaviour. The singly-diagonal kernels (SDIRK2,
/// TR-BDF2) call [`solve_substage_reuse`] instead.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_substage(
    ev: &dyn Evaluator,
    t: f64,
    p: &[f64],
    coef: f64,
    h: f64,
    base: &[f64],
    y: &mut [f64],
    work: &mut NewtonWork,
    scratch: &mut [f64],
    rtol: f64,
    atol: f64,
) -> BaseOutcome {
    let n = y.len();
    work.ensure(n);
    // A fresh factorization invalidates any cached one: this path leaves `mat`/`piv`
    // holding a factorization at the current shift, but it is not tracked for reuse.
    work.cached_shift = None;
    let NewtonWork {
        fy,
        jac,
        mat,
        piv,
        delta,
        cached_shift: _,
    } = work;
    newton_substage(
        ev, t, p, coef, h, base, y, fy, jac, mat, piv, delta, scratch, rtol, atol,
    )
}

/// Solve an implicit substage `y = base + coef·h·f(t, y)` **reusing** the cached
/// Jacobian factorization in `work` when its shift `coef·h` matches, and re-forming
/// only when it does not (or when a reused factor fails to converge).
///
/// The reuse contract (see the [module docs](self)):
///
/// 1. If `work.cached_shift == Some(coef·h)` (a bit-equal match), run modified
///    Newton against the cached `mat`/`piv` directly — no Jacobian evaluation, no
///    LU. This is the common case for the second substage of a singly-diagonal
///    step and for the first substage of a step that kept the same `h`.
/// 2. If that reused solve does **not** converge, *refresh*: re-freeze `J` at the
///    current iterate, re-factorize at the same shift, and retry the Newton loop.
///    A reused factorization can therefore never cause a spurious step failure —
///    it falls back to exactly the fresh solve [`solve_substage`] would have done.
/// 3. On a shift mismatch (or no cache), form fresh from the start.
///
/// On success the (possibly fresh) factorization is left in `work` and
/// `work.cached_shift` is set to `coef·h`, so the *next* substage with the same
/// shift reuses it. On any non-`Ok` outcome the cache is invalidated.
///
/// Correctness: the iteration matrix is held fixed during each Newton solve, so
/// this is an ordinary modified Newton converging (when it converges) to the same
/// root, to the same [`NEWTON_TOL`]; the accepted stage value is preserved to the
/// solver's iteration tolerance regardless of which (equally valid) frozen `J` was
/// used.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_substage_reuse(
    ev: &dyn Evaluator,
    t: f64,
    p: &[f64],
    coef: f64,
    h: f64,
    base: &[f64],
    y: &mut [f64],
    work: &mut NewtonWork,
    scratch: &mut [f64],
    rtol: f64,
    atol: f64,
) -> BaseOutcome {
    let n = y.len();
    work.ensure(n);
    let shift = coef * h;
    let NewtonWork {
        fy,
        jac,
        mat,
        piv,
        delta,
        cached_shift,
    } = work;

    // 1. Reuse a matching cached factorization (the fast path): run modified Newton
    // against the cached LU with no Jacobian evaluation / re-factorization.
    if *cached_shift == Some(shift) {
        if let BaseOutcome::Ok = newton_iterate(
            ev, t, p, coef, h, base, y, fy, mat, piv, delta, scratch, n, rtol, atol,
        ) {
            return BaseOutcome::Ok;
        }
        // 2. The reused (quasi-)Newton stalled or went non-finite. Fall through to
        // re-form J at the current iterate and retry — the refactor-on-degradation
        // trigger that keeps robustness identical to the always-re-form path. `y`
        // already holds the latest iterate, so the re-form is a better-conditioned
        // restart, not a fresh start.
    }

    // 3. Form fresh (cache miss, or the reuse retry above): freeze J at the current
    // `y`, factor at this shift, and run the Newton loop.
    *cached_shift = None;
    match form_factorization(ev, t, p, coef, h, y, fy, jac, mat, piv, scratch, n) {
        BaseOutcome::Ok => {}
        other => return other,
    }
    match newton_iterate(
        ev, t, p, coef, h, base, y, fy, mat, piv, delta, scratch, n, rtol, atol,
    ) {
        BaseOutcome::Ok => {
            // The factorization in `mat`/`piv` is valid at `shift`; offer it for
            // reuse by the next substage / step with the same shift.
            *cached_shift = Some(shift);
            BaseOutcome::Ok
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tsdyn_ir::Evaluator;

    /// A scalar mock evaluator `f(y) = k·y²` with the *true* analytic Jacobian
    /// `f'(y) = 2·k·y`, counting the RHS evaluations so a test can prove the
    /// contraction-rate early-out abandons a diverging substage before the full
    /// [`NEWTON_MAX_ITERS`] budget is spent. `dim = 1`, no parameters, no scratch.
    ///
    /// The counter is an [`AtomicUsize`] because [`Evaluator`] requires `Sync`.
    struct Quadratic {
        k: f64,
        rhs_evals: AtomicUsize,
    }

    impl Quadratic {
        fn new(k: f64) -> Self {
            Quadratic {
                k,
                rhs_evals: AtomicUsize::new(0),
            }
        }
        fn evals(&self) -> usize {
            self.rhs_evals.load(Ordering::Relaxed)
        }
    }

    impl Evaluator for Quadratic {
        fn dim(&self) -> usize {
            1
        }
        fn n_param(&self) -> usize {
            0
        }
        fn n_scratch(&self) -> usize {
            0
        }
        fn has_jacobian(&self) -> bool {
            true
        }
        fn eval(&self, u: &[f64], _p: &[f64], _t: f64, _scratch: &mut [f64], deriv: &mut [f64]) {
            self.rhs_evals.fetch_add(1, Ordering::Relaxed);
            deriv[0] = self.k * u[0] * u[0];
        }
        fn eval_jac(
            &self,
            u: &[f64],
            _p: &[f64],
            _t: f64,
            _scratch: &mut [f64],
            deriv: &mut [f64],
            jac: &mut [f64],
        ) {
            deriv[0] = self.k * u[0] * u[0];
            jac[0] = 2.0 * self.k * u[0];
        }
    }

    /// A linear mock `f(y) = a·y + b` with the exact Jacobian `a`. Modified Newton
    /// against the (exact, constant) frozen Jacobian solves a linear substage in a
    /// single iteration, so it is the clean "converging" case for the safeguard.
    struct Affine {
        a: f64,
        b: f64,
    }

    impl Evaluator for Affine {
        fn dim(&self) -> usize {
            1
        }
        fn n_param(&self) -> usize {
            0
        }
        fn n_scratch(&self) -> usize {
            0
        }
        fn has_jacobian(&self) -> bool {
            true
        }
        fn eval(&self, u: &[f64], _p: &[f64], _t: f64, _scratch: &mut [f64], deriv: &mut [f64]) {
            deriv[0] = self.a * u[0] + self.b;
        }
        fn eval_jac(
            &self,
            u: &[f64],
            _p: &[f64],
            _t: f64,
            _scratch: &mut [f64],
            deriv: &mut [f64],
            jac: &mut [f64],
        ) {
            deriv[0] = self.a * u[0] + self.b;
            jac[0] = self.a;
        }
    }

    /// A converging linear substage must reach its root *exactly* and report
    /// [`BaseOutcome::Ok`] — the divergence safeguard must never fire on it (the
    /// increment shrinks to zero, so the contraction rate stays well below 1). This
    /// pins the answer-preserving contract: the safeguard only short-circuits
    /// diverging steps, never a converging one.
    #[test]
    fn converging_substage_solves_exactly_and_accepts() {
        // Substage y = base + coef·h·f(y), f(y) = a·y + b.
        // Root: y* = (base + coef·h·b) / (1 − coef·h·a).
        let (a, b) = (-2.0, 1.0); // dissipative — a well-conditioned stiff-like substage
        let ev = Affine { a, b };
        let (coef, h, base) = (1.0, 0.1, [0.5_f64]);
        let mut y = [2.5_f64]; // a guess away from the root, so an iteration is exercised
        let mut work = NewtonWork::default();
        let mut scratch: Vec<f64> = Vec::new();

        let out = solve_substage(
            &ev,
            0.0,
            &[],
            coef,
            h,
            &base,
            &mut y,
            &mut work,
            &mut scratch,
            1e-6,
            1e-9,
        );
        assert_eq!(out, BaseOutcome::Ok);

        let root = (base[0] + coef * h * b) / (1.0 - coef * h * a);
        assert!(
            (y[0] - root).abs() < 1e-12,
            "converged y {} should equal the analytic root {root}",
            y[0]
        );
    }

    /// A genuinely diverging substage must return [`BaseOutcome::Recoverable`]
    /// **and** the contraction-rate early-out must abandon it well before the full
    /// [`NEWTON_MAX_ITERS`] RHS evaluations are spent (the perf fix). With the frozen
    /// Jacobian near zero (guess ≈ 0) the iteration matrix is ≈ 1, so the strongly
    /// expanding quadratic map `y ← base + coef·h·k·y²` blows up and the increment
    /// grows (`rate ≥ 1`), tripping the early return.
    #[test]
    fn diverging_substage_bails_before_full_budget() {
        let ev = Quadratic::new(50.0);
        // Large coef·h and a non-trivial base drive the quadratic iteration to blow
        // up; the guess ≈ 0 makes the frozen Jacobian ≈ 0 (iteration matrix ≈ 1).
        let (coef, h, base) = (1.0, 1.0, [3.0_f64]);
        let mut y = [1e-8_f64];
        let mut work = NewtonWork::default();
        let mut scratch: Vec<f64> = Vec::new();

        let out = solve_substage(
            &ev,
            0.0,
            &[],
            coef,
            h,
            &base,
            &mut y,
            &mut work,
            &mut scratch,
            1e-6,
            1e-9,
        );
        assert_eq!(
            out,
            BaseOutcome::Recoverable,
            "a diverging substage must be reported recoverable"
        );
        let evals = ev.evals();
        // The early-out catches the divergence in a handful of iterations; without it
        // the flat budget would burn all NEWTON_MAX_ITERS (20) RHS evaluations. The
        // generous ceiling keeps the test robust to the exact iteration the rate
        // guard trips while still proving it short-circuits well short of the budget.
        assert!(
            evals <= 5,
            "early-out should bail in ≲5 iterations (vs the full {NEWTON_MAX_ITERS} \
             budget), but ran {evals} RHS evals"
        );
    }

    /// The contraction-rate guard is a *divergence-only* short-circuit: a slowly but
    /// monotonically converging substage still lands on [`BaseOutcome::Ok`]. Here a
    /// mildly dissipative linear substage converges in one iteration (exact J), and
    /// the eval-count is the single iteration — confirming no premature abandonment.
    #[test]
    fn slowly_converging_substage_is_not_abandoned() {
        let ev = Affine { a: -0.5, b: 0.3 };
        let (coef, h, base) = (1.0, 0.5, [0.2_f64]);
        let mut y = [base[0]];
        let mut work = NewtonWork::default();
        let mut scratch: Vec<f64> = Vec::new();

        let out = solve_substage(
            &ev,
            0.0,
            &[],
            coef,
            h,
            &base,
            &mut y,
            &mut work,
            &mut scratch,
            1e-6,
            1e-9,
        );
        assert_eq!(out, BaseOutcome::Ok);
        let root = (base[0] + coef * h * 0.3) / (1.0 - coef * h * (-0.5));
        assert!((y[0] - root).abs() < 1e-12);
    }

    /// [`wrms`] is the tolerance-weighted RMS the convergence test gates on: a zero
    /// increment is zero, and scaling every component by its tolerance gives a unit
    /// norm (the sanity that makes the fixed [`NEWTON_TOL`] meaningful across rtol).
    #[test]
    fn wrms_is_tolerance_weighted() {
        assert_eq!(wrms(&[0.0, 0.0], &[1.0, 1.0], 1e-3, 1e-6), 0.0);
        // Each component equal to its own scale ⇒ every term is 1 ⇒ RMS is 1.
        let rtol = 1e-3;
        let atol = 1e-6;
        let reference: [f64; 2] = [10.0, 100.0];
        let v = [
            atol + rtol * reference[0].abs(),
            atol + rtol * reference[1].abs(),
        ];
        assert!((wrms(&v, &reference, rtol, atol) - 1.0).abs() < 1e-12);
    }
}
