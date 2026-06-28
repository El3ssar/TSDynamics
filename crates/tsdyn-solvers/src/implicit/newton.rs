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

/// Newton convergence threshold on the weighted-RMS norm of the increment. Tight
/// relative to the integration tolerance so the residual Newton error stays well
/// below the truncation error the step-doubling estimate measures.
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
/// Returns [`BaseOutcome::Ok`] when the increment falls under [`NEWTON_TOL`],
/// [`BaseOutcome::Recoverable`] on a non-finite residual / iterate or on
/// exhausting the iteration budget (the caller may then refactor and retry, or
/// shrink `h`).
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
    for _ in 0..NEWTON_MAX_ITERS {
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
        if wrms(delta, y, rtol, atol) <= NEWTON_TOL {
            return BaseOutcome::Ok;
        }
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
