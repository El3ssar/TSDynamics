//! `bdf` — a variable-order (1–5), variable-step Backward Differentiation
//! Formula: the engine's high-order stiff workhorse.
//!
//! The Rosenbrock-W and TR-BDF2 kernels ([`super::rosenbrock`] /
//! [`super::trbdf2`]) are *fixed* order (1 and 2). On a stiff transient they are
//! accuracy-bound: holding the local error at the requested tolerance forces many
//! small steps even where the solution is smooth. A BDF adapts its **order** as
//! well as its step, so it can run at order 5 through a smooth stiff phase and
//! take far fewer, larger steps — the property a variable-order BDF (e.g. the
//! classic LSODA/`ode15s` family) wins stiff problems with.
//!
//! # The method
//!
//! This is the **fixed-leading-coefficient** (FLC) quasi-constant-step BDF: the
//! solution history is held as a *difference array* `D` (the modified divided
//! differences, scaled by the step size), the corrector solves the implicit BDF
//! equation by a modified Newton iteration reusing the analytic Jacobian, and the
//! local error is estimated from the corrector increment with the FLC error
//! constants. Order and step are chosen after each accepted step by comparing the
//! error a one-lower, equal, and one-higher order step would have produced. A
//! step-size change rescales `D` through the `R`-matrix recurrence rather than
//! recomputing the history, which is what makes variable step cheap.
//!
//! The formulation, coefficients (`kappa`, `gamma`, `alpha`, the error constants)
//! and the `R`-matrix rescaling follow Shampine & Reichelt's fixed-leading-
//! coefficient treatment; the underlying method is Gear's BDF.
//!
//! References:
//! - C. W. Gear, *Numerical Initial Value Problems in Ordinary Differential
//!   Equations*, Prentice-Hall, 1971 (the BDF family).
//! - L. F. Shampine & M. W. Reichelt, "The MATLAB ODE Suite", SIAM J. Sci.
//!   Comput. 18(1), 1997 (the fixed-leading-coefficient variable-order formulation
//!   and `R`-matrix step-change recurrence this kernel ports).
//! - E. Hairer & G. Wanner, *Solving Ordinary Differential Equations II: Stiff
//!   and Differential-Algebraic Problems*, 2nd ed., Springer, §V (multistep BDF,
//!   order/step control).
//!
//! # Fitting the one-step `Solver` seam
//!
//! A multistep method carries history *across* steps, whereas the frozen
//! [`Solver`] trait exposes a single `step(ev, st, h)`. The kernel therefore owns
//! its difference array and order in `&mut self` (one kernel instance per
//! integrating worker, exactly as the engine builds them), and each `step` call is
//! one *attempt* at the engine-chosen `h`:
//!
//! - The engine drives the accept/reject loop. On a rejected attempt the kernel
//!   leaves `st.u`/`st.t` untouched and returns a smaller suggested `h` — and,
//!   crucially, **does not commit the trial to `D`**, so the history always
//!   reflects the last *accepted* state.
//! - The engine may hand a smaller `h` than the kernel suggested (it caps the step
//!   to land exactly on each output time). The kernel detects any change from the
//!   `h` its `D` is currently scaled to and rescales `D` accordingly, so output
//!   gridding never corrupts the multistep history. Under a fine output grid this
//!   means the kernel effectively runs at the grid step in smooth phases —
//!   constant step, where the iteration matrix `I − c·J` is reused across many
//!   steps with neither a Jacobian evaluation nor a refactorization (the central
//!   per-step economy that makes BDF cheap).
//!
//! Like the other implicit kernels it **refuses a Jacobian-less evaluator** rather
//! than silently degrading to an unstable explicit step.

use super::linalg::{build_shifted, lu_factor, lu_solve};
use crate::caps::{Caps, ProblemKind, ProblemKinds};
use crate::register_solver;
use crate::solver::{Solver, SolverState, StepOutcome};
use tsdyn_ir::Evaluator;

/// Maximum BDF order. Orders above 6 are not zero-stable; 5 is the standard
/// production ceiling (order 6 is only conditionally stable), matching the
/// variable-order stiff integrators this kernel mirrors.
const MAX_ORDER: usize = 5;
/// Modified-Newton iteration budget per attempt. BDF correctors converge in 1–2
/// iterations from the predictor when the iteration matrix is fresh; exceeding
/// this signals the step (or a stale Jacobian) is the problem, handled by the
/// refresh-then-shrink logic rather than more iterations.
const NEWTON_MAXITER: usize = 4;
/// Smallest step-shrink factor on a rejected step (Hairer–Wanner; also the floor
/// the other implicit kernels use).
const MIN_FACTOR: f64 = 0.2;
/// Largest step-growth factor on an accepted step. BDF tolerates aggressive
/// growth because the order machinery guards accuracy; the classic value is 10.
const MAX_FACTOR: f64 = 10.0;

/// Weighted-RMS norm `sqrt(mean( (v_i / scale_i)² ))` — the scaled size of a
/// vector relative to per-component tolerances. The single norm the corrector and
/// error control measure in (matching the other implicit kernels' `wrms`).
fn wrms(v: &[f64], scale: &[f64]) -> f64 {
    let n = v.len();
    let mut acc = 0.0;
    for i in 0..n {
        let e = v[i] / scale[i];
        acc += e * e;
    }
    (acc / n as f64).sqrt()
}

/// The FLC coefficient tables, indexed by order `0..=MAX_ORDER`.
///
/// `gamma[k] = Σ_{i=1}^{k} 1/i` (with `gamma[0] = 0`); `alpha[k] = (1 −
/// kappa[k])·gamma[k]` sets the corrector's leading coefficient (`c = h/alpha`);
/// `error_const[k] = kappa[k]·gamma[k] + 1/(k+1)` scales the corrector increment
/// into a local-error estimate. The `kappa` test coefficients are the standard
/// fixed-leading-coefficient values.
struct Coefficients {
    gamma: [f64; MAX_ORDER + 1],
    alpha: [f64; MAX_ORDER + 1],
    error_const: [f64; MAX_ORDER + 1],
}

impl Coefficients {
    fn new() -> Self {
        // Fixed-leading-coefficient test coefficients (Shampine & Reichelt 1997).
        let kappa: [f64; MAX_ORDER + 1] = [0.0, -0.1850, -1.0 / 9.0, -0.0823, -0.0415, 0.0];
        let mut gamma = [0.0; MAX_ORDER + 1];
        let mut alpha = [0.0; MAX_ORDER + 1];
        let mut error_const = [0.0; MAX_ORDER + 1];
        for k in 1..=MAX_ORDER {
            gamma[k] = gamma[k - 1] + 1.0 / k as f64;
        }
        for k in 0..=MAX_ORDER {
            alpha[k] = (1.0 - kappa[k]) * gamma[k];
            error_const[k] = kappa[k] * gamma[k] + 1.0 / (k as f64 + 1.0);
        }
        Coefficients {
            gamma,
            alpha,
            error_const,
        }
    }
}

/// The corrector's modified-Newton iteration (the FLC BDF system solve).
///
/// Solves `y − y_predict − c·f(t_new, y) + psi = 0` for `y`, starting from the
/// predictor, using the pre-factored iteration matrix `(I − c·J)` in `lu`/`piv`.
/// `y` carries the predictor in and the corrected state out; `d` accumulates the
/// total increment `y − y_predict` (the quantity the error estimate and `D`-update
/// consume). Returns `(converged, n_iter)`.
///
/// The convergence test is the standard contraction-rate estimate: the iteration
/// is declared converged when the predicted remaining error `rate/(1−rate)·‖Δy‖`
/// falls below the Newton tolerance, and abandoned early if the rate is ≥ 1 or the
/// projected error cannot reach tolerance within the remaining budget — so a
/// non-converging step is caught in one or two iterations, not `NEWTON_MAXITER`.
#[allow(clippy::too_many_arguments)]
fn solve_bdf_system(
    ev: &dyn Evaluator,
    t_new: f64,
    p: &[f64],
    c: f64,
    y_predict: &[f64],
    psi: &[f64],
    scale: &[f64],
    lu: &[f64],
    piv: &[usize],
    y: &mut [f64],
    d: &mut [f64],
    f: &mut [f64],
    dy: &mut [f64],
    scratch: &mut [f64],
    newton_tol: f64,
) -> (bool, usize) {
    let n = y.len();
    y.copy_from_slice(y_predict);
    d.iter_mut().for_each(|x| *x = 0.0);
    let mut dy_norm_old: Option<f64> = None;
    let mut converged = false;
    let mut iters = 0;
    for k in 0..NEWTON_MAXITER {
        iters = k + 1;
        ev.eval(y, p, t_new, scratch, f);
        if !f.iter().all(|x| x.is_finite()) {
            break;
        }
        // Newton residual RHS: (I − c·J) Δy = c·f − psi − d.
        for i in 0..n {
            dy[i] = c * f[i] - psi[i] - d[i];
        }
        lu_solve(lu, n, piv, dy);
        if !dy.iter().all(|x| x.is_finite()) {
            break;
        }
        let dy_norm = wrms(dy, scale);
        let rate = dy_norm_old.map(|old| if old > 0.0 { dy_norm / old } else { 0.0 });
        if let Some(rate) = rate {
            // Diverging, or cannot reach tolerance within the remaining budget.
            if rate >= 1.0
                || rate.powi((NEWTON_MAXITER - k) as i32) / (1.0 - rate) * dy_norm > newton_tol
            {
                break;
            }
        }
        for i in 0..n {
            y[i] += dy[i];
            d[i] += dy[i];
        }
        if dy_norm == 0.0 {
            converged = true;
            break;
        }
        if let Some(rate) = rate {
            if rate / (1.0 - rate) * dy_norm < newton_tol {
                converged = true;
                break;
            }
        }
        dy_norm_old = Some(dy_norm);
    }
    (converged, iters)
}

/// The variable-order, variable-step BDF kernel. See the [module docs](self).
pub struct Bdf {
    rtol: f64,
    atol: f64,
    coef: Coefficients,
    /// Newton convergence tolerance, derived from `rtol` (Shampine & Reichelt).
    newton_tol: f64,

    /// System dimension; `0` until the first step sizes the buffers.
    n: usize,
    /// Current BDF order (`1..=MAX_ORDER`).
    order: usize,
    /// Number of consecutive steps taken at the current order/step — the order
    /// machinery only reconsiders order once enough equal steps have accrued.
    n_equal_steps: usize,
    /// The step size the difference array [`Self::d`] is currently scaled to.
    h_d: f64,
    /// Whether the history has been seeded from the first step's state.
    initialized: bool,

    /// Difference array: `(MAX_ORDER + 3)` rows of length `n`, row-major. Row `j`
    /// holds the `j`-th modified divided difference (step-scaled). Rows `0..=order`
    /// form the predictor; `order+1`/`order+2` carry the error/order-raise terms.
    d: Vec<f64>,
    /// Analytic Jacobian `∂f/∂u`, row-major `n*n`, reused across steps.
    jac: Vec<f64>,
    /// Whether [`Self::jac`] has ever been evaluated.
    have_jac: bool,
    /// The factored iteration matrix `(I − c·J)`.
    lu: Vec<f64>,
    /// LU pivots for [`Self::lu`].
    piv: Vec<usize>,
    /// Whether [`Self::lu`] is a valid factorization for [`Self::c_lu`]/current `J`.
    lu_valid: bool,
    /// The `c = h/alpha[order]` the factored [`Self::lu`] was built for.
    c_lu: f64,

    // Per-step work buffers, grown to `n` on first use.
    y_predict: Vec<f64>,
    psi: Vec<f64>,
    scale: Vec<f64>,
    y: Vec<f64>,
    d_corr: Vec<f64>,
    f: Vec<f64>,
    dy: Vec<f64>,
    /// Scratch row buffer for the `change_D` rescale: `(MAX_ORDER + 1)` rows of `n`.
    d_scratch: Vec<f64>,
}

impl Bdf {
    /// A kernel with the default tolerances (`rtol = 1e-6`, `atol = 1e-9`), the
    /// same defaults the other implicit kernels carry.
    pub fn new() -> Self {
        Self::with_tolerances(1e-6, 1e-9)
    }

    /// A kernel with explicit relative / absolute tolerances.
    pub fn with_tolerances(rtol: f64, atol: f64) -> Self {
        // Newton tolerance per Shampine & Reichelt: tight enough that the residual
        // Newton error stays well under the truncation error the step controls,
        // but never tighter than rounding allows for a loose rtol.
        let newton_tol = (10.0 * f64::EPSILON / rtol).max(0.03_f64.min(rtol.sqrt()));
        Bdf {
            rtol,
            atol,
            coef: Coefficients::new(),
            newton_tol,
            n: 0,
            order: 1,
            n_equal_steps: 0,
            h_d: 0.0,
            initialized: false,
            d: Vec::new(),
            jac: Vec::new(),
            have_jac: false,
            lu: Vec::new(),
            piv: Vec::new(),
            lu_valid: false,
            c_lu: f64::NAN,
            y_predict: Vec::new(),
            psi: Vec::new(),
            scale: Vec::new(),
            y: Vec::new(),
            d_corr: Vec::new(),
            f: Vec::new(),
            dy: Vec::new(),
            d_scratch: Vec::new(),
        }
    }

    fn ensure(&mut self, n: usize) {
        if self.n == n {
            return;
        }
        self.n = n;
        self.d = vec![0.0; (MAX_ORDER + 3) * n];
        self.jac = vec![0.0; n * n];
        self.lu = vec![0.0; n * n];
        self.piv = vec![0; n];
        self.y_predict = vec![0.0; n];
        self.psi = vec![0.0; n];
        self.scale = vec![0.0; n];
        self.y = vec![0.0; n];
        self.d_corr = vec![0.0; n];
        self.f = vec![0.0; n];
        self.dy = vec![0.0; n];
        self.d_scratch = vec![0.0; (MAX_ORDER + 1) * n];
    }

    /// Row `j` of the difference array (length `n`).
    #[inline]
    fn d_row(&self, j: usize) -> &[f64] {
        &self.d[j * self.n..(j + 1) * self.n]
    }

    /// Seed the history from the current point at order 1: `D[0] = u`, `D[1] = h·f`.
    fn initialize(&mut self, ev: &dyn Evaluator, st: &SolverState, h: f64) {
        let n = st.u.len();
        self.ensure(n);
        self.d.iter_mut().for_each(|x| *x = 0.0);
        self.d[0..n].copy_from_slice(&st.u);
        // f(t0, u0) seeds the first difference D[1] = h·f.
        let mut scratch = vec![0.0; ev.n_scratch()];
        ev.eval(&st.u, &st.p, st.t, &mut scratch, &mut self.f);
        for i in 0..n {
            self.d[n + i] = h * self.f[i];
        }
        self.order = 1;
        self.h_d = h;
        self.n_equal_steps = 0;
        self.have_jac = false;
        self.lu_valid = false;
        self.initialized = true;
    }

    /// Rescale the difference array for a step-size change by `factor = h_new/h_d`,
    /// at the given `order`, via the `R`-matrix recurrence (Shampine & Reichelt).
    ///
    /// Maps the modified divided differences `D[0..=order]` from their current step
    /// scaling to the new one, the cheap alternative to recomputing the history.
    fn change_d(&mut self, order: usize, factor: f64) {
        let n = self.n;
        // R = compute_R(order, factor); U = compute_R(order, 1); RU = R·U.
        let mut r = [0.0; (MAX_ORDER + 1) * (MAX_ORDER + 1)];
        let mut u = [0.0; (MAX_ORDER + 1) * (MAX_ORDER + 1)];
        compute_r(order, factor, &mut r);
        compute_r(order, 1.0, &mut u);
        // RU[a][b] = Σ_k R[a][k]·U[k][b], a,b ∈ 0..=order.
        let m = order + 1;
        let mut ru = [0.0; (MAX_ORDER + 1) * (MAX_ORDER + 1)];
        for a in 0..m {
            for b in 0..m {
                let mut s = 0.0;
                for k in 0..m {
                    s += r[a * (MAX_ORDER + 1) + k] * u[k * (MAX_ORDER + 1) + b];
                }
                ru[a * (MAX_ORDER + 1) + b] = s;
            }
        }
        // newD[i] = Σ_k RU[k][i]·D[k]  (i.e. RUᵀ · D over rows 0..=order).
        for i in 0..m {
            for c in 0..n {
                let mut s = 0.0;
                for k in 0..m {
                    s += ru[k * (MAX_ORDER + 1) + i] * self.d[k * n + c];
                }
                self.d_scratch[i * n + c] = s;
            }
        }
        for i in 0..m {
            self.d[i * n..(i + 1) * n].copy_from_slice(&self.d_scratch[i * n..(i + 1) * n]);
        }
    }
}

/// Fill `out` (row-major `(order+1)×(MAX_ORDER+1)`, only the top-left
/// `(order+1)×(order+1)` block used) with the `R` matrix of Shampine & Reichelt:
/// `M[0][:] = 1`, `M[i][j] = (i − 1 − factor·j)/i` for `i,j ≥ 1` (else 0), then
/// `R = cumprod(M)` down each column.
// Row/column index arithmetic into the packed matrix reads more clearly than an
// enumerate over a strided slice.
#[allow(clippy::needless_range_loop)]
fn compute_r(order: usize, factor: f64, out: &mut [f64]) {
    let stride = MAX_ORDER + 1;
    let m = order + 1;
    // Build M into out, then take the running column product in place.
    for v in out.iter_mut() {
        *v = 0.0;
    }
    for j in 0..m {
        out[j] = 1.0; // row 0 is all ones
    }
    for i in 1..m {
        for j in 1..m {
            out[i * stride + j] = (i as f64 - 1.0 - factor * j as f64) / i as f64;
        }
        // column 0 stays 0 for i ≥ 1 (M[i][0] = 0)
    }
    // Cumulative product down each column: R[i][j] = R[i-1][j]·M[i][j].
    for i in 1..m {
        for j in 0..m {
            out[i * stride + j] *= out[(i - 1) * stride + j];
        }
    }
}

impl Default for Bdf {
    fn default() -> Self {
        Bdf::new()
    }
}

impl Solver for Bdf {
    fn name(&self) -> &'static str {
        "bdf"
    }

    fn caps(&self) -> Caps {
        Caps::implicit(ProblemKinds::of(ProblemKind::Ode)).adaptive()
    }

    // The step is a faithful single-attempt port of the fixed-leading-coefficient
    // BDF: rescale the history to the engine-chosen `h`, predict, correct by
    // modified Newton (reusing J/LU where possible), estimate error, then commit
    // + reselect order on accept or suggest a smaller step on reject. The
    // disjoint-buffer range loops mirror the other implicit kernels.
    #[allow(clippy::needless_range_loop)]
    fn step(&mut self, ev: &dyn Evaluator, st: &mut SolverState, h: f64) -> StepOutcome {
        // Defence in depth: an implicit kernel must never run Jacobian-free (the
        // engine boundary rejects this first; this protects direct callers).
        if !ev.has_jacobian() {
            return StepOutcome::Failed;
        }
        if !(h.is_finite() && h > 0.0) {
            return StepOutcome::Failed;
        }

        if !self.initialized {
            self.initialize(ev, st, h);
        }
        let n = self.n;

        // Rescale the history if the engine asks for a step different from the one
        // D is scaled to (it caps the step to land on output times). In smooth
        // phases under a fine grid this is the identity (h == h_d), so D and the
        // factored LU survive untouched across many steps.
        if h != self.h_d {
            let factor = h / self.h_d;
            self.change_d(self.order, factor);
            self.h_d = h;
            self.n_equal_steps = 0;
            self.lu_valid = false;
        }

        let order = self.order;
        let alpha = self.coef.alpha[order];
        let c = h / alpha;

        // Predictor y_predict = Σ_{k=0}^{order} D[k]; psi = (Σ_{k=1}^{order}
        // gamma[k]·D[k]) / alpha.
        for i in 0..n {
            let mut yp = 0.0;
            let mut ps = 0.0;
            for k in 0..=order {
                let dk = self.d[k * n + i];
                yp += dk;
                if k >= 1 {
                    ps += self.coef.gamma[k] * dk;
                }
            }
            self.y_predict[i] = yp;
            self.psi[i] = ps / alpha;
        }
        if !self.y_predict.iter().all(|x| x.is_finite()) {
            return StepOutcome::Failed;
        }
        for i in 0..n {
            self.scale[i] = self.atol + self.rtol * self.y_predict[i].abs();
        }
        let t_new = st.t + h;

        // Modified-Newton corrector with at most one Jacobian refresh, mirroring
        // the production stiff solvers: try the (reused) Jacobian first, refresh
        // and retry once on non-convergence, then reject.
        let mut jac_current = false; // self.jac is from an earlier step
                                     // The corrector loop breaks with `Some(n_iter)` on convergence, `None` when
                                     // even a fresh Jacobian cannot make Newton converge at this step size.
        let corrected = loop {
            // Ensure we hold a Jacobian.
            if !self.have_jac {
                self.eval_jacobian(ev, st, t_new);
                jac_current = true;
            }
            // (Re)factor (I − c·J) when stale or built for a different c.
            if !self.lu_valid || self.c_lu != c {
                build_shifted(&mut self.lu, &self.jac, n, c);
                if !lu_factor(&mut self.lu, n, &mut self.piv) {
                    // Singular iteration matrix ⇒ refresh J once, else reject.
                    if !jac_current {
                        self.eval_jacobian(ev, st, t_new);
                        jac_current = true;
                        continue;
                    }
                    self.lu_valid = false;
                    return self.reject(0.5 * h);
                }
                self.lu_valid = true;
                self.c_lu = c;
            }

            let (conv, iters) = {
                let Bdf {
                    y_predict,
                    psi,
                    scale,
                    lu,
                    piv,
                    y,
                    d_corr,
                    f,
                    dy,
                    newton_tol,
                    ..
                } = self;
                solve_bdf_system(
                    ev,
                    t_new,
                    &st.p,
                    c,
                    y_predict,
                    psi,
                    scale,
                    lu,
                    piv,
                    y,
                    d_corr,
                    f,
                    dy,
                    &mut st.scratch,
                    *newton_tol,
                )
            };
            if conv {
                break Some(iters);
            }
            if !jac_current {
                // Stale Jacobian: refresh at the predictor and retry once.
                self.eval_jacobian(ev, st, t_new);
                jac_current = true;
                self.lu_valid = false;
                continue;
            }
            // Fresh Jacobian still failed: the step is too large.
            self.lu_valid = false;
            break None;
        };

        let n_iter = match corrected {
            Some(iters) => iters,
            None => return self.reject(0.5 * h),
        };
        if !self.y.iter().all(|x| x.is_finite()) {
            return StepOutcome::Failed;
        }

        // Adaptive safety factor (Shampine & Reichelt): a step that needed more
        // Newton iterations is trusted less, so its growth is throttled.
        let safety = 0.9 * (2.0 * NEWTON_MAXITER as f64 + 1.0)
            / (2.0 * NEWTON_MAXITER as f64 + n_iter as f64);

        // Error estimate from the corrector increment, scaled to the corrected y.
        for i in 0..n {
            self.scale[i] = self.atol + self.rtol * self.y[i].abs();
        }
        let error_const = self.coef.error_const[order];
        let mut err_acc = 0.0;
        for i in 0..n {
            let e = error_const * self.d_corr[i] / self.scale[i];
            err_acc += e * e;
        }
        let error_norm = (err_acc / n as f64).sqrt();

        if !error_norm.is_finite() {
            return self.reject(MIN_FACTOR * h);
        }
        if error_norm > 1.0 {
            // Reject: suggest a smaller step. D is untouched (still the last
            // accepted state, scaled to h); the next attempt rescales from it.
            let factor = MIN_FACTOR.max(safety * error_norm.powf(-1.0 / (order as f64 + 1.0)));
            self.lu_valid = false;
            return self.reject(factor * h);
        }

        // --- Accept ---
        // Commit the corrected state and fold the increment into the history.
        st.u.copy_from_slice(&self.y);
        st.t = t_new;
        // D[order+2] = d − D[order+1]; D[order+1] = d; then prefix-sum down.
        for i in 0..n {
            let d_i = self.d_corr[i];
            self.d[(order + 2) * n + i] = d_i - self.d[(order + 1) * n + i];
            self.d[(order + 1) * n + i] = d_i;
        }
        for k in (0..=order).rev() {
            for i in 0..n {
                self.d[k * n + i] += self.d[(k + 1) * n + i];
            }
        }
        self.n_equal_steps += 1;

        // Keep the step (and the factored LU) until enough equal steps accrue to
        // justify reconsidering the order — the phase where BDF reuses one
        // factorization across many cheap steps.
        if self.n_equal_steps < order + 1 {
            return StepOutcome::Accepted { h_next: h };
        }

        // Order selection: compare the step factor a one-lower, equal, and
        // one-higher order would earn, and move toward the most efficient.
        let error_m_norm = if order > 1 {
            self.scaled_norm(self.coef.error_const[order - 1], order, n) // uses D[order]
        } else {
            f64::INFINITY
        };
        let error_p_norm = if order < MAX_ORDER {
            self.scaled_norm(self.coef.error_const[order + 1], order + 2, n) // uses D[order+2]
        } else {
            f64::INFINITY
        };

        // Step factor for each candidate: error^(−1/(k+1)) with k the candidate's
        // order (order−1, order, order+1).
        let f_lower = factor_for(error_m_norm, order); // exponent −1/order
        let f_same = factor_for(error_norm, order + 1); // exponent −1/(order+1)
        let f_higher = factor_for(error_p_norm, order + 2); // exponent −1/(order+2)

        let mut best = f_same;
        let mut delta: isize = 0;
        if f_lower > best {
            best = f_lower;
            delta = -1;
        }
        if f_higher > best {
            best = f_higher;
            delta = 1;
        }
        self.order = (order as isize + delta) as usize;

        let factor = MAX_FACTOR.min(safety * best);
        // Defer the rescale: keep D at h_d = h, and let the next step's entry
        // rescale by the actual h the engine hands back (== factor·h when it
        // honours the suggestion). The order field is already updated, so the
        // entry rescale uses the new order — identical to applying it here.
        //
        // The factored LU is *not* invalidated here on purpose: the next step
        // rebuilds it only if its iteration constant c = h/alpha[order] actually
        // changes — which the entry checks already catch (an order change moves
        // alpha, so `c_lu != c`; an h change triggers `change_d` + invalidation).
        // When the engine caps the step to a fine output grid it keeps handing
        // back the same h and the order settles, so c is unchanged and the same
        // factorization is reused across many steps — the per-step economy that
        // makes BDF cheap on the grid-limited warm path (benches/REPORT.md).
        self.n_equal_steps = 0;
        StepOutcome::Accepted { h_next: factor * h }
    }
}

impl Bdf {
    /// Evaluate the analytic Jacobian at the predictor point `(t_new, y_predict)`
    /// (the corrector's working point), refreshing [`Self::jac`] and forcing a
    /// refactor.
    fn eval_jacobian(&mut self, ev: &dyn Evaluator, st: &mut SolverState, t_new: f64) {
        let Bdf {
            y_predict, f, jac, ..
        } = self;
        ev.eval_jac(y_predict, &st.p, t_new, &mut st.scratch, f, jac);
        self.have_jac = true;
        self.lu_valid = false;
    }

    /// `‖error_const · D[row]‖` in the current (corrected-y) scale — the candidate
    /// error norm an order change is judged on.
    #[allow(clippy::needless_range_loop)]
    fn scaled_norm(&self, error_const: f64, row: usize, n: usize) -> f64 {
        let mut acc = 0.0;
        let d = self.d_row(row);
        for i in 0..n {
            let e = error_const * d[i] / self.scale[i];
            acc += e * e;
        }
        (acc / n as f64).sqrt()
    }

    /// Build a rejection outcome from a suggested next step, downgrading a
    /// collapsed (non-finite / non-positive) step to [`StepOutcome::Failed`] so the
    /// run fails loudly rather than spinning.
    fn reject(&self, h_next: f64) -> StepOutcome {
        if h_next.is_finite() && h_next > 0.0 {
            StepOutcome::Rejected { h_next }
        } else {
            StepOutcome::Failed
        }
    }
}

/// The elementary step-size factor a candidate order earns from its error norm:
/// `error_norm^(−1/(k+1))`, the larger the more attractive that order.
///
/// IEEE arithmetic gives exactly the right limits, so no special-casing is needed
/// (and special-casing infinity is a trap — see below):
/// - `error_norm = 0` (an exact step) → `+∞`, i.e. grow freely (the caller clamps
///   with the safety factor and `MAX_FACTOR`);
/// - `error_norm = +∞` (an *unavailable* candidate, e.g. order below 1 or above
///   `MAX_ORDER`, whose norm is set to infinity) → `0`, i.e. never selected.
///
/// The infinity → 0 mapping is essential: if an unavailable lower-order candidate
/// scored a large factor it would be picked and drive the order to 0, where
/// `alpha[0] = 0` makes `c = h/alpha` non-finite and the next step blows up.
fn factor_for(error_norm: f64, k_plus_1: usize) -> f64 {
    if error_norm.is_nan() {
        return 0.0; // a NaN candidate must never win the max
    }
    error_norm.powf(-1.0 / k_plus_1 as f64)
}

register_solver!(
    "bdf",
    Caps::implicit(ProblemKinds::of(ProblemKind::Ode)).adaptive(),
    || Box::new(Bdf::new())
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::SolverKind;

    #[test]
    fn metadata_is_consistent() {
        let s = Bdf::new();
        assert_eq!(s.name(), "bdf");
        let c = s.caps();
        assert_eq!(c.kind, SolverKind::Implicit);
        assert!(c.adaptive);
        assert!(c.needs_jacobian);
        assert!(c.supports(ProblemKind::Ode));
        assert!(!c.supports(ProblemKind::Sde));
    }

    #[test]
    fn tolerances_builder_overrides_defaults() {
        let s = Bdf::with_tolerances(1e-8, 1e-11);
        assert_eq!(s.rtol, 1e-8);
        assert_eq!(s.atol, 1e-11);
    }

    #[test]
    fn coefficients_match_known_values() {
        let c = Coefficients::new();
        // gamma[k] = sum 1/i: 0, 1, 1.5, 1.8333…, 2.0833…, 2.2833…
        assert!((c.gamma[1] - 1.0).abs() < 1e-15);
        assert!((c.gamma[2] - 1.5).abs() < 1e-15);
        assert!((c.gamma[5] - (1.0 + 0.5 + 1.0 / 3.0 + 0.25 + 0.2)).abs() < 1e-15);
        // alpha[1] = (1 - kappa[1])·gamma[1] = 1.185.
        assert!((c.alpha[1] - 1.185).abs() < 1e-12);
        // error_const[1] = kappa[1]·gamma[1] + 1/2 = -0.185 + 0.5 = 0.315.
        assert!((c.error_const[1] - 0.315).abs() < 1e-12);
        // error_const[k] > 0 for all orders (it scales a positive error estimate).
        for k in 1..=MAX_ORDER {
            assert!(
                c.error_const[k] > 0.0,
                "error_const[{k}] = {}",
                c.error_const[k]
            );
            assert!(c.alpha[k] > 0.0, "alpha[{k}] = {}", c.alpha[k]);
        }
    }

    #[test]
    fn compute_r_is_identity_at_factor_one_after_change_d() {
        // change_D with factor 1 must leave the difference array unchanged (it is
        // the RU = U·U = I property the rescale relies on). Drive it through a
        // small kernel instance with a hand-set D.
        let mut s = Bdf::new();
        s.ensure(2);
        s.order = 3;
        // Arbitrary difference rows.
        for k in 0..=s.order {
            for i in 0..2 {
                s.d[k * 2 + i] = (k as f64 + 1.0) * (i as f64 + 1.0) - 0.37 * k as f64;
            }
        }
        let before = s.d.clone();
        s.change_d(s.order, 1.0);
        for (a, b) in before.iter().zip(s.d.iter()) {
            assert!((a - b).abs() < 1e-12, "change_D(1) changed D: {a} vs {b}");
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn change_d_halving_then_doubling_round_trips() {
        // change_D(0.5) then change_D(2.0) returns to the original differences (the
        // rescale recurrence is exactly invertible for a constant-step history).
        let mut s = Bdf::new();
        s.ensure(2);
        s.order = 2;
        for k in 0..=s.order {
            for i in 0..2 {
                s.d[k * 2 + i] = 1.0 + 0.5 * k as f64 + 0.25 * i as f64;
            }
        }
        let before = s.d[0..(s.order + 1) * 2].to_vec();
        s.change_d(s.order, 0.5);
        s.change_d(s.order, 2.0);
        for i in 0..(s.order + 1) * 2 {
            assert!(
                (before[i] - s.d[i]).abs() < 1e-10,
                "round-trip drift at {i}: {} vs {}",
                before[i],
                s.d[i]
            );
        }
    }
}
