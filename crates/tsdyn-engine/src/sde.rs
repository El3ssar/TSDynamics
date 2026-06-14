//! Stochastic (SDE) integration — diagonal-Itô drift + diffusion (stream E-SDE).
//!
//! A diagonal-Itô SDE is `dX_k = f_k(X, t) dt + g_k(X, t) dW_k` with one
//! independent Wiener process per state component (ROADMAP §11). This module is
//! the SDE analogue of [`crate::integrate`] / [`crate::ensemble`]: it drives a
//! fixed-step [`SdeKernel`] (Euler–Maruyama / Milstein, `tsdyn-solvers`) over
//! **two** evaluators — a drift `f` and a diffusion `g`, each a `dim → dim`
//! [`Evaluator`] — and owns the seeded randomness.
//!
//! # Why a dedicated loop (not [`crate::integrate`])
//!
//! The ODE driver steps one evaluator with the frozen [`Solver`](tsdyn_solvers::Solver)
//! trait, which carries neither a second right-hand side nor a noise increment.
//! An SDE step needs both, so — exactly as the discrete-map family got
//! [`crate::map`] — the SDE family gets its own loop here and its own
//! [`SdeKernel`] trait in `tsdyn-solvers`, leaving the frozen ODE seams untouched.
//!
//! # Where the randomness lives (the determinism contract)
//!
//! The kernels are RNG-free and pure: this loop draws the diagonal Wiener
//! increment ([`fill_wiener`]) each step and hands the kernel a `dw` slice. The
//! ensemble seeds worker `i` from [`seed_for`]`(base_seed, i)`, so a trajectory's
//! entire random stream depends only on its **index**, never on thread
//! scheduling — [`sde_ensemble_final`] is therefore **parallel == serial**
//! bit-for-bit, the same guarantee [`crate::ensemble`] gives (ROADMAP §4c).
//!
//! # Fixed step, sub-stepping to the grid
//!
//! These schemes are fixed-step: the step size *is* `dt`, and `dt` sets the noise
//! scale `√dt`. Output at a requested time that is not on the `dt` lattice is
//! produced by a shorter final sub-step whose increment is drawn `~ N(0, h)` for
//! that `h` — a valid Euler–Maruyama/Milstein step on a non-uniform partition
//! (Brownian increments over sub-intervals are independent and sum correctly), so
//! the discretization stays consistent. As with the ODE driver, a non-finite
//! state is reported, never returned as plausible data ([`SdeError`]).

use rayon::prelude::*;
use tsdyn_ir::Evaluator;
use tsdyn_solvers::sde::SdeKernel;
use tsdyn_solvers::{SolverState, StepOutcome};

use crate::integrate::DEFAULT_MAX_STEPS;
use crate::rng::{fill_wiener, seed_for, SplitMix64};

/// Knobs for the fixed-step SDE loop.
///
/// Build with [`SdeConfig::new`] (the step `dt` plus the default step cap) and
/// refine with [`with_max_steps`](SdeConfig::with_max_steps).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdeConfig {
    /// The fixed step size — *and* the noise scale (each step's diagonal Wiener
    /// increment is drawn `~ N(0, dt)`). Must be finite and `> 0`.
    pub dt: f64,
    /// Cap on steps **per output segment** (mirrors [`crate::IntegrateConfig`]),
    /// a backstop against a pathological `dt`/span. See [`DEFAULT_MAX_STEPS`].
    pub max_steps: usize,
}

impl SdeConfig {
    /// A config with step `dt` and the default step cap ([`DEFAULT_MAX_STEPS`]).
    pub fn new(dt: f64) -> Self {
        SdeConfig {
            dt,
            max_steps: DEFAULT_MAX_STEPS,
        }
    }

    /// Set the per-segment step cap.
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }
}

/// Why an SDE integration stopped short of its target time.
///
/// Mirrors [`crate::IntegrateError`]'s contract: a diverging trajectory surfaces
/// as one of these (carrying where the trouble was), never as plausible numbers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SdeError {
    /// The state went non-finite — the drift/diffusion blew up. Carries the time.
    NonFinite {
        /// The time at which non-finiteness was detected.
        t: f64,
    },
    /// The per-segment step cap ([`SdeConfig::max_steps`]) was hit before
    /// reaching the target time (a pathological `dt`/span).
    StepLimit {
        /// Time reached when the cap was hit.
        t: f64,
        /// The cap that was hit.
        steps: usize,
    },
}

impl core::fmt::Display for SdeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SdeError::NonFinite { t } => {
                write!(f, "non-finite state at t = {t} (the SDE diverged)")
            }
            SdeError::StepLimit { t, steps } => {
                write!(f, "hit the {steps}-step limit at t = {t}")
            }
        }
    }
}

impl std::error::Error for SdeError {}

/// Advance `st` from its current time to `t_end` with fixed step `cfg.dt`,
/// drawing one diagonal Wiener increment per step from `rng` into `dw`.
///
/// A landing step (when `dt` would reach/pass `t_end`) is shortened to the
/// remaining span and its increment drawn `~ N(0, h)` for that `h`; the time is
/// then pinned exactly to `t_end` so grid points never drift. `dw` is a caller-
/// owned scratch of length `dim`, reused across steps.
#[allow(clippy::too_many_arguments)] // drift + diffusion + the state/rng/dw/time/cfg the loop threads.
fn sde_advance_to(
    drift: &dyn Evaluator,
    diffusion: &dyn Evaluator,
    kernel: &mut dyn SdeKernel,
    rng: &mut SplitMix64,
    st: &mut SolverState,
    dw: &mut [f64],
    t_end: f64,
    cfg: &SdeConfig,
) -> Result<(), SdeError> {
    // Hard assert (not debug-only): a non-positive/non-finite step is caller
    // error that would otherwise spin to the step limit or step the wrong way.
    assert!(
        cfg.dt.is_finite() && cfg.dt > 0.0,
        "SDE step dt must be finite and positive, got {}",
        cfg.dt
    );
    let mut steps = 0usize;
    while st.t < t_end {
        if steps >= cfg.max_steps {
            return Err(SdeError::StepLimit { t: st.t, steps });
        }
        let remaining = t_end - st.t;
        let landing = cfg.dt >= remaining;
        let h = if landing { remaining } else { cfg.dt };
        steps += 1;

        fill_wiener(rng, h, dw);
        match kernel.step(drift, diffusion, st, dw, h) {
            StepOutcome::Accepted { .. } => {
                if !st.u.iter().all(|x| x.is_finite()) || !st.t.is_finite() {
                    return Err(SdeError::NonFinite { t: st.t });
                }
                if landing {
                    // The kernel advanced t by `remaining` (up to rounding); pin
                    // it to the target so grid points never drift.
                    st.t = t_end;
                }
            }
            // The fixed-step SDE kernels never reject; anything but Accepted is an
            // abort (the kernel signalled a blow-up).
            _ => return Err(SdeError::NonFinite { t: st.t }),
        }
    }
    Ok(())
}

/// Integrate one SDE trajectory from `t0` to `t1`, returning the final state.
///
/// `u0`/`p` seed a fresh [`SolverState`]; `kernel` is stepped with `cfg.dt` until
/// the trajectory reaches `t1`, drawing its noise from `rng`. Forward only
/// (`t1 ≥ t0`); a non-increasing span returns `u0` unchanged. Returns an
/// [`SdeError`] if the trajectory diverged or stalled.
#[allow(clippy::too_many_arguments)] // drift + diffusion + the usual integrate args; the pair is the SDE contract.
pub fn sde_integrate_final(
    drift: &dyn Evaluator,
    diffusion: &dyn Evaluator,
    kernel: &mut dyn SdeKernel,
    rng: &mut SplitMix64,
    u0: &[f64],
    p: &[f64],
    t0: f64,
    t1: f64,
    cfg: &SdeConfig,
) -> Result<Vec<f64>, SdeError> {
    debug_assert_eq!(u0.len(), drift.dim(), "u0 length must equal the dimension");
    debug_assert_eq!(diffusion.dim(), drift.dim(), "drift/diffusion dim mismatch");
    debug_assert_eq!(p.len(), drift.n_param(), "p length must equal n_param");
    debug_assert!(t1 >= t0, "integration is forward only: need t1 >= t0");
    let mut st = SolverState {
        u: u0.to_vec(),
        t: t0,
        p: p.to_vec(),
        scratch: Vec::new(), // SDE kernels own their scratch; see SdeKernel::step
    };
    let mut dw = vec![0.0; drift.dim()];
    sde_advance_to(drift, diffusion, kernel, rng, &mut st, &mut dw, t1, cfg)?;
    Ok(st.u)
}

/// Integrate one SDE trajectory through the non-decreasing times `t_eval`,
/// recording the state at each into a flat row-major `(t_eval.len(), dim)`
/// buffer.
///
/// `u0` is the state at `t_eval[0]` (the first output row), matching the usual
/// dense-trajectory convention; the integration steps from each time to the
/// next, landing exactly on each, carrying the noise stream across segments.
/// Returns an [`SdeError`] if any segment diverges or stalls.
#[allow(clippy::too_many_arguments)] // drift + diffusion + the usual grid args.
pub fn sde_integrate_grid(
    drift: &dyn Evaluator,
    diffusion: &dyn Evaluator,
    kernel: &mut dyn SdeKernel,
    rng: &mut SplitMix64,
    u0: &[f64],
    p: &[f64],
    t_eval: &[f64],
    cfg: &SdeConfig,
) -> Result<Vec<f64>, SdeError> {
    debug_assert_eq!(u0.len(), drift.dim(), "u0 length must equal the dimension");
    debug_assert_eq!(diffusion.dim(), drift.dim(), "drift/diffusion dim mismatch");
    debug_assert_eq!(p.len(), drift.n_param(), "p length must equal n_param");
    let dim = drift.dim();
    let mut out = vec![0.0; t_eval.len() * dim];
    if t_eval.is_empty() {
        return Ok(out);
    }
    let mut st = SolverState {
        u: u0.to_vec(),
        t: t_eval[0],
        p: p.to_vec(),
        scratch: Vec::new(),
    };
    let mut dw = vec![0.0; dim];
    for (k, (chunk, &target)) in out.chunks_mut(dim).zip(t_eval).enumerate() {
        if k > 0 {
            debug_assert!(target >= st.t, "t_eval must be non-decreasing");
            sde_advance_to(drift, diffusion, kernel, rng, &mut st, &mut dw, target, cfg)?;
        }
        chunk.copy_from_slice(&st.u);
    }
    Ok(out)
}

/// The fate of one ensemble SDE trajectory.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SdeTrajStatus {
    /// Reached the target time; its row in [`SdeEnsembleFinal::states`] is the
    /// final state.
    Ok,
    /// Diverged or stalled; its row is `NaN` and the reason is carried here.
    Failed(SdeError),
}

impl SdeTrajStatus {
    /// Whether the trajectory completed successfully.
    pub fn is_ok(&self) -> bool {
        matches!(self, SdeTrajStatus::Ok)
    }
}

/// The result of an ensemble SDE integration to a final time.
///
/// `states` is a row-major `(n_ic, dim)` buffer of final states (one row per
/// initial condition, in input order); failed rows are all-`NaN`. `status[i]`
/// records each trajectory's fate.
#[derive(Clone, Debug)]
pub struct SdeEnsembleFinal {
    /// System dimension (row width of [`states`](SdeEnsembleFinal::states)).
    pub dim: usize,
    /// Row-major `(n_ic, dim)` final states; `NaN` rows for failed trajectories.
    pub states: Vec<f64>,
    /// Per-trajectory fate, length `n_ic`, in input order.
    pub status: Vec<SdeTrajStatus>,
}

impl SdeEnsembleFinal {
    /// Number of initial conditions integrated.
    pub fn n_ic(&self) -> usize {
        self.status.len()
    }

    /// The final state of trajectory `i` (a `NaN`-filled slice if it failed).
    ///
    /// Panics if `i >= ` [`n_ic`](SdeEnsembleFinal::n_ic).
    pub fn row(&self, i: usize) -> &[f64] {
        &self.states[i * self.dim..(i + 1) * self.dim]
    }

    /// How many trajectories failed to reach the target time.
    pub fn n_failed(&self) -> usize {
        self.status.iter().filter(|s| !s.is_ok()).count()
    }
}

/// Integrate a batch of SDE initial conditions to `t1` in parallel, returning
/// each final state — **parallel == serial** bit-for-bit.
///
/// `u0_batch` is a row-major `(n_ic, dim)` buffer; `dim` is taken from `drift`.
/// `make_kernel()` builds a fresh, RNG-free kernel for each worker (it takes no
/// index — unlike the ODE ensemble, the *engine* owns the seeding here). Worker
/// `i` draws its noise from a [`SplitMix64`] seeded with
/// [`seed_for`]`(base_seed, i)`, so the result depends only on the index, not on
/// thread count; results are collected in input order. A diverging trajectory
/// yields a `NaN` row and an [`SdeTrajStatus::Failed`] rather than aborting the
/// whole batch.
#[allow(clippy::too_many_arguments)] // drift + diffusion + factory + seed + the usual ensemble args.
pub fn sde_ensemble_final<F>(
    drift: &dyn Evaluator,
    diffusion: &dyn Evaluator,
    make_kernel: F,
    base_seed: u64,
    u0_batch: &[f64],
    p: &[f64],
    t0: f64,
    t1: f64,
    cfg: &SdeConfig,
) -> SdeEnsembleFinal
where
    F: Fn() -> Box<dyn SdeKernel> + Sync,
{
    let dim = drift.dim();
    // Hard asserts on the calling thread, before the fan-out: a ragged batch,
    // zero dim, or bad dt is caller error; silently flooring would drop work.
    assert!(dim > 0, "evaluator dimension must be positive");
    assert_eq!(diffusion.dim(), dim, "drift/diffusion dim mismatch");
    assert!(
        cfg.dt.is_finite() && cfg.dt > 0.0,
        "SDE step dt must be finite and positive, got {}",
        cfg.dt
    );
    assert_eq!(
        u0_batch.len() % dim,
        0,
        "u0_batch length {} is not a multiple of dim {dim}",
        u0_batch.len()
    );
    let n_ic = u0_batch.len() / dim;

    // map → collect preserves index order, so the output never depends on the
    // order rayon finishes workers in.
    let per_traj: Vec<(Vec<f64>, SdeTrajStatus)> = (0..n_ic)
        .into_par_iter()
        .map(|i| {
            let u0 = &u0_batch[i * dim..(i + 1) * dim];
            let mut kernel = make_kernel();
            // Randomness keyed by trajectory index — the determinism contract.
            let mut rng = SplitMix64::new(seed_for(base_seed, i as u64));
            match sde_integrate_final(drift, diffusion, &mut *kernel, &mut rng, u0, p, t0, t1, cfg)
            {
                Ok(uf) => (uf, SdeTrajStatus::Ok),
                Err(e) => (vec![f64::NAN; dim], SdeTrajStatus::Failed(e)),
            }
        })
        .collect();

    let mut states = Vec::with_capacity(n_ic * dim);
    let mut status = Vec::with_capacity(n_ic);
    for (uf, s) in per_traj {
        states.extend_from_slice(&uf);
        status.push(s);
    }
    SdeEnsembleFinal {
        dim,
        states,
        status,
    }
}

/// A diagonal-Itô SDE ready to integrate: a drift and a diffusion evaluator plus
/// their shared parameters.
///
/// The SDE analogue of [`OdeProblem`](crate::OdeProblem) /
/// [`MapProblem`](crate::map::MapProblem). Borrows both evaluators (built once,
/// shared across an ensemble's rayon workers since `Evaluator: Sync`) and owns
/// the parameter vector. Construct one, then call
/// [`integrate_final`](SdeProblem::integrate_final),
/// [`integrate_grid`](SdeProblem::integrate_grid) or
/// [`ensemble_final`](SdeProblem::ensemble_final).
pub struct SdeProblem<'e> {
    drift: &'e dyn Evaluator,
    diffusion: &'e dyn Evaluator,
    p: Vec<f64>,
}

impl<'e> SdeProblem<'e> {
    /// Bundle a drift and diffusion evaluator with their shared parameters.
    ///
    /// The two evaluators must agree on `dim` and `n_param`, and `p.len()` must
    /// equal that `n_param` (the drift and diffusion tapes share one parameter
    /// layout, so one vector feeds both).
    pub fn new(drift: &'e dyn Evaluator, diffusion: &'e dyn Evaluator, p: Vec<f64>) -> Self {
        debug_assert_eq!(
            drift.dim(),
            diffusion.dim(),
            "drift and diffusion must share the system dimension"
        );
        debug_assert_eq!(
            drift.n_param(),
            diffusion.n_param(),
            "drift and diffusion must share the parameter layout"
        );
        debug_assert_eq!(
            p.len(),
            drift.n_param(),
            "parameter vector length must equal the evaluators' n_param"
        );
        SdeProblem {
            drift,
            diffusion,
            p,
        }
    }

    /// The system dimension.
    pub fn dim(&self) -> usize {
        self.drift.dim()
    }

    /// The borrowed drift evaluator.
    pub fn drift(&self) -> &dyn Evaluator {
        self.drift
    }

    /// The borrowed diffusion evaluator.
    pub fn diffusion(&self) -> &dyn Evaluator {
        self.diffusion
    }

    /// The parameter vector.
    pub fn params(&self) -> &[f64] {
        &self.p
    }

    /// Integrate from `t0` to `t1`, returning the final state.
    /// See [`sde_integrate_final`].
    pub fn integrate_final(
        &self,
        kernel: &mut dyn SdeKernel,
        rng: &mut SplitMix64,
        u0: &[f64],
        t0: f64,
        t1: f64,
        cfg: &SdeConfig,
    ) -> Result<Vec<f64>, SdeError> {
        sde_integrate_final(
            self.drift,
            self.diffusion,
            kernel,
            rng,
            u0,
            &self.p,
            t0,
            t1,
            cfg,
        )
    }

    /// Integrate through `t_eval`, returning a flat `(t_eval.len(), dim)` buffer.
    /// See [`sde_integrate_grid`].
    pub fn integrate_grid(
        &self,
        kernel: &mut dyn SdeKernel,
        rng: &mut SplitMix64,
        u0: &[f64],
        t_eval: &[f64],
        cfg: &SdeConfig,
    ) -> Result<Vec<f64>, SdeError> {
        sde_integrate_grid(
            self.drift,
            self.diffusion,
            kernel,
            rng,
            u0,
            &self.p,
            t_eval,
            cfg,
        )
    }

    /// Integrate a batch of initial conditions to `t1` in parallel (seeded by
    /// trajectory index). See [`sde_ensemble_final`].
    pub fn ensemble_final<F>(
        &self,
        make_kernel: F,
        base_seed: u64,
        u0_batch: &[f64],
        t0: f64,
        t1: f64,
        cfg: &SdeConfig,
    ) -> SdeEnsembleFinal
    where
        F: Fn() -> Box<dyn SdeKernel> + Sync,
    {
        sde_ensemble_final(
            self.drift,
            self.diffusion,
            make_kernel,
            base_seed,
            u0_batch,
            &self.p,
            t0,
            t1,
            cfg,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::VmEval;
    use tsdyn_ir::TapeBuilder;
    use tsdyn_solvers::sde::{EulerMaruyama, Milstein};
    use tsdyn_vm::Interpreter;

    // --- Hand-built drift/diffusion evaluators (real interpreter tapes) -------

    /// Ornstein–Uhlenbeck drift `f(x) = θ(μ − x)`.
    fn ou_drift(theta: f64, mu: f64) -> VmEval {
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let th = b.constant(theta);
        let m = b.constant(mu);
        let mmx = b.sub(m, x);
        let f = b.mul(th, mmx);
        VmEval::new(Interpreter::new(b.finish(&[f], &[], 1, 0).unwrap()))
    }

    /// Constant (additive) diffusion `g(x) = σ`, with a (zero) Jacobian so
    /// Milstein can request `eval_jac`.
    fn const_diffusion(sigma: f64) -> VmEval {
        let mut b = TapeBuilder::new();
        let s = b.constant(sigma);
        let zero = b.constant(0.0);
        VmEval::new(Interpreter::new(b.finish(&[s], &[zero], 1, 0).unwrap()))
    }

    /// Geometric-Brownian-motion drift `f(x) = μ x`.
    fn gbm_drift(mu: f64) -> VmEval {
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let m = b.constant(mu);
        let f = b.mul(m, x);
        VmEval::new(Interpreter::new(b.finish(&[f], &[], 1, 0).unwrap()))
    }

    /// Multiplicative diffusion `g(x) = σ x`, with diagonal Jacobian `∂g/∂x = σ`.
    fn gbm_diffusion(sigma: f64) -> VmEval {
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let s = b.constant(sigma);
        let g = b.mul(s, x);
        // jac_outputs[0] = ∂g/∂x = σ, i.e. the register holding the constant σ.
        VmEval::new(Interpreter::new(b.finish(&[g], &[s], 1, 0).unwrap()))
    }

    /// Sample mean and (population) variance of a slice.
    fn mean_var(xs: &[f64]) -> (f64, f64) {
        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        (mean, var)
    }

    #[test]
    fn ou_converges_to_its_stationary_mean_and_variance() {
        // dX = θ(μ − X) dt + σ dW. Stationary law: N(μ, σ²/(2θ)). Starting at μ
        // (so the mean is stationary from t=0) and integrating well past the
        // relaxation time 1/θ, the ensemble mean ≈ μ and variance ≈ σ²/(2θ).
        let (theta, mu, sigma) = (1.0, 2.0, 0.5);
        let drift = ou_drift(theta, mu);
        let diffusion = const_diffusion(sigma);
        let cfg = SdeConfig::new(0.005);

        let n = 20_000;
        let u0 = vec![mu; n]; // start at the mean
        let ens = sde_ensemble_final(
            &drift,
            &diffusion,
            || Box::new(EulerMaruyama::new()),
            0xABCDEF,
            &u0,
            &[],
            0.0,
            10.0, // ≫ 1/θ
            &cfg,
        );
        assert_eq!(ens.n_failed(), 0);
        let finals: Vec<f64> = (0..n).map(|i| ens.row(i)[0]).collect();
        let (m, v) = mean_var(&finals);

        let want_var = sigma * sigma / (2.0 * theta); // 0.125
                                                      // MC standard error of the mean ≈ √(var/n) ≈ 0.0025; use loose bounds.
        assert!((m - mu).abs() < 0.02, "OU mean {m} vs {mu}");
        assert!((v - want_var).abs() < 0.02, "OU var {v} vs {want_var}");
    }

    #[test]
    fn gbm_reproduces_its_analytic_mean_with_both_schemes() {
        // dX = μX dt + σX dW ⇒ E[X_T] = X0 e^{μT}. Both Euler–Maruyama and
        // Milstein are weakly consistent, so each ensemble mean ≈ X0 e^{μT}.
        let (mu, sigma, x0, tf) = (0.15, 0.3, 1.0, 1.0);
        let drift = gbm_drift(mu);
        let diffusion = gbm_diffusion(sigma);
        let cfg = SdeConfig::new(0.002);
        let n = 40_000;
        let u0 = vec![x0; n];
        let want_mean = x0 * (mu * tf).exp();

        for (label, kernel_factory) in [
            (
                "euler_maruyama",
                &(|| Box::new(EulerMaruyama::new()) as Box<dyn SdeKernel>)
                    as &(dyn Fn() -> Box<dyn SdeKernel> + Sync),
            ),
            (
                "milstein",
                &(|| Box::new(Milstein::new()) as Box<dyn SdeKernel>),
            ),
        ] {
            let ens = sde_ensemble_final(
                &drift,
                &diffusion,
                kernel_factory,
                0x1234,
                &u0,
                &[],
                0.0,
                tf,
                &cfg,
            );
            assert_eq!(ens.n_failed(), 0, "{label}: trajectories failed");
            let finals: Vec<f64> = (0..n).map(|i| ens.row(i)[0]).collect();
            let (m, _) = mean_var(&finals);
            // Std error of the lognormal mean here ≈ mean·√((e^{σ²T}−1)/n) ≈ 0.0016.
            assert!(
                (m - want_mean).abs() < 0.02,
                "{label}: GBM mean {m} vs {want_mean}"
            );
        }
    }

    #[test]
    fn gbm_second_moment_matches_analytic() {
        // E[X_T²] = X0² e^{(2μ+σ²)T} for GBM — a moment that, unlike the mean,
        // actually sees the noise, so it checks the diffusion term's magnitude.
        let (mu, sigma, x0, tf) = (0.1, 0.4, 1.0, 1.0);
        let drift = gbm_drift(mu);
        let diffusion = gbm_diffusion(sigma);
        let cfg = SdeConfig::new(0.002);
        let n = 60_000;
        let u0 = vec![x0; n];
        let ens = sde_ensemble_final(
            &drift,
            &diffusion,
            || Box::new(Milstein::new()),
            0x55,
            &u0,
            &[],
            0.0,
            tf,
            &cfg,
        );
        assert_eq!(ens.n_failed(), 0);
        let second = (0..n).map(|i| ens.row(i)[0].powi(2)).sum::<f64>() / n as f64;
        let want = x0 * x0 * ((2.0 * mu + sigma * sigma) * tf).exp();
        // Sampling error on the 2nd moment is larger; ~3% bound is non-flaky here.
        assert!(
            (second / want - 1.0).abs() < 0.03,
            "GBM E[X²] {second} vs {want}"
        );
    }

    #[test]
    fn seeded_ensemble_is_parallel_equals_serial() {
        // The headline guarantee: a seeded SDE ensemble gives bit-for-bit the
        // same states as a serial loop building the same per-index seeds, even
        // under a forced multi-thread pool.
        let drift = gbm_drift(0.1);
        let diffusion = gbm_diffusion(0.5);
        let cfg = SdeConfig::new(0.01);
        let base = 0xC0FFEE;
        let n = 200;
        let u0 = vec![1.0; n];

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        let parallel = pool.install(|| {
            sde_ensemble_final(
                &drift,
                &diffusion,
                || Box::new(Milstein::new()),
                base,
                &u0,
                &[],
                0.0,
                1.0,
                &cfg,
            )
        });

        for i in 0..n {
            let mut kernel = Milstein::new();
            let mut rng = SplitMix64::new(seed_for(base, i as u64));
            let serial = sde_integrate_final(
                &drift,
                &diffusion,
                &mut kernel,
                &mut rng,
                &[1.0],
                &[],
                0.0,
                1.0,
                &cfg,
            )
            .unwrap();
            assert_eq!(
                parallel.row(i)[0].to_bits(),
                serial[0].to_bits(),
                "trajectory {i}: parallel != serial"
            );
        }
    }

    #[test]
    fn same_seed_reproduces_across_runs_and_noise_decorrelates_indices() {
        let drift = ou_drift(1.0, 0.0);
        let diffusion = const_diffusion(0.4);
        let cfg = SdeConfig::new(0.01);
        let u0 = vec![1.0; 16];

        let a = sde_ensemble_final(
            &drift,
            &diffusion,
            || Box::new(EulerMaruyama::new()),
            7,
            &u0,
            &[],
            0.0,
            1.0,
            &cfg,
        );
        let b = sde_ensemble_final(
            &drift,
            &diffusion,
            || Box::new(EulerMaruyama::new()),
            7,
            &u0,
            &[],
            0.0,
            1.0,
            &cfg,
        );
        assert_eq!(a.states, b.states, "same seed must reproduce");
        // Distinct indices draw distinct noise streams ⇒ distinct trajectories.
        assert_ne!(
            a.row(0)[0].to_bits(),
            a.row(1)[0].to_bits(),
            "distinct indices gave identical draws"
        );
    }

    #[test]
    fn milstein_has_higher_strong_order_than_euler_maruyama_on_gbm() {
        // Strong (pathwise) convergence: drive each kernel over a *known* Brownian
        // path (summing the drawn increments to W_T) and compare to the exact GBM
        // X_T = X0 exp((μ − σ²/2)T + σ W_T). Halving dt should cut Milstein's error
        // ~2× (order 1.0) but Euler–Maruyama's only ~√2 (order 0.5).
        let (mu, sigma, x0, tf) = (0.2, 0.5, 1.0, 1.0);
        let drift = gbm_drift(mu);
        let diffusion = gbm_diffusion(sigma);

        // Integrate one trajectory at fixed dt with a fresh kernel, returning the
        // mean strong error over `paths` independent Brownian paths.
        #[allow(clippy::too_many_arguments)] // a self-contained test helper; the GBM params are the contract.
        fn strong_error(
            drift: &dyn Evaluator,
            diffusion: &dyn Evaluator,
            mut kernel: Box<dyn SdeKernel>,
            x0: f64,
            mu: f64,
            sigma: f64,
            tf: f64,
            dt: f64,
            paths: usize,
        ) -> f64 {
            let n_steps = (tf / dt).round() as usize;
            let mut total = 0.0;
            for path in 0..paths {
                let mut rng = SplitMix64::new(seed_for(0xBEEF, path as u64));
                let mut st = SolverState {
                    u: vec![x0],
                    t: 0.0,
                    p: vec![],
                    scratch: Vec::new(),
                };
                let mut dw = [0.0];
                let mut w_t = 0.0; // accumulated Brownian increment
                for _ in 0..n_steps {
                    fill_wiener(&mut rng, dt, &mut dw);
                    w_t += dw[0];
                    kernel.step(drift, diffusion, &mut st, &dw, dt);
                }
                let exact = x0 * ((mu - 0.5 * sigma * sigma) * tf + sigma * w_t).exp();
                total += (st.u[0] - exact).abs();
            }
            total / paths as f64
        }

        let paths = 4000;
        let (dt_coarse, dt_fine) = (0.02, 0.005); // 4× refinement

        let em_coarse = strong_error(
            &drift,
            &diffusion,
            Box::new(EulerMaruyama::new()),
            x0,
            mu,
            sigma,
            tf,
            dt_coarse,
            paths,
        );
        let em_fine = strong_error(
            &drift,
            &diffusion,
            Box::new(EulerMaruyama::new()),
            x0,
            mu,
            sigma,
            tf,
            dt_fine,
            paths,
        );
        let mil_coarse = strong_error(
            &drift,
            &diffusion,
            Box::new(Milstein::new()),
            x0,
            mu,
            sigma,
            tf,
            dt_coarse,
            paths,
        );
        let mil_fine = strong_error(
            &drift,
            &diffusion,
            Box::new(Milstein::new()),
            x0,
            mu,
            sigma,
            tf,
            dt_fine,
            paths,
        );

        // Empirical strong order = log(coarse/fine) / log(4).
        let em_order = (em_coarse / em_fine).log2() / 4f64.log2();
        let mil_order = (mil_coarse / mil_fine).log2() / 4f64.log2();

        // Generous bands (4000 paths leaves Monte-Carlo scatter): EM near 0.5,
        // Milstein near 1.0, and Milstein strictly the more accurate at fine dt.
        assert!(
            (0.3..0.75).contains(&em_order),
            "Euler–Maruyama strong order {em_order} (errors {em_coarse}, {em_fine})"
        );
        assert!(
            mil_order > 0.85,
            "Milstein strong order {mil_order} (errors {mil_coarse}, {mil_fine})"
        );
        assert!(
            mil_fine < em_fine,
            "Milstein ({mil_fine}) should beat Euler–Maruyama ({em_fine}) at fine dt"
        );
    }

    #[test]
    fn grid_first_row_is_the_initial_condition_and_lands_on_times() {
        let drift = ou_drift(1.0, 0.0);
        let diffusion = const_diffusion(0.3);
        let mut kernel = EulerMaruyama::new();
        let mut rng = SplitMix64::new(1);
        let cfg = SdeConfig::new(0.01);
        let t_eval = [0.0, 0.5, 1.0, 2.0];
        let out = sde_integrate_grid(
            &drift,
            &diffusion,
            &mut kernel,
            &mut rng,
            &[0.7],
            &[],
            &t_eval,
            &cfg,
        )
        .unwrap();
        assert_eq!(out.len(), 4);
        assert_eq!(out[0], 0.7, "first grid row must be the initial condition");
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn divergence_is_reported_not_silently_returned() {
        // f(x) = x², g = 0 ⇒ deterministic finite-time blow-up past t = 1 from
        // x0 = 1 (the noise is off, so this isolates the divergence reporting).
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let f = b.mul(x, x);
        let drift = VmEval::new(Interpreter::new(b.finish(&[f], &[], 1, 0).unwrap()));
        let diffusion = const_diffusion(0.0);

        let mut kernel = EulerMaruyama::new();
        let mut rng = SplitMix64::new(0);
        let cfg = SdeConfig::new(0.01);
        let err = sde_integrate_final(
            &drift,
            &diffusion,
            &mut kernel,
            &mut rng,
            &[1.0],
            &[],
            0.0,
            2.0,
            &cfg,
        )
        .unwrap_err();
        assert!(matches!(err, SdeError::NonFinite { .. }), "got {err:?}");
    }

    #[test]
    fn ensemble_isolates_a_diverged_trajectory() {
        // x ← drift x² with g = 0: x0 = 1 blows up before t = 2; x0 = -1 decays.
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let f = b.mul(x, x);
        let drift = VmEval::new(Interpreter::new(b.finish(&[f], &[], 1, 0).unwrap()));
        let diffusion = const_diffusion(0.0);
        let cfg = SdeConfig::new(0.01);
        let u0 = [1.0, -1.0];
        let ens = sde_ensemble_final(
            &drift,
            &diffusion,
            || Box::new(EulerMaruyama::new()),
            0,
            &u0,
            &[],
            0.0,
            2.0,
            &cfg,
        );
        assert!(matches!(ens.status[0], SdeTrajStatus::Failed(_)));
        assert!(ens.row(0)[0].is_nan(), "failed row must be NaN");
        assert!(ens.status[1].is_ok());
        assert!(ens.row(1)[0].is_finite());
        assert_eq!(ens.n_failed(), 1);
    }

    #[test]
    fn problem_facade_carries_drift_diffusion_and_params() {
        let drift = ou_drift(1.5, 1.0);
        let diffusion = const_diffusion(0.2);
        let prob = SdeProblem::new(&drift, &diffusion, vec![]);
        assert_eq!(prob.dim(), 1);
        assert!(prob.params().is_empty());

        let mut kernel = EulerMaruyama::new();
        let mut rng = SplitMix64::new(99);
        let cfg = SdeConfig::new(0.001);
        // Mean-reverting to μ = 1 from x0 = 1 with tiny noise stays near 1.
        let uf = prob
            .integrate_final(&mut kernel, &mut rng, &[1.0], 0.0, 1.0, &cfg)
            .unwrap();
        assert!((uf[0] - 1.0).abs() < 0.1, "stayed near the mean: {}", uf[0]);
    }

    #[test]
    fn zero_span_returns_initial_condition() {
        let drift = gbm_drift(0.1);
        let diffusion = gbm_diffusion(0.3);
        let mut kernel = EulerMaruyama::new();
        let mut rng = SplitMix64::new(3);
        let cfg = SdeConfig::new(0.01);
        let got = sde_integrate_final(
            &drift,
            &diffusion,
            &mut kernel,
            &mut rng,
            &[2.0],
            &[],
            1.0,
            1.0,
            &cfg,
        )
        .unwrap();
        assert_eq!(got, vec![2.0]);
    }

    #[test]
    fn empty_batch_is_handled() {
        let drift = gbm_drift(0.1);
        let diffusion = gbm_diffusion(0.3);
        let cfg = SdeConfig::new(0.01);
        let ens = sde_ensemble_final(
            &drift,
            &diffusion,
            || Box::new(EulerMaruyama::new()),
            0,
            &[],
            &[],
            0.0,
            1.0,
            &cfg,
        );
        assert_eq!(ens.n_ic(), 0);
        assert!(ens.states.is_empty());
    }
}
