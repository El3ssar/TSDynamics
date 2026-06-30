//! The [`Solver`] trait — the second pluggability seam (with
//! [`Evaluator`](tsdyn_ir::Evaluator)) the engine hangs on (ROADMAP §3 / §4d).
//!
//! A solver is one integration kernel (a Runge–Kutta method, a Rosenbrock
//! method, Euler–Maruyama, …).  It advances an integration [`SolverState`] by
//! one step of a requested size, calling an [`Evaluator`](tsdyn_ir::Evaluator)
//! for RHS (and, for implicit kernels, Jacobian) values.  Each kernel is its own
//! module/file (`tsdyn-solvers/explicit/**`, `…/implicit/**`, `…/sde/**`) and
//! self-registers by name (see [`crate::register_solver!`]), so adding a method
//! is adding one file with no central table to edit — that is what lets streams
//! E3, E4 and E-SDE run without colliding.
//!
//! The trait is **object-safe**: the engine drives a `&mut dyn Solver` so a
//! `method=` string can select any registered kernel at run time.

use crate::caps::Caps;
use tsdyn_ir::Evaluator;

/// The evolving state of one integration, owned by the engine and threaded
/// through [`Solver::step`].
///
/// One `SolverState` belongs to one integrating worker; ensemble fan-out builds
/// one per trajectory, mirroring v2's per-worker `Workspace`.  It bundles
/// everything a step reads or advances:
///
/// - [`u`](SolverState::u) / [`t`](SolverState::t) — the current point, advanced
///   in place by an accepted step;
/// - [`p`](SolverState::p) — the parameter vector (read-only to the solver by
///   convention; the engine owns the truth);
/// - [`scratch`](SolverState::scratch) — the evaluator's working buffer, sized
///   from [`Evaluator::n_scratch`] and reused across every step with no
///   allocation.
///
/// Per-method stage buffers (the `k_i` of a Runge–Kutta method, Newton work for
/// an implicit method, the Wiener increment for an SDE) are *not* here: a kernel
/// owns those as its own `&mut self` fields, so the shared state stays minimal.
pub struct SolverState {
    /// Current state vector, length `dim`.  Advanced in place on an accepted step.
    pub u: Vec<f64>,
    /// Current independent variable (time).  Advanced in place on an accepted step.
    pub t: f64,
    /// Parameter vector, length [`Evaluator::n_param`].  Read-only by convention.
    pub p: Vec<f64>,
    /// Evaluator scratch, length [`Evaluator::n_scratch`].  Caller-owned and
    /// reused; contents between steps are unspecified.
    pub scratch: Vec<f64>,
}

impl SolverState {
    /// Build state for `ev`, sizing [`scratch`](SolverState::scratch) from
    /// [`Evaluator::n_scratch`].  `u`, `t`, `p` are the integration's start.
    pub fn for_evaluator(ev: &dyn Evaluator, u: Vec<f64>, t: f64, p: Vec<f64>) -> SolverState {
        let scratch = vec![0.0; ev.n_scratch()];
        SolverState { u, t, p, scratch }
    }

    /// System dimension (`u.len()`).
    #[inline]
    pub fn dim(&self) -> usize {
        self.u.len()
    }
}

/// The result of a single [`Solver::step`].
///
/// The engine drives the step/accept/retry loop off this:
///
/// - On [`Accepted`](StepOutcome::Accepted) the kernel has advanced
///   [`SolverState::u`]/[`t`](SolverState::t) and suggests the next step size
///   (for a fixed-step kernel, the same size it was given).
/// - On [`Rejected`](StepOutcome::Rejected) (adaptive kernels only) the state is
///   **left unchanged**; the engine retries with the suggested smaller size.
/// - On [`Failed`](StepOutcome::Failed) the RHS went non-finite or the step is
///   unrecoverable; the engine raises rather than return silent garbage (the v2
///   contract: diverging trajectories never masquerade as data).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StepOutcome {
    /// Step taken; `h_next` is the suggested size for the following step.
    Accepted {
        /// Suggested next step size (`== h` for fixed-step kernels).
        h_next: f64,
    },
    /// Step rejected by error control; state unchanged, retry with `h_next`.
    Rejected {
        /// Suggested (smaller) retry size.
        h_next: f64,
    },
    /// Unrecoverable: non-finite RHS / state.  The engine aborts the run.
    Failed,
}

/// One integration kernel.  See the [module docs](self).
///
/// Implemented in streams E3 (explicit family), E4 (implicit/stiff family) and
/// E-SDE (stochastic family); consumed as `&mut dyn Solver` by the engine (E5).
///
/// # `Send`
///
/// The [`Send`] supertrait makes `Box<dyn Solver>` movable across threads, so the
/// engine's rayon ensemble (E5) can build a kernel per worker and hand it off —
/// the natural fan-out shape.  It costs nothing: a kernel is a plain struct of
/// `Vec<f64>` stage buffers.  It is deliberately **not** `Sync` — each worker
/// owns its own `&mut` kernel, so shared access is never needed and `Sync` would
/// over-constrain (the *evaluator*, by contrast, is shared `&self` and so is
/// [`Sync`](tsdyn_ir::Evaluator)).
pub trait Solver: Send {
    /// The kernel's registry name — the string a user passes as `method=`.
    ///
    /// Must be unique across registered kernels.  Because registration is
    /// link-time (no central table), a clash between two kernels — e.g. one each
    /// from streams E3 and E4 — is **not** rejected at registration; it is caught
    /// instead by [`duplicates`](crate::duplicates) / the registry uniqueness
    /// test, which see every kernel linked into the binary.
    fn name(&self) -> &'static str;

    /// This kernel's [capabilities](Caps): explicit/implicit, adaptive,
    /// Jacobian need, and supported problem families.
    ///
    /// **Must equal the caps the kernel declares in its
    /// [`register_solver!`](crate::register_solver) line**
    /// ([`SolverRegistration::caps`](crate::SolverRegistration)). The two copies
    /// are read by different engine paths — the Jacobian / implicit-kind guards
    /// read the registered copy, while the dense-output / event path reads this
    /// method — so a drift between them is a silent correctness divergence. The
    /// `registered_caps_match_instance_caps` registry test asserts they agree
    /// for every linked kernel, so an out-of-sync edit fails CI.
    fn caps(&self) -> Caps;

    /// Advance `st` by one step of size `h`, using `ev` for RHS/Jacobian values.
    ///
    /// On [`Accepted`](StepOutcome::Accepted) the kernel has written the new
    /// point into `st.u`/`st.t`.  On [`Rejected`](StepOutcome::Rejected) it must
    /// leave `st.u`/`st.t` **exactly as it found them** so the engine can retry
    /// with a smaller `h` — this is the kernel's obligation, not enforced by the
    /// `&mut` signature, so adaptive kernels should compute the trial step into
    /// their own `&mut self` buffers and commit to `st` only once the step is
    /// accepted (the standard embedded-RK pattern).  On
    /// [`Failed`](StepOutcome::Failed) the run is aborted by the engine.
    fn step(&mut self, ev: &dyn Evaluator, st: &mut SolverState, h: f64) -> StepOutcome;

    /// Dense output: interpolate the state *within the most recently accepted
    /// step*.
    ///
    /// A kernel that carries a native continuous extension — a dense-output
    /// formula, advertised by [`Caps::dense`](crate::Caps::dense) — writes the
    /// interpolated state `u(t0 + theta·h)` for `theta ∈ [0, 1]` into `out` and
    /// returns `true`; `u0` is the state at the start of that step and `h` the
    /// (positive) step size it just took. `out` has length
    /// [`SolverState::dim`](SolverState::dim).
    ///
    /// The default returns `false` and writes nothing: the kernel has no native
    /// interpolant. The engine then falls back to a cubic-Hermite continuous
    /// extension built from the step endpoints and their derivatives — which
    /// needs only the [`Evaluator`] — so every kernel gets O(h⁴) dense output
    /// regardless, and existing kernels need no change.
    ///
    /// This is the additive event/dense-output capability of ROADMAP §13d: a new
    /// **defaulted** method behind a [`Caps`](crate::Caps) flag, never a change
    /// to [`step`](Solver::step). Calling it is only meaningful immediately after
    /// `step` returned [`Accepted`](StepOutcome::Accepted) for a kernel whose
    /// [`caps`](Solver::caps) report `dense == true`; the engine checks the flag
    /// before calling, and treats a `false` return as "use the Hermite fallback".
    fn interpolate(&self, u0: &[f64], h: f64, theta: f64, out: &mut [f64]) -> bool {
        let _ = (u0, h, theta, out);
        false
    }
}
