//! The single-trajectory integrate loop — the step / accept / retry / fail
//! driver every family runs on (ROADMAP §4c).
//!
//! A [`Solver`] knows how to take *one* step; this module turns that into a full
//! integration: it caps each step so the trajectory lands exactly on the next
//! requested output time, carries the adaptive step size across output segments,
//! retries rejected steps, and — the v2 contract — **raises rather than return
//! silent garbage** when the right-hand side blows up ([`IntegrateError`]).
//!
//! # Output: interpolated where the kernel can, stepped-to otherwise
//!
//! [`integrate_grid`] has **two** output modes, selected by
//! [`IntegrateConfig::dense`] *and* the kernel's
//! [`Caps::dense`](tsdyn_solvers::Caps::dense) flag:
//!
//! - **Dense** (`cfg.dense && solver.caps().dense && t_eval.len() > 2`): the
//!   adaptive controller steps freely, and each strictly-interior output sample
//!   is produced by the kernel's own continuous extension
//!   ([`Solver::interpolate`]). The march still lands *exactly* on
//!   `t_eval.last()`, so the final row is always the integrated state.
//! - **Landing** (everything else): each step is capped so the trajectory lands
//!   exactly on the next requested time. This needs nothing beyond `step` and is
//!   correct for every kernel — it is what a kernel with no continuous extension
//!   keeps doing, bit-for-bit.
//!
//! There is deliberately **no universal cubic-Hermite fallback** in this module:
//! measured against the native interpolants, an endpoint Hermite extension is
//! 13–384× worse for the order-5 kernels (and orders of magnitude worse for an
//! order-8 one), so a Hermite floor would trade "the answer depends on the output
//! grid" for "the answer is far worse than the stepper computed".
//! [`Caps::dense`](tsdyn_solvers::Caps::dense) therefore gates *whether dense
//! output happens at all*, not merely which interpolant is used.
//!
//! [`Solver::interpolate`]: tsdyn_solvers::Solver::interpolate
//!
//! # Forward integration
//!
//! The engine integrates forward (`t1 ≥ t0`), matching the Wiener substrate
//! ([`crate::rng`]) and the DDE method-of-steps to come. A non-increasing span
//! is a no-op (debug builds assert it).

use tsdyn_ir::Evaluator;
use tsdyn_solvers::{Solver, SolverState, StepOutcome};

use crate::alloc::{try_zeroed, AllocFailed};
use crate::interrupt::Poller;

/// Default per-segment cap on solver steps — a backstop against a kernel that
/// never makes progress (e.g. rejects forever). Large enough never to bite a
/// well-behaved integration: a fixed 1e-6 step covers a span of 100 in 1e8
/// steps.
pub const DEFAULT_MAX_STEPS: usize = 100_000_000;

/// State magnitude at which a trajectory is declared to have escaped, whatever
/// the kernel still thinks about its error estimate.
///
/// Waiting for an *actual* `inf` is a bad backstop: the state has to climb every
/// remaining decade to `f64::MAX ≈ 1.8e308` first, and an adaptive controller
/// shrinks the step as it goes, so a blow-up that is obvious by `1e150` can cost
/// tens of seconds (or grind all the way to the step cap) before it is finally
/// reported. Checking magnitude instead catches it in the step that crosses the
/// threshold.
///
/// `1e150` is deliberately an *overflow* scale, not a tuned physical one: it is
/// picked as `√f64::MAX` so that the very next squaring — the fastest growth an
/// ODE right-hand side can plausibly produce — is what actually overflows. No
/// system whose state reaches `1e150` is going to come back, and none of the
/// catalogue's legitimately large-amplitude systems come within a hundred orders
/// of magnitude of it, so the guard cannot false-positive on real dynamics.
pub const OVERFLOW_SCALE: f64 = 1e150;

/// Knobs for the integrate loop, shared by single and ensemble paths.
///
/// Build with [`IntegrateConfig::new`] (which sets safe defaults for everything
/// but the first step) and refine with the chained setters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntegrateConfig {
    /// The initial step size. For a fixed-step kernel this *is* the step; for an
    /// adaptive kernel it is the first trial, then the kernel adapts. Must be
    /// finite and `> 0`.
    pub first_step: f64,
    /// Adaptive step floor: a *rejected* step whose suggested retry size falls
    /// below this aborts the run with [`IntegrateError::StepCollapsed`] instead
    /// of grinding to a halt. `0.0` (the default) disables the floor, leaving
    /// [`max_steps`](IntegrateConfig::max_steps) as the only backstop.
    ///
    /// **No production path sets this today**: the PyO3 bridge never calls
    /// [`with_min_step`](IntegrateConfig::with_min_step), so every FFI-driven run
    /// uses `0.0`. That is a *knob that was never wired*, not a dead variant —
    /// [`IntegrateError::StepCollapsed`] still fires on a non-finite or
    /// non-positive suggested retry size (an underflowed `h` after a long run of
    /// rejections), which the `|| h_next < cfg.min_step` disjunct is independent
    /// of. Exposing it needs its own API decision (what default floor? relative
    /// to what time scale?), so it is deliberately left unexposed rather than
    /// given an arbitrary value here.
    pub min_step: f64,
    /// Upper bound on any single solver step. `f64::INFINITY` (the default)
    /// means no ceiling — the adaptive controller chooses freely. Must be
    /// finite-or-infinite and `> 0`.
    ///
    /// This is a step **size**; [`max_steps`](IntegrateConfig::max_steps) beside
    /// it is a step **count**. They are one character apart and carry opposite
    /// units — the naming follows SciPy's `solve_ivp(max_step=)` precedent
    /// deliberately, so keep them documented adjacently.
    ///
    /// The ceiling clamps only the *trial* step, never the stored natural `h`
    /// (matching the "a forced short step must not shrink `h`" convention), and
    /// `h.min(f64::INFINITY) == h` bit-for-bit for every finite `h` — so the
    /// default is provably inert.
    pub max_step: f64,
    /// Cap on solver steps **per output segment** (see [`DEFAULT_MAX_STEPS`]).
    /// [`integrate_grid`] applies it to each consecutive `t_eval` interval
    /// independently, so an `N`-point grid permits up to `N · max_steps` steps in
    /// total — the guard bounds work *within* a segment (catching a kernel that
    /// stalls between two output times), not across the whole run.
    pub max_steps: usize,
    /// Emit strictly-interior output samples by *interpolation* instead of by
    /// forcing the step to land on them.
    ///
    /// Honoured only when the kernel also reports
    /// [`Caps::dense`](tsdyn_solvers::Caps::dense) and the grid has more than two
    /// points; see [`integrate_grid_polled`]. `false` (the default) reproduces
    /// the land-on-every-sample behaviour bit-for-bit.
    pub dense: bool,
}

impl IntegrateConfig {
    /// A config with the given first step and default guards
    /// (`min_step = 0`, `max_step = ∞`, `max_steps = `[`DEFAULT_MAX_STEPS`],
    /// `dense = false`).
    pub fn new(first_step: f64) -> Self {
        IntegrateConfig {
            first_step,
            min_step: 0.0,
            max_step: f64::INFINITY,
            max_steps: DEFAULT_MAX_STEPS,
            dense: false,
        }
    }

    /// Set the per-step size ceiling (see [`max_step`](IntegrateConfig::max_step)).
    pub fn with_max_step(mut self, max_step: f64) -> Self {
        self.max_step = max_step;
        self
    }

    /// Request interpolated interior output (see [`dense`](IntegrateConfig::dense)).
    pub fn with_dense(mut self, dense: bool) -> Self {
        self.dense = dense;
        self
    }

    /// Set the adaptive step floor (see [`min_step`](IntegrateConfig::min_step)).
    pub fn with_min_step(mut self, min_step: f64) -> Self {
        self.min_step = min_step;
        self
    }

    /// Set the per-segment step cap (see [`max_steps`](IntegrateConfig::max_steps)).
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }
}

/// Why an integration stopped short of its target time.
///
/// Every variant carries the time at which the trouble was detected so a caller
/// (or the ensemble layer, which turns these into per-trajectory status) can
/// report *where* a trajectory failed. The unifying contract: a diverging
/// trajectory surfaces as one of these, never as plausible-looking numbers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IntegrateError {
    /// The state (or time) went non-finite — the right-hand side diverged, or
    /// the kernel reported [`StepOutcome::Failed`]. Carries the last good time.
    NonFinite {
        /// The integration time at which non-finiteness was detected.
        t: f64,
    },
    /// The adaptive step size collapsed without the kernel accepting — the
    /// dynamics are too stiff/singular here for this kernel and tolerance.
    ///
    /// Fires when a rejected step's suggested retry size is non-finite or
    /// non-positive (`h` underflowed after a long run of rejections), **or** when
    /// it falls below [`IntegrateConfig::min_step`]. The latter half is currently
    /// unreachable from the FFI, which never sets a floor — see
    /// [`min_step`](IntegrateConfig::min_step).
    StepCollapsed {
        /// Time at which the step collapsed.
        t: f64,
        /// The (rejected) step size that tripped the floor.
        h: f64,
    },
    /// The per-segment step cap ([`IntegrateConfig::max_steps`]) was hit before
    /// reaching the target time, with the state still perfectly finite.
    ///
    /// This is **not** a divergence and must not be reported as one: the model
    /// is fine, the *budget* ran out. It means the kernel is taking steps far
    /// smaller than the span needs — normally an explicit method on a stiff
    /// problem, or a tolerance tighter than the dynamics can meet. The caller
    /// fixes it by changing a knob (a looser `rtol`/`atol`, an implicit method,
    /// a step floor that fails fast), not by fixing their equations.
    StepLimit {
        /// Time reached when the cap was hit.
        t: f64,
        /// The cap that was hit.
        steps: usize,
    },
    /// The state's magnitude crossed [`OVERFLOW_SCALE`] — the trajectory is
    /// escaping, caught before it reaches an actual `inf`.
    Escaped {
        /// The integration time at which the escape was detected.
        t: f64,
        /// The offending component's magnitude.
        magnitude: f64,
    },
    /// The embedder's interrupt hook asked the run to stop (see
    /// [`crate::interrupt`]) — normally a Ctrl-C at the Python prompt.
    ///
    /// Carries no diagnosis of its own: the interrupting condition belongs to
    /// the embedder, which re-raises it (a `KeyboardInterrupt`) at the FFI seam.
    Interrupted {
        /// The integration time reached when the interrupt was observed.
        t: f64,
    },
    /// The output buffer could not be allocated — see [`AllocFailed`].
    AllocFailed(AllocFailed),
}

impl core::fmt::Display for IntegrateError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            IntegrateError::NonFinite { t } => {
                write!(f, "non-finite state at t = {t} (the RHS diverged)")
            }
            IntegrateError::StepCollapsed { t, h } => {
                write!(f, "step size collapsed to {h} at t = {t}")
            }
            IntegrateError::StepLimit { t, steps } => {
                write!(f, "hit the {steps}-step limit at t = {t}")
            }
            IntegrateError::Escaped { t, magnitude } => {
                write!(
                    f,
                    "state magnitude reached {magnitude:e} at t = {t} (the RHS is diverging)"
                )
            }
            IntegrateError::Interrupted { t } => {
                write!(f, "interrupted at t = {t}")
            }
            IntegrateError::AllocFailed(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for IntegrateError {}

/// Say *which* escape the step loop's single-comparison guard just caught.
///
/// Split out and marked `#[cold]` so the hot loop keeps only the one comparison
/// per component: a run that never blows up never executes any of this.
/// Shared with [`crate::event`], whose two step loops run the identical guard —
/// a Poincaré march must not be the one path where a blow-up still costs every
/// decade up to `inf`.
#[cold]
#[inline(never)]
pub(crate) fn classify_escape(u: &[f64], t: f64) -> IntegrateError {
    if !t.is_finite() || u.iter().any(|x| !x.is_finite()) {
        // An actual `inf`/`NaN`: the right-hand side has already overflowed.
        return IntegrateError::NonFinite { t };
    }
    // Still finite, but past the overflow scale — escaping, caught early.
    let magnitude = u.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    IntegrateError::Escaped { t, magnitude }
}

/// Decide this step's *trial* size from the kernel's natural step `h`, the
/// configured [`max_step`](IntegrateConfig::max_step) ceiling and the `remaining`
/// span to the target time.
///
/// Returns `(landing, h_try)`: `landing` says the trial reaches (or passes) the
/// target, so the caller snaps the time afterwards. The ceiling clamps only the
/// trial — the stored natural `h` is untouched, matching the "a forced short step
/// must not shrink `h`" convention every march in the engine follows.
///
/// Written once and shared by all three step marches ([`advance_to`] here and
/// both [`crate::event`] loops): they were four near-copies of the same two
/// lines, which is exactly how one of them gets missed.
///
/// `max_step = f64::INFINITY` (the default) is provably inert: `h` is asserted
/// finite and positive on entry and only ever reassigned from a checked-finite
/// `h_next`, so `NaN` can never reach the `min`, and `h.min(INFINITY) == h`
/// bit-for-bit for every finite `h`.
#[inline]
pub(crate) fn trial_step(h: f64, max_step: f64, remaining: f64) -> (bool, f64) {
    let pre = h.min(max_step);
    let landing = pre >= remaining;
    (landing, if landing { remaining } else { pre })
}

/// Advance `st` from its current time to `t_end`, threading the running step
/// size `h` so an adaptive kernel keeps its learned step across calls (the grid
/// loop reuses this between output points).
///
/// `h` is the kernel's *natural* step; each individual step is additionally
/// capped so the trajectory lands exactly on `t_end`, but that cap never shrinks
/// `h` itself — only an accepted larger step or a rejection updates it.
///
/// `poll` is threaded in rather than created here because the caller may invoke
/// this once per *output point*: on a dense grid a segment is often a single
/// step, so a poller scoped to one call would reset before it ever reached
/// [`crate::interrupt::POLL_STRIDE`] and the run — the very long run a user
/// wants to Ctrl-C — would never check for signals at all.
fn advance_to(
    ev: &dyn Evaluator,
    solver: &mut dyn Solver,
    st: &mut SolverState,
    h: &mut f64,
    t_end: f64,
    cfg: &IntegrateConfig,
    poll: &mut Poller,
) -> Result<(), IntegrateError> {
    // A hard assert (not debug-only): a non-positive or non-finite first step is
    // caller error that would otherwise spin to the step limit (h = 0) or step
    // the wrong way (h < 0) instead of failing cleanly.
    assert!(
        h.is_finite() && *h > 0.0,
        "first step must be finite and positive, got {h}"
    );
    let mut steps = 0usize;
    while st.t < t_end {
        if steps >= cfg.max_steps {
            return Err(IntegrateError::StepLimit { t: st.t, steps });
        }
        if poll.tick() {
            return Err(IntegrateError::Interrupted { t: st.t });
        }
        let remaining = t_end - st.t;
        // This step is "landing" when the (ceiling-clamped) step would reach or
        // pass t_end: we then cap it to `remaining` and snap the time afterwards.
        let (landing, h_try) = trial_step(*h, cfg.max_step, remaining);
        steps += 1;

        match solver.step(ev, st, h_try) {
            StepOutcome::Accepted { h_next } => {
                // ONE comparison per component, exactly what the old
                // finiteness-only check cost. `!(|x| < OVERFLOW_SCALE)` is true
                // for `NaN` (every NaN comparison is false), for `±inf`, and
                // for a state that has merely grown past the overflow scale —
                // so the escape guard rides along for free and the branch that
                // tells the three cases apart stays on the cold path.
                if !st.u.iter().all(|x| x.abs() < OVERFLOW_SCALE) || !st.t.is_finite() {
                    return Err(classify_escape(&st.u, st.t));
                }
                if landing {
                    // The kernel advanced by `remaining` (up to rounding); pin
                    // the time to the target so grid points never drift, and
                    // leave the natural step `h` untouched (a forced short step
                    // must not shrink it).
                    st.t = t_end;
                } else if h_next.is_finite() && h_next > 0.0 {
                    *h = h_next;
                }
            }
            StepOutcome::Rejected { h_next } => {
                // State is unchanged (the kernel's contract). Adopt the smaller
                // retry size, unless it has collapsed below the floor.
                if !(h_next.is_finite() && h_next > 0.0) || h_next < cfg.min_step {
                    return Err(IntegrateError::StepCollapsed { t: st.t, h: h_next });
                }
                *h = h_next;
            }
            StepOutcome::Failed => return Err(IntegrateError::NonFinite { t: st.t }),
        }
    }
    Ok(())
}

/// Integrate from `t0` to `t1`, returning the final state.
///
/// `u0`/`p` are copied into a fresh [`SolverState`]; `solver` is stepped until
/// the trajectory reaches `t1`. Returns the `dim`-length final state, or an
/// [`IntegrateError`] if the trajectory diverged or stalled. Forward only
/// (`t1 ≥ t0`); a non-increasing span returns `u0` unchanged.
pub fn integrate_final(
    ev: &dyn Evaluator,
    solver: &mut dyn Solver,
    u0: &[f64],
    p: &[f64],
    t0: f64,
    t1: f64,
    cfg: &IntegrateConfig,
) -> Result<Vec<f64>, IntegrateError> {
    debug_assert_eq!(
        u0.len(),
        ev.dim(),
        "u0 length must equal the system dimension"
    );
    debug_assert_eq!(p.len(), ev.n_param(), "p length must equal n_param");
    debug_assert!(t1 >= t0, "integration is forward only: need t1 >= t0");
    let mut st = SolverState::for_evaluator(ev, u0.to_vec(), t0, p.to_vec());
    let mut h = cfg.first_step;
    let mut poll = Poller::new();
    advance_to(ev, solver, &mut st, &mut h, t1, cfg, &mut poll)?;
    Ok(st.u)
}

/// Integrate through the non-decreasing times `t_eval`, recording the state at
/// each into a flat row-major `(t_eval.len(), dim)` buffer.
///
/// `u0` is the state at `t_eval[0]` (so the first output row is `u0`), matching
/// the usual dense-trajectory convention; the integration then steps from each
/// time to the next, landing exactly on each. The adaptive step size is carried
/// across segments, so a long grid costs no more than the same span integrated
/// in one shot. Returns an [`IntegrateError`] if any segment diverges or stalls.
pub fn integrate_grid(
    ev: &dyn Evaluator,
    solver: &mut dyn Solver,
    u0: &[f64],
    p: &[f64],
    t_eval: &[f64],
    cfg: &IntegrateConfig,
) -> Result<Vec<f64>, IntegrateError> {
    // ONE poller for the whole grid: see `advance_to`. A dense grid calls
    // `advance_to` once per output point, often for a single step, so a
    // per-segment poller would never reach a stride.
    let mut poll = Poller::new();
    integrate_grid_polled(ev, solver, u0, p, t_eval, cfg, &mut poll)
}

/// [`integrate_grid`] with the interrupt poller supplied by the caller.
///
/// For the callers that drive *many* short grids in a loop of their own — the
/// Lyapunov chunk loop, the basin cell march — each of which is a handful of
/// solver steps. A poller created per grid would reset before it ever reached a
/// stride, so those loops (the longest-running calls the engine has) would never
/// check for signals at all. Passing one poller down instead makes the polling
/// cadence what it is everywhere else: one check per
/// [`crate::interrupt::POLL_STRIDE`] *solver steps*, however the caller happens
/// to have chopped the span up.
#[allow(clippy::too_many_arguments)]
pub fn integrate_grid_polled(
    ev: &dyn Evaluator,
    solver: &mut dyn Solver,
    u0: &[f64],
    p: &[f64],
    t_eval: &[f64],
    cfg: &IntegrateConfig,
    poll: &mut Poller,
) -> Result<Vec<f64>, IntegrateError> {
    debug_assert_eq!(
        u0.len(),
        ev.dim(),
        "u0 length must equal the system dimension"
    );
    debug_assert_eq!(p.len(), ev.n_param(), "p length must equal n_param");
    let dim = ev.dim();
    // Checked: `t_eval.len() * dim` is caller-sized and can both overflow
    // `usize` and outrun the machine (see `crate::alloc`).
    let mut out = try_zeroed(t_eval.len(), dim).map_err(IntegrateError::AllocFailed)?;
    if t_eval.is_empty() {
        return Ok(out);
    }
    // Dense output requires all three: the caller asked for it, the kernel has a
    // native continuous extension, and the grid has a strictly-interior point to
    // interpolate. A 2-node grid has none — it lands on its end — so every
    // 2-node caller (the Lyapunov chunk loop, the basin cell march, the resumable
    // stepper's `advance`) is bit-for-bit unchanged *by construction*, with no
    // opt-out flag needed. `two_node_grid_is_bit_identical_with_and_without_dense`
    // pins that.
    if cfg.dense && solver.caps().dense && t_eval.len() > 2 {
        return dense_grid(ev, solver, u0, p, t_eval, cfg, poll, out, dim);
    }

    let mut st = SolverState::for_evaluator(ev, u0.to_vec(), t_eval[0], p.to_vec());
    let mut h = cfg.first_step;
    for (k, (chunk, &target)) in out.chunks_mut(dim).zip(t_eval).enumerate() {
        if k > 0 {
            debug_assert!(target >= st.t, "t_eval must be non-decreasing");
            advance_to(ev, solver, &mut st, &mut h, target, cfg, poll)?;
        }
        chunk.copy_from_slice(&st.u);
    }
    Ok(out)
}

/// The dense-output march: step freely to `t_eval.last()`, emitting every
/// strictly-interior output sample from the kernel's continuous extension.
///
/// Three rules are load-bearing; changing any of them breaks a documented
/// contract, and each has a test that says so:
///
/// 1. **Land exactly on `t_eval.last()`.** The march's target is the final grid
///    time and the landing branch snaps `st.t` to it, so the last row is always
///    the *integrated* state — never an interpolant — and bit-identical to
///    [`integrate_final`] over the same span (`dense_grid_final_row_equals_integrate_final`).
/// 2. **Interpolate strictly-interior points only.** A sample that coincides with
///    a step endpoint is copied from the state, not interpolated. Combined with
///    the `len() > 2` gate above, this is what makes every 2-node caller
///    bit-for-bit unchanged. Do **not** "simplify" this to overshoot-and-
///    interpolate-the-end.
/// 3. **Emit before stepping.** `Solver::interpolate` reads the kernel's stage
///    buffers, which the *next* `step` overwrites — accepted or rejected. Every
///    sample a step covers is therefore emitted before the loop iterates.
/// 4. **Prepare at most once, and only when needed.** `Solver::prepare_dense` is
///    where a kernel with an extra-stage interpolant (`dop853`) spends its extra
///    RHS evaluations, so it is called once per accepted step and only when that
///    step actually covers a strictly-interior sample.
#[allow(clippy::too_many_arguments)]
fn dense_grid(
    ev: &dyn Evaluator,
    solver: &mut dyn Solver,
    u0: &[f64],
    p: &[f64],
    t_eval: &[f64],
    cfg: &IntegrateConfig,
    poll: &mut Poller,
    mut out: Vec<f64>,
    dim: usize,
) -> Result<Vec<f64>, IntegrateError> {
    assert!(
        cfg.first_step.is_finite() && cfg.first_step > 0.0,
        "first step must be finite and positive, got {}",
        cfg.first_step
    );
    let last = t_eval.len() - 1;
    let t_end = t_eval[last];
    let mut st = SolverState::for_evaluator(ev, u0.to_vec(), t_eval[0], p.to_vec());
    let mut h = cfg.first_step;
    let mut u0_local = vec![0.0; dim];

    // Row 0 is the initial condition; a degenerate grid may repeat `t_eval[0]`
    // (or, for a zero-length span, be entirely at it), so copy the start state
    // into every leading row at or before it before the march begins.
    let mut k = 0usize;
    while k <= last && t_eval[k] <= st.t {
        out[k * dim..(k + 1) * dim].copy_from_slice(&st.u);
        k += 1;
    }

    let mut steps = 0usize;
    while st.t < t_end {
        if steps >= cfg.max_steps {
            return Err(IntegrateError::StepLimit { t: st.t, steps });
        }
        if poll.tick() {
            return Err(IntegrateError::Interrupted { t: st.t });
        }
        let t0_local = st.t;
        u0_local.copy_from_slice(&st.u);
        let remaining = t_end - st.t;
        let (landing, h_try) = trial_step(h, cfg.max_step, remaining);
        steps += 1;

        match solver.step(ev, &mut st, h_try) {
            StepOutcome::Accepted { h_next } => {
                // The identical one-comparison-per-component escape guard the
                // landing loop runs (see `advance_to`).
                if !st.u.iter().all(|x| x.abs() < OVERFLOW_SCALE) || !st.t.is_finite() {
                    return Err(classify_escape(&st.u, st.t));
                }
                if landing {
                    st.t = t_end;
                } else if h_next.is_finite() && h_next > 0.0 {
                    h = h_next;
                }
                let t1_local = st.t;
                // The interpolant spans the step the kernel actually took, which
                // is `h_try`: its stage cache is keyed to that size. This equals
                // `t1_local - t0_local` (the landing snap pins `st.t = t_end =
                // t0_local + remaining = t0_local + h_try`), but `h_try` avoids a
                // floating-point subtraction and stays exact — the same reasoning
                // `event.rs` records for its own `h_step`.
                let h_step = h_try;

                // Rule 4: a kernel whose interpolant needs extra stages builds
                // them here — once per accepted step, and ONLY when this step
                // actually covers a strictly-interior sample, so a step nobody
                // interpolates inside pays nothing. (`rk45`/`tsit5` take the
                // no-op default; `dop853` computes its four contd8 stages.)
                if k <= last
                    && t_eval[k] < t1_local
                    && !solver.prepare_dense(ev, &mut st, &u0_local, t0_local, h_step)
                {
                    debug_assert!(
                        false,
                        "kernel advertises Caps::dense but prepare_dense() returned false"
                    );
                    return Err(IntegrateError::NonFinite { t: t0_local });
                }

                // Rule 3: emit everything this step covers, before the next step.
                while k <= last && t_eval[k] <= t1_local {
                    let chunk = &mut out[k * dim..(k + 1) * dim];
                    if t_eval[k] >= t1_local {
                        // Rule 2: an exact step endpoint is the integrated state.
                        chunk.copy_from_slice(&st.u);
                    } else {
                        let theta = ((t_eval[k] - t0_local) / h_step).clamp(0.0, 1.0);
                        if !solver.interpolate(&u0_local, h_step, theta, chunk) {
                            // A kernel advertising `Caps::dense` must interpolate;
                            // a `false` here is a kernel bug, not a divergence.
                            debug_assert!(
                                false,
                                "kernel advertises Caps::dense but interpolate() returned false"
                            );
                            return Err(IntegrateError::NonFinite { t: t_eval[k] });
                        }
                        if !chunk.iter().all(|x| x.is_finite()) {
                            return Err(IntegrateError::NonFinite { t: t_eval[k] });
                        }
                    }
                    k += 1;
                    // `max_steps` keeps its documented per-output-segment meaning:
                    // the budget resets each time an output point is delivered.
                    steps = 0;
                }
            }
            StepOutcome::Rejected { h_next } => {
                if !(h_next.is_finite() && h_next > 0.0) || h_next < cfg.min_step {
                    return Err(IntegrateError::StepCollapsed { t: st.t, h: h_next });
                }
                h = h_next;
            }
            StepOutcome::Failed => return Err(IntegrateError::NonFinite { t: st.t }),
        }
    }
    // The loop exits only at `st.t >= t_end`, and the emit pass above delivers
    // every sample `<= t1_local`, so `k > last` here. (A grid whose last entries
    // duplicate `t_end` is covered by the same pass.)
    debug_assert!(k > last, "the dense march must fill every output row");
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{ConstantField, Rk4, VmEval};
    use tsdyn_ir::TapeBuilder;
    use tsdyn_vm::Interpreter;

    /// dx/dt = -k x ⇒ x(t) = x0 e^{-k t}. One parameter, one state.
    fn decay() -> Interpreter {
        let mut b = TapeBuilder::new();
        let k = b.param(0);
        let x = b.state(0);
        let kx = b.mul(k, x);
        let dx = b.neg(kx);
        Interpreter::new(b.finish(&[dx], &[], 1, 1).unwrap())
    }

    /// Undamped harmonic oscillator dx=v, dv=-x ⇒ (cos t, -sin t) from (1, 0).
    fn oscillator() -> Interpreter {
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let v = b.state(1);
        let dx = v;
        let dv = b.neg(x);
        Interpreter::new(b.finish(&[dx, dv], &[], 2, 0).unwrap())
    }

    /// dx/dt = x² ⇒ x(t) = 1/(1 - t) from x0 = 1: a finite-time blow-up at t = 1.
    fn blowup() -> Interpreter {
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let dx = b.mul(x, x);
        Interpreter::new(b.finish(&[dx], &[], 1, 0).unwrap())
    }

    #[test]
    fn exponential_decay_final_state() {
        let ev = VmEval::new(decay());
        let mut s = Rk4::new();
        let cfg = IntegrateConfig::new(0.001);
        let got = integrate_final(&ev, &mut s, &[1.0], &[2.0], 0.0, 3.0, &cfg).unwrap();
        let want = (-2.0_f64 * 3.0).exp();
        assert!((got[0] - want).abs() < 1e-9, "got {}, want {want}", got[0]);
    }

    #[test]
    fn harmonic_oscillator_on_a_grid() {
        let ev = VmEval::new(oscillator());
        let mut s = Rk4::new();
        let cfg = IntegrateConfig::new(0.005);
        let t_eval: Vec<f64> = (0..=8)
            .map(|i| i as f64 * core::f64::consts::FRAC_PI_4)
            .collect();
        let out = integrate_grid(&ev, &mut s, &[1.0, 0.0], &[], &t_eval, &cfg).unwrap();
        for (k, &t) in t_eval.iter().enumerate() {
            let (x, v) = (out[2 * k], out[2 * k + 1]);
            assert!((x - t.cos()).abs() < 1e-7, "x at t={t}: {x} vs {}", t.cos());
            assert!(
                (v + t.sin()).abs() < 1e-7,
                "v at t={t}: {v} vs {}",
                -t.sin()
            );
        }
    }

    #[test]
    fn first_grid_row_is_the_initial_condition() {
        let ev = VmEval::new(oscillator());
        let mut s = Rk4::new();
        let cfg = IntegrateConfig::new(0.01);
        let out = integrate_grid(&ev, &mut s, &[0.3, -0.7], &[], &[5.0, 6.0], &cfg).unwrap();
        assert_eq!(&out[0..2], &[0.3, -0.7]);
    }

    #[test]
    fn grid_lands_exactly_on_requested_times() {
        // A constant field x(t) = x0 + t makes the expected value exact, so any
        // time drift at the grid points would show up immediately.
        let ev = ConstantField::new(vec![1.0]);
        let mut s = Rk4::new();
        let cfg = IntegrateConfig::new(0.07); // step does not divide the spacing
        let t_eval = [0.0, 0.1, 0.2, 0.30000000001, 1.0];
        let out = integrate_grid(&ev, &mut s, &[0.0], &[], &t_eval, &cfg).unwrap();
        for (k, &t) in t_eval.iter().enumerate() {
            assert!((out[k] - t).abs() < 1e-12, "row {k}: {} vs {t}", out[k]);
        }
    }

    #[test]
    fn divergence_is_reported_not_silently_returned() {
        let ev = VmEval::new(blowup());
        let mut s = Rk4::new();
        let cfg = IntegrateConfig::new(0.01);
        // Integrate past the t = 1 singularity. Either escape guard is a correct
        // report: `Escaped` is the magnitude threshold firing on the way up (the
        // usual case, and the point of having it), `NonFinite` an actual `inf`
        // in a single step. What must never happen is a silent finite answer.
        let err = integrate_final(&ev, &mut s, &[1.0], &[], 0.0, 2.0, &cfg).unwrap_err();
        assert!(
            matches!(
                err,
                IntegrateError::NonFinite { .. } | IntegrateError::Escaped { .. }
            ),
            "got {err:?}"
        );
    }

    /// The magnitude guard must fire *before* the state reaches an actual `inf`
    /// — that is the whole point: waiting for overflow costs the caller every
    /// remaining decade of shrinking steps.
    #[test]
    fn escape_is_caught_at_the_overflow_scale_not_at_infinity() {
        let ev = VmEval::new(blowup());
        let mut s = Rk4::new();
        let cfg = IntegrateConfig::new(0.01);
        let err = integrate_final(&ev, &mut s, &[1.0], &[], 0.0, 2.0, &cfg).unwrap_err();
        let IntegrateError::Escaped { magnitude, .. } = err else {
            panic!("expected the magnitude guard to fire, got {err:?}")
        };
        assert!(
            magnitude.is_finite() && magnitude >= OVERFLOW_SCALE,
            "the guard must report a finite over-threshold magnitude, got {magnitude}"
        );
    }

    /// An armed interrupt stops the integrate loop instead of running to `t1`.
    #[test]
    fn an_armed_interrupt_stops_the_integrate_loop() {
        let _stop = crate::interrupt::testing::force_stop();
        let _armed = crate::interrupt::arm();

        let ev = VmEval::new(decay());
        let mut s = Rk4::new();
        // A tiny step over a long span, so the run needs far more than one
        // poll stride and cannot finish before the first check.
        let cfg = IntegrateConfig::new(1e-4);
        let err = integrate_final(&ev, &mut s, &[1.0], &[1.0], 0.0, 1e4, &cfg).unwrap_err();
        assert!(
            matches!(err, IntegrateError::Interrupted { .. }),
            "got {err:?}"
        );
    }

    /// An unservable output grid must be an `Err`, not the `capacity overflow`
    /// panic / allocator abort `vec![0.0; t_eval.len() * dim]` produced.
    #[test]
    fn an_unservable_output_grid_is_an_error() {
        let ev = VmEval::new(decay());
        let mut s = Rk4::new();
        let cfg = IntegrateConfig::new(1e-3);
        // A slice this long cannot be materialised, so fake the shape check via
        // `alloc::try_zeroed` directly (the same call `integrate_grid` makes).
        let err = crate::alloc::try_zeroed(usize::MAX, ev.dim().max(1))
            .expect_err("the product overflows usize");
        let lifted = IntegrateError::AllocFailed(err);
        assert!(lifted.to_string().contains("cannot allocate"));
        // And the ordinary path still works.
        let t_eval = [0.0, 0.5, 1.0];
        assert!(integrate_grid(&ev, &mut s, &[1.0], &[1.0], &t_eval, &cfg).is_ok());
    }

    /// A legitimately large-amplitude trajectory must not trip the guard. `1e150`
    /// is an overflow scale, not a physical one — a system that merely reaches
    /// `1e6` has to integrate normally.
    #[test]
    fn large_but_bounded_amplitudes_do_not_trip_the_escape_guard() {
        // dx/dt = -k x with k = -1 (i.e. growth) from x0 = 1 over t ∈ [0, 13.8]
        // reaches ~1e6 and must be returned, not rejected.
        let ev = VmEval::new(decay());
        let mut s = Rk4::new();
        let cfg = IntegrateConfig::new(1e-3);
        let uf = integrate_final(&ev, &mut s, &[1.0], &[-1.0], 0.0, 13.8, &cfg)
            .expect("a large but finite amplitude integrates");
        assert!(uf[0] > 1e5 && uf[0] < 1e7, "reached {}", uf[0]);
    }

    #[test]
    fn zero_span_returns_initial_condition() {
        let ev = VmEval::new(oscillator());
        let mut s = Rk4::new();
        let cfg = IntegrateConfig::new(0.01);
        let got = integrate_final(&ev, &mut s, &[2.0, 5.0], &[], 1.0, 1.0, &cfg).unwrap();
        assert_eq!(got, vec![2.0, 5.0]);
    }

    #[test]
    fn collapsing_step_is_reported() {
        use crate::testkit::AlwaysReject;
        let ev = VmEval::new(oscillator());
        let mut s = AlwaysReject::new(); // halves h every call, never accepts
        let cfg = IntegrateConfig::new(1e-3).with_min_step(1e-6);
        let err = integrate_final(&ev, &mut s, &[1.0, 0.0], &[], 0.0, 1.0, &cfg).unwrap_err();
        assert!(
            matches!(err, IntegrateError::StepCollapsed { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn step_limit_is_reported() {
        let ev = VmEval::new(oscillator());
        let mut s = Rk4::new();
        let cfg = IntegrateConfig::new(1e-4).with_max_steps(10);
        let err = integrate_final(&ev, &mut s, &[1.0, 0.0], &[], 0.0, 1.0, &cfg).unwrap_err();
        assert!(
            matches!(err, IntegrateError::StepLimit { steps: 10, .. }),
            "got {err:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Dense output (v6) — the blast-radius gates
    // -----------------------------------------------------------------------

    use std::sync::atomic::{AtomicUsize, Ordering};
    use tsdyn_solvers::explicit::{CashKarp, Dop853, Rk45, Tsit5};
    use tsdyn_solvers::implicit::Bdf;

    /// An [`Evaluator`] wrapper that counts RHS evaluations. `Evaluator: Sync`,
    /// so the counter is an [`AtomicUsize`] (the same shape `rk45`'s FSAL
    /// eval-count test uses).
    struct Counting<'e> {
        inner: &'e dyn Evaluator,
        evals: AtomicUsize,
    }

    impl Counting<'_> {
        fn n(&self) -> usize {
            self.evals.load(Ordering::Relaxed)
        }
    }

    impl Evaluator for Counting<'_> {
        fn dim(&self) -> usize {
            self.inner.dim()
        }
        fn n_param(&self) -> usize {
            self.inner.n_param()
        }
        fn n_scratch(&self) -> usize {
            self.inner.n_scratch()
        }
        fn has_jacobian(&self) -> bool {
            self.inner.has_jacobian()
        }
        fn eval(&self, u: &[f64], p: &[f64], t: f64, scratch: &mut [f64], deriv: &mut [f64]) {
            self.evals.fetch_add(1, Ordering::Relaxed);
            self.inner.eval(u, p, t, scratch, deriv);
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
            self.inner.eval_jac(u, p, t, scratch, deriv, jac);
        }
    }

    fn grid(t0: f64, dt: f64, n: usize) -> Vec<f64> {
        (0..n).map(|i| t0 + i as f64 * dt).collect()
    }

    /// The oscillator with its (constant) Jacobian `[[0, 1], [-1, 0]]` attached,
    /// so an implicit kernel can be driven over it.
    fn oscillator_with_jacobian() -> Interpreter {
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let v = b.state(1);
        let dx = v;
        let dv = b.neg(x);
        let zero = b.constant(0.0);
        let one = b.constant(1.0);
        let minus_one = b.constant(-1.0);
        Interpreter::new(
            b.finish(&[dx, dv], &[zero, one, minus_one, zero], 2, 0)
                .unwrap(),
        )
    }

    /// **E1 — the de-risking gate.** A two-node grid has no strictly-interior
    /// output point, so it must be **bit-for-bit** identical with and without
    /// dense output, for every `(t0, dt, rtol)`. This is what makes the three
    /// chunked two-node callers in the engine — `lyapunov::classify`,
    /// `basin`'s cell march, and the bridge's resumable `OdeStepper::advance`,
    /// all of which build `[t, tf]` — provably untouched by the dense-output
    /// change, with no opt-out flag needed.
    #[test]
    fn two_node_grid_is_bit_identical_with_and_without_dense() {
        let ev = VmEval::new(oscillator());
        let mut cases = 0usize;
        for &t0 in &[0.0, 0.37, 5.0, -2.5] {
            for &dt in &[1e-3, 0.01, 0.1, 0.5, 1.0, 3.0, 7.5] {
                for &(rtol, atol) in &[(1e-3, 1e-6), (1e-6, 1e-9), (1e-9, 1e-12), (1e-12, 1e-14)] {
                    for landing in [false, true] {
                        let t_eval = [t0, t0 + dt];
                        let first = if landing { dt } else { dt * 0.017 };
                        let base = IntegrateConfig::new(first);
                        let mut a = Rk45::with_tolerances(rtol, atol);
                        let mut b = Rk45::with_tolerances(rtol, atol);
                        let off =
                            integrate_grid(&ev, &mut a, &[1.0, 0.0], &[], &t_eval, &base).unwrap();
                        let on = integrate_grid(
                            &ev,
                            &mut b,
                            &[1.0, 0.0],
                            &[],
                            &t_eval,
                            &base.with_dense(true),
                        )
                        .unwrap();
                        for (i, (&x, &y)) in off.iter().zip(&on).enumerate() {
                            assert_eq!(
                                x.to_bits(),
                                y.to_bits(),
                                "row-major slot {i} moved at t0={t0}, dt={dt}, rtol={rtol}"
                            );
                        }
                        cases += 1;
                    }
                }
            }
        }
        assert!(cases >= 200, "only {cases} combinations swept");
    }

    /// **E2.** The last row of a dense grid is the *integrated* state, never an
    /// interpolant: it must equal `integrate_final` over the same span
    /// bit-for-bit. (Rule 1 of the dense march.)
    #[test]
    fn dense_grid_final_row_equals_integrate_final() {
        let ev = VmEval::new(oscillator());
        for &n in &[3usize, 11, 101] {
            let t_eval = grid(0.0, 4.0 / (n - 1) as f64, n);
            let cfg = IntegrateConfig::new(0.05).with_dense(true);
            let mut a = Rk45::with_tolerances(1e-9, 1e-12);
            let mut b = Rk45::with_tolerances(1e-9, 1e-12);
            let dense = integrate_grid(&ev, &mut a, &[1.0, 0.0], &[], &t_eval, &cfg).unwrap();
            let final_only = integrate_final(
                &ev,
                &mut b,
                &[1.0, 0.0],
                &[],
                0.0,
                *t_eval.last().unwrap(),
                &cfg,
            )
            .unwrap();
            for (i, (&x, &y)) in dense[dense.len() - 2..].iter().zip(&final_only).enumerate() {
                assert_eq!(x.to_bits(), y.to_bits(), "n={n}, component {i}");
            }
        }
    }

    /// **E3 — the performance gate, as a *counting* test so it cannot flake.**
    ///
    /// The whole point of dense output is that the RHS-evaluation count stops
    /// depending on the output resolution. Refining `dt` by 10× and then 100×
    /// over the same span must cost **exactly** the same number of evaluations.
    /// The criterion benches measure the same saving in wall time and so can
    /// only be advisory; this is exact and machine-independent (written in the
    /// style of `rk45`'s `fsal_reuse_saves_one_rhs_eval_per_continued_step`).
    #[test]
    fn dense_grid_rhs_evals_are_independent_of_output_dt() {
        let inner = VmEval::new(oscillator());
        let cfg = IntegrateConfig::new(0.05).with_dense(true);
        let mut counts = Vec::new();
        for &n in &[11usize, 101, 1001] {
            let ev = Counting {
                inner: &inner,
                evals: AtomicUsize::new(0),
            };
            let t_eval = grid(0.0, 10.0 / (n - 1) as f64, n);
            let mut s = Rk45::with_tolerances(1e-8, 1e-10);
            integrate_grid(&ev, &mut s, &[1.0, 0.0], &[], &t_eval, &cfg).unwrap();
            counts.push(ev.n());
        }
        assert_eq!(
            counts[0], counts[1],
            "a 10x finer output grid changed the RHS-eval count: {counts:?}"
        );
        assert_eq!(
            counts[1], counts[2],
            "a 100x finer output grid changed the RHS-eval count: {counts:?}"
        );

        // And the control: WITHOUT dense output the finest grid costs far more,
        // which is the defect this change fixes.
        let ev = Counting {
            inner: &inner,
            evals: AtomicUsize::new(0),
        };
        let t_eval = grid(0.0, 10.0 / 1000.0, 1001);
        let mut s = Rk45::with_tolerances(1e-8, 1e-10);
        integrate_grid(
            &ev,
            &mut s,
            &[1.0, 0.0],
            &[],
            &t_eval,
            &IntegrateConfig::new(0.05),
        )
        .unwrap();
        // Measured on this problem: 805 evaluations at 11, 101 *and* 1001 output
        // points with dense output, against 6001 for the landing march at 1001
        // points — a 7.5x reduction that grows without bound as the grid refines.
        assert!(
            ev.n() > 4 * counts[2],
            "the landing march should be much more expensive on a fine grid: \
             {} vs {}",
            ev.n(),
            counts[2]
        );
    }

    /// **E4.** The interpolated samples converge at the interpolant's order.
    ///
    /// Measured on the harmonic oscillator with a *fixed* step (`first_step`
    /// large enough that the controller never shrinks it at a loose tolerance is
    /// not reliable, so drive the order through `max_step` instead, which pins
    /// the step exactly). Both dense kernels carry an order-4 continuous
    /// extension, i.e. a local `O(h⁵)` interpolation error — which is what the
    /// h-halving fit measures here. (Note this is **5**, not the ≥4.9-and-call-it-
    /// order-5 confusion: the *continuous extension* is one order below the
    /// propagated solution, the standard DP5/Tsit5 trade and the same one SciPy
    /// makes.)
    #[test]
    fn dense_grid_error_order_matches_the_method() {
        let ev = VmEval::new(oscillator());
        for name in ["rk45", "tsit5", "dop853"] {
            let mut errs = Vec::new();
            let hs = [0.4_f64, 0.2, 0.1];
            for &h in &hs {
                // Sample off-step-endpoint times: the step is pinned to `h` by
                // `max_step`, and the grid spacing is an irrational-ish fraction
                // of it so every interior sample is genuinely interpolated.
                let t_eval = grid(0.0, h / std::f64::consts::PI, 40);
                let cfg = IntegrateConfig::new(h).with_max_step(h).with_dense(true);
                let mut s: Box<dyn Solver> = match name {
                    "rk45" => Box::new(Rk45::with_tolerances(1.0, 1.0)),
                    "tsit5" => Box::new(Tsit5::with_tolerances(1.0, 1.0)),
                    _ => Box::new(Dop853::with_tolerances(1.0, 1.0)),
                };
                let out = integrate_grid(&ev, &mut *s, &[1.0, 0.0], &[], &t_eval, &cfg).unwrap();
                let e = t_eval
                    .iter()
                    .enumerate()
                    .map(|(i, &t)| {
                        (out[2 * i] - t.cos())
                            .abs()
                            .max((out[2 * i + 1] + t.sin()).abs())
                    })
                    .fold(0.0_f64, f64::max);
                errs.push(e);
            }
            // `rk45`/`tsit5` carry order-4 continuous extensions (local O(h^5));
            // `dop853` carries Hairer's order-7 contd8 (local O(h^8)), which is
            // the whole reason it pays four extra RHS evaluations for it. At the
            // step sizes used here contd8 reaches ~1e-14 and the fit saturates on
            // round-off, so its floor is set at 6 rather than 8.
            let want = if name == "dop853" { 6.0 } else { 4.5 };
            let order = (errs[0] / errs[2]).ln() / (hs[0] / hs[2]).ln();
            assert!(
                order > want,
                "{name}: measured dense-output order {order:.2} (errors {errs:?}), \
                 expected > {want}"
            );
        }
    }

    /// **E5.** A kernel with no native continuous extension ignores `cfg.dense`
    /// entirely — bit-for-bit. This is what keeps the whole rest of the kernel
    /// zoo (fixed-step, non-FSAL adaptive, every implicit/stiff method) out of
    /// the blast radius.
    #[test]
    fn non_dense_kernel_ignores_cfg_dense() {
        // The Jacobian-carrying tape so `bdf` (the implicit representative) can
        // run over the same problem as the explicit ones.
        let ev = VmEval::new(oscillator_with_jacobian());
        let t_eval = grid(0.0, 0.1, 41);
        let base = IntegrateConfig::new(0.02);
        /// `(registry name, a factory for a fresh instance)` — a named kernel to
        /// drive twice over the identical grid.
        type NamedKernel = (&'static str, fn() -> Box<dyn Solver>);
        let build: [NamedKernel; 3] = [
            ("rk4", || Box::new(Rk4::new())),
            ("cashkarp", || {
                Box::new(CashKarp::with_tolerances(1e-8, 1e-10))
            }),
            ("bdf", || Box::new(Bdf::with_tolerances(1e-8, 1e-10))),
        ];
        for (name, make) in build {
            let mut a = make();
            let mut b = make();
            assert!(
                !a.caps().dense,
                "{name} unexpectedly advertises Caps::dense"
            );
            let off = integrate_grid(&ev, &mut *a, &[1.0, 0.0], &[], &t_eval, &base).unwrap();
            let on = integrate_grid(
                &ev,
                &mut *b,
                &[1.0, 0.0],
                &[],
                &t_eval,
                &base.with_dense(true),
            )
            .unwrap();
            for (i, (&x, &y)) in off.iter().zip(&on).enumerate() {
                assert_eq!(x.to_bits(), y.to_bits(), "{name}: slot {i} moved");
            }
        }
    }

    /// **E6.** The `max_step` ceiling is honoured on the integrate march, and
    /// `f64::INFINITY` (the default) is provably inert.
    ///
    /// (The two `event.rs` marches run the identical `trial_step` helper and are
    /// covered by `event::tests::max_step_bounds_the_event_march`.)
    #[test]
    fn max_step_is_honoured_and_infinity_is_inert() {
        let ev = VmEval::new(oscillator());
        let t_eval = grid(0.0, 2.0, 6);

        // Inert: the explicit infinite ceiling reproduces the default config bit
        // for bit, on both the landing and the dense march.
        for dense in [false, true] {
            let base = IntegrateConfig::new(0.05).with_dense(dense);
            let mut a = Rk45::with_tolerances(1e-8, 1e-10);
            let mut b = Rk45::with_tolerances(1e-8, 1e-10);
            let plain = integrate_grid(&ev, &mut a, &[1.0, 0.0], &[], &t_eval, &base).unwrap();
            let inf = integrate_grid(
                &ev,
                &mut b,
                &[1.0, 0.0],
                &[],
                &t_eval,
                &base.with_max_step(f64::INFINITY),
            )
            .unwrap();
            for (i, (&x, &y)) in plain.iter().zip(&inf).enumerate() {
                assert_eq!(x.to_bits(), y.to_bits(), "dense={dense}: slot {i} moved");
            }
        }

        // Honoured: capping the step forces strictly more work than the free
        // march, and (a sharper check) a ceiling well below the natural step
        // makes the eval count scale like span/max_step.
        let inner = VmEval::new(oscillator());
        let mut n = Vec::new();
        for cap in [f64::INFINITY, 0.05, 0.01] {
            let evc = Counting {
                inner: &inner,
                evals: AtomicUsize::new(0),
            };
            let mut s = Rk45::with_tolerances(1e-6, 1e-9);
            let cfg = IntegrateConfig::new(0.05)
                .with_max_step(cap)
                .with_dense(true);
            integrate_grid(&evc, &mut s, &[1.0, 0.0], &[], &t_eval, &cfg).unwrap();
            n.push(evc.n());
        }
        assert!(
            n[0] < n[1] && n[1] < n[2],
            "max_step did not bound the step: {n:?}"
        );
        // 10 time units at <= 0.01 per step needs at least 1000 steps, i.e. at
        // least ~6000 RHS evaluations for a 7-stage FSAL kernel.
        assert!(
            n[2] >= 6000,
            "a 0.01 ceiling should force ~1000 steps, got {} evals",
            n[2]
        );
    }

    /// A dense march must still report divergence loudly (never a plausible
    /// interpolated number), through the same magnitude guard the landing march
    /// runs.
    #[test]
    fn dense_march_still_diverges_loudly() {
        let ev = VmEval::new(blowup());
        let mut s = Rk45::with_tolerances(1e-6, 1e-9);
        let cfg = IntegrateConfig::new(0.01).with_dense(true);
        let t_eval = grid(0.0, 0.05, 41); // through the t = 1 singularity
        let err = integrate_grid(&ev, &mut s, &[1.0], &[], &t_eval, &cfg).unwrap_err();
        assert!(
            matches!(
                err,
                IntegrateError::NonFinite { .. } | IntegrateError::Escaped { .. }
            ),
            "got {err:?}"
        );
    }

    /// A degenerate grid (repeated leading times, a zero-length span) must fill
    /// every row on the dense path exactly as it does on the landing path.
    #[test]
    fn dense_march_fills_degenerate_grids() {
        let ev = VmEval::new(oscillator());
        let cfg = IntegrateConfig::new(0.05).with_dense(true);
        for t_eval in [
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.5, 1.0],
            vec![1.0, 1.0, 1.0, 1.0],
        ] {
            let mut a = Rk45::with_tolerances(1e-9, 1e-12);
            let mut b = Rk45::with_tolerances(1e-9, 1e-12);
            let on = integrate_grid(&ev, &mut a, &[1.0, 0.0], &[], &t_eval, &cfg).unwrap();
            let off = integrate_grid(
                &ev,
                &mut b,
                &[1.0, 0.0],
                &[],
                &t_eval,
                &IntegrateConfig::new(0.05),
            )
            .unwrap();
            assert_eq!(on.len(), t_eval.len() * 2);
            for (i, (&x, &y)) in on.iter().zip(&off).enumerate() {
                assert!(
                    (x - y).abs() < 1e-9,
                    "{t_eval:?} slot {i}: dense {x} vs landing {y}"
                );
            }
        }
    }
}
