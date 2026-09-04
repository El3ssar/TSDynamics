//! The single-trajectory integrate loop — the step / accept / retry / fail
//! driver every family runs on (ROADMAP §4c).
//!
//! A [`Solver`] knows how to take *one* step; this module turns that into a full
//! integration: it caps each step so the trajectory lands exactly on the next
//! requested output time, carries the adaptive step size across output segments,
//! retries rejected steps, and — the v2 contract — **raises rather than return
//! silent garbage** when the right-hand side blows up ([`IntegrateError`]).
//!
//! # Output by stepping to the grid, not by interpolation
//!
//! The frozen [`Solver`] trait (stream F2) exposes only `step`; it has no dense
//! interpolation hook. So [`integrate_grid`] produces output at each requested
//! time by *limiting the step* to land on that time exactly, which needs nothing
//! beyond `step` and is correct for every kernel. (A future dense-output trait
//! extension could interpolate between native steps for efficiency; that is an
//! interface change, out of scope for E5.)
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
    pub min_step: f64,
    /// Cap on solver steps **per output segment** (see [`DEFAULT_MAX_STEPS`]).
    /// [`integrate_grid`] applies it to each consecutive `t_eval` interval
    /// independently, so an `N`-point grid permits up to `N · max_steps` steps in
    /// total — the guard bounds work *within* a segment (catching a kernel that
    /// stalls between two output times), not across the whole run.
    pub max_steps: usize,
}

impl IntegrateConfig {
    /// A config with the given first step and default guards
    /// (`min_step = 0`, `max_steps = `[`DEFAULT_MAX_STEPS`]).
    pub fn new(first_step: f64) -> Self {
        IntegrateConfig {
            first_step,
            min_step: 0.0,
            max_steps: DEFAULT_MAX_STEPS,
        }
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
    /// An adaptive kernel shrank the step below
    /// [`IntegrateConfig::min_step`] without accepting — the dynamics are too
    /// stiff/singular here for this kernel and tolerance.
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
        // This step is "landing" when the natural step would reach (or pass)
        // t_end: we then cap it to `remaining` and snap the time afterwards.
        let landing = *h >= remaining;
        let h_try = if landing { remaining } else { *h };
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
}
