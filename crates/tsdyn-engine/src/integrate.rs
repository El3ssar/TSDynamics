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

/// Default per-segment cap on solver steps — a backstop against a kernel that
/// never makes progress (e.g. rejects forever). Large enough never to bite a
/// well-behaved integration: a fixed 1e-6 step covers a span of 100 in 1e8
/// steps.
pub const DEFAULT_MAX_STEPS: usize = 100_000_000;

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
    /// reaching the target time.
    StepLimit {
        /// Time reached when the cap was hit.
        t: f64,
        /// The cap that was hit.
        steps: usize,
    },
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
        }
    }
}

impl std::error::Error for IntegrateError {}

/// Advance `st` from its current time to `t_end`, threading the running step
/// size `h` so an adaptive kernel keeps its learned step across calls (the grid
/// loop reuses this between output points).
///
/// `h` is the kernel's *natural* step; each individual step is additionally
/// capped so the trajectory lands exactly on `t_end`, but that cap never shrinks
/// `h` itself — only an accepted larger step or a rejection updates it.
fn advance_to(
    ev: &dyn Evaluator,
    solver: &mut dyn Solver,
    st: &mut SolverState,
    h: &mut f64,
    t_end: f64,
    cfg: &IntegrateConfig,
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
        let remaining = t_end - st.t;
        // This step is "landing" when the natural step would reach (or pass)
        // t_end: we then cap it to `remaining` and snap the time afterwards.
        let landing = *h >= remaining;
        let h_try = if landing { remaining } else { *h };
        steps += 1;

        match solver.step(ev, st, h_try) {
            StepOutcome::Accepted { h_next } => {
                if !st.u.iter().all(|x| x.is_finite()) || !st.t.is_finite() {
                    return Err(IntegrateError::NonFinite { t: st.t });
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
    advance_to(ev, solver, &mut st, &mut h, t1, cfg)?;
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
    debug_assert_eq!(
        u0.len(),
        ev.dim(),
        "u0 length must equal the system dimension"
    );
    debug_assert_eq!(p.len(), ev.n_param(), "p length must equal n_param");
    let dim = ev.dim();
    let mut out = vec![0.0; t_eval.len() * dim];
    if t_eval.is_empty() {
        return Ok(out);
    }
    let mut st = SolverState::for_evaluator(ev, u0.to_vec(), t_eval[0], p.to_vec());
    let mut h = cfg.first_step;
    for (k, (chunk, &target)) in out.chunks_mut(dim).zip(t_eval).enumerate() {
        if k > 0 {
            debug_assert!(target >= st.t, "t_eval must be non-decreasing");
            advance_to(ev, solver, &mut st, &mut h, target, cfg)?;
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
        // Integrate past the t = 1 singularity.
        let err = integrate_final(&ev, &mut s, &[1.0], &[], 0.0, 2.0, &cfg).unwrap_err();
        assert!(
            matches!(err, IntegrateError::NonFinite { .. }),
            "got {err:?}"
        );
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
