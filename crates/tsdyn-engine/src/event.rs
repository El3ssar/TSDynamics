//! Event detection and dense output — the engine-level layer that finds and
//! refines zero-crossings of an event function `g(u, t) = 0` along a trajectory
//! (ROADMAP §13c, stream E-EVENT).
//!
//! Poincaré sections, recurrence and event-driven stopping all reduce to "march
//! the flow, watch a scalar `g(u, t)`, and pin the instant it crosses zero". The
//! v2 library does this in Python ([`PoincareMap`] refines crossings with a cubic
//! Hermite interpolant of the bracketing samples). This module brings that into
//! the Rust engine so the analysis layer (A-ORBIT / A-BASIN / A-RQA) can drive
//! events natively, at the same O(dt⁴) accuracy.
//!
//! # The two seams it uses
//!
//! - **The event function is a lowered expression**, evaluated through the
//!   ordinary [`Evaluator`] seam: a `g(u, t)` is a tape with a single (scalar)
//!   output, so it needs no new IR — the F1 contract already separates the input
//!   state width (`n_state`) from the output width (`outputs.len()`), and a
//!   one-output tape *is* an event channel. [`EventSpec`] holds such an
//!   evaluator alongside the crossing [`EventDirection`] and a `terminal` flag.
//! - **Dense output is a kernel capability**, advertised by
//!   [`Caps::dense`](tsdyn_solvers::Caps::dense) and supplied through
//!   [`Solver::interpolate`](tsdyn_solvers::Solver::interpolate). A kernel that
//!   carries a native continuous extension uses it; **every other kernel** (i.e.
//!   all of them today) falls back to the endpoint cubic-Hermite extension
//!   ([`HermiteStep`]), which needs only the [`Evaluator`]. Existing kernels are
//!   therefore unchanged — the capability is purely additive (ROADMAP §13d).
//!
//! # Accuracy
//!
//! Cubic-Hermite interpolation of the state across a step, using the right-hand
//! side at both endpoints, is O(h⁴) accurate — so the refined crossing time and
//! state are O(h⁴) too, matching the v2 Python refinement formula exactly. The
//! crossing within the bracket is located by a safeguarded false-position
//! ([`bracketed_root`]) to a tight abscissa tolerance, so the interpolation error
//! dominates, not the root solve.
//!
//! # What it does *not* do (matching v2)
//!
//! Detection is by **sign change at the step endpoints**, so an even number of
//! crossings inside one step is missed (the classic limitation: the march step
//! must be small enough not to skip a crossing). This mirrors v2's
//! [`PoincareMap`] exactly.
//!
//! [`PoincareMap`]: https://docs.rs/tsdynamics
//! [`Evaluator`]: tsdyn_ir::Evaluator

use tsdyn_ir::Evaluator;
use tsdyn_solvers::{Solver, SolverState, StepOutcome};

use crate::integrate::{IntegrateConfig, IntegrateError};

/// Abscissa tolerance for the in-bracket crossing solve, in the local step
/// fraction `s ∈ [0, 1]`. Tight enough that the O(h⁴) interpolation error — not
/// the root solve — bounds the crossing accuracy (matching v2's `brentq`
/// `xtol = 1e-14`).
const ROOT_XTOL: f64 = 1e-14;

/// Hard cap on root-solve iterations — a backstop only. Safeguarded
/// false-position converges on a smooth bracketed `g` in a handful of steps;
/// even pure bisection of `[0, 1]` to [`ROOT_XTOL`] needs < 50.
const ROOT_MAX_ITERS: usize = 128;

/// Which way `g` must cross zero for a crossing to count.
///
/// The predicate matches v2's [`PoincareMap`] exactly: `Rising` is
/// `g_prev < 0 ≤ g_now`, `Falling` is `g_prev > 0 ≥ g_now`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventDirection {
    /// Count crossings where `g` goes from negative to non-negative (`+1`).
    Rising,
    /// Count crossings where `g` goes from positive to non-positive (`-1`).
    Falling,
    /// Count crossings in either direction.
    Either,
}

impl EventDirection {
    /// Classify the transition `g_prev → g_now`. Returns the crossing sign
    /// (`+1` rising, `-1` falling) if it counts for this direction, else `None`.
    #[inline]
    fn crosses(self, g_prev: f64, g_now: f64) -> Option<i8> {
        let up = g_prev < 0.0 && g_now >= 0.0;
        let down = g_prev > 0.0 && g_now <= 0.0;
        match self {
            EventDirection::Rising => up.then_some(1),
            EventDirection::Falling => down.then_some(-1),
            EventDirection::Either => {
                if up {
                    Some(1)
                } else if down {
                    Some(-1)
                } else {
                    None
                }
            }
        }
    }
}

/// One event to watch along the integration.
///
/// `g` is a scalar-output [`Evaluator`] (a tape with `outputs.len() == 1`)
/// describing the event function `g(u, t)`; it shares the system's parameter
/// vector (it may declare fewer parameters, in which case it reads the leading
/// slice). A crossing of `g = 0` in the requested [`direction`](EventSpec::direction)
/// is recorded; if [`terminal`](EventSpec::terminal) the integration stops at the
/// (refined) crossing.
pub struct EventSpec<'a> {
    /// The event function `g(u, t)`: a single-output evaluator over the full
    /// system state.
    pub g: &'a dyn Evaluator,
    /// Which crossing direction counts.
    pub direction: EventDirection,
    /// Stop the integration at the first crossing of this event.
    pub terminal: bool,
}

impl<'a> EventSpec<'a> {
    /// A non-terminal event watching `g` in `direction`.
    pub fn new(g: &'a dyn Evaluator, direction: EventDirection) -> Self {
        EventSpec {
            g,
            direction,
            terminal: false,
        }
    }

    /// A terminal event: the integration stops at the first crossing.
    pub fn terminal(g: &'a dyn Evaluator, direction: EventDirection) -> Self {
        EventSpec {
            g,
            direction,
            terminal: true,
        }
    }
}

/// One refined zero-crossing.
#[derive(Clone, Debug, PartialEq)]
pub struct EventHit {
    /// Index of the crossed event within the `events` slice passed to
    /// [`integrate_events`].
    pub event: usize,
    /// Refined crossing time.
    pub t: f64,
    /// Full-dimensional state at the crossing.
    pub u: Vec<f64>,
    /// Crossing direction: `+1` rising, `-1` falling.
    pub direction: i8,
}

/// The result of an event-aware integration.
#[derive(Clone, Debug, PartialEq)]
pub struct EventOutcome {
    /// Every recorded crossing, in increasing time order.
    pub hits: Vec<EventHit>,
    /// Time the integration stopped: `t1`, or the terminal crossing time.
    pub t_final: f64,
    /// State at `t_final`.
    pub u_final: Vec<f64>,
    /// Whether a terminal event stopped the run before `t1`.
    pub terminated: bool,
}

/// A cubic-Hermite continuous extension of one integration step.
///
/// Built from the step endpoints `(u0, u1)` and the right-hand side there
/// `(f0, f1)`, it reproduces the unique cubic with those values and slopes — the
/// kernel-agnostic dense output the engine uses when a solver carries no native
/// interpolant. The formula is identical to v2's Python Poincaré refinement, so
/// the two agree to O(h⁴).
pub struct HermiteStep {
    u0: Vec<f64>,
    f0: Vec<f64>,
    u1: Vec<f64>,
    f1: Vec<f64>,
    h: f64,
}

impl HermiteStep {
    /// Build the interpolant for a step of size `h` from `(t0, u0)` to
    /// `(t0 + h, u1)`, with derivatives `f0 = f(u0, t0)` and `f1 = f(u1, t0+h)`.
    /// The endpoint slices are copied in, so the interpolant owns its data and
    /// outlives the buffers it was built from.
    pub fn new(u0: &[f64], f0: &[f64], u1: &[f64], f1: &[f64], h: f64) -> Self {
        debug_assert_eq!(u0.len(), f0.len());
        debug_assert_eq!(u0.len(), u1.len());
        debug_assert_eq!(u0.len(), f1.len());
        HermiteStep {
            u0: u0.to_vec(),
            f0: f0.to_vec(),
            u1: u1.to_vec(),
            f1: f1.to_vec(),
            h,
        }
    }

    /// The interpolated state at local fraction `s ∈ [0, 1]` (i.e. at time
    /// `t0 + s·h`), written into `out`.
    #[inline]
    pub fn eval(&self, s: f64, out: &mut [f64]) {
        let s2 = s * s;
        let s3 = s2 * s;
        // Hermite basis on [0, 1].
        let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
        let h10 = s3 - 2.0 * s2 + s;
        let h01 = -2.0 * s3 + 3.0 * s2;
        let h11 = s3 - s2;
        for (d, o) in out.iter_mut().enumerate() {
            *o = h00 * self.u0[d]
                + h10 * self.h * self.f0[d]
                + h01 * self.u1[d]
                + h11 * self.h * self.f1[d];
        }
    }
}

/// Locate a root of `f` in the bracket `[lo, hi]`, given the endpoint values
/// `flo = f(lo)`, `fhi = f(hi)` of opposite sign, to abscissa tolerance `xtol`.
///
/// A safeguarded false-position (Illinois) iteration: it keeps `[lo, hi]` a true
/// bracket throughout and applies the Illinois down-weighting that restores
/// superlinear convergence (plain false-position can stall, retaining one
/// endpoint forever). The interpolated estimate is clamped back to the interior
/// and replaced by the bisection midpoint if it would step outside, so the method
/// is globally convergent — it never escapes the bracket.
fn bracketed_root(
    mut f: impl FnMut(f64) -> f64,
    lo0: f64,
    hi0: f64,
    flo0: f64,
    fhi0: f64,
    xtol: f64,
) -> f64 {
    // Order the bracket so `lo <= hi`; carry each endpoint's function value.
    let (mut lo, mut hi, mut flo, fhi) = if lo0 <= hi0 {
        (lo0, hi0, flo0, fhi0)
    } else {
        (hi0, lo0, fhi0, flo0)
    };
    // Exact-endpoint roots: nothing to refine.
    if flo == 0.0 {
        return lo;
    }
    if fhi == 0.0 {
        return hi;
    }
    debug_assert!(flo * fhi < 0.0, "bracketed_root: endpoints must straddle 0");

    // `wlo`/`whi` are the (possibly Illinois-down-weighted) values used only in
    // the false-position estimate. The bracket invariant — `flo` and the current
    // high-end value have opposite signs — is maintained structurally, so only
    // `flo`'s sign is consulted to decide which side the root falls on.
    let mut wlo = flo;
    let mut whi = fhi;

    for _ in 0..ROOT_MAX_ITERS {
        if hi - lo <= xtol {
            break;
        }
        // False-position estimate; fall back to bisection if it leaves the
        // interior (a degenerate `whi - wlo`, or a clamp at an endpoint).
        let denom = whi - wlo;
        let mut c = if denom != 0.0 {
            (lo * whi - hi * wlo) / denom
        } else {
            0.5 * (lo + hi)
        };
        if !(c > lo && c < hi) {
            c = 0.5 * (lo + hi);
        }

        let fc = f(c);
        if fc == 0.0 {
            return c;
        }
        if (fc < 0.0) == (flo < 0.0) {
            // Same sign as the low end → root in [c, hi]; advance `lo` to `c`
            // and down-weight the retained high end (Illinois).
            lo = c;
            flo = fc;
            wlo = fc;
            whi *= 0.5;
        } else {
            // Root in [lo, c]; advance `hi` to `c` and down-weight the low end.
            hi = c;
            whi = fc;
            wlo *= 0.5;
        }
    }
    0.5 * (lo + hi)
}

/// Evaluate the scalar event function `g(u, t)` through the [`Evaluator`] seam.
///
/// `g` is a single-output evaluator; it shares the system parameter vector (it
/// may declare fewer parameters and read the leading slice). `scratch` is `g`'s
/// own register buffer (length `g.n_scratch()`).
#[inline]
fn eval_event(g: &dyn Evaluator, u: &[f64], p: &[f64], t: f64, scratch: &mut [f64]) -> f64 {
    debug_assert_eq!(g.dim(), 1, "an event function must have a single output");
    debug_assert!(
        g.n_param() <= p.len(),
        "event function declares more params than the system"
    );
    let mut out = [0.0f64];
    g.eval(u, &p[..g.n_param()], t, scratch, &mut out);
    out[0]
}

/// Integrate from `t0` to `t1`, detecting and refining crossings of each event's
/// `g(u, t) = 0`.
///
/// The solver takes its natural (adaptive) steps; after each accepted step the
/// engine checks every event for a sign change at the step endpoints, and for
/// each crossing refines the instant inside the step with the kernel's dense
/// output (native if [`Caps::dense`](tsdyn_solvers::Caps::dense), else the
/// endpoint cubic-Hermite fallback) plus a bracketed root solve. Crossings are
/// returned in time order. A `terminal` event stops the run at its (refined)
/// crossing.
///
/// Forward only (`t1 ≥ t0`). The "diverge loudly, never silently"
/// [`IntegrateError`] contract is preserved: a non-finite state, a collapsed
/// step, or the step cap surfaces as an error rather than plausible-looking data.
// The argument list mirrors `integrate_grid` (evaluator + solver + problem data
// + config) plus the `events` slice; bundling them into a struct would add a
// type without removing a parameter (cf. `Tape::from_parts`/`adaptive_step`).
#[allow(clippy::too_many_arguments)]
pub fn integrate_events(
    ev: &dyn Evaluator,
    solver: &mut dyn Solver,
    u0: &[f64],
    p: &[f64],
    t0: f64,
    t1: f64,
    events: &[EventSpec<'_>],
    cfg: &IntegrateConfig,
) -> Result<EventOutcome, IntegrateError> {
    let dim = ev.dim();
    debug_assert_eq!(u0.len(), dim, "u0 length must equal the system dimension");
    debug_assert_eq!(p.len(), ev.n_param(), "p length must equal n_param");
    debug_assert!(t1 >= t0, "integrate_events is forward only: need t1 >= t0");
    assert!(
        cfg.first_step.is_finite() && cfg.first_step > 0.0,
        "first step must be finite and positive, got {}",
        cfg.first_step
    );

    let mut st = SolverState::for_evaluator(ev, u0.to_vec(), t0, p.to_vec());
    let mut h = cfg.first_step;

    // The dense-output source is a kernel property, fixed for the whole run.
    let use_native = solver.caps().dense;

    // Per-event scratch and the previous-endpoint value of each `g`.
    let mut g_scratch: Vec<Vec<f64>> = events.iter().map(|e| vec![0.0; e.g.n_scratch()]).collect();
    let mut g_prev: Vec<f64> = events
        .iter()
        .enumerate()
        .map(|(i, e)| eval_event(e.g, &st.u, p, st.t, &mut g_scratch[i]))
        .collect();

    // Hermite endpoint derivatives, only when there is no native interpolant.
    let (mut f_prev, mut f_now) = if use_native {
        (Vec::new(), Vec::new())
    } else {
        let mut f0 = vec![0.0; dim];
        ev.eval(&st.u, &st.p, st.t, &mut st.scratch, &mut f0);
        (f0, vec![0.0; dim])
    };

    let mut hits: Vec<EventHit> = Vec::new();
    let mut pending: Vec<EventHit> = Vec::new();
    let mut u0_local = vec![0.0; dim];
    let mut ubuf = vec![0.0; dim];
    let mut steps = 0usize;

    while st.t < t1 {
        if steps >= cfg.max_steps {
            return Err(IntegrateError::StepLimit { t: st.t, steps });
        }
        let t0_local = st.t;
        u0_local.copy_from_slice(&st.u);
        let remaining = t1 - st.t;
        let landing = h >= remaining;
        let h_try = if landing { remaining } else { h };
        steps += 1;

        match solver.step(ev, &mut st, h_try) {
            StepOutcome::Accepted { h_next } => {
                if !st.u.iter().all(|x| x.is_finite()) || !st.t.is_finite() {
                    return Err(IntegrateError::NonFinite { t: st.t });
                }
                if landing {
                    // Pin the time to the target so grid points never drift, and
                    // leave the natural step `h` untouched (a forced short step
                    // must not shrink it) — mirrors the grid integrator.
                    st.t = t1;
                } else if h_next.is_finite() && h_next > 0.0 {
                    h = h_next;
                }
                let t1_local = st.t;
                // The interpolant spans the step the kernel actually took, which
                // is `h_try` — its cached stage data is keyed to that size. This
                // equals `t1_local - t0_local` (the landing snap above pins
                // `st.t = t1 = t0_local + remaining = t0_local + h_try`), but
                // `h_try` avoids a floating-point subtraction and stays exact.
                let h_step = h_try;

                if !use_native {
                    ev.eval(&st.u, &st.p, t1_local, &mut st.scratch, &mut f_now);
                }

                pending.clear();
                for (e_idx, ev_spec) in events.iter().enumerate() {
                    let g1 = eval_event(ev_spec.g, &st.u, p, t1_local, &mut g_scratch[e_idx]);
                    // `crosses` only fires when both endpoint values are finite
                    // and straddle zero (the comparisons are false for NaN), so
                    // `g0`/`g1` entering the refinement are guaranteed finite.
                    if let Some(dir) = ev_spec.direction.crosses(g_prev[e_idx], g1) {
                        let g0 = g_prev[e_idx];
                        let mut u_cross = vec![0.0; dim];
                        let s_star = if use_native {
                            let s = bracketed_root(
                                |s| {
                                    solver.interpolate(&u0_local, h_step, s, &mut ubuf);
                                    eval_event(
                                        ev_spec.g,
                                        &ubuf,
                                        p,
                                        t0_local + s * h_step,
                                        &mut g_scratch[e_idx],
                                    )
                                },
                                0.0,
                                1.0,
                                g0,
                                g1,
                                ROOT_XTOL,
                            );
                            let ok = solver.interpolate(&u0_local, h_step, s, &mut u_cross);
                            debug_assert!(
                                ok,
                                "kernel advertises Caps::dense but interpolate() returned false"
                            );
                            s
                        } else {
                            let hermite =
                                HermiteStep::new(&u0_local, &f_prev, &st.u, &f_now, h_step);
                            let s = bracketed_root(
                                |s| {
                                    hermite.eval(s, &mut ubuf);
                                    eval_event(
                                        ev_spec.g,
                                        &ubuf,
                                        p,
                                        t0_local + s * h_step,
                                        &mut g_scratch[e_idx],
                                    )
                                },
                                0.0,
                                1.0,
                                g0,
                                g1,
                                ROOT_XTOL,
                            );
                            hermite.eval(s, &mut u_cross);
                            s
                        };
                        // Diverge loudly: an event function that goes singular at an
                        // interior point (overflow in the interpolant, a pole in `g`)
                        // can poison the root solve into a non-finite crossing.
                        // Surface that as an error rather than a plausible-looking hit.
                        let t_cross = t0_local + s_star * h_step;
                        let g_cross =
                            eval_event(ev_spec.g, &u_cross, p, t_cross, &mut g_scratch[e_idx]);
                        if !(s_star.is_finite()
                            && g_cross.is_finite()
                            && u_cross.iter().all(|x| x.is_finite()))
                        {
                            return Err(IntegrateError::NonFinite { t: t0_local });
                        }
                        pending.push(EventHit {
                            event: e_idx,
                            t: t_cross,
                            u: u_cross,
                            direction: dir,
                        });
                    }
                    g_prev[e_idx] = g1;
                }

                if !use_native {
                    std::mem::swap(&mut f_prev, &mut f_now);
                }

                // Commit this step's crossings in time order; a terminal event
                // stops the run at its crossing (later same-step hits dropped).
                // Break ties so a non-terminal at the *same* refined time as a
                // terminal is recorded before the terminal stops the run (`false`
                // sorts before `true`).
                pending.sort_by(|a, b| {
                    a.t.total_cmp(&b.t)
                        .then_with(|| events[a.event].terminal.cmp(&events[b.event].terminal))
                });
                for hit in pending.drain(..) {
                    let is_terminal = events[hit.event].terminal;
                    if is_terminal {
                        let (t_final, u_final) = (hit.t, hit.u.clone());
                        hits.push(hit);
                        return Ok(EventOutcome {
                            hits,
                            t_final,
                            u_final,
                            terminated: true,
                        });
                    }
                    hits.push(hit);
                }
            }
            StepOutcome::Rejected { h_next } => {
                // State unchanged (the kernel's contract); carry-over endpoint
                // values (g_prev, f_prev) stay valid only because of it — assert
                // it in debug builds so a misbehaving kernel is caught here rather
                // than silently corrupting the next step's Hermite interpolant.
                debug_assert!(
                    st.u == u0_local && st.t == t0_local,
                    "a Rejected step must leave SolverState unchanged"
                );
                // Adopt the smaller retry, unless it has collapsed below the floor.
                if !(h_next.is_finite() && h_next > 0.0) || h_next < cfg.min_step {
                    return Err(IntegrateError::StepCollapsed { t: st.t, h: h_next });
                }
                h = h_next;
            }
            StepOutcome::Failed => return Err(IntegrateError::NonFinite { t: st.t }),
        }
    }

    Ok(EventOutcome {
        hits,
        t_final: st.t,
        u_final: st.u,
        terminated: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{Rk4, VmEval};
    use tsdyn_ir::TapeBuilder;
    use tsdyn_solvers::{Caps, ProblemKind, ProblemKinds};
    use tsdyn_vm::Interpreter;

    // --- evaluators / event functions as one-output tapes ---------------------

    /// Undamped oscillator dx = v, dv = -x ⇒ (cos t, -sin t) from (1, 0).
    fn oscillator() -> VmEval {
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let v = b.state(1);
        let dx = v;
        let dv = b.neg(x);
        VmEval::new(Interpreter::new(b.finish(&[dx, dv], &[], 2, 0).unwrap()))
    }

    /// A flow whose solution is the cubic u(t) = u0 + t³: du/dt = 3 t².
    /// RK4 integrates a degree-2 RHS in t exactly, and cubic Hermite reproduces a
    /// cubic exactly, so the refined crossing is the *analytic* root to ~1e-13.
    fn cubic_flow() -> VmEval {
        let mut b = TapeBuilder::new();
        let three = b.constant(3.0);
        let t = b.time();
        let t2 = b.mul(t, t);
        let du = b.mul(three, t2);
        VmEval::new(Interpreter::new(b.finish(&[du], &[], 1, 0).unwrap()))
    }

    /// Event g(u) = u[component] - c (a coordinate plane), as a one-output tape
    /// over a `dim`-dimensional state.
    fn plane_event(dim: usize, component: usize, c: f64) -> VmEval {
        let mut b = TapeBuilder::new();
        let x = b.state(component);
        let off = b.constant(c);
        let g = b.sub(x, off);
        VmEval::new(Interpreter::new(b.finish(&[g], &[], dim, 0).unwrap()))
    }

    fn cfg(first_step: f64) -> IntegrateConfig {
        IntegrateConfig::new(first_step)
    }

    // --- HermiteStep ----------------------------------------------------------

    #[test]
    fn hermite_reproduces_a_cubic_exactly() {
        // u(s) = 1 + 2s + 3s² + 4s³ on [0,1]; with h = 1, f = du/ds.
        let u0 = [1.0];
        let u1 = [1.0 + 2.0 + 3.0 + 4.0];
        let f0 = [2.0]; // u'(0)
        let f1 = [2.0 + 6.0 + 12.0]; // u'(1) = 2 + 6 + 12
        let herm = HermiteStep::new(&u0, &f0, &u1, &f1, 1.0);
        let mut out = [0.0];
        for &s in &[0.0, 0.13, 0.5, 0.777, 1.0] {
            herm.eval(s, &mut out);
            let want = 1.0 + 2.0 * s + 3.0 * s * s + 4.0 * s * s * s;
            assert!((out[0] - want).abs() < 1e-14, "s={s}: {} vs {want}", out[0]);
        }
    }

    // --- root finder ----------------------------------------------------------

    #[test]
    fn bracketed_root_solves_smooth_functions() {
        // A cubic with a single root in (0,1): g(s) = s³ - 0.3.
        let root = bracketed_root(|s| s * s * s - 0.3, 0.0, 1.0, -0.3, 0.7, ROOT_XTOL);
        assert!((root - 0.3_f64.cbrt()).abs() < 1e-12, "root = {root}");

        // A steep function that would stall plain false-position.
        let g = |s: f64| (10.0 * s).exp() - 2.0;
        let s_star = 2.0_f64.ln() / 10.0;
        let root = bracketed_root(g, 0.0, 1.0, g(0.0), g(1.0), ROOT_XTOL);
        assert!((root - s_star).abs() < 1e-12, "root = {root}");
    }

    #[test]
    fn bracketed_root_returns_exact_endpoint_roots() {
        assert_eq!(bracketed_root(|s| s, 0.0, 1.0, 0.0, 1.0, ROOT_XTOL), 0.0);
        assert_eq!(
            bracketed_root(|s| s - 1.0, 0.0, 1.0, -1.0, 0.0, ROOT_XTOL),
            1.0
        );
    }

    // --- event detection ------------------------------------------------------

    #[test]
    fn detects_falling_crossing_of_the_oscillator() {
        // x(t) = cos t falls through 0 at t = π/2, with v = -sin(π/2) = -1.
        let ev = oscillator();
        let g = plane_event(2, 0, 0.0);
        let mut s = Rk4::new();
        let events = [EventSpec::new(&g, EventDirection::Falling)];
        let out =
            integrate_events(&ev, &mut s, &[1.0, 0.0], &[], 0.0, 3.0, &events, &cfg(0.01)).unwrap();
        assert_eq!(out.hits.len(), 1, "exactly one falling crossing in [0,3]");
        let hit = &out.hits[0];
        assert_eq!(hit.direction, -1);
        assert!(
            (hit.t - core::f64::consts::FRAC_PI_2).abs() < 1e-6,
            "t = {}",
            hit.t
        );
        assert!(hit.u[0].abs() < 1e-6, "x at crossing = {}", hit.u[0]);
        assert!(
            (hit.u[1] + 1.0).abs() < 1e-6,
            "v at crossing = {}",
            hit.u[1]
        );
    }

    #[test]
    fn direction_filter_selects_the_right_crossings() {
        // Over [0, 2π], x = cos t falls at π/2 and rises at 3π/2.
        let ev = oscillator();
        let g = plane_event(2, 0, 0.0);
        let span = 2.0 * core::f64::consts::PI + 0.1;

        let mut s = Rk4::new();
        let rising = integrate_events(
            &ev,
            &mut s,
            &[1.0, 0.0],
            &[],
            0.0,
            span,
            &[EventSpec::new(&g, EventDirection::Rising)],
            &cfg(0.01),
        )
        .unwrap();
        assert_eq!(rising.hits.len(), 1);
        assert_eq!(rising.hits[0].direction, 1);
        assert!((rising.hits[0].t - 3.0 * core::f64::consts::FRAC_PI_2).abs() < 1e-5);

        let mut s = Rk4::new();
        let either = integrate_events(
            &ev,
            &mut s,
            &[1.0, 0.0],
            &[],
            0.0,
            span,
            &[EventSpec::new(&g, EventDirection::Either)],
            &cfg(0.01),
        )
        .unwrap();
        assert_eq!(either.hits.len(), 2);
        assert_eq!(either.hits[0].direction, -1); // falling first (π/2)
        assert_eq!(either.hits[1].direction, 1); // rising second (3π/2)
        assert!(either.hits[0].t < either.hits[1].t);
    }

    #[test]
    fn cubic_flow_crossing_is_found_to_machine_precision() {
        // u(t) = t³; event u = 0.5 ⇒ t* = 0.5^(1/3). This validates the *combined*
        // pipeline (RK4 step + Hermite interpolant + root solve) on a case where
        // every stage is exact: RK4 integrates this degree-2-in-t RHS exactly, and
        // cubic Hermite reproduces the resulting cubic exactly — so only the root
        // tolerance limits the crossing. (RK4 step accuracy is not isolated here;
        // the Hermite formula alone is covered by `hermite_reproduces_a_cubic_exactly`
        // and the order by `poincare_crossing_is_fourth_order_accurate`.)
        let ev = cubic_flow();
        let g = plane_event(1, 0, 0.5);
        let mut s = Rk4::new();
        let events = [EventSpec::new(&g, EventDirection::Rising)];
        let out = integrate_events(&ev, &mut s, &[0.0], &[], 0.0, 1.0, &events, &cfg(0.1)).unwrap();
        assert_eq!(out.hits.len(), 1);
        let t_star = 0.5_f64.cbrt();
        assert!(
            (out.hits[0].t - t_star).abs() < 1e-11,
            "t = {}",
            out.hits[0].t
        );
        assert!(
            (out.hits[0].u[0] - 0.5).abs() < 1e-11,
            "u = {}",
            out.hits[0].u[0]
        );
    }

    #[test]
    fn poincare_crossing_is_fourth_order_accurate() {
        // The refined crossing time of x = cos t through 0 converges as O(h⁴):
        // RK4 global O(h⁴) + cubic-Hermite O(h⁴). This is the ROADMAP #77
        // acceptance signature ("matches the Python Hermite refinement to O(dt⁴)").
        // Fit the empirical order from three step sizes via a log-log slope — a
        // two-point ratio window would admit an O(h³) method (ratio 8); the fitted
        // slope must land near 4, which O(h³) cannot reach.
        let ev = oscillator();
        let g = plane_event(2, 0, 0.0);
        let t_star = core::f64::consts::FRAC_PI_2;
        let err = |h: f64| {
            let mut s = Rk4::new();
            let out = integrate_events(
                &ev,
                &mut s,
                &[1.0, 0.0],
                &[],
                0.0,
                3.0,
                &[EventSpec::new(&g, EventDirection::Falling)],
                &cfg(h),
            )
            .unwrap();
            (out.hits[0].t - t_star).abs()
        };
        let (h1, h2, h3) = (0.2, 0.1, 0.05);
        let (e1, e2, e3) = (err(h1), err(h2), err(h3));
        // Empirical order p from each successive halving: error ∝ h^p ⇒
        // p = log(e_coarse / e_fine) / log(h_coarse / h_fine).
        let p_a = (e1 / e2).ln() / (h1 / h2).ln();
        let p_b = (e2 / e3).ln() / (h2 / h3).ln();
        let slope = 0.5 * (p_a + p_b);
        assert!(
            (3.6..4.4).contains(&slope),
            "expected O(h⁴) convergence (order ≈ 4), got {slope:.2} \
             (e1={e1:e}, e2={e2:e}, e3={e3:e})"
        );
    }

    #[test]
    fn terminal_event_stops_at_the_crossing() {
        // Stop the oscillator at its first falling crossing (t = π/2).
        let ev = oscillator();
        let g = plane_event(2, 0, 0.0);
        let mut s = Rk4::new();
        let events = [EventSpec::terminal(&g, EventDirection::Falling)];
        let out = integrate_events(
            &ev,
            &mut s,
            &[1.0, 0.0],
            &[],
            0.0,
            100.0,
            &events,
            &cfg(0.01),
        )
        .unwrap();
        assert!(out.terminated);
        assert_eq!(out.hits.len(), 1);
        assert!((out.t_final - core::f64::consts::FRAC_PI_2).abs() < 1e-6);
        assert_eq!(out.u_final, out.hits[0].u);
        // The run stopped well before t = 100.
        assert!(out.t_final < 2.0);
    }

    #[test]
    fn no_crossing_returns_final_state_unterminated() {
        // x = cos t never reaches the plane x = 5; no crossing, run to t1.
        let ev = oscillator();
        let g = plane_event(2, 0, 5.0);
        let mut s = Rk4::new();
        let events = [EventSpec::new(&g, EventDirection::Either)];
        let out =
            integrate_events(&ev, &mut s, &[1.0, 0.0], &[], 0.0, 1.0, &events, &cfg(0.01)).unwrap();
        assert!(out.hits.is_empty());
        assert!(!out.terminated);
        assert!((out.t_final - 1.0).abs() < 1e-12);
        assert!((out.u_final[0] - 1.0_f64.cos()).abs() < 1e-7);
    }

    #[test]
    fn many_crossings_are_collected_in_time_order() {
        // Count rising crossings of x = cos t over several periods.
        let ev = oscillator();
        let g = plane_event(2, 0, 0.0);
        let mut s = Rk4::new();
        let periods = 5;
        let span = periods as f64 * 2.0 * core::f64::consts::PI + 0.1;
        let out = integrate_events(
            &ev,
            &mut s,
            &[1.0, 0.0],
            &[],
            0.0,
            span,
            &[EventSpec::new(&g, EventDirection::Rising)],
            &cfg(0.01),
        )
        .unwrap();
        assert_eq!(out.hits.len(), periods);
        for w in out.hits.windows(2) {
            assert!(w[0].t < w[1].t, "hits must be time-ordered");
        }
        // The k-th rising crossing is at (2k + 3/2)π.
        for (k, hit) in out.hits.iter().enumerate() {
            let want = (2.0 * k as f64 + 1.5) * core::f64::consts::PI;
            assert!(
                (hit.t - want).abs() < 1e-4,
                "crossing {k}: {} vs {want}",
                hit.t
            );
        }
    }

    #[test]
    fn divergence_during_event_search_is_reported() {
        // dx/dt = x² blows up at t = 1; integrating past it must error, not
        // silently return, even with an event armed.
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let dx = b.mul(x, x);
        let ev = VmEval::new(Interpreter::new(b.finish(&[dx], &[], 1, 0).unwrap()));
        let g = plane_event(1, 0, 1e9);
        let mut s = Rk4::new();
        let err = integrate_events(
            &ev,
            &mut s,
            &[1.0],
            &[],
            0.0,
            2.0,
            &[EventSpec::new(&g, EventDirection::Rising)],
            &cfg(0.01),
        )
        .unwrap_err();
        assert!(
            matches!(err, IntegrateError::NonFinite { .. }),
            "got {err:?}"
        );
    }

    // --- native dense-output capability path ----------------------------------

    /// A forward-Euler kernel that *also* advertises a native dense output: a
    /// linear interpolant `u(s) = u0 + s·(u1 - u0)`. Forward Euler's step is
    /// exactly linear in `s`, so this interpolant is the method's true continuous
    /// extension — and it lets a test prove the engine drives
    /// [`Solver::interpolate`] (not the Hermite fallback) when `Caps::dense`.
    struct DenseEuler {
        deriv: Vec<f64>,
        // Counts interpolate() calls so a test can prove the engine drove the
        // native dense path, not the Hermite fallback.
        interp_calls: std::cell::Cell<usize>,
    }
    impl DenseEuler {
        fn new() -> Self {
            DenseEuler {
                deriv: Vec::new(),
                interp_calls: std::cell::Cell::new(0),
            }
        }
    }
    impl Solver for DenseEuler {
        fn name(&self) -> &'static str {
            "testkit-dense-euler"
        }
        fn caps(&self) -> Caps {
            Caps::explicit(ProblemKinds::of(ProblemKind::Ode)).with_dense()
        }
        fn step(&mut self, ev: &dyn Evaluator, st: &mut SolverState, h: f64) -> StepOutcome {
            let dim = st.u.len();
            if self.deriv.len() != dim {
                self.deriv = vec![0.0; dim];
            }
            let SolverState { u, t, p, scratch } = st;
            ev.eval(u, p, *t, scratch, &mut self.deriv);
            if self.deriv.iter().any(|x| !x.is_finite()) {
                return StepOutcome::Failed;
            }
            for (ui, &di) in u.iter_mut().zip(self.deriv.iter()) {
                *ui += h * di;
            }
            *t += h;
            StepOutcome::Accepted { h_next: h }
        }
        fn interpolate(&self, u0: &[f64], h: f64, theta: f64, out: &mut [f64]) -> bool {
            // u(theta) = u0 + theta * h * f(u0)  (Euler's exact step extension).
            self.interp_calls.set(self.interp_calls.get() + 1);
            for (o, (&u0d, &fd)) in out.iter_mut().zip(u0.iter().zip(self.deriv.iter())) {
                *o = u0d + theta * h * fd;
            }
            true
        }
    }

    #[test]
    fn native_dense_output_path_is_used_and_correct() {
        // A constant-velocity flow dx = c: Euler is exact, so the *linear* native
        // interpolant pins the crossing of x = c0 exactly — exercising the
        // `Caps::dense` → `Solver::interpolate` branch (no endpoint derivatives).
        let mut b = TapeBuilder::new();
        let _x = b.state(0);
        let c = b.constant(2.0);
        let ev = VmEval::new(Interpreter::new(b.finish(&[c], &[], 1, 0).unwrap()));
        let g = plane_event(1, 0, 1.0); // x crosses 1 at t = 0.5 from x0 = 0
        let mut s = DenseEuler::new();
        assert!(
            s.caps().dense,
            "the kernel must advertise native dense output"
        );
        let out = integrate_events(
            &ev,
            &mut s,
            &[0.0],
            &[],
            0.0,
            1.0,
            &[EventSpec::new(&g, EventDirection::Rising)],
            &cfg(0.3),
        )
        .unwrap();
        assert_eq!(out.hits.len(), 1);
        assert!((out.hits[0].t - 0.5).abs() < 1e-12, "t = {}", out.hits[0].t);
        assert!(
            (out.hits[0].u[0] - 1.0).abs() < 1e-12,
            "u = {}",
            out.hits[0].u[0]
        );
        // Prove the engine actually drove Solver::interpolate (the native dense
        // path), not the Hermite fallback — otherwise this test would pass even
        // if the `Caps::dense` branch were dead.
        assert!(
            s.interp_calls.get() > 0,
            "engine must call interpolate() for a dense-capable kernel"
        );
    }

    #[test]
    fn adaptive_kernel_drives_events() {
        // The natural-step loop must work with a real adaptive kernel (rejections
        // + carried step size), not just fixed-step RK4. Drive rk45 from the
        // registry and check the oscillator crossing still lands at π/2. The
        // crossing inherits rk45's (default-tolerance) trajectory accuracy, so we
        // test compatibility of the loop, not the O(h⁴) refinement (covered by
        // the RK4 tests above).
        let ev = oscillator();
        let g = plane_event(2, 0, 0.0);
        let mut s = tsdyn_solvers::make("rk45").expect("rk45 is registered");
        let out = integrate_events(
            &ev,
            s.as_mut(),
            &[1.0, 0.0],
            &[],
            0.0,
            3.0,
            &[EventSpec::new(&g, EventDirection::Falling)],
            &cfg(0.05),
        )
        .unwrap();
        assert_eq!(out.hits.len(), 1);
        assert_eq!(out.hits[0].direction, -1);
        assert!(
            (out.hits[0].t - core::f64::consts::FRAC_PI_2).abs() < 1e-3,
            "t = {}",
            out.hits[0].t
        );
    }

    #[test]
    fn matches_an_independent_hermite_reference() {
        // ROADMAP #77's acceptance is "matches the Python Hermite refinement to
        // O(dt⁴)". The live Python cross-check awaits the FFI from a later wiring
        // stream; here we prove the *substance* — the engine uses the same cubic
        // Hermite + bracketed root as v2 — by reimplementing v2's PoincareMap._refine
        // independently (its `u_at` formula + a plain bisection) over the same RK4
        // bracket and confirming the two crossings agree to ~1e-11.
        let ev = oscillator();
        let g = plane_event(2, 0, 0.0);
        let h = 0.05;

        let mut s = Rk4::new();
        let out = integrate_events(
            &ev,
            &mut s,
            &[1.0, 0.0],
            &[],
            0.0,
            3.0,
            &[EventSpec::new(&g, EventDirection::Falling)],
            &cfg(h),
        )
        .unwrap();
        let (engine_t, engine_u) = (out.hits[0].t, out.hits[0].u.clone());

        // Independent fixed-step RK4 march (same arithmetic as the testkit kernel),
        // then the v2 cubic-Hermite refinement with a bisection root solve.
        let f = |u: &[f64]| [u[1], -u[0]]; // oscillator RHS
        let rk4 = |u: &[f64; 2], h: f64| -> [f64; 2] {
            let k1 = f(u);
            let u2 = [u[0] + 0.5 * h * k1[0], u[1] + 0.5 * h * k1[1]];
            let k2 = f(&u2);
            let u3 = [u[0] + 0.5 * h * k2[0], u[1] + 0.5 * h * k2[1]];
            let k3 = f(&u3);
            let u4 = [u[0] + h * k3[0], u[1] + h * k3[1]];
            let k4 = f(&u4);
            let sixth = h / 6.0;
            [
                u[0] + sixth * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
                u[1] + sixth * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
            ]
        };
        let mut u_prev = [1.0_f64, 0.0];
        let mut t_prev = 0.0_f64;
        let (ref_t, ref_u) = loop {
            let u = rk4(&u_prev, h);
            let t = t_prev + h;
            if u_prev[0] > 0.0 && u[0] <= 0.0 {
                // falling bracket on g = x
                let f0 = f(&u_prev);
                let f1 = f(&u);
                let u_at = |s: f64| -> [f64; 2] {
                    let (s2, s3) = (s * s, s * s * s);
                    let (h00, h10, h01, h11) = (
                        2.0 * s3 - 3.0 * s2 + 1.0,
                        s3 - 2.0 * s2 + s,
                        -2.0 * s3 + 3.0 * s2,
                        s3 - s2,
                    );
                    [
                        h00 * u_prev[0] + h10 * h * f0[0] + h01 * u[0] + h11 * h * f1[0],
                        h00 * u_prev[1] + h10 * h * f0[1] + h01 * u[1] + h11 * h * f1[1],
                    ]
                };
                let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
                for _ in 0..200 {
                    let mid = 0.5 * (lo + hi);
                    if u_at(lo)[0] * u_at(mid)[0] <= 0.0 {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                let s_star = 0.5 * (lo + hi);
                break (t_prev + s_star * h, u_at(s_star));
            }
            u_prev = u;
            t_prev = t;
        };
        assert!(
            (engine_t - ref_t).abs() < 1e-11,
            "t: engine {engine_t} vs reference {ref_t}"
        );
        assert!((engine_u[0] - ref_u[0]).abs() < 1e-11);
        assert!((engine_u[1] - ref_u[1]).abs() < 1e-11);
    }

    #[test]
    fn two_distinct_events_are_detected_and_time_ordered() {
        // x = cos t (falls at π/2), y = -sin t (= v, falls through 0 at t = π).
        // Two independent event functions over one run; hits must be time-ordered
        // and tagged with the right event index.
        let ev = oscillator();
        let gx = plane_event(2, 0, 0.0); // x = 0
        let gy = plane_event(2, 1, 0.0); // v = 0
        let mut s = Rk4::new();
        let events = [
            EventSpec::new(&gx, EventDirection::Either),
            EventSpec::new(&gy, EventDirection::Either),
        ];
        let span = core::f64::consts::PI + 0.2;
        let out = integrate_events(
            &ev,
            &mut s,
            &[1.0, 0.0],
            &[],
            0.0,
            span,
            &events,
            &cfg(0.01),
        )
        .unwrap();
        // Expect x-crossing near π/2 (event 0) and v-crossing near π (event 1).
        assert_eq!(out.hits.len(), 2);
        for w in out.hits.windows(2) {
            assert!(w[0].t <= w[1].t, "hits must be time-ordered");
        }
        assert_eq!(out.hits[0].event, 0);
        assert!((out.hits[0].t - core::f64::consts::FRAC_PI_2).abs() < 1e-4);
        assert_eq!(out.hits[1].event, 1);
        assert!((out.hits[1].t - core::f64::consts::PI).abs() < 1e-4);
    }

    #[test]
    fn terminal_event_among_several_stops_the_run() {
        // Arm a non-terminal (v=0 at t=π) and a terminal (x=0 at t=π/2). The
        // terminal fires first, so the run stops at π/2 and the v-crossing never
        // happens.
        let ev = oscillator();
        let gx = plane_event(2, 0, 0.0);
        let gv = plane_event(2, 1, 0.0);
        let mut s = Rk4::new();
        let events = [
            EventSpec::new(&gv, EventDirection::Either),
            EventSpec::terminal(&gx, EventDirection::Falling),
        ];
        let out = integrate_events(
            &ev,
            &mut s,
            &[1.0, 0.0],
            &[],
            0.0,
            100.0,
            &events,
            &cfg(0.01),
        )
        .unwrap();
        assert!(out.terminated);
        assert_eq!(out.hits.len(), 1, "only the terminal crossing is recorded");
        assert_eq!(out.hits[0].event, 1);
        assert!((out.t_final - core::f64::consts::FRAC_PI_2).abs() < 1e-6);
    }

    #[test]
    fn exact_endpoint_crossing_is_handled() {
        // A constant-velocity flow dx = 1 from x0 = -0.3 with step h = 0.3 lands
        // *exactly* on the plane x = 0 at t = 0.3 (RK4 is exact for a constant
        // RHS), so the bracket's high endpoint has g = 0 — the root is the
        // endpoint itself, which bracketed_root must return without misbehaving.
        let mut b = TapeBuilder::new();
        let one = b.constant(1.0);
        let ev = VmEval::new(Interpreter::new(b.finish(&[one], &[], 1, 0).unwrap()));
        let g = plane_event(1, 0, 0.0);
        let mut s = Rk4::new();
        let out = integrate_events(
            &ev,
            &mut s,
            &[-0.3],
            &[],
            0.0,
            1.0,
            &[EventSpec::new(&g, EventDirection::Rising)],
            &cfg(0.3),
        )
        .unwrap();
        assert_eq!(out.hits.len(), 1);
        assert!((out.hits[0].t - 0.3).abs() < 1e-12, "t = {}", out.hits[0].t);
        assert!(out.hits[0].u[0].abs() < 1e-12, "u = {}", out.hits[0].u[0]);
    }

    #[test]
    fn step_collapse_during_events_is_reported() {
        // The "diverge loudly" contract must hold on the event loop too: an
        // adaptive kernel that shrinks below the floor without accepting aborts.
        use crate::testkit::AlwaysReject;
        let ev = oscillator();
        let g = plane_event(2, 0, 0.0);
        let mut s = AlwaysReject::new();
        let conf = IntegrateConfig::new(1e-3).with_min_step(1e-6);
        let err = integrate_events(
            &ev,
            &mut s,
            &[1.0, 0.0],
            &[],
            0.0,
            3.0,
            &[EventSpec::new(&g, EventDirection::Falling)],
            &conf,
        )
        .unwrap_err();
        assert!(
            matches!(err, IntegrateError::StepCollapsed { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn step_limit_during_events_is_reported() {
        let ev = oscillator();
        let g = plane_event(2, 0, 0.0);
        let mut s = Rk4::new();
        let conf = IntegrateConfig::new(1e-4).with_max_steps(5);
        let err = integrate_events(
            &ev,
            &mut s,
            &[1.0, 0.0],
            &[],
            0.0,
            3.0,
            &[EventSpec::new(&g, EventDirection::Falling)],
            &conf,
        )
        .unwrap_err();
        assert!(
            matches!(err, IntegrateError::StepLimit { steps: 5, .. }),
            "got {err:?}"
        );
    }
}
