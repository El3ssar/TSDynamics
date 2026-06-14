//! Delay differential equations — the engine's method-of-steps integrator
//! (stream **E-DDE**).
//!
//! A DDE is `du/dt = f(u(t), u(t − τ₁), …, u(t − τ_m), t)`: the right-hand side
//! reads the solution at one or more *past* times.  We integrate it by the
//! **method of steps** (Bellen & Zennaro 2003): treat each delayed term as a
//! known function of the established history, so over one step the problem is an
//! ordinary ODE that any explicit [`Solver`](tsdyn_solvers::Solver) kernel can
//! advance.  The two pieces this needs beyond the ODE driver are
//!
//! 1. a **history buffer** that stores the solution densely enough to be sampled
//!    at any past time (cubic-Hermite interpolation between stored nodes), and
//! 2. a **delayed-input assembly** step that, each time the kernel asks for the
//!    RHS, fills the delayed arguments from that history.
//!
//! Both live here; the kernels (E3's explicit family) are reused unchanged.
//!
//! # The delay-slot tape contract (with `engine.compile.lower_dde`)
//!
//! The frozen IR has no delay opcode (ROADMAP §4a), so the Python lowering turns
//! each distinct delayed access `u(component, t − τ)` into an **extra state
//! input** appended after the `dim` real components: a tape over `dim + m` inputs
//! whose first `dim` outputs are the derivatives.  Slot `k` (input index
//! `dim + k`) is described by a [`DelaySlot`] — *which* component it samples and
//! by *how much* delay.  The engine fills those extra inputs from the history on
//! every evaluation; the tape itself is an ordinary RHS.  This keeps the IR
//! untouched: delays are *data*, not a new instruction.
//!
//! # Why the step is capped at the smallest delay
//!
//! Within one step `[t, t + h]` the RK stages evaluate the RHS at times
//! `t + cᵢ·h` (`cᵢ ∈ [0, 1]`), so a delayed argument is at `t + cᵢ·h − τ`.  For
//! that to lie in the *already computed* history (`≤ t`) we need `h ≤ τ` for
//! every delay, i.e. `h ≤ τ_min`.  Capping the step there means every delayed
//! lookup is a plain interpolation of established history — no iteration, no
//! extrapolation (the simplest robust method-of-steps; ROADMAP §10 "vendor/borrow
//! a vetted method-of-steps").  For the built-in delay systems the delays dwarf
//! the accuracy-limited step, so this cap never actually binds; it only matters
//! for very small delays.
//!
//! # The t₀ derivative discontinuity
//!
//! A DDE's past (for `s ≤ t₀`) and its forward solution (for `s ≥ t₀`) generally
//! have *different* slopes at `t₀` (the past's derivative versus `f` at `t₀`).
//! We keep the two separate — a fixed user past plus the growing forward
//! solution — so each is interpolated with its own correct slopes and the
//! discontinuity is represented exactly rather than smeared across `t₀`.
//!
//! # Only constant delays
//!
//! This integrator handles **constant** delays (`τ` independent of the state).
//! State-dependent delays (`τ = g(u)`) are not yet lowered by
//! `engine.compile.lower_dde` (it raises), so they stay on the v2 backend until
//! the IR/compile seam grows dynamic delay slots — see that module and the
//! E-DDE PR notes.

use std::collections::VecDeque;

use tsdyn_ir::Evaluator;
use tsdyn_solvers::{Solver, SolverState, StepOutcome};

use crate::integrate::{IntegrateConfig, IntegrateError};

/// One delayed-state input of a lowered DDE tape.
///
/// Slot `k` occupies extended-input index `dim + k` and is filled each evaluation
/// with [`component`](DelaySlot::component) of the history at time `t − `[`delay`](DelaySlot::delay).
/// Mirrors `tsdynamics.engine.compile.DelaySlot` (Python side of the contract).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DelaySlot {
    /// Which true-state component (`0 ≤ component < dim`) is delayed.
    pub component: usize,
    /// The positive, constant delay magnitude τ.
    pub delay: f64,
}

/// A stored solution point for cubic-Hermite interpolation: the time, the state
/// there, and the true RHS derivative there (the Hermite tangent).
#[derive(Clone, Debug)]
struct Node {
    t: f64,
    y: Vec<f64>,
    dy: Vec<f64>,
}

/// Cubic Hermite interpolation on `[ta, tb]` at `s`, from endpoint values
/// `(ya, yb)` and tangents `(ma, mb)`.  Reduces to the endpoints at `s = ta`/`tb`
/// and is `C¹` across nodes (matching tangents), giving `O(h⁴)` dense output —
/// the standard cheap continuous extension for a 4th/5th-order RK history.
#[inline]
fn hermite(s: f64, ta: f64, ya: f64, ma: f64, tb: f64, yb: f64, mb: f64) -> f64 {
    let dt = tb - ta;
    if dt <= 0.0 {
        return ya;
    }
    let th = (s - ta) / dt;
    let th2 = th * th;
    let th3 = th2 * th;
    let h00 = 2.0 * th3 - 3.0 * th2 + 1.0;
    let h10 = th3 - 2.0 * th2 + th;
    let h01 = -2.0 * th3 + 3.0 * th2;
    let h11 = th3 - th2;
    h00 * ya + h10 * dt * ma + h01 * yb + h11 * dt * mb
}

/// Largest index `i` with `t[i] <= s`, assuming `t` is ascending and
/// `t[0] <= s < t[n-1]` (the interior case; ends are handled by the callers).
#[inline]
fn bracket(times: impl Fn(usize) -> f64, n: usize, s: f64) -> usize {
    let (mut lo, mut hi) = (0usize, n - 1);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if times(mid) <= s {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// The integration history: the fixed user past on `[t₀ − max_delay, t₀]` and the
/// growing, front-pruned forward solution.  Kept apart so the `t₀` slope
/// discontinuity (see the module docs) is exact.
struct History {
    dim: usize,
    t0: f64,
    max_delay: f64,
    /// Past sample times (ascending, length `n_past ≥ 1`).  A single sample means
    /// a constant past.
    past_t: Vec<f64>,
    /// Past sample states, flat row-major `(n_past, dim)`.
    past_y: Vec<f64>,
    /// Hermite tangents at the past samples (finite differences); all zero for a
    /// constant past.  Same shape as `past_y`.
    past_slope: Vec<f64>,
    /// The forward solution nodes (ascending `t`), first node at `t₀`, pruned to
    /// `[t_now − max_delay, t_now]` so memory stays bounded on long runs.
    forward: VecDeque<Node>,
}

impl History {
    /// Build the history from the user past samples.  `past_t` must be ascending
    /// with `past_t.len() == past_y.len() / dim ≥ 1`.  Forward nodes are added by
    /// the integrator via [`push_forward`](History::push_forward).
    fn new(dim: usize, t0: f64, max_delay: f64, past_t: &[f64], past_y: &[f64]) -> History {
        let n_past = past_t.len();
        let past_slope = finite_difference_slopes(past_t, past_y, dim);
        History {
            dim,
            t0,
            max_delay,
            past_t: past_t.to_vec(),
            past_y: past_y.to_vec(),
            past_slope,
            forward: VecDeque::with_capacity(n_past.max(16)),
        }
    }

    /// Append a forward solution node (time strictly after the last).
    fn push_forward(&mut self, node: Node) {
        self.forward.push_back(node);
    }

    /// Drop forward nodes older than `t_now − max_delay`, keeping the one node
    /// just below the cutoff so any future query at `≥ t_now − max_delay` is still
    /// bracketed.  Bounds memory without ever discarding a node a later step needs.
    fn prune(&mut self, t_now: f64) {
        let cutoff = t_now - self.max_delay;
        while self.forward.len() >= 2 && self.forward[1].t <= cutoff {
            self.forward.pop_front();
        }
    }

    /// Interpolate `component` of the history at past time `s`.
    ///
    /// `s ≥ t₀` reads the forward solution; `s < t₀` reads the user past.  The
    /// integrator guarantees `s` lies within the retained range (the step cap
    /// keeps every delayed lookup `≤ t_now`), but the ends are clamped so
    /// floating-point grazing at a boundary returns the endpoint rather than
    /// extrapolating.
    fn interp(&self, component: usize, s: f64) -> f64 {
        if s >= self.t0 {
            self.forward_interp(component, s)
        } else {
            self.past_interp(component, s)
        }
    }

    fn forward_interp(&self, comp: usize, s: f64) -> f64 {
        let f = &self.forward;
        let n = f.len();
        debug_assert!(n >= 1, "forward history is seeded with the t0 node");
        if s <= f[0].t {
            return f[0].y[comp];
        }
        if s >= f[n - 1].t {
            return f[n - 1].y[comp];
        }
        let i = bracket(|j| f[j].t, n, s);
        let a = &f[i];
        let b = &f[i + 1];
        hermite(s, a.t, a.y[comp], a.dy[comp], b.t, b.y[comp], b.dy[comp])
    }

    fn past_interp(&self, comp: usize, s: f64) -> f64 {
        let n = self.past_t.len();
        let row = |i: usize| self.past_y[i * self.dim + comp];
        if n == 1 {
            return row(0); // constant past
        }
        if s <= self.past_t[0] {
            return row(0);
        }
        if s >= self.past_t[n - 1] {
            return row(n - 1);
        }
        let i = bracket(|j| self.past_t[j], n, s);
        let slope = |k: usize| self.past_slope[k * self.dim + comp];
        hermite(
            s,
            self.past_t[i],
            row(i),
            slope(i),
            self.past_t[i + 1],
            row(i + 1),
            slope(i + 1),
        )
    }
}

/// Central finite-difference tangents for the past samples (one-sided at the
/// ends), used as the Hermite slopes of a callable past.  Returns all zeros for a
/// single sample (a constant past).
fn finite_difference_slopes(t: &[f64], y: &[f64], dim: usize) -> Vec<f64> {
    let n = t.len();
    let mut slope = vec![0.0; n * dim];
    if n < 2 {
        return slope;
    }
    for c in 0..dim {
        let val = |i: usize| y[i * dim + c];
        // Forward / backward difference at the ends.
        slope[c] = (val(1) - val(0)) / (t[1] - t[0]);
        slope[(n - 1) * dim + c] = (val(n - 1) - val(n - 2)) / (t[n - 1] - t[n - 2]);
        // Central difference in the interior.
        for i in 1..n - 1 {
            slope[i * dim + c] = (val(i + 1) - val(i - 1)) / (t[i + 1] - t[i - 1]);
        }
    }
    slope
}

/// An [`Evaluator`] over the *true* state that fills the lowered tape's delayed
/// inputs from the [`History`] before delegating to the underlying tape evaluator.
///
/// To the RK kernel this looks like an ordinary `dim`-dimensional RHS: it reports
/// `dim()` as the true dimension and writes a `dim`-length derivative.  Inside
/// [`eval`](DdeStageEval::eval) it builds the `dim + m` extended input — the true
/// state followed by each delay slot's interpolated past value — and runs the
/// tape over it.  It borrows the history immutably, so it is `Sync` with no
/// interior mutability: the integrator only ever *reads* the history during a
/// step and appends to it between steps.
struct DdeStageEval<'a> {
    /// The lowered DDE tape evaluator (over `dim + m` inputs, `dim` outputs).
    inner: &'a dyn Evaluator,
    history: &'a History,
    slots: &'a [DelaySlot],
    dim: usize,
    /// `dim + slots.len()` — the tape's input width.
    n_state: usize,
}

impl Evaluator for DdeStageEval<'_> {
    fn dim(&self) -> usize {
        self.dim
    }
    fn n_param(&self) -> usize {
        self.inner.n_param()
    }
    fn n_scratch(&self) -> usize {
        // The inner tape's register file, plus room to assemble the extended input.
        self.inner.n_scratch() + self.n_state
    }
    fn has_jacobian(&self) -> bool {
        false
    }
    fn eval(&self, u: &[f64], p: &[f64], t: f64, scratch: &mut [f64], deriv: &mut [f64]) {
        let inner_n = self.inner.n_scratch();
        let (inner_scratch, tail) = scratch.split_at_mut(inner_n);
        let u_ext = &mut tail[..self.n_state];
        u_ext[..self.dim].copy_from_slice(&u[..self.dim]);
        for (k, slot) in self.slots.iter().enumerate() {
            u_ext[self.dim + k] = self.history.interp(slot.component, t - slot.delay);
        }
        self.inner.eval(u_ext, p, t, inner_scratch, deriv);
    }
    fn eval_jac(
        &self,
        _u: &[f64],
        _p: &[f64],
        _t: f64,
        _scratch: &mut [f64],
        _deriv: &mut [f64],
        _jac: &mut [f64],
    ) {
        // The DDE engine drives explicit kernels only; none calls eval_jac.
        unreachable!("the DDE method-of-steps uses explicit kernels (no Jacobian)");
    }
}

/// Integrate a DDE through the non-decreasing output grid `t_eval`, returning a
/// flat row-major `(t_eval.len(), dim)` buffer (the first row is `ic`).
///
/// # Arguments
///
/// - `inner` — the lowered DDE tape evaluator: `dim` outputs over `dim + slots.len()`
///   inputs (the extra inputs are the delay slots; parameters are folded into the
///   tape, so `inner.n_param() == 0`).
/// - `solver` — an **explicit** kernel (DP45, Tsit5, DOP853, RK4); the caller
///   rejects implicit kernels (the method of steps has no delayed Jacobian here).
/// - `dim`, `slots` — the true dimension and the delay-slot descriptors (slot `k`
///   is extended input `dim + k`).
/// - `ic` — the true state at `t_eval[0]` (`= history(t₀)` for a callable past).
/// - `past_t`, `past_y` — the user past on `[t₀ − max_delay, t₀]`: ascending
///   times and a flat `(n_past, dim)` value buffer.  A single sample is a
///   constant past.
/// - `cfg` — first step / step floor / per-segment step cap (shared with the ODE
///   driver).  The first step is additionally capped at the smallest delay.
///
/// Diverges loudly: a non-finite state, a collapsed step, or the per-segment step
/// cap surfaces as the matching [`IntegrateError`] rather than plausible numbers.
#[allow(clippy::too_many_arguments)]
pub fn integrate_dde_grid(
    inner: &dyn Evaluator,
    solver: &mut dyn Solver,
    dim: usize,
    slots: &[DelaySlot],
    ic: &[f64],
    past_t: &[f64],
    past_y: &[f64],
    t_eval: &[f64],
    cfg: &IntegrateConfig,
) -> Result<Vec<f64>, IntegrateError> {
    let n_slots = slots.len();
    let n_state = dim + n_slots;
    let mut out = vec![0.0; t_eval.len() * dim];
    if t_eval.is_empty() {
        return Ok(out);
    }

    let t0 = t_eval[0];
    let max_delay = slots.iter().map(|s| s.delay).fold(0.0_f64, f64::max);
    // The smallest delay caps the step (see the module docs). With no slots there
    // is nothing to cap against — fall back to the ODE behaviour (no cap).
    let tau_min = slots.iter().map(|s| s.delay).fold(f64::INFINITY, f64::min);

    let mut history = History::new(dim, t0, max_delay, past_t, past_y);

    // Per-worker buffers: scratch sized for the *stage* evaluator (inner register
    // file + extended-input assembly), and a derivative buffer for seeding /
    // appending Hermite tangents.
    let stage_scratch_len = inner.n_scratch() + n_state;
    let mut dy = vec![0.0; dim];

    // Seed the forward history with the t₀ node: its tangent is the true RHS at t₀.
    {
        let ev = DdeStageEval {
            inner,
            history: &history,
            slots,
            dim,
            n_state,
        };
        let mut scratch = vec![0.0; stage_scratch_len];
        ev.eval(&ic[..dim], &[], t0, &mut scratch, &mut dy);
    }
    history.push_forward(Node {
        t: t0,
        y: ic[..dim].to_vec(),
        dy: dy.clone(),
    });

    out[..dim].copy_from_slice(&ic[..dim]); // first output row is the IC

    let mut st = SolverState {
        u: ic[..dim].to_vec(),
        t: t0,
        p: Vec::new(),
        scratch: vec![0.0; stage_scratch_len],
    };
    let mut h = cfg.first_step.min(tau_min);

    for (k, &target) in t_eval.iter().enumerate() {
        if k == 0 {
            continue;
        }
        debug_assert!(target >= st.t, "t_eval must be non-decreasing");
        advance_to(
            inner,
            solver,
            &mut st,
            &mut history,
            slots,
            dim,
            n_state,
            &mut h,
            target,
            tau_min,
            cfg,
            &mut dy,
        )?;
        out[k * dim..(k + 1) * dim].copy_from_slice(&st.u);
    }
    Ok(out)
}

/// Advance `st` from its current time to `t_end`, growing the history one node per
/// accepted step.  Mirrors the ODE [`advance_to`](crate::integrate) loop with two
/// additions: the step is capped at `tau_min` so delayed lookups stay in the
/// established past, and each accepted step appends a Hermite node.
#[allow(clippy::too_many_arguments)]
fn advance_to(
    inner: &dyn Evaluator,
    solver: &mut dyn Solver,
    st: &mut SolverState,
    history: &mut History,
    slots: &[DelaySlot],
    dim: usize,
    n_state: usize,
    h: &mut f64,
    t_end: f64,
    tau_min: f64,
    cfg: &IntegrateConfig,
    dy: &mut [f64],
) -> Result<(), IntegrateError> {
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
        // The natural step, never larger than the smallest delay.
        let pre = (*h).min(tau_min);
        // "Landing" when this step reaches the output time; "tau-capped" when the
        // delay cap (not the natural step) shrinks it. In both cases the step is
        // forced shorter than the kernel's natural `h`, which must not be shrunk
        // by a forced-short step (the v2/ODE convention).
        let landing = pre >= remaining;
        let h_try = if landing { remaining } else { pre };
        let tau_capped = !landing && tau_min < *h;
        steps += 1;

        // Borrow the history immutably only for the duration of the step.
        let outcome = {
            let ev = DdeStageEval {
                inner,
                history,
                slots,
                dim,
                n_state,
            };
            solver.step(&ev, st, h_try)
        };

        match outcome {
            StepOutcome::Accepted { h_next } => {
                if !st.u.iter().all(|x| x.is_finite()) || !st.t.is_finite() {
                    return Err(IntegrateError::NonFinite { t: st.t });
                }
                if landing {
                    st.t = t_end; // pin to the grid time; never let it drift
                }
                // The Hermite tangent of the new node is the true RHS there. The
                // delayed lookups for this evaluation are at `st.t − τ ≤ st.t − tau_min
                // ≤ prev_node.t`, i.e. inside the established history.
                {
                    let ev = DdeStageEval {
                        inner,
                        history,
                        slots,
                        dim,
                        n_state,
                    };
                    ev.eval(&st.u, &st.p, st.t, &mut st.scratch, dy);
                }
                history.push_forward(Node {
                    t: st.t,
                    y: st.u.clone(),
                    dy: dy.to_vec(),
                });
                history.prune(st.t);
                // Only a full natural step updates the learned step size.
                if !landing && !tau_capped && h_next.is_finite() && h_next > 0.0 {
                    *h = h_next;
                }
            }
            StepOutcome::Rejected { h_next } => {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::VmEval;
    use tsdyn_ir::TapeBuilder;
    use tsdyn_solvers::explicit::Rk45;
    use tsdyn_vm::Interpreter;

    /// Lower the scalar DDE `y'(t) = −y(t − 1)` to a one-output tape over two
    /// inputs: `u0 = y(t)`, `u1 = y(t − 1)` (the single delay slot). Output is
    /// `−u1`.
    fn scalar_neg_delay() -> Interpreter {
        let mut b = TapeBuilder::new();
        let _y = b.state(0);
        let y_tau = b.state(1); // delay slot 0
        let dy = b.neg(y_tau);
        Interpreter::new(b.finish(&[dy], &[], 2, 0).unwrap())
    }

    /// The analytic method-of-steps solution of `y'=−y(t−1)`, constant past 1:
    /// on `[0,1]` `y=1−t`; on `[1,2]` `y=t²/2−2t+3/2`.
    fn neg_delay_exact(t: f64) -> f64 {
        if t <= 1.0 {
            1.0 - t
        } else {
            0.5 * t * t - 2.0 * t + 1.5
        }
    }

    #[test]
    fn scalar_linear_dde_matches_method_of_steps_closed_form() {
        let inner = VmEval::new(scalar_neg_delay());
        let slots = [DelaySlot {
            component: 0,
            delay: 1.0,
        }];
        let t_eval: Vec<f64> = (0..=20).map(|i| i as f64 * 0.1).collect();
        let mut solver = Rk45::with_tolerances(1e-9, 1e-11);
        let cfg = IntegrateConfig::new(0.05);
        // Constant past = 1 on [−1, 0]: a single sample.
        let out = integrate_dde_grid(
            &inner,
            &mut solver,
            1,
            &slots,
            &[1.0],
            &[0.0],
            &[1.0],
            &t_eval,
            &cfg,
        )
        .unwrap();
        assert_eq!(out[0], 1.0); // first row is the IC
        for (k, &t) in t_eval.iter().enumerate() {
            let want = neg_delay_exact(t);
            assert!(
                (out[k] - want).abs() < 1e-6,
                "t = {t}: got {}, want {want}",
                out[k]
            );
        }
        // Spot-check the two exact checkpoints.
        assert!((out[10] - 0.0).abs() < 1e-6, "y(1) should be 0"); // index 10 → t=1
        assert!((out[20] - (-0.5)).abs() < 1e-6, "y(2) should be -0.5");
    }

    /// Two independent scalar delays with different constant pasts, exercising
    /// `dim > 1` and per-component delayed lookup:
    /// `y0' = −y0(t−1)` (past 1), `y1' = −y1(t−1)` (past 2).
    /// Analytic at t=2: `y0(2) = −0.5`, `y1(2) = −1`.
    fn two_component_neg_delay() -> Interpreter {
        let mut b = TapeBuilder::new();
        let _y0 = b.state(0);
        let _y1 = b.state(1);
        let y0_tau = b.state(2); // slot 0 → component 0
        let y1_tau = b.state(3); // slot 1 → component 1
        let d0 = b.neg(y0_tau);
        let d1 = b.neg(y1_tau);
        Interpreter::new(b.finish(&[d0, d1], &[], 4, 0).unwrap())
    }

    #[test]
    fn two_component_dde_handles_per_component_delays() {
        let inner = VmEval::new(two_component_neg_delay());
        let slots = [
            DelaySlot {
                component: 0,
                delay: 1.0,
            },
            DelaySlot {
                component: 1,
                delay: 1.0,
            },
        ];
        let t_eval = [0.0, 1.0, 2.0];
        let mut solver = Rk45::with_tolerances(1e-9, 1e-11);
        let cfg = IntegrateConfig::new(0.05);
        let out = integrate_dde_grid(
            &inner,
            &mut solver,
            2,
            &slots,
            &[1.0, 2.0],
            &[0.0],
            &[1.0, 2.0],
            &t_eval,
            &cfg,
        )
        .unwrap();
        // Row 2 = states at t = 2.
        let (y0, y1) = (out[4], out[5]);
        assert!((y0 - (-0.5)).abs() < 1e-6, "y0(2): {y0}");
        assert!((y1 - (-1.0)).abs() < 1e-6, "y1(2): {y1}");
    }

    #[test]
    fn callable_past_is_interpolated() {
        // `y'(t) = −y(t − 1)` with a non-constant past `y(s) = 1 + s` on [−1, 0].
        // On [0,1]: y'(t) = −(1 + (t−1)) = −t ⇒ y(t) = 1 − t²/2 (y(0)=1).
        let inner = VmEval::new(scalar_neg_delay());
        let slots = [DelaySlot {
            component: 0,
            delay: 1.0,
        }];
        // Dense past samples of 1 + s on [−1, 0].
        let n = 201;
        let past_t: Vec<f64> = (0..n).map(|i| -1.0 + i as f64 / (n - 1) as f64).collect();
        let past_y: Vec<f64> = past_t.iter().map(|&s| 1.0 + s).collect();
        let t_eval: Vec<f64> = (0..=10).map(|i| i as f64 * 0.1).collect();
        let mut solver = Rk45::with_tolerances(1e-9, 1e-11);
        let cfg = IntegrateConfig::new(0.05);
        let out = integrate_dde_grid(
            &inner,
            &mut solver,
            1,
            &slots,
            &[1.0], // y(0) = 1 + 0
            &past_t,
            &past_y,
            &t_eval,
            &cfg,
        )
        .unwrap();
        for (k, &t) in t_eval.iter().enumerate() {
            let want = 1.0 - 0.5 * t * t;
            assert!(
                (out[k] - want).abs() < 1e-5,
                "t = {t}: got {}, want {want}",
                out[k]
            );
        }
    }

    #[test]
    fn small_delay_forces_the_step_cap() {
        // A tiny delay (τ = 0.01) forces the step well below the accuracy-limited
        // step, exercising the tau-cap branch. `y'(t) = −y(t − 0.01)`, constant
        // past 1: for t ≪ 1 this tracks the un-delayed decay closely; we only
        // assert it stays finite, bounded, and decreasing initially.
        let inner = VmEval::new(scalar_neg_delay());
        let slots = [DelaySlot {
            component: 0,
            delay: 0.01,
        }];
        let t_eval: Vec<f64> = (0..=50).map(|i| i as f64 * 0.01).collect();
        let mut solver = Rk45::with_tolerances(1e-8, 1e-10);
        let cfg = IntegrateConfig::new(1.0); // huge first step → must be capped to τ
        let out = integrate_dde_grid(
            &inner,
            &mut solver,
            1,
            &slots,
            &[1.0],
            &[0.0],
            &[1.0],
            &t_eval,
            &cfg,
        )
        .unwrap();
        assert!(out.iter().all(|x| x.is_finite()));
        assert!(out[0] == 1.0 && out[50] < 1.0 && out[50] > 0.0);
    }

    #[test]
    fn divergence_is_reported() {
        // `y'(t) = exp(y(t − 1))` with constant past 1 grows super-exponentially
        // and overflows f64 to `inf` well before the horizon — the engine must
        // surface that as an error, never as plausible numbers.
        let mut b = TapeBuilder::new();
        let _y = b.state(0);
        let y_tau = b.state(1);
        let dy = b.exp(y_tau);
        let inner = VmEval::new(Interpreter::new(b.finish(&[dy], &[], 2, 0).unwrap()));
        let slots = [DelaySlot {
            component: 0,
            delay: 1.0,
        }];
        // Long horizon so the doubly-exponential growth overflows f64 to inf.
        let t_eval: Vec<f64> = (0..=400).map(|i| i as f64 * 0.25).collect();
        let mut solver = Rk45::with_tolerances(1e-6, 1e-9);
        let cfg = IntegrateConfig::new(0.1);
        let err = integrate_dde_grid(
            &inner,
            &mut solver,
            1,
            &slots,
            &[1.0],
            &[0.0],
            &[1.0],
            &t_eval,
            &cfg,
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                IntegrateError::NonFinite { .. }
                    | IntegrateError::StepCollapsed { .. }
                    | IntegrateError::StepLimit { .. }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn first_row_is_the_initial_condition() {
        let inner = VmEval::new(scalar_neg_delay());
        let slots = [DelaySlot {
            component: 0,
            delay: 1.0,
        }];
        let mut solver = Rk45::new();
        let cfg = IntegrateConfig::new(0.05);
        let out = integrate_dde_grid(
            &inner,
            &mut solver,
            1,
            &slots,
            &[0.7],
            &[0.0],
            &[0.7],
            &[3.0, 4.0],
            &cfg,
        )
        .unwrap();
        assert_eq!(out[0], 0.7);
    }

    #[test]
    fn pruning_keeps_long_runs_bounded_and_correct() {
        // Integrate the linear DDE to a long horizon; the closed form past t = 2
        // is not elementary, but the solution decays to 0, so just assert it stays
        // finite and small — the point is that pruning (max_delay = 1) over a long
        // grid does not corrupt the interpolation.
        let inner = VmEval::new(scalar_neg_delay());
        let slots = [DelaySlot {
            component: 0,
            delay: 1.0,
        }];
        let t_eval: Vec<f64> = (0..=500).map(|i| i as f64 * 0.1).collect();
        let mut solver = Rk45::with_tolerances(1e-8, 1e-10);
        let cfg = IntegrateConfig::new(0.05);
        let out = integrate_dde_grid(
            &inner,
            &mut solver,
            1,
            &slots,
            &[1.0],
            &[0.0],
            &[1.0],
            &t_eval,
            &cfg,
        )
        .unwrap();
        assert!(out.iter().all(|x| x.is_finite()));
        // y'=-y(t-1) has decaying oscillatory solutions; by t = 50 it is tiny.
        assert!(out.last().unwrap().abs() < 0.5);
    }

    #[test]
    fn hermite_reduces_to_endpoints() {
        assert_eq!(hermite(0.0, 0.0, 2.0, 5.0, 1.0, 3.0, 7.0), 2.0);
        assert_eq!(hermite(1.0, 0.0, 2.0, 5.0, 1.0, 3.0, 7.0), 3.0);
        // A straight line y = 2 + t on [0,1] with matching slopes is reproduced.
        assert!((hermite(0.5, 0.0, 2.0, 1.0, 1.0, 3.0, 1.0) - 2.5).abs() < 1e-15);
    }
}
