//! Discrete-map iteration — the engine's native loop for maps (stream E-MAP).
//!
//! A discrete map is `u_{n+1} = f(u_n)`: repeated *function application*, with no
//! step size, no error control, and no [`Solver`](tsdyn_solvers::Solver). So this
//! module does not reuse the ODE [`integrate`](crate::integrate) loop — it drives
//! an [`Evaluator`] directly, treating each [`Evaluator::eval`] call as one map
//! step (the lowered next-state map writes the *next state* into the `deriv`
//! buffer, exactly as a lowered RHS would write `du/dt`).
//!
//! # Output convention
//!
//! [`iterate_dense`] records the iterates *after* `u0`, matching the v2 Numba
//! loop and the [`reference`](tsdyn_ir::reference) map path: row `i` is
//! `f^{i+1}(u0)`, so an `n_steps`-row buffer holds `f(u0) … f^{n_steps}(u0)` and
//! never the initial condition itself. [`iterate_final`] returns `f^{n_steps}(u0)`
//! (and `u0` unchanged for `n_steps == 0`).
//!
//! # Autonomous in the IR
//!
//! Maps lower with no `Time` leaf (their `_step` has no time argument), so the
//! evaluator's `t` is unused; this loop passes `t = 0.0` on every step, which is
//! exactly what the pure-Python reference path does — so the native and reference
//! iterations agree to the same tolerance the interpreter already meets against
//! the reference evaluator (stream E1).
//!
//! # Diverge loudly
//!
//! As with the ODE driver, a non-finite iterate is reported, never returned as
//! plausible data: [`iterate_dense`]/[`iterate_final`] stop at the first
//! non-finite state with [`MapError::NonFinite`], and the ensemble path isolates
//! a diverged trajectory as a `NaN` row plus a [`MapTrajStatus::Failed`].

use rayon::prelude::*;
use tsdyn_ir::Evaluator;

/// Why a map iteration stopped short of its requested step count.
///
/// The single failure mode for a (deterministic, step-free) map is a non-finite
/// iterate — the map blew up. The carried index is the *step* at which it was
/// detected (0-based: `step` iterates of `f` had been applied), so a caller can
/// report where the orbit escaped.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MapError {
    /// An iterate went non-finite (the map diverged). `step` is the number of
    /// the iterate that produced the non-finite state (`f^{step+1}` was the
    /// offending application).
    NonFinite {
        /// The iterate index (0-based) at which non-finiteness was detected.
        step: usize,
    },
}

impl core::fmt::Display for MapError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MapError::NonFinite { step } => {
                write!(f, "non-finite state at iterate {step} (the map diverged)")
            }
        }
    }
}

impl std::error::Error for MapError {}

/// One step of the map: `next ← f(cur)`, with a finiteness guard.
///
/// `scratch` is the evaluator's working buffer (the interpreter's register file;
/// empty for the JIT), reused across steps so the loop allocates nothing.
#[inline]
fn map_step(
    ev: &dyn Evaluator,
    cur: &[f64],
    p: &[f64],
    scratch: &mut [f64],
    next: &mut [f64],
    step: usize,
) -> Result<(), MapError> {
    // Maps are autonomous in the IR (no Time leaf) — t = 0.0 matches the
    // reference path exactly.
    ev.eval(cur, p, 0.0, scratch, next);
    if !next.iter().all(|x| x.is_finite()) {
        return Err(MapError::NonFinite { step });
    }
    Ok(())
}

/// Iterate the map `n_steps` times from `u0`, recording every iterate.
///
/// Returns a row-major `(n_steps, dim)` buffer whose row `i` is `f^{i+1}(u0)`
/// (the initial condition is *not* included — matching the v2 Numba loop). `p`
/// is the parameter vector (`p.len() == ev.n_param()`; empty for a lowered map,
/// whose parameters are folded into the tape). Stops at the first non-finite
/// iterate with [`MapError::NonFinite`].
pub fn iterate_dense(
    ev: &dyn Evaluator,
    u0: &[f64],
    p: &[f64],
    n_steps: usize,
) -> Result<Vec<f64>, MapError> {
    let dim = ev.dim();
    debug_assert_eq!(u0.len(), dim, "u0 length must equal the system dimension");
    debug_assert_eq!(p.len(), ev.n_param(), "p length must equal n_param");

    let mut out = vec![0.0; n_steps * dim];
    if n_steps == 0 {
        return Ok(out);
    }
    let mut scratch = vec![0.0; ev.n_scratch()];
    // Ping-pong the current state through the output buffer: write iterate `i`
    // into its row, then read that row as the input for iterate `i + 1`. This
    // keeps the loop allocation-free and avoids a separate `cur` copy per step.
    let mut cur = u0.to_vec();
    for i in 0..n_steps {
        let (head, tail) = out.split_at_mut(i * dim);
        let next = &mut tail[..dim];
        let prev: &[f64] = if i == 0 { &cur } else { &head[(i - 1) * dim..] };
        map_step(ev, prev, p, &mut scratch, next, i)?;
        if i == 0 {
            // `cur` was only needed to seed the first step.
            cur.clear();
        }
    }
    Ok(out)
}

/// Iterate the map `n_steps` times from `u0`, returning only the final state
/// `f^{n_steps}(u0)`.
///
/// `n_steps == 0` returns `u0` unchanged. Stops at the first non-finite iterate
/// with [`MapError::NonFinite`]. Unlike [`iterate_dense`] this keeps only two
/// `dim`-length buffers regardless of `n_steps`.
pub fn iterate_final(
    ev: &dyn Evaluator,
    u0: &[f64],
    p: &[f64],
    n_steps: usize,
) -> Result<Vec<f64>, MapError> {
    let dim = ev.dim();
    debug_assert_eq!(u0.len(), dim, "u0 length must equal the system dimension");
    debug_assert_eq!(p.len(), ev.n_param(), "p length must equal n_param");

    let mut cur = u0.to_vec();
    if n_steps == 0 {
        return Ok(cur);
    }
    let mut scratch = vec![0.0; ev.n_scratch()];
    let mut next = vec![0.0; dim];
    for i in 0..n_steps {
        map_step(ev, &cur, p, &mut scratch, &mut next, i)?;
        std::mem::swap(&mut cur, &mut next);
    }
    Ok(cur)
}

/// The fate of one map trajectory in an ensemble.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MapTrajStatus {
    /// Reached `n_steps`; its row in [`MapEnsembleFinal::states`] is the final
    /// state.
    Ok,
    /// Diverged; its row is `NaN` and the offending iterate is carried here.
    Failed(MapError),
}

impl MapTrajStatus {
    /// Whether the trajectory completed successfully.
    pub fn is_ok(&self) -> bool {
        matches!(self, MapTrajStatus::Ok)
    }
}

/// The result of an ensemble map iteration to a final iterate.
///
/// `states` is a row-major `(n_ic, dim)` buffer of final states (one row per
/// initial condition, in input order); failed rows are all-`NaN`. `status[i]`
/// records each trajectory's fate.
#[derive(Clone, Debug)]
pub struct MapEnsembleFinal {
    /// System dimension (row width of [`states`](MapEnsembleFinal::states)).
    pub dim: usize,
    /// Row-major `(n_ic, dim)` final states; `NaN` rows for diverged trajectories.
    pub states: Vec<f64>,
    /// Per-trajectory fate, length `n_ic`, in input order.
    pub status: Vec<MapTrajStatus>,
}

impl MapEnsembleFinal {
    /// Number of initial conditions iterated.
    pub fn n_ic(&self) -> usize {
        self.status.len()
    }

    /// The final state of trajectory `i` (a `NaN`-filled slice if it diverged).
    ///
    /// Panics if `i >= ` [`n_ic`](MapEnsembleFinal::n_ic).
    pub fn row(&self, i: usize) -> &[f64] {
        &self.states[i * self.dim..(i + 1) * self.dim]
    }

    /// How many trajectories diverged before reaching `n_steps`.
    pub fn n_failed(&self) -> usize {
        self.status.iter().filter(|s| !s.is_ok()).count()
    }
}

/// Iterate a batch of initial conditions to their `f^{n_steps}` in parallel.
///
/// `u0_batch` is a row-major `(n_ic, dim)` buffer; `dim` is taken from `ev`.
/// Each row is iterated independently on a rayon worker over the shared
/// (`Sync`) evaluator — the v2 build-once / share-many pattern. Maps carry no
/// per-step randomness, so the result is identical regardless of thread count
/// (results are collected in input order). A diverging trajectory yields a `NaN`
/// row and a [`MapTrajStatus::Failed`] rather than aborting the batch.
pub fn iterate_ensemble_final(
    ev: &dyn Evaluator,
    u0_batch: &[f64],
    p: &[f64],
    n_steps: usize,
) -> MapEnsembleFinal {
    let dim = ev.dim();
    // Hard asserts on the calling thread, before the fan-out: a ragged batch or
    // zero dim is caller error, and silently flooring `n_ic` would drop
    // trajectories without a trace.
    assert!(dim > 0, "evaluator dimension must be positive");
    assert_eq!(
        u0_batch.len() % dim,
        0,
        "u0_batch length {} is not a multiple of dim {dim}",
        u0_batch.len()
    );
    let n_ic = u0_batch.len() / dim;

    let per_traj: Vec<(Vec<f64>, MapTrajStatus)> = (0..n_ic)
        .into_par_iter()
        .map(|i| {
            let u0 = &u0_batch[i * dim..(i + 1) * dim];
            match iterate_final(ev, u0, p, n_steps) {
                Ok(uf) => (uf, MapTrajStatus::Ok),
                Err(e) => (vec![f64::NAN; dim], MapTrajStatus::Failed(e)),
            }
        })
        .collect();

    let mut states = Vec::with_capacity(n_ic * dim);
    let mut status = Vec::with_capacity(n_ic);
    for (uf, s) in per_traj {
        states.extend_from_slice(&uf);
        status.push(s);
    }
    MapEnsembleFinal {
        dim,
        states,
        status,
    }
}

/// A discrete map ready to iterate: a shared evaluator plus its parameters.
///
/// The map analogue of [`OdeProblem`](crate::OdeProblem). Borrows the
/// [`Evaluator`] (built once, shared across an ensemble's rayon workers since
/// `Evaluator: Sync`) and owns the parameter vector — for a lowered map this is
/// empty, as map parameters are folded into the tape. Construct one, then call
/// [`iterate_dense`](MapProblem::iterate_dense),
/// [`iterate_final`](MapProblem::iterate_final) or
/// [`iterate_ensemble_final`](MapProblem::iterate_ensemble_final).
pub struct MapProblem<'e> {
    ev: &'e dyn Evaluator,
    p: Vec<f64>,
}

impl<'e> MapProblem<'e> {
    /// Bundle an evaluator with its parameter vector (`p.len()` must equal
    /// `ev.n_param()` — typically `0` for a lowered map).
    pub fn new(ev: &'e dyn Evaluator, p: Vec<f64>) -> Self {
        debug_assert_eq!(
            p.len(),
            ev.n_param(),
            "parameter vector length must equal the evaluator's n_param"
        );
        MapProblem { ev, p }
    }

    /// The system dimension.
    pub fn dim(&self) -> usize {
        self.ev.dim()
    }

    /// The borrowed evaluator.
    pub fn evaluator(&self) -> &dyn Evaluator {
        self.ev
    }

    /// The parameter vector.
    pub fn params(&self) -> &[f64] {
        &self.p
    }

    /// Iterate `n_steps` times from `u0`, recording every iterate. See
    /// [`iterate_dense`].
    pub fn iterate_dense(&self, u0: &[f64], n_steps: usize) -> Result<Vec<f64>, MapError> {
        iterate_dense(self.ev, u0, &self.p, n_steps)
    }

    /// Iterate `n_steps` times from `u0`, returning the final state. See
    /// [`iterate_final`].
    pub fn iterate_final(&self, u0: &[f64], n_steps: usize) -> Result<Vec<f64>, MapError> {
        iterate_final(self.ev, u0, &self.p, n_steps)
    }

    /// Iterate a batch of initial conditions to `f^{n_steps}` in parallel. See
    /// [`iterate_ensemble_final`].
    pub fn iterate_ensemble_final(&self, u0_batch: &[f64], n_steps: usize) -> MapEnsembleFinal {
        iterate_ensemble_final(self.ev, u0_batch, &self.p, n_steps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::VmEval;
    use tsdyn_ir::TapeBuilder;
    use tsdyn_vm::Interpreter;

    /// Logistic map `x ← r x (1 - x)` with `r` folded in as a constant (a lowered
    /// map has `n_param == 0`).
    fn logistic(r: f64) -> Interpreter {
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let rc = b.constant(r);
        let one = b.constant(1.0);
        let omx = b.sub(one, x);
        let rx = b.mul(rc, x);
        let nx = b.mul(rx, omx);
        Interpreter::new(b.finish(&[nx], &[], 1, 0).unwrap())
    }

    /// Hénon map `(x, y) ← (1 - a x² + y, b x)` with `a, b` folded in.
    fn henon(a: f64, bcoef: f64) -> Interpreter {
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let y = b.state(1);
        let one = b.constant(1.0);
        let ac = b.constant(a);
        let bc = b.constant(bcoef);
        let xx = b.mul(x, x);
        let axx = b.mul(ac, xx);
        let omaxx = b.sub(one, axx);
        let nx = b.add(omaxx, y);
        let ny = b.mul(bc, x);
        Interpreter::new(b.finish(&[nx, ny], &[], 2, 0).unwrap())
    }

    /// `x ← 2 x` — a deterministic blow-up under iteration.
    fn doubling() -> Interpreter {
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let two = b.constant(2.0);
        let nx = b.mul(two, x);
        Interpreter::new(b.finish(&[nx], &[], 1, 0).unwrap())
    }

    #[test]
    fn dense_records_iterates_after_the_initial_condition() {
        // Logistic at r = 2 has the closed form x_{n+1} = 2 x_n (1 - x_n); check
        // the first few iterates by hand from x0 = 0.1.
        let ev = VmEval::new(logistic(2.0));
        let out = iterate_dense(&ev, &[0.1], &[], 3).unwrap();
        let x1 = 2.0 * 0.1 * (1.0 - 0.1); // 0.18
        let x2 = 2.0 * x1 * (1.0 - x1); // 0.2952
        let x3 = 2.0 * x2 * (1.0 - x2);
        assert!((out[0] - x1).abs() < 1e-14, "x1: {} vs {x1}", out[0]);
        assert!((out[1] - x2).abs() < 1e-14, "x2: {} vs {x2}", out[1]);
        assert!((out[2] - x3).abs() < 1e-14, "x3: {} vs {x3}", out[2]);
    }

    #[test]
    fn logistic_r2_converges_to_one_half() {
        // r = 2 has a stable fixed point at x* = 1/2 for any x0 in (0, 1).
        let ev = VmEval::new(logistic(2.0));
        let xf = iterate_final(&ev, &[0.1], &[], 200).unwrap();
        assert!((xf[0] - 0.5).abs() < 1e-12, "converged to {}", xf[0]);
    }

    #[test]
    fn final_matches_last_dense_row() {
        let ev = VmEval::new(henon(1.4, 0.3));
        let dense = iterate_dense(&ev, &[0.1, 0.2], &[], 50).unwrap();
        let last = &dense[49 * 2..];
        let fin = iterate_final(&ev, &[0.1, 0.2], &[], 50).unwrap();
        assert_eq!(
            fin.as_slice(),
            last,
            "final state must equal the last dense row"
        );
    }

    #[test]
    fn henon_first_step_is_exact() {
        // (x, y) ← (1 - a x² + y, b x) from (0, 0) gives (1, 0); next gives
        // (1 - 1.4, 0.3) = (-0.4, 0.3).
        let ev = VmEval::new(henon(1.4, 0.3));
        let out = iterate_dense(&ev, &[0.0, 0.0], &[], 2).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-15);
        assert!((out[1] - 0.0).abs() < 1e-15);
        assert!((out[2] - (-0.4)).abs() < 1e-15, "x2 = {}", out[2]);
        assert!((out[3] - 0.3).abs() < 1e-15, "y2 = {}", out[3]);
    }

    #[test]
    fn zero_steps_is_identity() {
        let ev = VmEval::new(henon(1.4, 0.3));
        assert!(iterate_dense(&ev, &[0.3, -0.1], &[], 0).unwrap().is_empty());
        assert_eq!(
            iterate_final(&ev, &[0.3, -0.1], &[], 0).unwrap(),
            vec![0.3, -0.1],
            "zero steps returns the initial condition unchanged"
        );
    }

    #[test]
    fn divergence_is_reported_not_silently_returned() {
        // x ← 2 x from x0 = 1 overflows to +inf after ~1024 steps.
        let ev = VmEval::new(doubling());
        let err = iterate_final(&ev, &[1.0], &[], 100_000).unwrap_err();
        assert!(matches!(err, MapError::NonFinite { .. }), "got {err:?}");
        // The dense path reports the same way and stops at the offending iterate.
        let err = iterate_dense(&ev, &[1.0], &[], 100_000).unwrap_err();
        let MapError::NonFinite { step } = err;
        assert!(step < 100_000, "should fail before the full horizon");
    }

    #[test]
    fn ensemble_matches_a_serial_loop() {
        let ev = VmEval::new(henon(1.4, 0.3));
        let n_ic = 64;
        // A spread of finite ICs near the Hénon attractor.
        let u0: Vec<f64> = (0..n_ic)
            .flat_map(|i| {
                let s = i as f64;
                [0.01 * (s % 7.0) - 0.03, 0.005 * (s % 5.0)]
            })
            .collect();
        let ens = iterate_ensemble_final(&ev, &u0, &[], 30);
        assert_eq!(ens.n_ic(), n_ic);
        assert_eq!(ens.n_failed(), 0);
        for i in 0..n_ic {
            let want = iterate_final(&ev, &u0[i * 2..(i + 1) * 2], &[], 30).unwrap();
            assert_eq!(ens.row(i), want.as_slice(), "trajectory {i} differs");
        }
    }

    #[test]
    fn ensemble_is_parallel_equals_serial_under_threads() {
        let ev = VmEval::new(henon(1.4, 0.3));
        let n_ic = 200;
        let u0: Vec<f64> = (0..n_ic)
            .flat_map(|i| [0.0001 * (i as f64 % 11.0), 0.0])
            .collect();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        let parallel = pool.install(|| iterate_ensemble_final(&ev, &u0, &[], 40));
        // Serial reference.
        let mut serial = vec![0.0; n_ic * 2];
        for i in 0..n_ic {
            let want = iterate_final(&ev, &u0[i * 2..(i + 1) * 2], &[], 40).unwrap();
            serial[i * 2..(i + 1) * 2].copy_from_slice(&want);
        }
        for (k, &s) in serial.iter().enumerate() {
            assert_eq!(
                parallel.states[k].to_bits(),
                s.to_bits(),
                "element {k}: parallel {} != serial {s}",
                parallel.states[k],
            );
        }
    }

    #[test]
    fn ensemble_isolates_a_diverged_trajectory() {
        let ev = VmEval::new(doubling());
        // x ← 2x: only the fixed point x0 = 0 stays finite; any nonzero x0
        // overflows over this horizon. Ordering (ok, fail, ok) checks the NaN is
        // isolated to the diverged row, not smeared across its neighbours.
        let u0 = [0.0, 1.0, 0.0];
        let ens = iterate_ensemble_final(&ev, &u0, &[], 100_000);
        assert!(ens.status[0].is_ok());
        assert_eq!(ens.row(0), &[0.0]);
        assert!(matches!(ens.status[1], MapTrajStatus::Failed(_)));
        assert!(ens.row(1)[0].is_nan(), "diverged row must be NaN");
        assert!(ens.status[2].is_ok());
        assert_eq!(ens.row(2), &[0.0]);
        assert_eq!(ens.n_failed(), 1);
    }

    #[test]
    fn problem_facade_carries_params() {
        // A one-parameter map kept as a runtime parameter (n_param = 1) to
        // exercise the facade's parameter plumbing, even though lowered maps
        // fold their parameters in.
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let r = b.param(0);
        let one = b.constant(1.0);
        let omx = b.sub(one, x);
        let rx = b.mul(r, x);
        let nx = b.mul(rx, omx);
        let ev = VmEval::new(Interpreter::new(b.finish(&[nx], &[], 1, 1).unwrap()));
        let prob = MapProblem::new(&ev, vec![2.0]);
        assert_eq!(prob.dim(), 1);
        assert_eq!(prob.params(), &[2.0]);
        let xf = prob.iterate_final(&[0.1], 200).unwrap();
        assert!((xf[0] - 0.5).abs() < 1e-12, "converged to {}", xf[0]);
    }

    #[test]
    fn empty_ensemble_is_handled() {
        let ev = VmEval::new(logistic(2.0));
        let ens = iterate_ensemble_final(&ev, &[], &[], 10);
        assert_eq!(ens.n_ic(), 0);
        assert!(ens.states.is_empty());
    }
}
