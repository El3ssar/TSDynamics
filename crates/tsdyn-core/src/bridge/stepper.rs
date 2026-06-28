//! The resumable ODE stepper handle (stream WS-STEPPER): a durable
//! single-trajectory stepper that owns its evaluator + solver config and carries
//! the live integration point across `advance` / `advance_to_event` calls.
//!
//! Moved verbatim from the original `bridge.rs` (the bridge-split reorg); the
//! per-`dt` byte-identity contract, the resumable event search and the
//! validation are unchanged. Shared plumbing lives in [`super::marshal`].

use tsdyn_engine::event::advance_to_event;
use tsdyn_engine::{integrate_grid, EventHit, IntegrateConfig};
use tsdyn_ir::{Evaluator, Tape};
use tsdyn_solvers::SolverState;

use super::marshal::{
    build_evaluator, build_evaluator_send, build_solver, diverge_msg, event_direction,
    first_step_from_grid, guard_continuous, require_jacobian_if_needed, resolve_solver,
    EngineError,
};

/// A durable, resumable single-trajectory ODE stepper — the Python-free core of
/// the `tsdynamics._rust.OdeStepper` handle (stream WS-STEPPER).
///
/// It owns the built [`Evaluator`] (the interpreter or the Cranelift JIT — the
/// expensive-to-build artefact) and the resolved solver name/tolerances once, and
/// carries the live integration point `(u, t)` across calls. Marching a flow with
/// repeated [`advance`](OdeStepper::advance) / [`advance_to_event`](OdeStepper::advance_to_event)
/// calls therefore never rebuilds or re-marshals the tape — only the live state is
/// threaded — which is what makes a per-`dt` stepping loop (Poincaré refinement,
/// basins over flows) cheap.
///
/// # Why a fresh solver per `advance`
///
/// [`advance`](OdeStepper::advance) builds a **fresh** `Box<dyn Solver>`
/// and a **fresh** [`SolverState`] for each call, then runs the *single-segment* grid
/// integration `[t, t + dt]` with the first step set to `dt` — exactly what the
/// batch [`super::ode::integrate_dense`] path does for one `dt` span. This is
/// deliberate: it makes `advance(dt)` **byte-for-byte identical** to the per-`dt`
/// `integrate_dense` the released `ContinuousSystem.step()` runs, including for the
/// multistep / stiff kernels (`bdf`, the Adams family) whose `Solver` carries
/// history *across* steps — a reused solver would accumulate that history and
/// silently diverge from the per-`dt` path. Building a kernel is cheap (it only
/// allocates its stage buffers); the costly artefact is the evaluator, which is
/// built once and reused. So the handle amortises the real per-step cost (tape
/// build / JIT compile / FFI tape marshalling) while staying answer-exact.
///
/// (Carrying the adaptive step `h` across `advance` calls would be faster still,
/// but a chunked adaptive integration is *not* equal to N single-`dt` integrations
/// — the controller carries its step/error state across nodes — which is exactly
/// the WS-STEPBUF batch-ahead error that broke sensitive consumers such as
/// `max_lyapunov`. The resumable `h` *is* used inside one
/// [`advance_to_event`](OdeStepper::advance_to_event) span, where there is no
/// per-`dt` numerics to preserve.)
pub struct OdeStepper {
    /// The built evaluator (interpreter or JIT), owning its tape — `'static` and
    /// `Send` so the handle (and an `&mut` to it) can cross `Python::detach`.
    ev: Box<dyn Evaluator + Send>,
    /// The resolved registry solver name (from [`resolve_solver`]).
    method: &'static str,
    /// User-requested tolerances for the adaptive kernels.
    rtol: f64,
    atol: f64,
    /// The live integration point, advanced in place by each accepted segment.
    u: Vec<f64>,
    t: f64,
}

impl OdeStepper {
    /// Build a stepper over a validated ODE tape, starting from `(ic, t0)`.
    ///
    /// `method` is resolved through the solver registry (so `"RK45"` reaches
    /// `"rk45"`); an implicit kernel needs a tape compiled `with_jacobian=True` (the
    /// same guard the batch path applies, checked up front so a misbuilt stepper
    /// fails at construction, not mid-march). `jit` selects the Cranelift evaluator.
    pub fn new(
        tape: Tape,
        ic: &[f64],
        t0: f64,
        method: &str,
        rtol: f64,
        atol: f64,
        jit: bool,
    ) -> Result<Self, EngineError> {
        guard_continuous(&tape)?;
        let dim = tape.dim();
        if dim == 0 {
            return Err(EngineError::BadShape(
                "system dimension is zero (the tape has no outputs)".to_string(),
            ));
        }
        if ic.len() < dim {
            return Err(EngineError::BadShape(format!(
                "initial state has length {}, need dim = {dim}",
                ic.len()
            )));
        }
        if let Some(&bad) = ic.iter().take(dim).find(|x| !x.is_finite()) {
            return Err(EngineError::BadShape(format!(
                "initial state must be finite, found {bad}"
            )));
        }
        if !t0.is_finite() {
            return Err(EngineError::BadShape(format!(
                "initial time must be finite, got {t0}"
            )));
        }
        let name = resolve_solver(method)?;
        require_jacobian_if_needed(&tape, name)?;
        let ev = build_evaluator_send(tape, jit)?;
        Ok(OdeStepper {
            ev,
            method: name,
            rtol,
            atol,
            u: ic[..dim].to_vec(),
            t: t0,
        })
    }

    /// System dimension.
    pub fn dim(&self) -> usize {
        self.ev.dim()
    }

    /// The current state (a copy).
    pub fn state(&self) -> Vec<f64> {
        self.u.clone()
    }

    /// The current time.
    pub fn time(&self) -> f64 {
        self.t
    }

    /// Reset the live point to `(u, t)` without rebuilding the evaluator.
    ///
    /// The resume / reseat primitive: a derived stepper (e.g. a Poincaré map
    /// restarting from a refined crossing) reseats the live state cheaply, keeping
    /// the built evaluator and solver configuration.
    pub fn set_state(&mut self, u: &[f64], t: f64) -> Result<(), EngineError> {
        let dim = self.ev.dim();
        if u.len() < dim {
            return Err(EngineError::BadShape(format!(
                "state has length {}, need dim = {dim}",
                u.len()
            )));
        }
        if let Some(&bad) = u.iter().take(dim).find(|x| !x.is_finite()) {
            return Err(EngineError::BadShape(format!(
                "state must be finite, found {bad}"
            )));
        }
        if !t.is_finite() {
            return Err(EngineError::BadShape(format!(
                "time must be finite, got {t}"
            )));
        }
        self.u[..dim].copy_from_slice(&u[..dim]);
        self.t = t;
        Ok(())
    }

    /// Validate the live parameter vector against the evaluator's `n_param`.
    ///
    /// Parameters are passed in per call (not stored) so a live parameter change on
    /// the Python side takes effect on the next `advance`, exactly as the released
    /// `ContinuousSystem.step()` reads `params_vec()` each step.
    fn check_params(&self, p: &[f64]) -> Result<(), EngineError> {
        if p.len() < self.ev.n_param() {
            return Err(EngineError::BadShape(format!(
                "parameter vector has length {}, need n_param = {}",
                p.len(),
                self.ev.n_param()
            )));
        }
        Ok(())
    }

    /// Advance the live point by one `dt` segment and return the new state.
    ///
    /// **Byte-for-byte identical** to the batch [`super::ode::integrate_dense`]
    /// over the two-node grid `[t, t + dt]` (last row): a fresh solver and fresh
    /// [`SolverState`] are built, the first step is set to `(t + dt) − t` (the
    /// grid-derived step the batch path uses), and the single segment is integrated.
    /// The live `(u, t)` is then advanced. `p` is read live each call so a
    /// mid-march parameter change still takes effect. Divergence raises
    /// ([`EngineError::Diverged`]).
    pub fn advance(&mut self, dt: f64, p: &[f64]) -> Result<Vec<f64>, EngineError> {
        self.check_params(p)?;
        if !(dt.is_finite() && dt > 0.0) {
            return Err(EngineError::BadShape(format!(
                "advance step dt must be finite and positive, got {dt}"
            )));
        }
        let tf = self.t + dt;
        // Mirror the batch path exactly: the two-node grid is `[t, tf]` and the
        // first step is `first_step_from_grid([t, tf]) = tf - t` (NOT the raw `dt` —
        // the subtraction recovers the same float the grid path sees).
        let t_eval = [self.t, tf];
        let dim = self.ev.dim();
        let mut solver = build_solver(self.method, self.rtol, self.atol);
        let cfg = IntegrateConfig::new(first_step_from_grid(&t_eval));
        let out = integrate_grid(&*self.ev, &mut *solver, &self.u[..dim], p, &t_eval, &cfg)
            .map_err(|e| EngineError::Diverged(diverge_msg(&e)))?;
        // `out` is the flat `(2, dim)` buffer; the last row is the advanced state.
        let last = &out[dim..2 * dim];
        self.u[..dim].copy_from_slice(last);
        self.t = tf;
        Ok(self.u.clone())
    }

    /// Resumably march toward `t + max_span`, stopping at the first refined crossing
    /// of the event function `g(u, t) = 0` in `direction`.
    ///
    /// The durable analogue of [`super::events::integrate_events_dense`] for one
    /// event, threaded through the engine's [`advance_to_event`]: the live point and
    /// the adaptive step are carried across the *whole* event search (one resumable
    /// integration, not a re-seeded segment per `dt`), so a Poincaré map marches
    /// crossing by crossing without rebuilding the integration. `first_step` seeds
    /// the solver (with the fixed-step `rk4` it *is* the detection step `dt`,
    /// reproducing the Python `PoincareMap` dt-grid march). `g` is a single-output
    /// tape over the full state.
    ///
    /// Returns `(found, t_cross, u_cross, dir)`: `found` is whether a crossing
    /// occurred before `t + max_span`; on a hit `(t_cross, u_cross)` is the refined
    /// crossing and `dir` its sign (`+1`/`-1`), and the live point is advanced one
    /// marching step *past* the crossing (so a repeated call finds the *next*
    /// crossing, never the same one); with no hit the live point is advanced to
    /// `t + max_span` and `found` is `false`. Divergence raises
    /// ([`EngineError::Diverged`]).
    pub fn advance_to_event(
        &mut self,
        g: Tape,
        max_span: f64,
        first_step: f64,
        direction: i32,
        p: &[f64],
    ) -> Result<(bool, f64, Vec<f64>, i32), EngineError> {
        self.check_params(p)?;
        let dim = self.ev.dim();
        if g.dim() != 1 {
            return Err(EngineError::BadShape(format!(
                "event function must have a single output, got {}",
                g.dim()
            )));
        }
        if g.n_state() > dim {
            return Err(EngineError::BadShape(format!(
                "event function reads {} state inputs but the system has dim = {dim}",
                g.n_state()
            )));
        }
        if g.n_param() > p.len() {
            return Err(EngineError::BadShape(format!(
                "event function declares {} parameters but the system has {}",
                g.n_param(),
                p.len()
            )));
        }
        if !(max_span.is_finite() && max_span > 0.0) {
            return Err(EngineError::BadShape(format!(
                "max_span must be finite and positive, got {max_span}"
            )));
        }
        if !(first_step.is_finite() && first_step > 0.0) {
            return Err(EngineError::BadShape(format!(
                "first step (the detection dt) must be finite and positive, got {first_step}"
            )));
        }
        let dir = event_direction(direction)?;
        let g_ev = build_evaluator(g, false)?;
        let mut solver = build_solver(self.method, self.rtol, self.atol);
        let cfg = IntegrateConfig::new(first_step);

        // Build a resumable SolverState seeded from the live point, march it, then
        // copy the advanced live point back. The adaptive step is carried *within*
        // this one event search (its own integration), which is correct: there is no
        // per-`dt` numerics to preserve for the event search (unlike `advance`).
        let mut st =
            SolverState::for_evaluator(&*self.ev, self.u[..dim].to_vec(), self.t, p.to_vec());
        let mut h = cfg.first_step;
        let t1 = self.t + max_span;
        let hit: Option<EventHit> = advance_to_event(
            &*self.ev,
            &mut *solver,
            &mut st,
            &mut h,
            t1,
            &*g_ev,
            dir,
            &cfg,
        )
        .map_err(|e| EngineError::Diverged(diverge_msg(&e)))?;

        // Commit the advanced live point (one step past the crossing, or `t1`).
        self.u[..dim].copy_from_slice(&st.u[..dim]);
        self.t = st.t;

        match hit {
            Some(h) => Ok((true, h.t, h.u, h.direction as i32)),
            None => Ok((false, self.t, vec![0.0; dim], 0)),
        }
    }
}
