//! `tsdyn-engine` — wires an [`Evaluator`](tsdyn_ir::Evaluator) and a
//! [`Solver`](tsdyn_solvers::Solver) to a problem and runs it.
//!
//! This crate owns the *driving* layer of the engine: the integrate loop, the
//! rayon ensemble fan-out, and the seeded RNG / Wiener substrate that makes
//! stochastic runs reproducible. It is deliberately agnostic about *which*
//! evaluator (interpreter vs JIT) and *which* solver kernel it drives — it sees
//! both only through their frozen traits (stream F2), so the same loop serves
//! every family and every backend.
//!
//! # What stream E5 lands here
//!
//! - [`rng`] — the [`SplitMix64`](rng::SplitMix64) generator, per-stream seeding
//!   ([`seed_for`](rng::seed_for)), standard-normal draws, and diagonal Wiener
//!   increments — the substrate the SDE family (stream E-SDE) builds on.
//! - [`integrate`] — the single-trajectory step/accept/retry/fail driver:
//!   [`integrate_final`] and [`integrate_grid`], with the
//!   "diverge loudly, never silently" [`IntegrateError`] contract.
//! - [`ensemble`] — [`ensemble_final`], the rayon fan-out over initial
//!   conditions, with a per-trajectory-index solver factory so seeded runs are
//!   **parallel == serial** bit-for-bit.
//! - [`OdeProblem`] — the ergonomic ODE bundle (evaluator + parameters) the
//!   binding layer and derived systems integrate through.
//! - [`check_solver_registry`] — the startup tripwire for duplicate solver
//!   names (the registry contract asks the engine to run this once).
//!
//! # What later streams add (alongside, not editing E5's files)
//!
//! The DDE engine (method of steps, history buffers) is stream **E-DDE**; the
//! SDE problem/kernels that consume [`rng`] are stream **E-SDE**; the discrete
//! map loop is stream **E-MAP**; event detection + dense output is stream
//! **E-EVENT** ([`event`]). Each adds its own module plus one `pub mod`
//! line here (kept append-only). See ROADMAP §4a.

pub mod ensemble;
pub mod integrate;
pub mod problem;
pub mod rng;
// Appended by stream E-MAP (kept append-only — see the module docs above).
pub mod map;
// Appended by stream E-SDE (kept append-only — see the module docs above).
pub mod sde;
// Appended by stream E-DDE (method-of-steps DDE integrator).
pub mod dde;
// Appended by stream E-EVENT (event detection + dense output).
pub mod event;

#[cfg(test)]
mod testkit;

pub use ensemble::{ensemble_final, EnsembleFinal, TrajStatus};
pub use integrate::{integrate_final, integrate_grid, IntegrateConfig, IntegrateError};
pub use problem::OdeProblem;
// Appended by stream E-MAP.
pub use map::{
    iterate_dense, iterate_ensemble_final, iterate_final as iterate_map_final, MapEnsembleFinal,
    MapError, MapProblem, MapTrajStatus,
};
// Appended by stream E-SDE.
pub use sde::{
    sde_ensemble_final, sde_integrate_final, sde_integrate_grid, SdeConfig, SdeEnsembleFinal,
    SdeError, SdeProblem, SdeTrajStatus,
};
// Appended by stream E-DDE.
pub use dde::{integrate_dde_grid, DelaySlot};
// Appended by stream E-EVENT.
pub use event::{integrate_events, EventDirection, EventHit, EventOutcome, EventSpec, HermiteStep};

/// Check that the linked solver registry has no duplicate names, returning the
/// clashing names if any.
///
/// Solver kernels self-register at link time with no central table, so two
/// kernels (e.g. one each from streams E3 and E4) could register the same
/// `method=` name and silently shadow each other. The registry cannot reject
/// that at registration; this is the agreed tripwire the *engine* runs once at
/// startup (the Python binding, stream E7, should call it during module init and
/// turn a non-empty result into a hard error). `Ok(())` means every registered
/// name is unique.
pub fn check_solver_registry() -> Result<(), Vec<&'static str>> {
    let dups = tsdyn_solvers::duplicates();
    if dups.is_empty() {
        Ok(())
    } else {
        Err(dups)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solver_registry_has_no_duplicate_names() {
        // A tripwire for the parallel solver streams: if E3/E4/E-SDE ever
        // register two kernels under one name, this fails in the workspace test
        // run instead of silently shadowing one of them.
        assert_eq!(check_solver_registry(), Ok(()));
    }
}
