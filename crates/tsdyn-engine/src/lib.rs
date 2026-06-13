//! `tsdyn-engine` — wires an `Evaluator` and a `Solver` to a `Problem`.
//!
//! Owns the per-family problem definitions (ODE / DDE / SDE / map), the
//! integrate loop, the rayon ensemble fan-out (already proven race-free in v2),
//! the seeded RNG / Wiener substrate, DDE history ring buffers and event
//! detection. Determinism is a contract: every stochastic/sampling entry takes
//! a seed and ensembles seed per-trajectory-index, so parallel == serial
//! bit-for-bit (ROADMAP §4c).
//!
//! Skeleton only (stream F0). Core integrate + ensembles + RNG land in **stream
//! E5**; the DDE engine in **E-DDE**; the SDE engine in **E-SDE**; the map loop
//! in **E-MAP**. See ROADMAP §4a.
