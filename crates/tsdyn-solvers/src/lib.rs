//! `tsdyn-solvers` — the `Solver` trait and its kernels, one method per module
//! (explicit RK family, implicit/stiff family, SDE, symplectic, ...).
//!
//! Each kernel calls an `Evaluator` (from `tsdyn-ir`) for RHS/Jacobian values;
//! dense output, error control and step adaption live here. Solvers are
//! auto-registered by name (a `register!`-per-file / `inventory`-style registry)
//! so adding a method is adding one file with no central table to edit — that is
//! what lets two solver streams (E3, E4) run without colliding (ROADMAP §4d).
//!
//! Skeleton only (stream F0). The trait + registry mechanism land in **stream
//! F2**; the explicit family in **E3**, the implicit/stiff family in **E4**,
//! the SDE family in **E-SDE**. See ROADMAP §4a/§4d.
