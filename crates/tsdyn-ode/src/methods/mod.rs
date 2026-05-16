//! Implemented integrators (Butcher data, adaptive drivers).
//!
//! - **explicit** DP5 RK4 [`explicit_dp5_rk4`] + tableau [`butcher`]
//! - **embedded** DP8 Tsit BS3 [`embedded_pairs`], Verner 9 [`vern9::integrate_vern9`]
//! - **implicit** Rosenbrock family [`implicit::rosenbrock`]
//!
//! Add a timestepper by extending [`crate::method::Method`] and branching in [`crate::driver`].
//! **N5** DDE crates will depend on **`tsdyn-solver-base`** for output grids; delay-specific stepping
//! should live in a sibling **`tsdyn-dde`** crate alongside this tree.

pub(crate) mod butcher;
pub(crate) mod embedded_pairs;
pub(crate) mod explicit_dp5_rk4;
pub(crate) mod implicit;
pub(crate) mod vern9;
