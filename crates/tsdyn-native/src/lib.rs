//! PyO3 bindings for the TSDynamics Rust compute layer.
//!
//! Compiles to `tsdynamics._native._core`. The Python facade in
//! `src/tsdynamics/_native/__init__.py` re-exports the user-facing symbols.
//!
//! # Adding R-track kernels
//!
//! New Track-C / Track-E crates are rlibs linked into this cdylib. Register
//! their PyO3 functions here so Python sees them under `_native._core`.
//! Do NOT create a second cdylib — maturin's multi-extension-module support
//! is non-standard; keeping everything in one `.so` is the policy.

use pyo3::prelude::*;

mod maps;
mod ode;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // N1 map kernels.
    m.add_function(wrap_pyfunction!(maps::iterate_map, m)?)?;
    m.add_function(wrap_pyfunction!(maps::lyapunov_spectrum_map, m)?)?;

    // N2 ODE stepper.
    m.add_function(wrap_pyfunction!(ode::eval_ode_rhs, m)?)?;
    m.add_function(wrap_pyfunction!(ode::eval_ode_jacobian, m)?)?;
    m.add_function(wrap_pyfunction!(ode::eval_ode_rhs_batch, m)?)?;
    m.add_function(wrap_pyfunction!(ode::integrate_ode, m)?)?;
    m.add_function(wrap_pyfunction!(ode::lyapunov_spectrum_ode, m)?)?;
    Ok(())
}
