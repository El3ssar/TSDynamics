//! PyO3 bindings for the TSDynamics Rust compute layer.
//!
//! Compiles to `tsdynamics._native._core`. The Python facade in
//! `src/tsdynamics/_native/__init__.py` re-exports the user-facing symbols.

use pyo3::prelude::*;

mod maps;
mod smoke;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Smoke-test holdover from R1; kept so the round-trip test keeps passing.
    m.add_function(wrap_pyfunction!(smoke::add_one, m)?)?;

    // N1 map kernels.
    m.add_function(wrap_pyfunction!(maps::iterate_map, m)?)?;
    m.add_function(wrap_pyfunction!(maps::lyapunov_spectrum_map, m)?)?;
    Ok(())
}
