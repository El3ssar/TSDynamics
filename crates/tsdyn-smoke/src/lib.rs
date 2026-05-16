//! Smoke-test PyO3 extension for milestone R1.
//!
//! Its only job is to prove that the Rust → maturin → Python pipeline works.
//! All real APIs land in later milestones.

use pyo3::prelude::*;

// Drag the placeholder type from tsdyn-core in so the workspace dependency
// graph is exercised end-to-end.
use tsdyn_core::ProblemHandle;

#[pyfunction]
fn add_one(x: i64) -> i64 {
    let _h = ProblemHandle;
    x + 1
}

#[pymodule]
fn _smoke(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add_one, m)?)?;
    Ok(())
}
