//! R1 round-trip smoke test, kept alive so the build pipeline is exercised
//! end-to-end on every CI run.

use pyo3::prelude::*;
use tsdyn_core::ProblemHandle;

#[pyfunction]
pub fn add_one(x: i64) -> i64 {
    let _h = ProblemHandle;
    x + 1
}
