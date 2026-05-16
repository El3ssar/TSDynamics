//! PyO3 bindings for the ODE IR evaluator (N2.a).
//!
//! Only the right-hand-side evaluator is exposed at this stage — the
//! Rust stepper suite lands in N2.b. The Python side uses
//! `eval_ode_rhs` to unit-test the SymEngine → IR lowering against
//! JiTCODE's symbolic RHS on the full continuous catalogue.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use tsdyn_core::ir::CompiledOde;

fn decode_ode(bytecode: &Bound<'_, PyBytes>) -> PyResult<CompiledOde> {
    let bytes: &[u8] = bytecode.as_bytes();
    CompiledOde::from_bytes(bytes).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("malformed ODE IR bytecode: {e}"))
    })
}

/// Evaluate the IR-lowered RHS at a single ``(t, y, params)`` triple.
///
/// Returns a fresh ``(dim,)`` ndarray.
#[pyfunction]
pub fn eval_ode_rhs<'py>(
    py: Python<'py>,
    bytecode: &Bound<'py, PyBytes>,
    t: f64,
    y: PyReadonlyArray1<'py, f64>,
    params: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let ode = decode_ode(bytecode)?;
    let y_slice = y.as_slice()?;
    let params_slice = params.as_slice()?;
    if y_slice.len() != ode.dim {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "y length {} != ode dim {}",
            y_slice.len(),
            ode.dim
        )));
    }
    if params_slice.len() != ode.n_params {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "params length {} != ode n_params {}",
            params_slice.len(),
            ode.n_params
        )));
    }
    let mut out = vec![0.0; ode.dim];
    let mut scratch: Vec<f64> = Vec::with_capacity(32);
    py.detach(|| {
        ode.eval_rhs(t, y_slice, params_slice, &mut out, &mut scratch);
    });
    Ok(PyArray1::from_vec(py, out))
}

/// Evaluate the RHS on a batch of ``(t_k, y_k)`` samples sharing the
/// same parameter vector. Used by the regression-test harness to
/// compare against JiTCODE on hundreds of random samples per system
/// without paying the PyO3 trampoline cost per sample.
///
/// ``ts`` has shape ``(N,)`` and ``ys`` has shape ``(N, dim)``.
/// Returns ``(N, dim)``.
#[pyfunction]
pub fn eval_ode_rhs_batch<'py>(
    py: Python<'py>,
    bytecode: &Bound<'py, PyBytes>,
    ts: PyReadonlyArray1<'py, f64>,
    ys: PyReadonlyArray2<'py, f64>,
    params: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let ode = decode_ode(bytecode)?;
    let ts_slice = ts.as_slice()?;
    let ys_arr = ys.as_array();
    let params_slice = params.as_slice()?;
    let n = ts_slice.len();
    let dim = ode.dim;
    if ys_arr.shape() != [n, dim] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "ys shape {:?} doesn't match (N={}, dim={})",
            ys_arr.shape(),
            n,
            dim
        )));
    }
    if params_slice.len() != ode.n_params {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "params length {} != ode n_params {}",
            params_slice.len(),
            ode.n_params
        )));
    }
    let ys_owned = ys_arr.to_owned();
    let mut out_flat = vec![0.0; n * dim];
    let mut scratch: Vec<f64> = Vec::with_capacity(32);
    let mut y_row = vec![0.0; dim];
    py.detach(|| {
        for k in 0..n {
            for j in 0..dim {
                y_row[j] = ys_owned[(k, j)];
            }
            let row = &mut out_flat[k * dim..(k + 1) * dim];
            ode.eval_rhs(ts_slice[k], &y_row, params_slice, row, &mut scratch);
        }
    });
    let arr = numpy::ndarray::Array2::from_shape_vec((n, dim), out_flat)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("reshape: {e}")))?;
    Ok(arr.into_pyarray(py))
}
