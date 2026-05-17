//! PyO3 bindings for the ODE IR evaluator and the Rust stepper (N2.b).

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use tsdyn_core::ir::CompiledOde;
use tsdyn_ode::integrate_ode_bytes;
use tsdyn_ode::lyapunov_spectrum_ode_bytes;

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

/// Evaluate the IR Jacobian \(\\partial f_i/\\partial y_j\) as ``(dim, dim)`` row-major.
#[pyfunction]
pub fn eval_ode_jacobian<'py>(
    py: Python<'py>,
    bytecode: &Bound<'py, PyBytes>,
    t: f64,
    y: PyReadonlyArray1<'py, f64>,
    params: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let ode = decode_ode(bytecode)?;
    if ode.jacobian.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "ODE bytecode has no Jacobian (has_jacobian=0)",
        ));
    }
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
    let mut jac_flat = vec![0.0_f64; ode.dim * ode.dim];
    let mut scratch: Vec<f64> = Vec::with_capacity(32);
    py.detach(|| {
        ode.eval_jacobian(t, y_slice, params_slice, &mut jac_flat, &mut scratch);
    });
    let arr2 = numpy::ndarray::Array2::from_shape_vec((ode.dim, ode.dim), jac_flat)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("reshape Jacobian: {e}")))?;
    Ok(arr2.into_pyarray(py))
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

/// Integrate an IR-encoded ODE on a uniform output grid (``dt_output``).
///
/// Returns ``(t, y)`` where ``y`` has shape ``(len(t), dim)``.
#[pyfunction]
pub fn integrate_ode<'py>(
    py: Python<'py>,
    bytecode: &Bound<'py, PyBytes>,
    t0: f64,
    tf: f64,
    y0: PyReadonlyArray1<'py, f64>,
    params: PyReadonlyArray1<'py, f64>,
    method: &str,
    dt_output: f64,
    rtol: f64,
    atol: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    let bytes = bytecode.as_bytes();
    let y0s = y0.as_slice()?;
    let pars = params.as_slice()?;
    let (tvec, yflat) = py
        .detach(|| integrate_ode_bytes(bytes, pars, t0, tf, y0s, dt_output, method, rtol, atol))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let n = tvec.len();
    let dim = if n > 0 { yflat.len() / n } else { 0 };
    let arr2 = numpy::ndarray::Array2::from_shape_vec((n, dim), yflat)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("reshape: {e}")))?;
    Ok((PyArray1::from_vec(py, tvec), arr2.into_pyarray(py)))
}

/// Variational QR Lyapunov spectrum on IR bytecode (**N3**).
///
/// ``dt_reortho`` is the wall-clock interval between QR renormalisations (typically
/// match the Python ``dt`` sampling argument). Requires ``has_jacobian=1`` bytecode.
#[pyfunction]
pub fn lyapunov_spectrum_ode<'py>(
    py: Python<'py>,
    bytecode: &Bound<'py, PyBytes>,
    ic: PyReadonlyArray1<'py, f64>,
    params: PyReadonlyArray1<'py, f64>,
    n_exp: usize,
    burn_in: f64,
    final_time: f64,
    dt_reortho: f64,
    method: &str,
    rtol: f64,
    atol: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let bytes = bytecode.as_bytes();
    let ic_s = ic.as_slice()?;
    let pars = params.as_slice()?;
    let exps = py
        .detach(|| {
            lyapunov_spectrum_ode_bytes(
                bytes,
                pars,
                ic_s,
                burn_in,
                final_time,
                dt_reortho,
                n_exp,
                method,
                rtol,
                atol,
            )
        })
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyArray1::from_vec(py, exps))
}
