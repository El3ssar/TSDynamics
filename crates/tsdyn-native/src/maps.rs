//! PyO3 bindings for the map kernels in `tsdyn-maps`.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use tsdyn_core::ir::CompiledMap;
use tsdyn_maps::{iterate, lyapunov_spectrum};

fn decode_map(bytecode: &Bound<'_, PyBytes>) -> PyResult<CompiledMap> {
    let bytes: &[u8] = bytecode.as_bytes();
    CompiledMap::from_bytes(bytes).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("malformed IR bytecode: {e}"))
    })
}

#[pyfunction]
pub fn iterate_map<'py>(
    py: Python<'py>,
    bytecode: &Bound<'py, PyBytes>,
    ic: PyReadonlyArray1<'py, f64>,
    params: PyReadonlyArray1<'py, f64>,
    steps: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let map = decode_map(bytecode)?;
    let ic_slice = ic.as_slice()?;
    let params_slice = params.as_slice()?;
    if ic_slice.len() != map.dim {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "ic length {} != map dim {}",
            ic_slice.len(),
            map.dim
        )));
    }
    if params_slice.len() != map.n_params {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "params length {} != map n_params {}",
            params_slice.len(),
            map.n_params
        )));
    }
    let out = py.detach(|| iterate(&map, ic_slice, params_slice, steps));
    let out = out.map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("map iterate failed: {e}"))
    })?;
    let arr = Array2::from_shape_vec((steps, map.dim), out)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("reshape: {e}")))?;
    Ok(arr.into_pyarray(py))
}

#[pyfunction]
pub fn lyapunov_spectrum_map<'py>(
    py: Python<'py>,
    bytecode: &Bound<'py, PyBytes>,
    ic: PyReadonlyArray1<'py, f64>,
    params: PyReadonlyArray1<'py, f64>,
    steps: usize,
    n_exp: usize,
    reortho_interval: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let map = decode_map(bytecode)?;
    let ic_slice = ic.as_slice()?;
    let params_slice = params.as_slice()?;
    if ic_slice.len() != map.dim {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "ic length {} != map dim {}",
            ic_slice.len(),
            map.dim
        )));
    }
    if params_slice.len() != map.n_params {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "params length {} != map n_params {}",
            params_slice.len(),
            map.n_params
        )));
    }
    if n_exp == 0 || n_exp > map.dim {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "n_exp {} must be in 1..={}",
            n_exp, map.dim
        )));
    }
    let result = py.detach(|| {
        lyapunov_spectrum(&map, ic_slice, params_slice, steps, n_exp, reortho_interval)
    });
    let result = result.map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("lyapunov_spectrum failed: {e}"))
    })?;
    Ok(result.into_pyarray(py))
}
