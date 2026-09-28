//! NumPy boundary for a prepared, immutable native numerical evaluator.

use numpy::{
    IxDyn, PyArray1, PyArrayDyn, PyArrayMethods, PyReadonlyArray1, PyReadonlyArrayDyn,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;

use crate::{bridge, detached, to_py_err, OwnedTape};

/// Own a validated tape and evaluator; input/output/scratch storage is per call.
#[pyclass(name = "PreparedEvaluator", module = "tsdynamics._rust", frozen)]
pub(crate) struct PyPreparedEvaluator {
    inner: bridge::PreparedEvaluator,
}

impl PyPreparedEvaluator {
    fn evaluate<'py, const CHECKED: bool>(
        &self,
        py: Python<'py>,
        arguments: PyReadonlyArrayDyn<'py, f64>,
        jacobian: bool,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let width = self.inner.input_width();
        let mut shape = arguments.shape().to_vec();
        if shape.last().copied() != Some(width) {
            return Err(PyValueError::new_err(format!(
                "expected trailing argument width {width}, got {shape:?}"
            )));
        }
        let result_width = self.inner.result_width(jacobian).map_err(to_py_err)?;
        let view = arguments.as_array();
        let mut copied = tsdyn_engine::alloc::try_zeroed(1, view.len()).map_err(|_| {
            PyMemoryError::new_err(format!(
                "cannot copy {} input values for prepared evaluation",
                view.len()
            ))
        })?;
        // ndarray's as_slice requires standard row-major layout. The iterator
        // fallback copies logical rows for Fortran, reversed and strided views.
        if let Some(contiguous) = view.as_slice() {
            copied.copy_from_slice(contiguous);
        } else {
            for (destination, source) in copied.iter_mut().zip(view.iter()) {
                *destination = *source;
            }
        }
        let output = detached(py, || {
            if CHECKED {
                self.inner.evaluate_checked(&copied, jacobian)
            } else {
                self.inner.evaluate(&copied, jacobian)
            }
        })
        .map_err(to_py_err)?;
        *shape.last_mut().expect("argument rank checked above") = result_width;
        PyArray1::from_vec(py, output).reshape(IxDyn(&shape))
    }
}

#[pymethods]
impl PyPreparedEvaluator {
    /// Copy and validate the tape once; retain its chosen interpreter/JIT code.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (ops, a, b, imm, outputs, jac_outputs, n_state, n_param, jit=true))]
    fn new(
        py: Python<'_>,
        ops: PyReadonlyArray1<i32>,
        a: PyReadonlyArray1<i32>,
        b: PyReadonlyArray1<i32>,
        imm: PyReadonlyArray1<f64>,
        outputs: PyReadonlyArray1<i32>,
        jac_outputs: PyReadonlyArray1<i32>,
        n_state: usize,
        n_param: usize,
        jit: bool,
    ) -> PyResult<Self> {
        let tape =
            OwnedTape::copy_in(&ops, &a, &b, &imm, &outputs, &jac_outputs, n_state, n_param)?;
        let inner = detached(py, || bridge::PreparedEvaluator::new(tape.build()?, jit))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Evaluate raw primal components with the input's leading batch dimensions.
    fn eval_rhs<'py>(
        &self,
        py: Python<'py>,
        arguments: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        self.evaluate::<false>(py, arguments, false)
    }

    /// Return raw primal components followed by the row-major Jacobian per row.
    fn eval_jac<'py>(
        &self,
        py: Python<'py>,
        arguments: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        self.evaluate::<false>(py, arguments, true)
    }

    /// Require finite arguments and RHS outputs, preserving the raw method above.
    fn eval_rhs_checked<'py>(
        &self,
        py: Python<'py>,
        arguments: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        self.evaluate::<true>(py, arguments, false)
    }

    /// Require finite arguments and the full fused primal/Jacobian output.
    fn eval_jac_checked<'py>(
        &self,
        py: Python<'py>,
        arguments: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        self.evaluate::<true>(py, arguments, true)
    }

    #[getter]
    fn input_width(&self) -> usize {
        self.inner.input_width()
    }

    #[getter]
    fn output_width(&self) -> usize {
        self.inner.output_width()
    }
}
