//! `tsdyn-core` — the PyO3 binding layer that builds the `tsdynamics._rust`
//! extension module (ROADMAP stream **E7**).
//!
//! Thin by design: every function here marshals NumPy arrays to/from `&[f64]` /
//! `&[i32]` slices (zero-copy in via [`numpy::PyReadonlyArray`]), releases the GIL
//! around the Rust work ([`Python::detach`]), and forwards to the Python-free
//! [`bridge`] module, which drives the engine (`tsdyn-engine`) over the frozen
//! `Evaluator` / `Solver` seams. This is the only crate that knows about Python;
//! everything below it is pure Rust.
//!
//! # The exposed surface (the Python `tsdynamics.engine.run` seam)
//!
//! The function names and positional argument order mirror exactly what
//! `tsdynamics/engine/run.py` calls on the `tsdynamics._rust` module, so that
//! file is the single place the contract is pinned:
//!
//! | function | returns | purpose |
//! |----------|---------|---------|
//! | [`eval_rhs`] | `(dim,)` | one RHS / next-state evaluation |
//! | [`eval_jac`] | `((dim,), (dim*dim,))` | RHS + row-major Jacobian |
//! | [`iterate_map`] | `(steps, dim)` | iterate a discrete map |
//! | [`integrate_dense`] | `(n_t, dim)` | one trajectory, sampled on a grid |
//! | [`integrate_ensemble_final`] | `(n_ic, dim)` | parallel batch → final states |
//! | [`solvers`] | `list[str]` | registered `method=` names (introspection) |
//!
//! Each leading call passes the tape wire arrays
//! `(ops, a, b, imm, outputs, jac_outputs, n_state, n_param)` — exactly the tuple
//! Python's `Tape.to_arrays()` yields — followed by the runtime vectors and
//! solver options.

mod bridge;

use bridge::EngineError;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Map an [`EngineError`] to the Python exception type its variant implies.
fn to_py_err(e: EngineError) -> PyErr {
    let msg = e.to_string();
    match e {
        EngineError::BadTape(_) | EngineError::BadShape(_) | EngineError::UnknownMethod(_) => {
            PyValueError::new_err(msg)
        }
        EngineError::Unsupported(_) => PyNotImplementedError::new_err(msg),
        EngineError::Diverged(_) => PyRuntimeError::new_err(msg),
    }
}

/// Copy a contiguous `i32` view into an owned `Vec` (so it can cross
/// [`Python::detach`], which forbids holding a `Bound` reference).
fn vec_i32(name: &str, a: &PyReadonlyArray1<i32>) -> PyResult<Vec<i32>> {
    Ok(a.as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a contiguous int32 array")))?
        .to_vec())
}

/// Copy a contiguous `f64` view into an owned `Vec`.
fn vec_f64(name: &str, a: &PyReadonlyArray1<f64>) -> PyResult<Vec<f64>> {
    Ok(a.as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a contiguous float64 array")))?
        .to_vec())
}

/// The six tape wire arrays, owned (ready to cross the GIL release).
struct OwnedTape {
    ops: Vec<i32>,
    a: Vec<i32>,
    b: Vec<i32>,
    imm: Vec<f64>,
    outputs: Vec<i32>,
    jac_outputs: Vec<i32>,
    n_state: usize,
    n_param: usize,
}

impl OwnedTape {
    #[allow(clippy::too_many_arguments)]
    fn copy_in(
        ops: &PyReadonlyArray1<i32>,
        a: &PyReadonlyArray1<i32>,
        b: &PyReadonlyArray1<i32>,
        imm: &PyReadonlyArray1<f64>,
        outputs: &PyReadonlyArray1<i32>,
        jac_outputs: &PyReadonlyArray1<i32>,
        n_state: usize,
        n_param: usize,
    ) -> PyResult<Self> {
        Ok(OwnedTape {
            ops: vec_i32("ops", ops)?,
            a: vec_i32("a", a)?,
            b: vec_i32("b", b)?,
            imm: vec_f64("imm", imm)?,
            outputs: vec_i32("outputs", outputs)?,
            jac_outputs: vec_i32("jac_outputs", jac_outputs)?,
            n_state,
            n_param,
        })
    }

    /// System dimension — the number of derivative / next-state outputs.
    fn dim(&self) -> usize {
        self.outputs.len()
    }

    /// Build the validated [`tsdyn_ir::Tape`] (inside the GIL release).
    fn build(&self) -> Result<tsdyn_ir::Tape, EngineError> {
        bridge::build_tape(
            &self.ops,
            &self.a,
            &self.b,
            &self.imm,
            &self.outputs,
            &self.jac_outputs,
            self.n_state,
            self.n_param,
        )
    }
}

// ---------------------------------------------------------------------------
// Pointwise evaluation
// ---------------------------------------------------------------------------

/// Evaluate `du/dt = f(u, p, t)` (or the next state, for a map tape) once.
///
/// Returns the `(dim,)` derivative. The strongest, divergence-free signal that a
/// system lowers correctly — used by `tsdynamics.engine.run.eval_rhs`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn eval_rhs<'py>(
    py: Python<'py>,
    ops: PyReadonlyArray1<i32>,
    a: PyReadonlyArray1<i32>,
    b: PyReadonlyArray1<i32>,
    imm: PyReadonlyArray1<f64>,
    outputs: PyReadonlyArray1<i32>,
    jac_outputs: PyReadonlyArray1<i32>,
    n_state: usize,
    n_param: usize,
    u: PyReadonlyArray1<f64>,
    p: PyReadonlyArray1<f64>,
    t: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let tape = OwnedTape::copy_in(&ops, &a, &b, &imm, &outputs, &jac_outputs, n_state, n_param)?;
    let u = vec_f64("u", &u)?;
    let p = vec_f64("p", &p)?;
    let deriv = py
        .detach(|| bridge::eval_rhs(tape.build()?, &u, &p, t))
        .map_err(to_py_err)?;
    Ok(deriv.into_pyarray(py))
}

/// Evaluate `(du/dt, Jacobian)` once; the Jacobian is the flat row-major
/// `(dim*dim,)` buffer (`jac[k*dim + j] = ∂f_k/∂u_j`).
///
/// Requires a tape carrying a Jacobian (built with `with_jacobian=True`).
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn eval_jac<'py>(
    py: Python<'py>,
    ops: PyReadonlyArray1<i32>,
    a: PyReadonlyArray1<i32>,
    b: PyReadonlyArray1<i32>,
    imm: PyReadonlyArray1<f64>,
    outputs: PyReadonlyArray1<i32>,
    jac_outputs: PyReadonlyArray1<i32>,
    n_state: usize,
    n_param: usize,
    u: PyReadonlyArray1<f64>,
    p: PyReadonlyArray1<f64>,
    t: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let tape = OwnedTape::copy_in(&ops, &a, &b, &imm, &outputs, &jac_outputs, n_state, n_param)?;
    let u = vec_f64("u", &u)?;
    let p = vec_f64("p", &p)?;
    let (deriv, jac) = py
        .detach(|| bridge::eval_jac(tape.build()?, &u, &p, t))
        .map_err(to_py_err)?;
    Ok((deriv.into_pyarray(py), jac.into_pyarray(py)))
}

// ---------------------------------------------------------------------------
// Map iteration
// ---------------------------------------------------------------------------

/// Iterate a discrete-map tape `steps` times from `ic`, returning the state after
/// each step as a `(steps, dim)` array (the initial condition is not included).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn iterate_map<'py>(
    py: Python<'py>,
    ops: PyReadonlyArray1<i32>,
    a: PyReadonlyArray1<i32>,
    b: PyReadonlyArray1<i32>,
    imm: PyReadonlyArray1<f64>,
    outputs: PyReadonlyArray1<i32>,
    jac_outputs: PyReadonlyArray1<i32>,
    n_state: usize,
    n_param: usize,
    ic: PyReadonlyArray1<f64>,
    steps: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let tape = OwnedTape::copy_in(&ops, &a, &b, &imm, &outputs, &jac_outputs, n_state, n_param)?;
    let dim = tape.dim();
    let ic = vec_f64("ic", &ic)?;
    let flat = py
        .detach(|| bridge::iterate_map(tape.build()?, &ic, steps))
        .map_err(to_py_err)?;
    PyArray1::from_vec(py, flat).reshape([steps, dim])
}

// ---------------------------------------------------------------------------
// Integration
// ---------------------------------------------------------------------------

/// Integrate one trajectory and sample at every `t_eval` point, returning an
/// `(n_t, dim)` array whose first row is the initial condition.
///
/// `method` resolves through the solver registry (case-insensitively); `rtol` /
/// `atol` configure the built-in adaptive kernels. `jit=True` is reserved for the
/// Cranelift evaluator (stream E2) and currently raises `NotImplementedError`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn integrate_dense<'py>(
    py: Python<'py>,
    ops: PyReadonlyArray1<i32>,
    a: PyReadonlyArray1<i32>,
    b: PyReadonlyArray1<i32>,
    imm: PyReadonlyArray1<f64>,
    outputs: PyReadonlyArray1<i32>,
    jac_outputs: PyReadonlyArray1<i32>,
    n_state: usize,
    n_param: usize,
    ic: PyReadonlyArray1<f64>,
    p: PyReadonlyArray1<f64>,
    t_eval: PyReadonlyArray1<f64>,
    method: String,
    rtol: f64,
    atol: f64,
    jit: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let tape = OwnedTape::copy_in(&ops, &a, &b, &imm, &outputs, &jac_outputs, n_state, n_param)?;
    let dim = tape.dim();
    let ic = vec_f64("ic", &ic)?;
    let p = vec_f64("p", &p)?;
    let t_eval = vec_f64("t_eval", &t_eval)?;
    let n_t = t_eval.len();
    let flat = py
        .detach(|| {
            bridge::integrate_dense(tape.build()?, &ic, &p, &t_eval, &method, rtol, atol, jit)
        })
        .map_err(to_py_err)?;
    PyArray1::from_vec(py, flat).reshape([n_t, dim])
}

/// Integrate a batch of initial conditions to `t1` in parallel (rayon), returning
/// each final state as a row of an `(n_ic, dim)` array.
///
/// `ics` is `(n_ic, dim)`. A diverging trajectory yields a row of `NaN` rather
/// than aborting the batch; the run is seeded/scheduling-independent.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn integrate_ensemble_final<'py>(
    py: Python<'py>,
    ops: PyReadonlyArray1<i32>,
    a: PyReadonlyArray1<i32>,
    b: PyReadonlyArray1<i32>,
    imm: PyReadonlyArray1<f64>,
    outputs: PyReadonlyArray1<i32>,
    jac_outputs: PyReadonlyArray1<i32>,
    n_state: usize,
    n_param: usize,
    ics: PyReadonlyArray2<f64>,
    p: PyReadonlyArray1<f64>,
    t0: f64,
    t1: f64,
    method: String,
    rtol: f64,
    atol: f64,
    jit: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let tape = OwnedTape::copy_in(&ops, &a, &b, &imm, &outputs, &jac_outputs, n_state, n_param)?;
    let dim = tape.dim();
    let ics_vec = ics
        .as_slice()
        .map_err(|_| PyValueError::new_err("ics must be a C-contiguous (n, dim) float64 array"))?
        .to_vec();
    let n_ic = ics.shape()[0];
    let p = vec_f64("p", &p)?;
    let flat = py
        .detach(|| {
            bridge::ensemble_final(
                tape.build()?,
                &ics_vec,
                &p,
                t0,
                t1,
                &method,
                rtol,
                atol,
                jit,
            )
        })
        .map_err(to_py_err)?;
    PyArray1::from_vec(py, flat).reshape([n_ic, dim])
}

// ---------------------------------------------------------------------------
// Introspection
// ---------------------------------------------------------------------------

/// The registered solver `method=` names, sorted.
///
/// Doubles as the dead-strip smoke test the registry contract asks E7 to add: a
/// non-empty result proves the link-time `register_solver!` records survived into
/// the `cdylib` (a linker that GC'd them would leave this empty).
#[pyfunction]
fn solvers() -> Vec<String> {
    tsdyn_solvers::available()
        .into_iter()
        .map(|s| s.to_string())
        .collect()
}

/// The crate version string — a toolchain/import sanity check.
#[pyfunction]
fn _version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// The `tsdynamics._rust` engine extension.
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // The duplicate-name tripwire the registry contract asks the engine to run
    // once at startup: link-time registration cannot reject a clash, so two
    // kernels sharing a name would silently shadow each other. Fail the import
    // loudly instead.
    if let Err(dups) = tsdyn_engine::check_solver_registry() {
        return Err(PyRuntimeError::new_err(format!(
            "solver registry has duplicate method names: {dups:?} — two kernels registered \
             under the same name (a build/link bug)"
        )));
    }
    m.add_function(wrap_pyfunction!(eval_rhs, m)?)?;
    m.add_function(wrap_pyfunction!(eval_jac, m)?)?;
    m.add_function(wrap_pyfunction!(iterate_map, m)?)?;
    m.add_function(wrap_pyfunction!(integrate_dense, m)?)?;
    m.add_function(wrap_pyfunction!(integrate_ensemble_final, m)?)?;
    m.add_function(wrap_pyfunction!(solvers, m)?)?;
    m.add_function(wrap_pyfunction!(_version, m)?)?;
    Ok(())
}
