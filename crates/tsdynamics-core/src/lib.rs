//! Optional Rust acceleration kernels for TSDynamics.
//!
//! `tsdynamics` is pure-Python and runs without this package; when it is
//! installed, the Python layer can offload right-hand-side evaluation and
//! ensemble integration to the tape VM defined in [`vm`].
//!
//! Nothing here re-implements the symbolic core — the Python side compiles a
//! system's `_equations` to a flat instruction tape and ships the tape arrays
//! across the FFI boundary.  This keeps one source of truth for the math and
//! lets Rust own only the hot numeric loops (RK4 stepping, rayon ensembles).

mod vm;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use vm::{integrate_dense, integrate_final, Tape, Workspace};

/// Toolchain sanity check — returns the crate version string.
#[pyfunction]
fn _version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Build a [`Tape`] from the flat arrays the Python emitter produces.
#[allow(clippy::too_many_arguments)]
fn make_tape(
    ops: PyReadonlyArray1<i32>,
    a: PyReadonlyArray1<i32>,
    b: PyReadonlyArray1<i32>,
    imm: PyReadonlyArray1<f64>,
    outputs: PyReadonlyArray1<i32>,
    n_state: usize,
    n_param: usize,
) -> Tape {
    Tape {
        ops: ops.as_slice().unwrap().to_vec(),
        a: a.as_slice().unwrap().to_vec(),
        b: b.as_slice().unwrap().to_vec(),
        imm: imm.as_slice().unwrap().to_vec(),
        outputs: outputs.as_slice().unwrap().to_vec(),
        n_state,
        n_param,
    }
}

/// Evaluate `du/dt = f(u, p, t)` once — exposed mainly for cross-checking the
/// tape against the Python/symbolic RHS in tests.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn eval_rhs<'py>(
    py: Python<'py>,
    ops: PyReadonlyArray1<i32>,
    a: PyReadonlyArray1<i32>,
    b: PyReadonlyArray1<i32>,
    imm: PyReadonlyArray1<f64>,
    outputs: PyReadonlyArray1<i32>,
    n_state: usize,
    n_param: usize,
    u: PyReadonlyArray1<f64>,
    p: PyReadonlyArray1<f64>,
    t: f64,
) -> Bound<'py, PyArray1<f64>> {
    let tape = make_tape(ops, a, b, imm, outputs, n_state, n_param);
    let u = u.as_slice().unwrap();
    let p = p.as_slice().unwrap();
    let mut regs = vec![0.0; tape.n_reg()];
    let mut deriv = vec![0.0; tape.dim()];
    tape.eval(u, p, t, &mut regs, &mut deriv);
    deriv.into_pyarray(py)
}

/// Integrate one trajectory with fixed-step RK4, returning state at every
/// `t_eval` point as an `(n_t, dim)` array.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn integrate_dense_py<'py>(
    py: Python<'py>,
    ops: PyReadonlyArray1<i32>,
    a: PyReadonlyArray1<i32>,
    b: PyReadonlyArray1<i32>,
    imm: PyReadonlyArray1<f64>,
    outputs: PyReadonlyArray1<i32>,
    n_state: usize,
    n_param: usize,
    u0: PyReadonlyArray1<f64>,
    p: PyReadonlyArray1<f64>,
    t_eval: PyReadonlyArray1<f64>,
    h: f64,
) -> Bound<'py, PyArray2<f64>> {
    let tape = make_tape(ops, a, b, imm, outputs, n_state, n_param);
    let dim = tape.dim();
    let u0 = u0.as_slice().unwrap().to_vec();
    let p = p.as_slice().unwrap().to_vec();
    let t_eval = t_eval.as_slice().unwrap().to_vec();
    let flat = py.detach(|| integrate_dense(&tape, &u0, &p, &t_eval, h));
    let n_t = t_eval.len();
    PyArray1::from_vec(py, flat).reshape([n_t, dim]).unwrap()
}

/// Integrate a batch of initial conditions from `t0` to `t1` in parallel
/// (rayon), returning the final state of each as an `(n_ic, dim)` array.
///
/// This is the basin/ensemble primitive: thousands of independent trajectories,
/// each a GIL-free RK4 sweep on its own worker, with no Python in the hot loop.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn integrate_ensemble_final_py<'py>(
    py: Python<'py>,
    ops: PyReadonlyArray1<i32>,
    a: PyReadonlyArray1<i32>,
    b: PyReadonlyArray1<i32>,
    imm: PyReadonlyArray1<f64>,
    outputs: PyReadonlyArray1<i32>,
    n_state: usize,
    n_param: usize,
    u0_batch: PyReadonlyArray2<f64>,
    p: PyReadonlyArray1<f64>,
    t0: f64,
    t1: f64,
    h: f64,
) -> Bound<'py, PyArray2<f64>> {
    let tape = make_tape(ops, a, b, imm, outputs, n_state, n_param);
    let dim = tape.dim();
    let batch = u0_batch.as_array().to_owned();
    let n_ic = batch.nrows();
    let p = p.as_slice().unwrap().to_vec();

    let flat = py.detach(|| {
        let rows: Vec<Vec<f64>> = (0..n_ic)
            .into_par_iter()
            .map_init(
                || Workspace::for_tape(&tape),
                |ws, i| {
                    let u0: Vec<f64> = batch.row(i).to_vec();
                    integrate_final(&tape, &u0, &p, t0, t1, h, ws)
                },
            )
            .collect();
        let mut out = Vec::with_capacity(n_ic * dim);
        for row in rows {
            out.extend_from_slice(&row);
        }
        out
    });
    PyArray1::from_vec(py, flat).reshape([n_ic, dim]).unwrap()
}

#[pymodule]
fn tsdynamics_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_version, m)?)?;
    m.add_function(wrap_pyfunction!(eval_rhs, m)?)?;
    m.add_function(wrap_pyfunction!(integrate_dense_py, m)?)?;
    m.add_function(wrap_pyfunction!(integrate_ensemble_final_py, m)?)?;
    Ok(())
}
