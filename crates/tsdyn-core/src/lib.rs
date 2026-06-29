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
//! | [`iterate_ensemble_final`] | `(n_ic, dim)` | parallel map batch → final states |
//! | [`integrate_dense`] | `(n_t, dim)` | one trajectory, sampled on a grid |
//! | [`integrate_dde_dense`] | `(n_t, dim)` | one DDE trajectory (method of steps) |
//! | [`integrate_ensemble_final`] | `(n_ic, dim)` | parallel ODE batch → final states |
//! | [`integrate_sde_dense`] | `(n_t, dim)` | one SDE trajectory (drift + diffusion) |
//! | [`integrate_sde_ensemble_final`] | `(n_ic, dim)` | parallel SDE batch → final states |
//! | [`integrate_events_dense`] | `(K,), (K, dim), …` | crossings of one event over a span |
//! | [`PyOdeStepper`] (`OdeStepper`) | handle | resumable per-`dt` ODE stepper (stream WS-STEPPER) |
//! | [`solvers`] | `list[str]` | registered `method=` names (introspection) |
//!
//! Unlike the stateless free functions above, [`OdeStepper`](PyOdeStepper) is a
//! durable *handle*: it builds the tape/evaluator + solver once and carries the
//! live integration point across `advance(dt)` / `advance_to_event(g)` calls, so a
//! per-`dt` stepping loop never re-marshals the tape. It backs
//! `ContinuousSystem.step()` and is GIL/lifetime-safe (owns its data, releases the
//! GIL during compute, `Send` so Python may finalize it on any thread).
//!
//! Each leading call passes the tape wire arrays
//! `(ops, a, b, imm, outputs, jac_outputs, n_state, n_param)` — exactly the tuple
//! Python's `Tape.to_arrays()` yields — followed by the runtime vectors and
//! solver options. The two SDE entry points pass **two** tapes back to back (the
//! drift, then the diffusion), since a diagonal-Itô step drives both.
//!
//! Both production evaluators are reachable: every continuous entry point takes a
//! `jit` flag selecting the interpreter (`false`) or the Cranelift JIT (`true`),
//! and the two are numerically identical (E-WIRE wired the JIT through the same
//! `&dyn Evaluator` seam the interpreter uses).

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
        EngineError::JitCompile(_) | EngineError::Diverged(_) => PyRuntimeError::new_err(msg),
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

    /// Build the validated [`tsdyn_ir::Tape`] (inside the GIL release),
    /// **consuming** the owned wire arrays.
    ///
    /// `OwnedTape` already holds owned `Vec`s (copied once, in
    /// [`copy_in`](OwnedTape::copy_in), so they could cross the
    /// [`Python::detach`] GIL release). Handing those `Vec`s **by value** to the
    /// engine — decoding `ops` to [`tsdyn_ir::Op`]s in place and moving every
    /// array into [`tsdyn_ir::Tape::from_parts`] — avoids the second allocation
    /// the slice path ([`tsdyn_ir::Tape::from_arrays`], which re-`to_vec`s each
    /// array) would make. The resulting `Tape` is byte-identical: `from_parts`
    /// runs the same validation `from_arrays` does after its copy. `build` takes
    /// `self` because each caller uses the `OwnedTape` exactly once (the system
    /// dimension is read via [`dim`](OwnedTape::dim) *before* the GIL release, and
    /// `build` is the tape's last use, inside the detached closure).
    fn build(self) -> Result<tsdyn_ir::Tape, EngineError> {
        bridge::marshal::build_tape_owned(
            self.ops,
            self.a,
            self.b,
            self.imm,
            self.outputs,
            self.jac_outputs,
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
    jit: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let tape = OwnedTape::copy_in(&ops, &a, &b, &imm, &outputs, &jac_outputs, n_state, n_param)?;
    let dim = tape.dim();
    let ic = vec_f64("ic", &ic)?;
    let flat = py
        .detach(|| bridge::iterate_map(tape.build()?, &ic, steps, jit))
        .map_err(to_py_err)?;
    PyArray1::from_vec(py, flat).reshape([steps, dim])
}

/// Iterate a batch of map initial conditions `steps` times in parallel (rayon),
/// returning each final state as a row of an `(n_ic, dim)` array.
///
/// `ics` is `(n_ic, dim)`. A diverging trajectory yields a row of `NaN` rather
/// than aborting the batch; maps carry no randomness, so the result is
/// independent of thread count.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn iterate_ensemble_final<'py>(
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
    steps: usize,
    jit: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let tape = OwnedTape::copy_in(&ops, &a, &b, &imm, &outputs, &jac_outputs, n_state, n_param)?;
    let dim = tape.dim();
    let ics_vec = ics
        .as_slice()
        .map_err(|_| PyValueError::new_err("ics must be a C-contiguous (n, dim) float64 array"))?
        .to_vec();
    let n_ic = ics.shape()[0];
    let flat = py
        .detach(|| bridge::map_ensemble_final(tape.build()?, &ics_vec, steps, jit))
        .map_err(to_py_err)?;
    PyArray1::from_vec(py, flat).reshape([n_ic, dim])
}

/// Run a whole map orbit-diagram parameter sweep in one call (stream
/// `perf/param-sweep-kernel`), returning the recorded asymptotic states and each
/// value's fate.
///
/// The leading tape arrays describe the lowered map with the **swept** parameter
/// kept as its single runtime `Param` input (`n_param == 1`) — the other
/// parameters fold into constants — so the sweep needs no per-value re-lowering or
/// per-value FFI round-trip. `base_params` is the full runtime parameter vector;
/// `sweep_index` which of its entries each `values[k]` overwrites; `ic` the base
/// initial condition; `components` the state-component indices to record;
/// `transient` / `n_record` the discard / record counts; `carry_state` whether
/// each value resumes from the previous value's final state; `jit` selects the
/// Cranelift evaluator (numerically identical to the interpreter).
///
/// Returns `(points, status)`: `points` is the `(n_values * n_record,
/// n_components)` recorded array, and `status` an `(n_values,)` i64 array of `0`
/// (finite) / `1` (diverged — the value's block is zero, dropped to an empty set
/// by the Python wiring). The per-iterate numerics are byte-for-byte the
/// [`iterate_map`] path.
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn map_param_sweep<'py>(
    py: Python<'py>,
    ops: PyReadonlyArray1<i32>,
    a: PyReadonlyArray1<i32>,
    b: PyReadonlyArray1<i32>,
    imm: PyReadonlyArray1<f64>,
    outputs: PyReadonlyArray1<i32>,
    jac_outputs: PyReadonlyArray1<i32>,
    n_state: usize,
    n_param: usize,
    base_params: PyReadonlyArray1<f64>,
    sweep_index: usize,
    values: PyReadonlyArray1<f64>,
    ic: PyReadonlyArray1<f64>,
    components: PyReadonlyArray1<i64>,
    transient: usize,
    n_record: usize,
    carry_state: bool,
    jit: bool,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<i64>>)> {
    let tape = OwnedTape::copy_in(&ops, &a, &b, &imm, &outputs, &jac_outputs, n_state, n_param)?;
    let base_params = vec_f64("base_params", &base_params)?;
    let values = vec_f64("values", &values)?;
    let ic = vec_f64("ic", &ic)?;
    let components: Vec<usize> = components
        .as_slice()
        .map_err(|_| PyValueError::new_err("components must be a contiguous int64 array"))?
        .iter()
        .map(|&c| c as usize)
        .collect();
    let n_values = values.len();
    let n_components = components.len();
    let (points, status) = py
        .detach(|| {
            bridge::map_param_sweep(
                tape.build()?,
                &base_params,
                sweep_index,
                &values,
                &ic,
                &components,
                transient,
                n_record,
                carry_state,
                jit,
            )
        })
        .map_err(to_py_err)?;
    let points = PyArray1::from_vec(py, points).reshape([n_values * n_record, n_components])?;
    Ok((points, PyArray1::from_vec(py, status)))
}

// ---------------------------------------------------------------------------
// Integration
// ---------------------------------------------------------------------------

/// Integrate one trajectory and sample at every `t_eval` point, returning an
/// `(n_t, dim)` array whose first row is the initial condition.
///
/// `method` resolves through the solver registry (case-insensitively); `rtol` /
/// `atol` configure the built-in adaptive kernels. `jit=True` selects the
/// Cranelift native-code evaluator (`tsdyn-jit`); it is numerically identical to
/// the interpreter (`jit=False`).
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

/// Integrate a delay differential equation through `t_eval`, returning an
/// `(n_t, dim)` array whose first row is the initial condition.
///
/// The leading tape arrays describe the lowered DDE right-hand side over
/// `dim + n_slots` inputs (the extra inputs are the delay slots; parameters fold
/// into the tape, so `n_param == 0`). `slot_components` / `slot_delays` describe
/// each delay slot, and `past_t` / `past_y` (flat `(n_past, dim)`) the user past
/// on `[t0 − max_delay, t0]` — a single sample is a constant past. Only explicit
/// `method`s are supported (the method of steps); `jit=True` selects the
/// Cranelift evaluator (numerically identical to the interpreter).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn integrate_dde_dense<'py>(
    py: Python<'py>,
    ops: PyReadonlyArray1<i32>,
    a: PyReadonlyArray1<i32>,
    b: PyReadonlyArray1<i32>,
    imm: PyReadonlyArray1<f64>,
    outputs: PyReadonlyArray1<i32>,
    jac_outputs: PyReadonlyArray1<i32>,
    n_state: usize,
    n_param: usize,
    slot_components: PyReadonlyArray1<i32>,
    slot_delays: PyReadonlyArray1<f64>,
    ic: PyReadonlyArray1<f64>,
    past_t: PyReadonlyArray1<f64>,
    past_y: PyReadonlyArray1<f64>,
    t_eval: PyReadonlyArray1<f64>,
    method: String,
    rtol: f64,
    atol: f64,
    jit: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let tape = OwnedTape::copy_in(&ops, &a, &b, &imm, &outputs, &jac_outputs, n_state, n_param)?;
    let dim = tape.dim();
    let slot_components = vec_i32("slot_components", &slot_components)?;
    let slot_delays = vec_f64("slot_delays", &slot_delays)?;
    let ic = vec_f64("ic", &ic)?;
    let past_t = vec_f64("past_t", &past_t)?;
    let past_y = vec_f64("past_y", &past_y)?;
    let t_eval = vec_f64("t_eval", &t_eval)?;
    let n_t = t_eval.len();
    let flat = py
        .detach(|| {
            bridge::integrate_dde_dense(
                tape.build()?,
                &slot_components,
                &slot_delays,
                &ic,
                &past_t,
                &past_y,
                &t_eval,
                &method,
                rtol,
                atol,
                jit,
            )
        })
        .map_err(to_py_err)?;
    PyArray1::from_vec(py, flat).reshape([n_t, dim])
}

/// Integrate a batch of initial conditions to `t1` in parallel (rayon), returning
/// each final state as a row of an `(n_ic, dim)` array.
///
/// `ics` is `(n_ic, dim)`. `first_step` is the integration cadence (the user's
/// `dt`): only the first trial step for an adaptive kernel, but the step for the
/// whole run for the fixed-step `rk4` — so it is threaded through to keep the
/// ensemble numerically identical to [`integrate_dense`] for fixed-step methods.
/// A diverging trajectory yields a row of `NaN` rather than aborting the batch;
/// the run is seeded/scheduling-independent.
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
    first_step: f64,
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
                first_step,
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
// SDE integration (two tapes: drift + diffusion)
// ---------------------------------------------------------------------------

/// Integrate one diagonal-Itô SDE trajectory and sample at every `t_eval` point,
/// returning an `(n_t, dim)` array whose first row is the state at `t_eval[0]`.
///
/// The two tapes are passed back to back — first the drift `f`, then the
/// diffusion `g` (the per-component diagonal noise coefficients, carrying `∂g/∂u`
/// for Milstein). `method` is the SDE kernel name (`euler_maruyama` / `milstein`,
/// resolved against the SDE registry), `dt` the fixed step *and* noise scale
/// `√dt`, `seed` the noise stream's seed, and `jit` selects the Cranelift
/// evaluator (numerically identical to the interpreter). Divergence raises
/// `RuntimeError`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn integrate_sde_dense<'py>(
    py: Python<'py>,
    ops_d: PyReadonlyArray1<i32>,
    a_d: PyReadonlyArray1<i32>,
    b_d: PyReadonlyArray1<i32>,
    imm_d: PyReadonlyArray1<f64>,
    outputs_d: PyReadonlyArray1<i32>,
    jac_outputs_d: PyReadonlyArray1<i32>,
    n_state_d: usize,
    n_param_d: usize,
    ops_g: PyReadonlyArray1<i32>,
    a_g: PyReadonlyArray1<i32>,
    b_g: PyReadonlyArray1<i32>,
    imm_g: PyReadonlyArray1<f64>,
    outputs_g: PyReadonlyArray1<i32>,
    jac_outputs_g: PyReadonlyArray1<i32>,
    n_state_g: usize,
    n_param_g: usize,
    ic: PyReadonlyArray1<f64>,
    p: PyReadonlyArray1<f64>,
    t_eval: PyReadonlyArray1<f64>,
    method: String,
    dt: f64,
    seed: u64,
    jit: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let drift = OwnedTape::copy_in(
        &ops_d,
        &a_d,
        &b_d,
        &imm_d,
        &outputs_d,
        &jac_outputs_d,
        n_state_d,
        n_param_d,
    )?;
    let diffusion = OwnedTape::copy_in(
        &ops_g,
        &a_g,
        &b_g,
        &imm_g,
        &outputs_g,
        &jac_outputs_g,
        n_state_g,
        n_param_g,
    )?;
    let dim = drift.dim();
    let ic = vec_f64("ic", &ic)?;
    let p = vec_f64("p", &p)?;
    let t_eval = vec_f64("t_eval", &t_eval)?;
    let n_t = t_eval.len();
    let flat = py
        .detach(|| {
            bridge::sde_integrate_dense(
                drift.build()?,
                diffusion.build()?,
                &ic,
                &p,
                &t_eval,
                &method,
                dt,
                seed,
                jit,
            )
        })
        .map_err(to_py_err)?;
    PyArray1::from_vec(py, flat).reshape([n_t, dim])
}

/// Integrate a batch of SDE initial conditions to `t1` in parallel (rayon),
/// returning each final state as a row of an `(n_ic, dim)` array.
///
/// Drift then diffusion tapes as in [`integrate_sde_dense`]; `ics` is
/// `(n_ic, dim)`. Worker `i` is seeded by `seed_for(seed, i)` so the batch is
/// **parallel == serial** bit-for-bit; a diverging trajectory yields a `NaN` row
/// rather than aborting the batch.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn integrate_sde_ensemble_final<'py>(
    py: Python<'py>,
    ops_d: PyReadonlyArray1<i32>,
    a_d: PyReadonlyArray1<i32>,
    b_d: PyReadonlyArray1<i32>,
    imm_d: PyReadonlyArray1<f64>,
    outputs_d: PyReadonlyArray1<i32>,
    jac_outputs_d: PyReadonlyArray1<i32>,
    n_state_d: usize,
    n_param_d: usize,
    ops_g: PyReadonlyArray1<i32>,
    a_g: PyReadonlyArray1<i32>,
    b_g: PyReadonlyArray1<i32>,
    imm_g: PyReadonlyArray1<f64>,
    outputs_g: PyReadonlyArray1<i32>,
    jac_outputs_g: PyReadonlyArray1<i32>,
    n_state_g: usize,
    n_param_g: usize,
    ics: PyReadonlyArray2<f64>,
    p: PyReadonlyArray1<f64>,
    t0: f64,
    t1: f64,
    method: String,
    dt: f64,
    seed: u64,
    jit: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let drift = OwnedTape::copy_in(
        &ops_d,
        &a_d,
        &b_d,
        &imm_d,
        &outputs_d,
        &jac_outputs_d,
        n_state_d,
        n_param_d,
    )?;
    let diffusion = OwnedTape::copy_in(
        &ops_g,
        &a_g,
        &b_g,
        &imm_g,
        &outputs_g,
        &jac_outputs_g,
        n_state_g,
        n_param_g,
    )?;
    let dim = drift.dim();
    let ics_vec = ics
        .as_slice()
        .map_err(|_| PyValueError::new_err("ics must be a C-contiguous (n, dim) float64 array"))?
        .to_vec();
    let n_ic = ics.shape()[0];
    let p = vec_f64("p", &p)?;
    let flat = py
        .detach(|| {
            bridge::sde_ensemble_final(
                drift.build()?,
                diffusion.build()?,
                &ics_vec,
                &p,
                t0,
                t1,
                &method,
                dt,
                seed,
                jit,
            )
        })
        .map_err(to_py_err)?;
    PyArray1::from_vec(py, flat).reshape([n_ic, dim])
}

// ---------------------------------------------------------------------------
// Event detection (two tapes: right-hand side + event function g)
// ---------------------------------------------------------------------------

/// Integrate `[t0, t1]` and return the crossings of the event function
/// `g(u, t) = 0` in `direction` (`+1` rising / `-1` falling / `0` either),
/// refined with the engine's O(h⁴) cubic-Hermite dense output.
///
/// The two tapes are passed back to back — first the right-hand side `f`, then the
/// single-output event function `g` (`g`'s `outputs.len() == 1`; it reads the full
/// state and may declare fewer parameters than the system, reading the leading
/// slice). `first_step` seeds the solver; with the fixed-step `rk4` it *is* the
/// detection step, so `method="rk4"`, `first_step=dt` reproduces the Python
/// `PoincareMap` dt-grid march and its refinement (answer-identical). `terminal`
/// stops the run at the first crossing. Divergence raises `RuntimeError`.
///
/// Returns `(times, states, t_final, u_final, terminated)`: the `(K,)` crossing
/// times, the `(K, dim)` crossing states, the time and state the run stopped at
/// (to resume the next span exactly), and whether a terminal event stopped it.
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn integrate_events_dense<'py>(
    py: Python<'py>,
    ops: PyReadonlyArray1<i32>,
    a: PyReadonlyArray1<i32>,
    b: PyReadonlyArray1<i32>,
    imm: PyReadonlyArray1<f64>,
    outputs: PyReadonlyArray1<i32>,
    jac_outputs: PyReadonlyArray1<i32>,
    n_state: usize,
    n_param: usize,
    ops_g: PyReadonlyArray1<i32>,
    a_g: PyReadonlyArray1<i32>,
    b_g: PyReadonlyArray1<i32>,
    imm_g: PyReadonlyArray1<f64>,
    outputs_g: PyReadonlyArray1<i32>,
    jac_outputs_g: PyReadonlyArray1<i32>,
    n_state_g: usize,
    n_param_g: usize,
    ic: PyReadonlyArray1<f64>,
    p: PyReadonlyArray1<f64>,
    t0: f64,
    t1: f64,
    first_step: f64,
    direction: i32,
    terminal: bool,
    method: String,
    rtol: f64,
    atol: f64,
    jit: bool,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    f64,
    Bound<'py, PyArray1<f64>>,
    bool,
)> {
    let rhs = OwnedTape::copy_in(&ops, &a, &b, &imm, &outputs, &jac_outputs, n_state, n_param)?;
    let g = OwnedTape::copy_in(
        &ops_g,
        &a_g,
        &b_g,
        &imm_g,
        &outputs_g,
        &jac_outputs_g,
        n_state_g,
        n_param_g,
    )?;
    let dim = rhs.dim();
    let ic = vec_f64("ic", &ic)?;
    let p = vec_f64("p", &p)?;
    let (times, states, t_final, u_final, terminated) = py
        .detach(|| {
            bridge::integrate_events_dense(
                rhs.build()?,
                g.build()?,
                &ic,
                &p,
                t0,
                t1,
                first_step,
                direction,
                terminal,
                &method,
                rtol,
                atol,
                jit,
            )
        })
        .map_err(to_py_err)?;
    let n_hits = times.len();
    let times = PyArray1::from_vec(py, times);
    let states = PyArray1::from_vec(py, states).reshape([n_hits, dim])?;
    let u_final = PyArray1::from_vec(py, u_final);
    Ok((times, states, t_final, u_final, terminated))
}

// ---------------------------------------------------------------------------
// Basin / attractor recurrence FSM (stream perf/basin-march)
// ---------------------------------------------------------------------------

use bridge::{basin_march_flow_bridge, basin_march_map_bridge};
use tsdyn_engine::basin::{BasinMarchOutcome, MarchConfig};

/// Marshal a [`BasinMarchOutcome`] into the flat NumPy tuple the Python wiring
/// rebuilds the mapper from.
///
/// Returns `(labels, att_keys, att_ids, bas_keys, bas_ids, point_ids,
/// point_counts, points_flat, dim)`:
///
/// - `labels` `(n_seeds,)` i64 — per-seed attractor id (or `-1` diverged);
/// - `att_keys`/`att_ids` `(A,)` i64 — the `att_cells` map (flat cell → id);
/// - `bas_keys`/`bas_ids` `(B,)` i64 — the `bas_cells` map;
/// - `point_ids`/`point_counts` `(P,)` i64 — one row per attractor id with its
///   point-cloud row count; `points_flat` `(sum(counts)*dim,)` f64 the row-major
///   clouds, sliced back per id by the cumulative counts; `dim` the row width.
#[allow(clippy::type_complexity)]
fn basin_outcome_to_py(
    py: Python<'_>,
    out: BasinMarchOutcome,
) -> (
    Bound<'_, PyArray1<i64>>,
    Bound<'_, PyArray1<i64>>,
    Bound<'_, PyArray1<i64>>,
    Bound<'_, PyArray1<i64>>,
    Bound<'_, PyArray1<i64>>,
    Bound<'_, PyArray1<i64>>,
    Bound<'_, PyArray1<i64>>,
    Bound<'_, PyArray1<f64>>,
    usize,
) {
    let labels = PyArray1::from_vec(py, out.labels);

    let (att_keys, att_ids): (Vec<i64>, Vec<i64>) = out
        .att_cells
        .into_iter()
        .map(|(k, v)| (k as i64, v))
        .unzip();
    let (bas_keys, bas_ids): (Vec<i64>, Vec<i64>) = out
        .bas_cells
        .into_iter()
        .map(|(k, v)| (k as i64, v))
        .unzip();

    let mut point_ids: Vec<i64> = Vec::with_capacity(out.att_points.len());
    let mut point_counts: Vec<i64> = Vec::with_capacity(out.att_points.len());
    let mut points_flat: Vec<f64> = Vec::new();
    for (id, m, flat) in out.att_points {
        point_ids.push(id);
        point_counts.push(m as i64);
        points_flat.extend_from_slice(&flat);
    }

    (
        labels,
        PyArray1::from_vec(py, att_keys),
        PyArray1::from_vec(py, att_ids),
        PyArray1::from_vec(py, bas_keys),
        PyArray1::from_vec(py, bas_ids),
        PyArray1::from_vec(py, point_ids),
        PyArray1::from_vec(py, point_counts),
        PyArray1::from_vec(py, points_flat),
        out.dim,
    )
}

/// Build the six FSM thresholds from the positional ints the Python wiring passes.
fn march_config(
    max_steps: usize,
    mx_fnd: usize,
    mx_loc: usize,
    mx_att: usize,
    mx_bas: usize,
    mx_lost: usize,
) -> MarchConfig {
    MarchConfig {
        max_steps,
        mx_fnd,
        mx_loc,
        mx_att,
        mx_bas,
        mx_lost,
    }
}

/// Run the basin recurrence FSM over a **flow** (ODE) — the whole per-IC march
/// (stepping + cell-binning + the shared-label early-out) for every seed in one
/// sequential engine call.
///
/// The leading tape arrays are the usual `Tape.to_arrays()` tuple; `p` the live
/// control parameters; `method` resolves through the solver registry (an implicit
/// kernel needs `with_jacobian=True`, rejected up front); `dt` the per-cell-check
/// step (byte-for-byte the `ContinuousSystem.step(dt)` numerics); `grid_lo` /
/// `grid_hi` / `grid_counts` the recurrence cell tessellation; `seeds` an
/// `(n_seeds, dim)` array of initial conditions; the six `mx_*` ints the FSM
/// thresholds; `jit` selects the Cranelift evaluator (numerically identical).
///
/// Returns the flat tuple [`basin_outcome_to_py`] documents — per-seed labels plus
/// the accumulated `att_cells`/`bas_cells`/`att_points` the Python wiring rebuilds
/// the mapper from. Divergence is a `-1` label, never an error; an engine
/// unavailability or a shape error raises (it must not become divergence).
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn basin_march_flow<'py>(
    py: Python<'py>,
    ops: PyReadonlyArray1<i32>,
    a: PyReadonlyArray1<i32>,
    b: PyReadonlyArray1<i32>,
    imm: PyReadonlyArray1<f64>,
    outputs: PyReadonlyArray1<i32>,
    jac_outputs: PyReadonlyArray1<i32>,
    n_state: usize,
    n_param: usize,
    p: PyReadonlyArray1<f64>,
    method: String,
    rtol: f64,
    atol: f64,
    dt: f64,
    grid_lo: PyReadonlyArray1<f64>,
    grid_hi: PyReadonlyArray1<f64>,
    grid_counts: PyReadonlyArray1<i64>,
    seeds: PyReadonlyArray2<f64>,
    max_steps: usize,
    mx_fnd: usize,
    mx_loc: usize,
    mx_att: usize,
    mx_bas: usize,
    mx_lost: usize,
    jit: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
)> {
    let tape = OwnedTape::copy_in(&ops, &a, &b, &imm, &outputs, &jac_outputs, n_state, n_param)?;
    let p = vec_f64("p", &p)?;
    let grid_lo = vec_f64("grid_lo", &grid_lo)?;
    let grid_hi = vec_f64("grid_hi", &grid_hi)?;
    let grid_counts = grid_counts
        .as_slice()
        .map_err(|_| PyValueError::new_err("grid_counts must be a contiguous int64 array"))?
        .to_vec();
    let seeds_vec = seeds
        .as_slice()
        .map_err(|_| {
            PyValueError::new_err("seeds must be a C-contiguous (n_seeds, dim) float64 array")
        })?
        .to_vec();
    let cfg = march_config(max_steps, mx_fnd, mx_loc, mx_att, mx_bas, mx_lost);
    let out = py
        .detach(|| {
            basin_march_flow_bridge(
                tape.build()?,
                &p,
                &method,
                rtol,
                atol,
                dt,
                &grid_lo,
                &grid_hi,
                &grid_counts,
                &seeds_vec,
                cfg,
                jit,
            )
        })
        .map_err(to_py_err)?;
    Ok(basin_outcome_to_py(py, out))
}

/// Run the basin recurrence FSM over a **map** (discrete system).
///
/// As [`basin_march_flow`], but each cell check applies the lowered `_step` map
/// once (no `method`/`rtol`/`atol`/`dt`). The map tape folds its parameters in, so
/// `p` is empty.
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn basin_march_map<'py>(
    py: Python<'py>,
    ops: PyReadonlyArray1<i32>,
    a: PyReadonlyArray1<i32>,
    b: PyReadonlyArray1<i32>,
    imm: PyReadonlyArray1<f64>,
    outputs: PyReadonlyArray1<i32>,
    jac_outputs: PyReadonlyArray1<i32>,
    n_state: usize,
    n_param: usize,
    p: PyReadonlyArray1<f64>,
    grid_lo: PyReadonlyArray1<f64>,
    grid_hi: PyReadonlyArray1<f64>,
    grid_counts: PyReadonlyArray1<i64>,
    seeds: PyReadonlyArray2<f64>,
    max_steps: usize,
    mx_fnd: usize,
    mx_loc: usize,
    mx_att: usize,
    mx_bas: usize,
    mx_lost: usize,
    jit: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    usize,
)> {
    let tape = OwnedTape::copy_in(&ops, &a, &b, &imm, &outputs, &jac_outputs, n_state, n_param)?;
    let p = vec_f64("p", &p)?;
    let grid_lo = vec_f64("grid_lo", &grid_lo)?;
    let grid_hi = vec_f64("grid_hi", &grid_hi)?;
    let grid_counts = grid_counts
        .as_slice()
        .map_err(|_| PyValueError::new_err("grid_counts must be a contiguous int64 array"))?
        .to_vec();
    let seeds_vec = seeds
        .as_slice()
        .map_err(|_| {
            PyValueError::new_err("seeds must be a C-contiguous (n_seeds, dim) float64 array")
        })?
        .to_vec();
    let cfg = march_config(max_steps, mx_fnd, mx_loc, mx_att, mx_bas, mx_lost);
    let out = py
        .detach(|| {
            basin_march_map_bridge(
                tape.build()?,
                &p,
                &grid_lo,
                &grid_hi,
                &grid_counts,
                &seeds_vec,
                cfg,
                jit,
            )
        })
        .map_err(to_py_err)?;
    Ok(basin_outcome_to_py(py, out))
}

// ---------------------------------------------------------------------------
// Resumable ODE stepper handle (stream WS-STEPPER)
// ---------------------------------------------------------------------------

/// A durable, resumable ODE stepper handle over the compiled engine.
///
/// Owns a built tape evaluator (interpreter or JIT) + the resolved solver/tolerances
/// once, and carries the live integration point `(u, t)` across calls — so a
/// per-`dt` stepping loop (`ContinuousSystem.step()`, Poincaré refinement, basins
/// over flows) never rebuilds or re-marshals the tape, only threads the live state.
///
/// `advance(dt, p)` is **byte-for-byte identical** to the batch
/// `integrate_dense([t, t+dt])` the released `step()` runs — a fresh solver and
/// state are built per call (the adaptive controller re-seeds each `dt`, by
/// design), so the multistep / stiff kernels stay answer-exact while the build /
/// marshalling cost is amortised. `advance_to_event(...)` is the resumable
/// crossing primitive (it *does* carry the adaptive step within one event search).
///
/// # GIL / lifetime safety
///
/// The handle owns all its data (the boxed `Send` evaluator, the live state
/// vectors) — it borrows nothing across the FFI boundary, so there is no dangling
/// reference risk. Each method copies its NumPy inputs into owned `Vec`s, then
/// releases the GIL ([`Python::detach`]) around the pure-Rust compute, holding no
/// `Bound`/`Py` reference inside. The class is `Send` (its boxed evaluator and
/// state vectors are all `Send`), so pyo3 may move or drop the handle on any
/// thread: concurrent access is serialized by the GIL and pyo3's runtime borrow
/// check, and `advance` mutates owned data while the GIL is released. It is
/// deliberately *not* `unsendable` — Python may finalize the owning object on a
/// GC/finalizer thread, and an `unsendable` handle raises when dropped off its
/// creating thread (observed under the parallel test runner).
#[pyclass(name = "OdeStepper", module = "tsdynamics._rust")]
struct PyOdeStepper {
    inner: bridge::OdeStepper,
}

#[pymethods]
impl PyOdeStepper {
    /// Build a stepper over an ODE tape, starting from `(ic, t0)`.
    ///
    /// The leading tape arrays are the usual `Tape.to_arrays()` tuple; `method`
    /// resolves through the solver registry; an implicit kernel needs a tape
    /// compiled `with_jacobian=True` (rejected here, at construction). `jit`
    /// selects the Cranelift evaluator (numerically identical to the interpreter).
    #[new]
    #[allow(clippy::too_many_arguments)]
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
        ic: PyReadonlyArray1<f64>,
        t0: f64,
        method: String,
        rtol: f64,
        atol: f64,
        jit: bool,
    ) -> PyResult<Self> {
        let tape =
            OwnedTape::copy_in(&ops, &a, &b, &imm, &outputs, &jac_outputs, n_state, n_param)?;
        let ic = vec_f64("ic", &ic)?;
        let inner = py
            .detach(|| bridge::OdeStepper::new(tape.build()?, &ic, t0, &method, rtol, atol, jit))
            .map_err(to_py_err)?;
        Ok(PyOdeStepper { inner })
    }

    /// The current state, as a `(dim,)` array (a copy).
    fn state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.state().into_pyarray(py)
    }

    /// The current integration time.
    fn time(&self) -> f64 {
        self.inner.time()
    }

    /// The system dimension.
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Reseat the live point to `(u, t)` without rebuilding the evaluator.
    fn set_state(&mut self, py: Python<'_>, u: PyReadonlyArray1<f64>, t: f64) -> PyResult<()> {
        let u = vec_f64("u", &u)?;
        py.detach(|| self.inner.set_state(&u, t)).map_err(to_py_err)
    }

    /// Advance the live point by one `dt` segment; return the new `(dim,)` state.
    ///
    /// `p` is the live control-parameter vector, read each call so a mid-loop
    /// parameter change still takes effect (mirroring `ContinuousSystem.step()`).
    /// Byte-identical to the batch `integrate_dense([t, t+dt])`. Divergence raises
    /// `RuntimeError`.
    fn advance<'py>(
        &mut self,
        py: Python<'py>,
        dt: f64,
        p: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let p = vec_f64("p", &p)?;
        let u = py
            .detach(|| self.inner.advance(dt, &p))
            .map_err(to_py_err)?;
        Ok(u.into_pyarray(py))
    }

    /// Resumably march up to `max_span` ahead, stopping at the first refined
    /// crossing of the single-output event tape `g(u, t) = 0` in `direction`
    /// (`+1` rising / `-1` falling / `0` either).
    ///
    /// The `g` tape arrays follow the constructor's tape convention; `first_step`
    /// seeds the solver (with `method="rk4"` it *is* the detection step `dt`, so the
    /// crossing reproduces the Python `PoincareMap` dt-grid march). Returns
    /// `(found, t_cross, u_cross, direction)`: on a hit the refined crossing and its
    /// sign, with the live point advanced one marching step *past* it (so a repeated
    /// call finds the *next* crossing); with no hit `found` is `False`, the live
    /// point is advanced to `t + max_span`, and `u_cross` is a zero placeholder.
    /// Divergence raises `RuntimeError`.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn advance_to_event<'py>(
        &mut self,
        py: Python<'py>,
        ops_g: PyReadonlyArray1<i32>,
        a_g: PyReadonlyArray1<i32>,
        b_g: PyReadonlyArray1<i32>,
        imm_g: PyReadonlyArray1<f64>,
        outputs_g: PyReadonlyArray1<i32>,
        jac_outputs_g: PyReadonlyArray1<i32>,
        n_state_g: usize,
        n_param_g: usize,
        max_span: f64,
        first_step: f64,
        direction: i32,
        p: PyReadonlyArray1<f64>,
    ) -> PyResult<(bool, f64, Bound<'py, PyArray1<f64>>, i32)> {
        let g = OwnedTape::copy_in(
            &ops_g,
            &a_g,
            &b_g,
            &imm_g,
            &outputs_g,
            &jac_outputs_g,
            n_state_g,
            n_param_g,
        )?;
        let p = vec_f64("p", &p)?;
        let (found, t_cross, u_cross, dir) = py
            .detach(|| {
                self.inner
                    .advance_to_event(g.build()?, max_span, first_step, direction, &p)
            })
            .map_err(to_py_err)?;
        Ok((found, t_cross, u_cross.into_pyarray(py), dir))
    }
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
    // loudly instead — for both the ODE/stiff `Solver` registry and the SDE
    // kernel registry (`tsdyn_solvers::sde`), which E-WIRE now reaches.
    if let Err(dups) = tsdyn_engine::check_solver_registry() {
        return Err(PyRuntimeError::new_err(format!(
            "solver registry has duplicate method names: {dups:?} — two kernels registered \
             under the same name (a build/link bug)"
        )));
    }
    let sde_dups = tsdyn_solvers::sde::duplicates();
    if !sde_dups.is_empty() {
        return Err(PyRuntimeError::new_err(format!(
            "SDE kernel registry has duplicate method names: {sde_dups:?} — two kernels \
             registered under the same name (a build/link bug)"
        )));
    }
    m.add_function(wrap_pyfunction!(eval_rhs, m)?)?;
    m.add_function(wrap_pyfunction!(eval_jac, m)?)?;
    m.add_function(wrap_pyfunction!(iterate_map, m)?)?;
    m.add_function(wrap_pyfunction!(iterate_ensemble_final, m)?)?;
    m.add_function(wrap_pyfunction!(map_param_sweep, m)?)?;
    m.add_function(wrap_pyfunction!(integrate_dense, m)?)?;
    m.add_function(wrap_pyfunction!(integrate_dde_dense, m)?)?;
    m.add_function(wrap_pyfunction!(integrate_ensemble_final, m)?)?;
    m.add_function(wrap_pyfunction!(integrate_sde_dense, m)?)?;
    m.add_function(wrap_pyfunction!(integrate_sde_ensemble_final, m)?)?;
    m.add_function(wrap_pyfunction!(integrate_events_dense, m)?)?;
    m.add_function(wrap_pyfunction!(basin_march_flow, m)?)?;
    m.add_function(wrap_pyfunction!(basin_march_map, m)?)?;
    m.add_class::<PyOdeStepper>()?;
    m.add_function(wrap_pyfunction!(solvers, m)?)?;
    m.add_function(wrap_pyfunction!(_version, m)?)?;
    Ok(())
}
