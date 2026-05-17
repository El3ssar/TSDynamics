//! Top-level [`integrate_ode`] dispatcher over the Rust method catalogue.
//!
//! To add an integrator (DiffEq-style extensibility):
//! 1. Implement the tableau / stage logic under [`crate::methods`].
//! 2. Extend [`crate::method::Method`] and its [`std::str::FromStr`] implementation.
//! 3. Append a match arm below.
//! 4. Extend Python `ContinuousSystem` (`_RUST_NATIVE_METHODS` and aliases) plus tests/docs.

use std::str::FromStr;

use tsdyn_core::ir::CompiledOde;

use tsdyn_solver_base::uniform_time_grid;

use crate::error::IntegrateError;
use crate::method::Method;
use crate::methods::embedded_pairs::{integrate_bs3, integrate_dp8, integrate_tsit5};
use crate::methods::explicit_dp5_rk4::{integrate_dp5, integrate_rk4};
use crate::methods::implicit::rosenbrock;
use crate::methods::vern6;
use crate::methods::vern7;
use crate::methods::vern8;
use crate::methods::vern9;
use crate::rhs::{IrOdeRhs, Rhs};
use crate::util::copy_from;

/// Integrate `dy/dt = f(t,y)` from `t0` to `tf` and return the endpoint only.
///
/// Used by the variational Lyapunov driver (**N3**) for burn-in and short
/// augmented-system segments between QR renormalisations.
pub(crate) fn integrate_segment_explicit<R: Rhs + ?Sized>(
    rhs: &mut R,
    t0: f64,
    tf: f64,
    y0: &[f64],
    method: Method,
    rtol: f64,
    atol: f64,
) -> Result<Vec<f64>, IntegrateError> {
    if !t0.is_finite() || !tf.is_finite() {
        return Err(IntegrateError::BadBytecode(
            "non-finite time in integrate_segment_explicit".into(),
        ));
    }
    if (tf - t0).abs() < 1e-15 {
        return Ok(y0.to_vec());
    }
    let t_grid = vec![t0, tf];
    let dim = rhs.dim();
    let mut y_rows: Vec<Vec<f64>> = vec![vec![0.0; dim]; 2];
    copy_from(y0, &mut y_rows[0]);

    match method {
        Method::Dp5 => {
            crate::methods::explicit_dp5_rk4::integrate_dp5(rhs, &t_grid, &mut y_rows, rtol, atol)?;
        }
        Method::Dp8 => {
            crate::methods::embedded_pairs::integrate_dp8(rhs, &t_grid, &mut y_rows, rtol, atol)?;
        }
        Method::Tsit5 => {
            crate::methods::embedded_pairs::integrate_tsit5(rhs, &t_grid, &mut y_rows, rtol, atol)?;
        }
        Method::Bs3 => {
            crate::methods::embedded_pairs::integrate_bs3(rhs, &t_grid, &mut y_rows, rtol, atol)?;
        }
        Method::Rk4 => {
            crate::methods::explicit_dp5_rk4::integrate_rk4(rhs, &t_grid, &mut y_rows)?;
        }
        Method::Vern6 => {
            vern6::integrate_vern6(rhs, &t_grid, &mut y_rows, rtol, atol)?;
        }
        Method::Vern7 => {
            vern7::integrate_vern7(rhs, &t_grid, &mut y_rows, rtol, atol)?;
        }
        Method::Vern8 => {
            vern8::integrate_vern8(rhs, &t_grid, &mut y_rows, rtol, atol)?;
        }
        Method::Vern9 => {
            vern9::integrate_vern9(rhs, &t_grid, &mut y_rows, rtol, atol)?;
        }
        Method::Rosenbrock23 | Method::Rosenbrock34 | Method::Rodas4 => {
            return Err(IntegrateError::VariationalNeedsExplicit(format!("{method:?}")));
        }
    }

    Ok(std::mem::take(&mut y_rows[1]))
}

/// Integrate `dy/dt = f(t,y)` on the uniform grid implied by `dt_output`.
pub fn integrate_ode(
    ode: &CompiledOde,
    params: &[f64],
    t0: f64,
    tf: f64,
    y0: &[f64],
    dt_output: f64,
    method: Method,
    rtol: f64,
    atol: f64,
) -> Result<(Vec<f64>, Vec<f64>), IntegrateError> {
    if params.len() != ode.n_params {
        return Err(IntegrateError::ParamsLen {
            expected: ode.n_params,
            got: params.len(),
        });
    }
    let dim = ode.dim;
    if y0.len() != dim {
        return Err(IntegrateError::BadBytecode(format!(
            "ic length {} != dim {}",
            y0.len(),
            dim
        )));
    }
    let t_grid = uniform_time_grid(t0, tf, dt_output);
    let nt = t_grid.len();
    let mut y_rows: Vec<Vec<f64>> = vec![vec![0.0; dim]; nt];
    copy_from(y0, &mut y_rows[0]);

    let mut scratch: Vec<f64> = Vec::with_capacity(64);
    let mut rhs = IrOdeRhs {
        ode,
        params,
        scratch: &mut scratch,
    };

    match method {
        Method::Dp5 => integrate_dp5(&mut rhs, &t_grid, &mut y_rows, rtol, atol),
        Method::Dp8 => integrate_dp8(&mut rhs, &t_grid, &mut y_rows, rtol, atol),
        Method::Tsit5 => integrate_tsit5(&mut rhs, &t_grid, &mut y_rows, rtol, atol),
        Method::Bs3 => integrate_bs3(&mut rhs, &t_grid, &mut y_rows, rtol, atol),
        Method::Rk4 => integrate_rk4(&mut rhs, &t_grid, &mut y_rows),
        Method::Vern6 => vern6::integrate_vern6(&mut rhs, &t_grid, &mut y_rows, rtol, atol),
        Method::Vern7 => vern7::integrate_vern7(&mut rhs, &t_grid, &mut y_rows, rtol, atol),
        Method::Vern8 => vern8::integrate_vern8(&mut rhs, &t_grid, &mut y_rows, rtol, atol),
        Method::Vern9 => vern9::integrate_vern9(&mut rhs, &t_grid, &mut y_rows, rtol, atol),
        Method::Rosenbrock23 | Method::Rosenbrock34 | Method::Rodas4 => {
            rosenbrock::integrate_stiff(ode, params, &t_grid, &mut y_rows, method, rtol, atol)
        }
    }?;

    let mut y_flat = Vec::with_capacity(nt * dim);
    for row in y_rows {
        y_flat.extend(row);
    }
    Ok((t_grid, y_flat))
}

/// Decode bytecode then [`integrate_ode`].
pub fn integrate_ode_bytes(
    bytecode: &[u8],
    params: &[f64],
    t0: f64,
    tf: f64,
    y0: &[f64],
    dt_output: f64,
    method: &str,
    rtol: f64,
    atol: f64,
) -> Result<(Vec<f64>, Vec<f64>), IntegrateError> {
    let ode = CompiledOde::from_bytes(bytecode)
        .map_err(|e| IntegrateError::BadBytecode(e.to_string()))?;
    let m = Method::from_str(method.trim())?;
    integrate_ode(&ode, params, t0, tf, y0, dt_output, m, rtol, atol)
}
