//! Uniform-output-grid ODE integration (N2.b).

mod butcher;
mod controller;
mod erk_extra;
mod lu;
pub mod rhs;
mod rosenbrock;
mod util;
mod vern9;
mod vern9_extra;
mod vern9_interp;
mod vern9_main;

use std::fmt;
use std::str::FromStr;

pub use rhs::{IrOdeRhs, Rhs};
use tsdyn_core::ir::CompiledOde;

use crate::butcher::dopri54;
use crate::controller::adapt_step;
use crate::erk_extra::{integrate_bs3, integrate_dp8, integrate_tsit5};
use crate::util::{axpy, copy_from, dense_dp5_eval};

#[derive(Debug, Clone)]
pub enum IntegrateError {
    BadBytecode(String),
    BadMethod(String),
    Diverged { t: f64 },
    ParamsLen { expected: usize, got: usize },
    MissingJacobian { method: String },
}

impl fmt::Display for IntegrateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadBytecode(s) => write!(f, "bad ODE bytecode: {s}"),
            Self::BadMethod(s) => write!(f, "unknown ODE method: {s}"),
            Self::Diverged { t } => write!(f, "non-finite state at t={t}"),
            Self::ParamsLen { expected, got } => {
                write!(f, "params length mismatch: expected {expected}, got {got}")
            }
            Self::MissingJacobian { method } => write!(
                f,
                "method {method} requires a Jacobian in the IR bytecode (has_jacobian=false)"
            ),
        }
    }
}

impl std::error::Error for IntegrateError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Method {
    Dp5,
    Dp8,
    Tsit5,
    Vern9,
    Bs3,
    Rk4,
    Rosenbrock23,
    Rosenbrock34,
    Rodas4,
}

impl FromStr for Method {
    type Err = IntegrateError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_uppercase().as_str() {
            "DP5" => Ok(Self::Dp5),
            "DP8" => Ok(Self::Dp8),
            "TSIT5" => Ok(Self::Tsit5),
            "VERN9" => Ok(Self::Vern9),
            "BS3" => Ok(Self::Bs3),
            "RK4" => Ok(Self::Rk4),
            "ROSENBROCK23" => Ok(Self::Rosenbrock23),
            "ROSENBROCK34" => Ok(Self::Rosenbrock34),
            "RODAS4" => Ok(Self::Rodas4),
            _ => Err(IntegrateError::BadMethod(s.to_string())),
        }
    }
}

fn output_grid(t0: f64, tf: f64, dt: f64) -> Vec<f64> {
    if dt <= 0.0 {
        return vec![t0, tf];
    }
    let mut t_arr = Vec::new();
    let mut t = t0;
    while t < tf - 1e-12 {
        t_arr.push(t);
        t += dt;
    }
    if t_arr.is_empty() || (tf - *t_arr.last().unwrap()).abs() > 1e-12 {
        t_arr.push(tf);
    }
    t_arr
}

pub(crate) fn all_finite(v: &[f64]) -> bool {
    v.iter().all(|x| x.is_finite())
}

pub(crate) fn h_init<R: Rhs + ?Sized>(
    rhs: &mut R,
    t0: f64,
    y0: &[f64],
    tf: f64,
    rtol: f64,
    atol: f64,
    f0: &mut [f64],
    y1: &mut [f64],
    f1: &mut [f64],
) -> f64 {
    let d = rhs.dim();
    rhs.eval(t0, y0, f0);
    let mut d0 = 0.0;
    let mut d1 = 0.0;
    for i in 0..d {
        let sci = atol + y0[i].abs() * rtol;
        d0 += (y0[i] / sci).powi(2);
        d1 += (f0[i] / sci).powi(2);
    }
    let tol = 1e-10_f64;
    let span = (tf - t0).abs();
    let mut h0 = if d0 < tol || d1 < tol {
        1e-6_f64
    } else {
        0.01 * (d0 / d1).sqrt()
    };
    h0 = h0.min(span);
    let dir = (tf - t0).signum();
    if dir == 0.0 {
        return span.max(1e-15);
    }
    for i in 0..d {
        y1[i] = y0[i] + dir * h0 * f0[i];
    }
    rhs.eval(t0 + dir * h0, y1, f1);
    let mut d2 = 0.0;
    for i in 0..d {
        let sci = atol + y0[i].abs() * rtol;
        let q = (f1[i] - f0[i]) / sci;
        d2 += q * q;
    }
    d2 = d2.sqrt() / h0.abs().max(1e-15);
    let h1 = if d1.sqrt().max(d2.abs()) <= 1e-15 {
        1e-6_f64.max(h0.abs() * 1e-3)
    } else {
        0.01 / d1.sqrt().max(d2)
    };
    let fac: f64 = 100.0;
    let rel_floor = (1e-12_f64 * span).max(1e-15_f64);
    dir * (fac * h0.abs()).min(h1.min(span)).max(rel_floor)
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
    let t_grid = output_grid(t0, tf, dt_output);
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

/// Decode bytecode then :func:`integrate_ode`.
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

// ─── DP5 ───────────────────────────────────────────────────────────────────

fn integrate_dp5<R: Rhs + ?Sized>(
    rhs: &mut R,
    t_grid: &[f64],
    y_out: &mut [Vec<f64>],
    rtol: f64,
    atol: f64,
) -> Result<(), IntegrateError> {
    let dim = rhs.dim();
    let tf = *t_grid.last().unwrap();
    let t0 = t_grid[0];

    let mut k = vec![vec![0.0_f64; dim]; 7];
    let mut y_tmp = vec![0.0_f64; dim];
    let mut y_next = vec![0.0_f64; dim];
    let mut errw = vec![0.0_f64; dim];
    let mut rcont: [Vec<f64>; 5] = std::array::from_fn(|_| vec![0.0_f64; dim]);

    let mut y = y_out[0].clone();
    let mut t = t0;
    rhs.eval(t, &y, &mut k[0]);

    let mut f_tmp = vec![0.0_f64; dim];
    let mut y1 = vec![0.0_f64; dim];
    let mut f1 = vec![0.0_f64; dim];
    let mut h = h_init(rhs, t0, &y, tf, rtol, atol, &mut f_tmp, &mut y1, &mut f1);
    rhs.eval(t, &y, &mut k[0]);

    let mut i_out = 1usize;

    while i_out < t_grid.len() {
        let t_target = t_grid[i_out];
        while t < t_target - 1e-14 {
            let mut last = false;
            let dir = (tf - t0).signum();
            if (t + 1.01 * h - tf) * dir >= 0.0 {
                h = tf - t;
                last = true;
            }

            let t_old = t;
            let h_step = h;

            // ERK stages — after this, `k[1]..k[6]` hold stages 2–7.
            for s in 1..7 {
                for i in 0..dim {
                    y_tmp[i] = y[i];
                }
                for j in 0..s {
                    let a = dopri54::a(s + 1, j + 1);
                    if a != 0.0 {
                        axpy(h_step * a, &k[j], &mut y_tmp);
                    }
                }
                rhs.eval(t + h_step * dopri54::c(s + 1), &y_tmp, &mut k[s]);
            }

            // High-order solution: row 7 of the tableau (depends on k[0]..k[5] only).
            for i in 0..dim {
                y_next[i] = y[i];
            }
            for j in 0..6 {
                let a = dopri54::a(7, j + 1);
                if a != 0.0 {
                    axpy(h_step * a, &k[j], &mut y_next);
                }
            }

            // Slot `k[1]` is repurposed to hold the 7th-stage derivative (FSAL).
            for i in 0..dim {
                k[1][i] = k[6][i];
            }

            // dense rcont[4] (uses repurposed k[1] == old k[6])
            for i in 0..dim {
                rcont[4][i] = h_step
                    * (dopri54::d(1) * k[0][i]
                        + dopri54::d(3) * k[2][i]
                        + dopri54::d(4) * k[3][i]
                        + dopri54::d(5) * k[4][i]
                        + dopri54::d(6) * k[5][i]
                        + dopri54::d(7) * k[1][i]);
            }

            // embedded error — reuse buffer `errw`, then copy into `k[3]` for RMS
            for i in 0..dim {
                errw[i] = h_step
                    * (dopri54::e(1) * k[0][i]
                        + dopri54::e(3) * k[2][i]
                        + dopri54::e(4) * k[3][i]
                        + dopri54::e(5) * k[4][i]
                        + dopri54::e(6) * k[5][i]
                        + dopri54::e(7) * k[1][i]);
            }
            copy_from(&errw, &mut k[3]);

            let mut err2 = 0.0_f64;
            for i in 0..dim {
                let sci = atol + y[i].abs().max(y_next[i].abs()) * rtol;
                let ri = k[3][i] / sci;
                err2 += ri * ri;
            }
            let err = (err2 / dim as f64).sqrt();

            let (acc, h_new) = adapt_step(err, h, 5, 0.9, 0.2, 10.0);
            if acc {
                if !all_finite(&y_next) {
                    return Err(IntegrateError::Diverged { t: t + h_step });
                }
                // dense rcont[0..3]
                for i in 0..dim {
                    let ydiff = y_next[i] - y[i];
                    let bspl = k[0][i] * h_step - ydiff;
                    rcont[0][i] = y[i];
                    rcont[1][i] = ydiff;
                    rcont[2][i] = bspl;
                    rcont[3][i] = -k[1][i] * h_step + ydiff - bspl;
                }

                let t_new = t + h_step;
                while i_out < t_grid.len() && t_grid[i_out] <= t_new + 1e-12 {
                    let tg = t_grid[i_out];
                    let theta = (tg - t_old) / h_step;
                    let theta1 = 1.0 - theta;
                    dense_dp5_eval(&rcont, theta, theta1, &mut y_out[i_out]);
                    if !all_finite(&y_out[i_out]) {
                        return Err(IntegrateError::Diverged { t: tg });
                    }
                    i_out += 1;
                }

                for i in 0..dim {
                    k[0][i] = k[1][i];
                }
                y.clone_from(&y_next);
                t = t_new;
                h = if last { h_new } else { h_new };
                if (t - tf).abs() < 1e-12 {
                    break;
                }
            } else {
                h = h_new;
            }
        }
        if i_out >= t_grid.len() {
            break;
        }
    }
    Ok(())
}

fn integrate_rk4<R: Rhs + ?Sized>(
    rhs: &mut R,
    t_grid: &[f64],
    y_out: &mut [Vec<f64>],
) -> Result<(), IntegrateError> {
    let dim = rhs.dim();
    let mut k1 = vec![0.0_f64; dim];
    let mut k2 = vec![0.0_f64; dim];
    let mut k3 = vec![0.0_f64; dim];
    let mut k4v = vec![0.0_f64; dim];
    let mut yt = vec![0.0_f64; dim];
    for i in 0..t_grid.len().saturating_sub(1) {
        let t0 = t_grid[i];
        let h = t_grid[i + 1] - t0;
        if !h.is_finite() || h == 0.0 {
            let src = y_out[i].clone();
            copy_from(&src, &mut y_out[i + 1]);
            continue;
        }
        copy_from(&y_out[i], &mut yt);
        rhs.eval(t0, &yt, &mut k1);
        for j in 0..dim {
            yt[j] = y_out[i][j] + 0.5 * h * k1[j];
        }
        rhs.eval(t0 + 0.5 * h, &yt, &mut k2);
        for j in 0..dim {
            yt[j] = y_out[i][j] + 0.5 * h * k2[j];
        }
        rhs.eval(t0 + 0.5 * h, &yt, &mut k3);
        for j in 0..dim {
            yt[j] = y_out[i][j] + h * k3[j];
        }
        rhs.eval(t0 + h, &yt, &mut k4v);
        for j in 0..dim {
            y_out[i + 1][j] =
                y_out[i][j] + (h / 6.0) * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4v[j]);
        }
        if !all_finite(&y_out[i + 1]) {
            return Err(IntegrateError::Diverged { t: t_grid[i + 1] });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_matches_python_arange() {
        let g = output_grid(0.0, 1.0, 0.3);
        assert!((g[0] - 0.0).abs() < 1e-15);
        assert!((g[g.len() - 1] - 1.0).abs() < 1e-12);
    }
}
