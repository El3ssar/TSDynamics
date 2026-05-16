//! Dormand–Prince 5(4) and classical RK4 on the uniform output grid.

use crate::controller::adapt_step;
use crate::error::IntegrateError;
use crate::methods::butcher::dopri54;
use crate::rhs::Rhs;
use crate::step_helpers::{all_finite, h_init};
use crate::util::{axpy, copy_from, dense_dp5_eval};

pub(crate) fn integrate_dp5<R: Rhs + ?Sized>(
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

pub(crate) fn integrate_rk4<R: Rhs + ?Sized>(
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
