//! Verner `(6,5)` pair with OrdinaryDiffEq-style dense interpolant (SciML tableau, MIT).
//!
//! Primary stages + extrapolation polynomials are generated into `*_generated.rs` from
//! `scripts/gen_verner_ode_coeffs.py` (upstream: `OrdinaryDiffEqVerner/verner_tableaus.jl`).

mod extra_generated;
mod interp_generated;
mod tableau_generated;

use extra_generated::*;
use interp_generated::*;
use tableau_generated::*;

use crate::controller::adapt_step;
use crate::error::IntegrateError;
use crate::rhs::Rhs;
use crate::step_helpers::{all_finite, h_init};
use crate::util::{axpy, copy_from, eval_poly};

#[inline]
fn dense_vern6(y0: &[f64], k: &[Vec<f64>], h: f64, theta: f64, acc: &mut [f64], out: &mut [f64]) {
    let dim = out.len();
    let th = theta;
    let th2 = th * th;
    acc.fill(0.0);

    let b1 = th * eval_poly(&[R011, R012, R013, R014, R015, R016], th);
    let b4 = th2 * eval_poly(&[R042, R043, R044, R045, R046], th);
    let b5 = th2 * eval_poly(&[R052, R053, R054, R055, R056], th);
    let b6 = th2 * eval_poly(&[R062, R063, R064, R065, R066], th);
    let b7 = th2 * eval_poly(&[R072, R073, R074, R075, R076], th);
    let b8 = th2 * eval_poly(&[R082, R083, R084, R085, R086], th);
    let b9 = th2 * eval_poly(&[R092, R093, R094, R095, R096], th);
    let b10 = th2 * eval_poly(&[R102, R103, R104, R105, R106], th);
    let b11 = th2 * eval_poly(&[R112, R113, R114, R115, R116], th);
    let b12 = th2 * eval_poly(&[R122, R123, R124, R125, R126], th);

    axpy(b1, &k[0], acc);
    axpy(b4, &k[3], acc);
    axpy(b5, &k[4], acc);
    axpy(b6, &k[5], acc);
    axpy(b7, &k[6], acc);
    axpy(b8, &k[7], acc);
    axpy(b9, &k[8], acc);
    axpy(b10, &k[9], acc);
    axpy(b11, &k[10], acc);
    axpy(b12, &k[11], acc);
    for i in 0..dim {
        out[i] = y0[i] + h * acc[i];
    }
}

pub(crate) fn integrate_vern6<R: Rhs + ?Sized>(
    rhs: &mut R,
    t_grid: &[f64],
    y_out: &mut [Vec<f64>],
    rtol: f64,
    atol: f64,
) -> Result<(), IntegrateError> {
    let dim = rhs.dim();
    let tf = *t_grid.last().unwrap();
    let t0 = t_grid[0];

    let mut k = vec![vec![0.0_f64; dim]; 12];
    let mut y_tmp = vec![0.0_f64; dim];
    let mut y_next = vec![0.0_f64; dim];
    let mut errw = vec![0.0_f64; dim];
    let mut dense_acc = vec![0.0_f64; dim];

    let mut y = y_out[0].clone();
    let mut t = t0;

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
            let y_stage_start = y.clone();
            let h_step = h;

            rhs.eval(t_old, &y_stage_start, &mut k[0]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A21, &k[0], &mut y_tmp);
            rhs.eval(t_old + h_step * C1, &y_tmp, &mut k[1]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A31, &k[0], &mut y_tmp);
            axpy(h_step * A32, &k[1], &mut y_tmp);
            rhs.eval(t_old + h_step * C2, &y_tmp, &mut k[2]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A41, &k[0], &mut y_tmp);
            axpy(h_step * A43, &k[2], &mut y_tmp);
            rhs.eval(t_old + h_step * C3, &y_tmp, &mut k[3]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A51, &k[0], &mut y_tmp);
            axpy(h_step * A53, &k[2], &mut y_tmp);
            axpy(h_step * A54, &k[3], &mut y_tmp);
            rhs.eval(t_old + h_step * C4, &y_tmp, &mut k[4]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A61, &k[0], &mut y_tmp);
            axpy(h_step * A63, &k[2], &mut y_tmp);
            axpy(h_step * A64, &k[3], &mut y_tmp);
            axpy(h_step * A65, &k[4], &mut y_tmp);
            rhs.eval(t_old + h_step * C5, &y_tmp, &mut k[5]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A71, &k[0], &mut y_tmp);
            axpy(h_step * A73, &k[2], &mut y_tmp);
            axpy(h_step * A74, &k[3], &mut y_tmp);
            axpy(h_step * A75, &k[4], &mut y_tmp);
            axpy(h_step * A76, &k[5], &mut y_tmp);
            rhs.eval(t_old + h_step * C6, &y_tmp, &mut k[6]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A81, &k[0], &mut y_tmp);
            axpy(h_step * A83, &k[2], &mut y_tmp);
            axpy(h_step * A84, &k[3], &mut y_tmp);
            axpy(h_step * A85, &k[4], &mut y_tmp);
            axpy(h_step * A86, &k[5], &mut y_tmp);
            axpy(h_step * A87, &k[6], &mut y_tmp);
            rhs.eval(t_old + h_step, &y_tmp, &mut k[7]);

            copy_from(&y_stage_start, &mut y_next);
            axpy(h_step * A91, &k[0], &mut y_next);
            axpy(h_step * A94, &k[3], &mut y_next);
            axpy(h_step * A95, &k[4], &mut y_next);
            axpy(h_step * A96, &k[5], &mut y_next);
            axpy(h_step * A97, &k[6], &mut y_next);
            axpy(h_step * A98, &k[7], &mut y_next);

            rhs.eval(t_old + h_step, &y_next, &mut k[8]);

            for i in 0..dim {
                errw[i] = h_step
                    * (BTILDE1 * k[0][i]
                        + BTILDE4 * k[3][i]
                        + BTILDE5 * k[4][i]
                        + BTILDE6 * k[5][i]
                        + BTILDE7 * k[6][i]
                        + BTILDE8 * k[7][i]
                        + BTILDE9 * k[8][i]);
            }

            let mut err2 = 0.0_f64;
            for i in 0..dim {
                let sci = atol + y_stage_start[i].abs().max(y_next[i].abs()) * rtol;
                let ri = errw[i] / sci;
                err2 += ri * ri;
            }
            let err = (err2 / dim as f64).sqrt();

            let (acc_step, h_new) = adapt_step(err, h_step, 5, 0.9, 0.2, 10.0);
            if acc_step {
                if !all_finite(&y_next) {
                    return Err(IntegrateError::Diverged { t: t_old + h_step });
                }

                copy_from(&y_stage_start, &mut y_tmp);
                axpy(h_step * A1001, &k[0], &mut y_tmp);
                axpy(h_step * A1004, &k[3], &mut y_tmp);
                axpy(h_step * A1005, &k[4], &mut y_tmp);
                axpy(h_step * A1006, &k[5], &mut y_tmp);
                axpy(h_step * A1007, &k[6], &mut y_tmp);
                axpy(h_step * A1008, &k[7], &mut y_tmp);
                axpy(h_step * A1009, &k[8], &mut y_tmp);
                rhs.eval(t_old + h_step * C10, &y_tmp, &mut k[9]);

                copy_from(&y_stage_start, &mut y_tmp);
                axpy(h_step * A1101, &k[0], &mut y_tmp);
                axpy(h_step * A1104, &k[3], &mut y_tmp);
                axpy(h_step * A1105, &k[4], &mut y_tmp);
                axpy(h_step * A1106, &k[5], &mut y_tmp);
                axpy(h_step * A1107, &k[6], &mut y_tmp);
                axpy(h_step * A1108, &k[7], &mut y_tmp);
                axpy(h_step * A1109, &k[8], &mut y_tmp);
                axpy(h_step * A1110, &k[9], &mut y_tmp);
                rhs.eval(t_old + h_step * C11, &y_tmp, &mut k[10]);

                copy_from(&y_stage_start, &mut y_tmp);
                axpy(h_step * A1201, &k[0], &mut y_tmp);
                axpy(h_step * A1204, &k[3], &mut y_tmp);
                axpy(h_step * A1205, &k[4], &mut y_tmp);
                axpy(h_step * A1206, &k[5], &mut y_tmp);
                axpy(h_step * A1207, &k[6], &mut y_tmp);
                axpy(h_step * A1208, &k[7], &mut y_tmp);
                axpy(h_step * A1209, &k[8], &mut y_tmp);
                axpy(h_step * A1210, &k[9], &mut y_tmp);
                axpy(h_step * A1211, &k[10], &mut y_tmp);
                rhs.eval(t_old + h_step * C12, &y_tmp, &mut k[11]);

                let t_new = t_old + h_step;
                while i_out < t_grid.len() && t_grid[i_out] <= t_new + 1e-12 {
                    let tg = t_grid[i_out];
                    let theta = (tg - t_old) / h_step;
                    dense_vern6(&y_stage_start, &k, h_step, theta, &mut dense_acc, &mut y_out[i_out]);
                    if !all_finite(&y_out[i_out]) {
                        return Err(IntegrateError::Diverged { t: tg });
                    }
                    i_out += 1;
                }

                y.clone_from(&y_next);
                t = t_new;
                h = if last { h_new } else { h_new };
                if (t - tf).abs() < 1e-12 {
                    break;
                }
            } else {
                h = h_new;
                if !h.is_finite() || h.abs() < 1e-28 {
                    return Err(IntegrateError::Diverged { t: t_old });
                }
            }
        }
        if i_out >= t_grid.len() {
            break;
        }
    }
    Ok(())
}
