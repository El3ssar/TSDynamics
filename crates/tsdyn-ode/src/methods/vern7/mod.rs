//! Verner `(7,6)` pair with dense OrdinaryDiffEq-style interpolant (SciML tableau, MIT).

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
fn dense_vern7(y0: &[f64], k: &[Vec<f64>], h: f64, theta: f64, acc: &mut [f64], out: &mut [f64]) {
    let dim = out.len();
    let th = theta;
    let th2 = th * th;
    acc.fill(0.0);

    let b1 = th * eval_poly(&[R011, R012, R013, R014, R015, R016, R017], th);
    let b4 = th2 * eval_poly(&[R042, R043, R044, R045, R046, R047], th);
    let b5 = th2 * eval_poly(&[R052, R053, R054, R055, R056, R057], th);
    let b6 = th2 * eval_poly(&[R062, R063, R064, R065, R066, R067], th);
    let b7 = th2 * eval_poly(&[R072, R073, R074, R075, R076, R077], th);
    let b8 = th2 * eval_poly(&[R082, R083, R084, R085, R086, R087], th);
    let b9 = th2 * eval_poly(&[R092, R093, R094, R095, R096, R097], th);
    let b11 = th2 * eval_poly(&[R112, R113, R114, R115, R116, R117], th);
    let b12 = th2 * eval_poly(&[R122, R123, R124, R125, R126, R127], th);
    let b13 = th2 * eval_poly(&[R132, R133, R134, R135, R136, R137], th);
    let b14 = th2 * eval_poly(&[R142, R143, R144, R145, R146, R147], th);
    let b15 = th2 * eval_poly(&[R152, R153, R154, R155, R156, R157], th);
    let b16 = th2 * eval_poly(&[R162, R163, R164, R165, R166, R167], th);

    axpy(b1, &k[0], acc);
    for i in [3usize, 4, 5, 6, 7, 8] {
        let bb = match i {
            3 => b4,
            4 => b5,
            5 => b6,
            6 => b7,
            7 => b8,
            8 => b9,
            _ => unreachable!(),
        };
        axpy(bb, &k[i], acc);
    }
    for i in [10usize, 11, 12, 13, 14, 15] {
        let bb = match i {
            10 => b11,
            11 => b12,
            12 => b13,
            13 => b14,
            14 => b15,
            15 => b16,
            _ => unreachable!(),
        };
        axpy(bb, &k[i], acc);
    }
    for ii in 0..dim {
        out[ii] = y0[ii] + h * acc[ii];
    }
}

pub(crate) fn integrate_vern7<R: Rhs + ?Sized>(
    rhs: &mut R,
    t_grid: &[f64],
    y_out: &mut [Vec<f64>],
    rtol: f64,
    atol: f64,
) -> Result<(), IntegrateError> {
    let dim = rhs.dim();
    let tf = *t_grid.last().unwrap();
    let t0 = t_grid[0];

    let mut k = vec![vec![0.0_f64; dim]; 16];
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
            axpy(h_step * A021, &k[0], &mut y_tmp);
            rhs.eval(t_old + h_step * C2, &y_tmp, &mut k[1]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A031, &k[0], &mut y_tmp);
            axpy(h_step * A032, &k[1], &mut y_tmp);
            rhs.eval(t_old + h_step * C3, &y_tmp, &mut k[2]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A041, &k[0], &mut y_tmp);
            axpy(h_step * A043, &k[2], &mut y_tmp);
            rhs.eval(t_old + h_step * C4, &y_tmp, &mut k[3]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A051, &k[0], &mut y_tmp);
            axpy(h_step * A053, &k[2], &mut y_tmp);
            axpy(h_step * A054, &k[3], &mut y_tmp);
            rhs.eval(t_old + h_step * C5, &y_tmp, &mut k[4]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A061, &k[0], &mut y_tmp);
            axpy(h_step * A063, &k[2], &mut y_tmp);
            axpy(h_step * A064, &k[3], &mut y_tmp);
            axpy(h_step * A065, &k[4], &mut y_tmp);
            rhs.eval(t_old + h_step * C6, &y_tmp, &mut k[5]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A071, &k[0], &mut y_tmp);
            axpy(h_step * A073, &k[2], &mut y_tmp);
            axpy(h_step * A074, &k[3], &mut y_tmp);
            axpy(h_step * A075, &k[4], &mut y_tmp);
            axpy(h_step * A076, &k[5], &mut y_tmp);
            rhs.eval(t_old + h_step * C7, &y_tmp, &mut k[6]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A081, &k[0], &mut y_tmp);
            axpy(h_step * A083, &k[2], &mut y_tmp);
            axpy(h_step * A084, &k[3], &mut y_tmp);
            axpy(h_step * A085, &k[4], &mut y_tmp);
            axpy(h_step * A086, &k[5], &mut y_tmp);
            axpy(h_step * A087, &k[6], &mut y_tmp);
            rhs.eval(t_old + h_step * C8, &y_tmp, &mut k[7]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A091, &k[0], &mut y_tmp);
            axpy(h_step * A093, &k[2], &mut y_tmp);
            axpy(h_step * A094, &k[3], &mut y_tmp);
            axpy(h_step * A095, &k[4], &mut y_tmp);
            axpy(h_step * A096, &k[5], &mut y_tmp);
            axpy(h_step * A097, &k[6], &mut y_tmp);
            axpy(h_step * A098, &k[7], &mut y_tmp);
            rhs.eval(t_old + h_step, &y_tmp, &mut k[8]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A101, &k[0], &mut y_tmp);
            axpy(h_step * A103, &k[2], &mut y_tmp);
            axpy(h_step * A104, &k[3], &mut y_tmp);
            axpy(h_step * A105, &k[4], &mut y_tmp);
            axpy(h_step * A106, &k[5], &mut y_tmp);
            axpy(h_step * A107, &k[6], &mut y_tmp);
            rhs.eval(t_old + h_step, &y_tmp, &mut k[9]);

            copy_from(&y_stage_start, &mut y_next);
            axpy(h_step * B1, &k[0], &mut y_next);
            axpy(h_step * B4, &k[3], &mut y_next);
            axpy(h_step * B5, &k[4], &mut y_next);
            axpy(h_step * B6, &k[5], &mut y_next);
            axpy(h_step * B7, &k[6], &mut y_next);
            axpy(h_step * B8, &k[7], &mut y_next);
            axpy(h_step * B9, &k[8], &mut y_next);

            for i in 0..dim {
                errw[i] = h_step
                    * (BTILDE1 * k[0][i]
                        + BTILDE4 * k[3][i]
                        + BTILDE5 * k[4][i]
                        + BTILDE6 * k[5][i]
                        + BTILDE7 * k[6][i]
                        + BTILDE8 * k[7][i]
                        + BTILDE9 * k[8][i]
                        + BTILDE10 * k[9][i]);
            }

            let mut err2 = 0.0_f64;
            for i in 0..dim {
                let sci = atol + y_stage_start[i].abs().max(y_next[i].abs()) * rtol;
                let ri = errw[i] / sci;
                err2 += ri * ri;
            }
            let err = (err2 / dim as f64).sqrt();

            let (acc_step, h_new) = adapt_step(err, h_step, 6, 0.9, 0.2, 10.0);
            if acc_step {
                if !all_finite(&y_next) {
                    return Err(IntegrateError::Diverged { t: t_old + h_step });
                }

                copy_from(&y_stage_start, &mut y_tmp);
                axpy(h_step * A1101, &k[0], &mut y_tmp);
                axpy(h_step * A1104, &k[3], &mut y_tmp);
                axpy(h_step * A1105, &k[4], &mut y_tmp);
                axpy(h_step * A1106, &k[5], &mut y_tmp);
                axpy(h_step * A1107, &k[6], &mut y_tmp);
                axpy(h_step * A1108, &k[7], &mut y_tmp);
                axpy(h_step * A1109, &k[8], &mut y_tmp);
                rhs.eval(t_old + h_step * C11, &y_tmp, &mut k[10]);

                copy_from(&y_stage_start, &mut y_tmp);
                axpy(h_step * A1201, &k[0], &mut y_tmp);
                axpy(h_step * A1204, &k[3], &mut y_tmp);
                axpy(h_step * A1205, &k[4], &mut y_tmp);
                axpy(h_step * A1206, &k[5], &mut y_tmp);
                axpy(h_step * A1207, &k[6], &mut y_tmp);
                axpy(h_step * A1208, &k[7], &mut y_tmp);
                axpy(h_step * A1209, &k[8], &mut y_tmp);
                axpy(h_step * A1211, &k[10], &mut y_tmp);
                rhs.eval(t_old + h_step * C12, &y_tmp, &mut k[11]);

                copy_from(&y_stage_start, &mut y_tmp);
                axpy(h_step * A1301, &k[0], &mut y_tmp);
                axpy(h_step * A1304, &k[3], &mut y_tmp);
                axpy(h_step * A1305, &k[4], &mut y_tmp);
                axpy(h_step * A1306, &k[5], &mut y_tmp);
                axpy(h_step * A1307, &k[6], &mut y_tmp);
                axpy(h_step * A1308, &k[7], &mut y_tmp);
                axpy(h_step * A1309, &k[8], &mut y_tmp);
                axpy(h_step * A1311, &k[10], &mut y_tmp);
                axpy(h_step * A1312, &k[11], &mut y_tmp);
                rhs.eval(t_old + h_step * C13, &y_tmp, &mut k[12]);

                copy_from(&y_stage_start, &mut y_tmp);
                axpy(h_step * A1401, &k[0], &mut y_tmp);
                axpy(h_step * A1404, &k[3], &mut y_tmp);
                axpy(h_step * A1405, &k[4], &mut y_tmp);
                axpy(h_step * A1406, &k[5], &mut y_tmp);
                axpy(h_step * A1407, &k[6], &mut y_tmp);
                axpy(h_step * A1408, &k[7], &mut y_tmp);
                axpy(h_step * A1409, &k[8], &mut y_tmp);
                axpy(h_step * A1411, &k[10], &mut y_tmp);
                axpy(h_step * A1412, &k[11], &mut y_tmp);
                axpy(h_step * A1413, &k[12], &mut y_tmp);
                rhs.eval(t_old + h_step * C14, &y_tmp, &mut k[13]);

                copy_from(&y_stage_start, &mut y_tmp);
                axpy(h_step * A1501, &k[0], &mut y_tmp);
                axpy(h_step * A1504, &k[3], &mut y_tmp);
                axpy(h_step * A1505, &k[4], &mut y_tmp);
                axpy(h_step * A1506, &k[5], &mut y_tmp);
                axpy(h_step * A1507, &k[6], &mut y_tmp);
                axpy(h_step * A1508, &k[7], &mut y_tmp);
                axpy(h_step * A1509, &k[8], &mut y_tmp);
                axpy(h_step * A1511, &k[10], &mut y_tmp);
                axpy(h_step * A1512, &k[11], &mut y_tmp);
                axpy(h_step * A1513, &k[12], &mut y_tmp);
                rhs.eval(t_old + h_step * C15, &y_tmp, &mut k[14]);

                copy_from(&y_stage_start, &mut y_tmp);
                axpy(h_step * A1601, &k[0], &mut y_tmp);
                axpy(h_step * A1604, &k[3], &mut y_tmp);
                axpy(h_step * A1605, &k[4], &mut y_tmp);
                axpy(h_step * A1606, &k[5], &mut y_tmp);
                axpy(h_step * A1607, &k[6], &mut y_tmp);
                axpy(h_step * A1608, &k[7], &mut y_tmp);
                axpy(h_step * A1609, &k[8], &mut y_tmp);
                axpy(h_step * A1611, &k[10], &mut y_tmp);
                axpy(h_step * A1612, &k[11], &mut y_tmp);
                axpy(h_step * A1613, &k[12], &mut y_tmp);
                rhs.eval(t_old + h_step * C16, &y_tmp, &mut k[15]);

                let t_new = t_old + h_step;
                while i_out < t_grid.len() && t_grid[i_out] <= t_new + 1e-12 {
                    let tg = t_grid[i_out];
                    let theta = (tg - t_old) / h_step;
                    dense_vern7(&y_stage_start, &k, h_step, theta, &mut dense_acc, &mut y_out[i_out]);
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
