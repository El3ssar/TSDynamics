//! Verner `(8,7)` pair with dense OrdinaryDiffEq-style interpolant (SciML tableau, MIT).

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
fn dense_vern8(y0: &[f64], k: &[Vec<f64>], h: f64, theta: f64, acc: &mut [f64], out: &mut [f64]) {
    let dim = out.len();
    let th = theta;
    let th2 = th * th;
    acc.fill(0.0);

    let b1 = th * eval_poly(&[R011, R012, R013, R014, R015, R016, R017, R018], th);
    let b6 = th2 * eval_poly(&[R062, R063, R064, R065, R066, R067, R068], th);
    let b7 = th2 * eval_poly(&[R072, R073, R074, R075, R076, R077, R078], th);
    let b8 = th2 * eval_poly(&[R082, R083, R084, R085, R086, R087, R088], th);
    let b9 = th2 * eval_poly(&[R092, R093, R094, R095, R096, R097, R098], th);
    let b10 = th2 * eval_poly(&[R102, R103, R104, R105, R106, R107, R108], th);
    let b11 = th2 * eval_poly(&[R112, R113, R114, R115, R116, R117, R118], th);
    let b12 = th2 * eval_poly(&[R122, R123, R124, R125, R126, R127, R128], th);
    let b14 = th2 * eval_poly(&[R142, R143, R144, R145, R146, R147, R148], th);
    let b15 = th2 * eval_poly(&[R152, R153, R154, R155, R156, R157, R158], th);
    let b16 = th2 * eval_poly(&[R162, R163, R164, R165, R166, R167, R168], th);
    let b17 = th2 * eval_poly(&[R172, R173, R174, R175, R176, R177, R178], th);
    let b18 = th2 * eval_poly(&[R182, R183, R184, R185, R186, R187, R188], th);
    let b19 = th2 * eval_poly(&[R192, R193, R194, R195, R196, R197, R198], th);
    let b20 = th2 * eval_poly(&[R202, R203, R204, R205, R206, R207, R208], th);
    let b21 = th2 * eval_poly(&[R212, R213, R214, R215, R216, R217, R218], th);

    axpy(b1, &k[0], acc);
    axpy(b6, &k[5], acc);
    axpy(b7, &k[6], acc);
    axpy(b8, &k[7], acc);
    axpy(b9, &k[8], acc);
    axpy(b10, &k[9], acc);
    axpy(b11, &k[10], acc);
    axpy(b12, &k[11], acc);
    axpy(b14, &k[13], acc);
    axpy(b15, &k[14], acc);
    axpy(b16, &k[15], acc);
    axpy(b17, &k[16], acc);
    axpy(b18, &k[17], acc);
    axpy(b19, &k[18], acc);
    axpy(b20, &k[19], acc);
    axpy(b21, &k[20], acc);

    for i in 0..dim {
        out[i] = y0[i] + h * acc[i];
    }
}

pub(crate) fn integrate_vern8<R: Rhs + ?Sized>(
    rhs: &mut R,
    t_grid: &[f64],
    y_out: &mut [Vec<f64>],
    rtol: f64,
    atol: f64,
) -> Result<(), IntegrateError> {
    let dim = rhs.dim();
    let tf = *t_grid.last().unwrap();
    let t0 = t_grid[0];

    let mut k = vec![vec![0.0_f64; dim]; 21];
    let mut y_tmp = vec![0.0_f64; dim];
    let mut g13_buf = vec![0.0_f64; dim];
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
            axpy(h_step * A0201, &k[0], &mut y_tmp);
            rhs.eval(t_old + h_step * C2, &y_tmp, &mut k[1]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A0301, &k[0], &mut y_tmp);
            axpy(h_step * A0302, &k[1], &mut y_tmp);
            rhs.eval(t_old + h_step * C3, &y_tmp, &mut k[2]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A0401, &k[0], &mut y_tmp);
            axpy(h_step * A0403, &k[2], &mut y_tmp);
            rhs.eval(t_old + h_step * C4, &y_tmp, &mut k[3]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A0501, &k[0], &mut y_tmp);
            axpy(h_step * A0503, &k[2], &mut y_tmp);
            axpy(h_step * A0504, &k[3], &mut y_tmp);
            rhs.eval(t_old + h_step * C5, &y_tmp, &mut k[4]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A0601, &k[0], &mut y_tmp);
            axpy(h_step * A0604, &k[3], &mut y_tmp);
            axpy(h_step * A0605, &k[4], &mut y_tmp);
            rhs.eval(t_old + h_step * C6, &y_tmp, &mut k[5]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A0701, &k[0], &mut y_tmp);
            axpy(h_step * A0704, &k[3], &mut y_tmp);
            axpy(h_step * A0705, &k[4], &mut y_tmp);
            axpy(h_step * A0706, &k[5], &mut y_tmp);
            rhs.eval(t_old + h_step * C7, &y_tmp, &mut k[6]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A0801, &k[0], &mut y_tmp);
            axpy(h_step * A0804, &k[3], &mut y_tmp);
            axpy(h_step * A0805, &k[4], &mut y_tmp);
            axpy(h_step * A0806, &k[5], &mut y_tmp);
            axpy(h_step * A0807, &k[6], &mut y_tmp);
            rhs.eval(t_old + h_step * C8, &y_tmp, &mut k[7]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A0901, &k[0], &mut y_tmp);
            axpy(h_step * A0904, &k[3], &mut y_tmp);
            axpy(h_step * A0905, &k[4], &mut y_tmp);
            axpy(h_step * A0906, &k[5], &mut y_tmp);
            axpy(h_step * A0907, &k[6], &mut y_tmp);
            axpy(h_step * A0908, &k[7], &mut y_tmp);
            rhs.eval(t_old + h_step * C9, &y_tmp, &mut k[8]);

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
            rhs.eval(t_old + h_step, &y_tmp, &mut k[11]);

            copy_from(&y_stage_start, &mut g13_buf);
            axpy(h_step * A1301, &k[0], &mut g13_buf);
            axpy(h_step * A1304, &k[3], &mut g13_buf);
            axpy(h_step * A1305, &k[4], &mut g13_buf);
            axpy(h_step * A1306, &k[5], &mut g13_buf);
            axpy(h_step * A1307, &k[6], &mut g13_buf);
            axpy(h_step * A1308, &k[7], &mut g13_buf);
            axpy(h_step * A1309, &k[8], &mut g13_buf);
            axpy(h_step * A1310, &k[9], &mut g13_buf);
            rhs.eval(t_old + h_step, &g13_buf, &mut k[12]);

            copy_from(&y_stage_start, &mut y_next);
            axpy(h_step * B1, &k[0], &mut y_next);
            axpy(h_step * B6, &k[5], &mut y_next);
            axpy(h_step * B7, &k[6], &mut y_next);
            axpy(h_step * B8, &k[7], &mut y_next);
            axpy(h_step * B9, &k[8], &mut y_next);
            axpy(h_step * B10, &k[9], &mut y_next);
            axpy(h_step * B11, &k[10], &mut y_next);
            axpy(h_step * B12, &k[11], &mut y_next);

            for i in 0..dim {
                errw[i] = h_step
                    * (BTILDE1 * k[0][i]
                        + BTILDE6 * k[5][i]
                        + BTILDE7 * k[6][i]
                        + BTILDE8 * k[7][i]
                        + BTILDE9 * k[8][i]
                        + BTILDE10 * k[9][i]
                        + BTILDE11 * k[10][i]
                        + BTILDE12 * k[11][i]
                        + BTILDE13 * k[12][i]);
            }

            let mut err2 = 0.0_f64;
            for i in 0..dim {
                let sci = atol + y_stage_start[i].abs().max(y_next[i].abs()) * rtol;
                let ri = errw[i] / sci;
                err2 += ri * ri;
            }
            let err = (err2 / dim as f64).sqrt();

            let (acc_step, h_new) = adapt_step(err, h_step, 7, 0.9, 0.2, 10.0);
            if acc_step {
                if !all_finite(&y_next) {
                    return Err(IntegrateError::Diverged { t: t_old + h_step });
                }

                copy_from(&y_stage_start, &mut y_tmp);
                axpy(h_step * A1401, &k[0], &mut y_tmp);
                axpy(h_step * A1406, &k[5], &mut y_tmp);
                axpy(h_step * A1407, &k[6], &mut y_tmp);
                axpy(h_step * A1408, &k[7], &mut y_tmp);
                axpy(h_step * A1409, &k[8], &mut y_tmp);
                axpy(h_step * A1410, &k[9], &mut y_tmp);
                axpy(h_step * A1411, &k[10], &mut y_tmp);
                axpy(h_step * A1412, &k[11], &mut y_tmp);
                rhs.eval(t_old + h_step * C14, &y_tmp, &mut k[13]);

                copy_from(&y_stage_start, &mut y_tmp);
                axpy(h_step * A1501, &k[0], &mut y_tmp);
                axpy(h_step * A1506, &k[5], &mut y_tmp);
                axpy(h_step * A1507, &k[6], &mut y_tmp);
                axpy(h_step * A1508, &k[7], &mut y_tmp);
                axpy(h_step * A1509, &k[8], &mut y_tmp);
                axpy(h_step * A1510, &k[9], &mut y_tmp);
                axpy(h_step * A1511, &k[10], &mut y_tmp);
                axpy(h_step * A1512, &k[11], &mut y_tmp);
                axpy(h_step * A1514, &k[13], &mut y_tmp);
                rhs.eval(t_old + h_step * C15, &y_tmp, &mut k[14]);

                copy_from(&y_stage_start, &mut y_tmp);
                axpy(h_step * A1601, &k[0], &mut y_tmp);
                axpy(h_step * A1606, &k[5], &mut y_tmp);
                axpy(h_step * A1607, &k[6], &mut y_tmp);
                axpy(h_step * A1608, &k[7], &mut y_tmp);
                axpy(h_step * A1609, &k[8], &mut y_tmp);
                axpy(h_step * A1610, &k[9], &mut y_tmp);
                axpy(h_step * A1611, &k[10], &mut y_tmp);
                axpy(h_step * A1612, &k[11], &mut y_tmp);
                axpy(h_step * A1614, &k[13], &mut y_tmp);
                axpy(h_step * A1615, &k[14], &mut y_tmp);
                rhs.eval(t_old + h_step * C16, &y_tmp, &mut k[15]);

                copy_from(&y_stage_start, &mut y_tmp);
                axpy(h_step * A1701, &k[0], &mut y_tmp);
                axpy(h_step * A1706, &k[5], &mut y_tmp);
                axpy(h_step * A1707, &k[6], &mut y_tmp);
                axpy(h_step * A1708, &k[7], &mut y_tmp);
                axpy(h_step * A1709, &k[8], &mut y_tmp);
                axpy(h_step * A1710, &k[9], &mut y_tmp);
                axpy(h_step * A1711, &k[10], &mut y_tmp);
                axpy(h_step * A1712, &k[11], &mut y_tmp);
                axpy(h_step * A1714, &k[13], &mut y_tmp);
                axpy(h_step * A1715, &k[14], &mut y_tmp);
                axpy(h_step * A1716, &k[15], &mut y_tmp);
                rhs.eval(t_old + h_step * C17, &y_tmp, &mut k[16]);

                copy_from(&y_stage_start, &mut y_tmp);
                axpy(h_step * A1801, &k[0], &mut y_tmp);
                axpy(h_step * A1806, &k[5], &mut y_tmp);
                axpy(h_step * A1807, &k[6], &mut y_tmp);
                axpy(h_step * A1808, &k[7], &mut y_tmp);
                axpy(h_step * A1809, &k[8], &mut y_tmp);
                axpy(h_step * A1810, &k[9], &mut y_tmp);
                axpy(h_step * A1811, &k[10], &mut y_tmp);
                axpy(h_step * A1812, &k[11], &mut y_tmp);
                axpy(h_step * A1814, &k[13], &mut y_tmp);
                axpy(h_step * A1815, &k[14], &mut y_tmp);
                axpy(h_step * A1816, &k[15], &mut y_tmp);
                axpy(h_step * A1817, &k[16], &mut y_tmp);
                rhs.eval(t_old + h_step * C18, &y_tmp, &mut k[17]);

                copy_from(&y_stage_start, &mut y_tmp);
                axpy(h_step * A1901, &k[0], &mut y_tmp);
                axpy(h_step * A1906, &k[5], &mut y_tmp);
                axpy(h_step * A1907, &k[6], &mut y_tmp);
                axpy(h_step * A1908, &k[7], &mut y_tmp);
                axpy(h_step * A1909, &k[8], &mut y_tmp);
                axpy(h_step * A1910, &k[9], &mut y_tmp);
                axpy(h_step * A1911, &k[10], &mut y_tmp);
                axpy(h_step * A1912, &k[11], &mut y_tmp);
                axpy(h_step * A1914, &k[13], &mut y_tmp);
                axpy(h_step * A1915, &k[14], &mut y_tmp);
                axpy(h_step * A1916, &k[15], &mut y_tmp);
                axpy(h_step * A1917, &k[16], &mut y_tmp);
                rhs.eval(t_old + h_step * C19, &y_tmp, &mut k[18]);

                copy_from(&y_stage_start, &mut y_tmp);
                axpy(h_step * A2001, &k[0], &mut y_tmp);
                axpy(h_step * A2006, &k[5], &mut y_tmp);
                axpy(h_step * A2007, &k[6], &mut y_tmp);
                axpy(h_step * A2008, &k[7], &mut y_tmp);
                axpy(h_step * A2009, &k[8], &mut y_tmp);
                axpy(h_step * A2010, &k[9], &mut y_tmp);
                axpy(h_step * A2011, &k[10], &mut y_tmp);
                axpy(h_step * A2012, &k[11], &mut y_tmp);
                axpy(h_step * A2014, &k[13], &mut y_tmp);
                axpy(h_step * A2015, &k[14], &mut y_tmp);
                axpy(h_step * A2016, &k[15], &mut y_tmp);
                axpy(h_step * A2017, &k[16], &mut y_tmp);
                rhs.eval(t_old + h_step * C20, &y_tmp, &mut k[19]);

                copy_from(&y_stage_start, &mut y_tmp);
                axpy(h_step * A2101, &k[0], &mut y_tmp);
                axpy(h_step * A2106, &k[5], &mut y_tmp);
                axpy(h_step * A2107, &k[6], &mut y_tmp);
                axpy(h_step * A2108, &k[7], &mut y_tmp);
                axpy(h_step * A2109, &k[8], &mut y_tmp);
                axpy(h_step * A2110, &k[9], &mut y_tmp);
                axpy(h_step * A2111, &k[10], &mut y_tmp);
                axpy(h_step * A2112, &k[11], &mut y_tmp);
                axpy(h_step * A2114, &k[13], &mut y_tmp);
                axpy(h_step * A2115, &k[14], &mut y_tmp);
                axpy(h_step * A2116, &k[15], &mut y_tmp);
                axpy(h_step * A2117, &k[16], &mut y_tmp);
                rhs.eval(t_old + h_step * C21, &y_tmp, &mut k[20]);

                let t_new = t_old + h_step;
                while i_out < t_grid.len() && t_grid[i_out] <= t_new + 1e-12 {
                    let tg = t_grid[i_out];
                    let theta = (tg - t_old) / h_step;
                    dense_vern8(&y_stage_start, &k, h_step, theta, &mut dense_acc, &mut y_out[i_out]);
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