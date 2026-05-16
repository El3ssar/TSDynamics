//! Verner `Vern9` explicit Runge–Kutta pair with 9th-order dense output (N2.b).
//!
//! Primary tableau, extra interpolant stages, and polynomial coefficients are
//! vendored from SciML [`OrdinaryDiffEq.jl`](https://github.com/SciML/OrdinaryDiffEq.jl)
//! (`lib/OrdinaryDiffEqVerner/src/verner_tableaus.jl`, MIT license — coefficients
//! attributed to J.H. Verner).

mod extra;
mod interp;
mod main_stage;

use crate::controller::adapt_step;
use crate::error::IntegrateError;
use crate::rhs::Rhs;
use crate::step_helpers::{all_finite, h_init};
use crate::util::{axpy, copy_from, eval_poly};

use extra::*;
use main_stage::*;

const K_EXTRA_START: usize = 16;

#[inline]
fn dense_vern9(y0: &[f64], k: &[Vec<f64>], h: f64, theta: f64, acc: &mut [f64], out: &mut [f64]) {
    let dim = out.len();
    let th = theta;
    let th2 = th * th;
    acc.fill(0.0);
    let b1 = th * eval_poly(interp::R01, th);
    axpy(b1, &k[0], acc);

    let polys: [&[f64]; 18] = [
        interp::R08,
        interp::R09,
        interp::R10,
        interp::R11,
        interp::R12,
        interp::R13,
        interp::R14,
        interp::R15,
        interp::R17,
        interp::R18,
        interp::R19,
        interp::R20,
        interp::R21,
        interp::R22,
        interp::R23,
        interp::R24,
        interp::R25,
        interp::R26,
    ];
    let kidx: [usize; 18] = [
        7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    ];
    for (poly, ki) in polys.iter().zip(kidx.iter()) {
        let b = th2 * eval_poly(poly, th);
        axpy(b, &k[*ki], acc);
    }
    for i in 0..dim {
        out[i] = y0[i] + h * acc[i];
    }
}

pub(crate) fn integrate_vern9<R: Rhs + ?Sized>(
    rhs: &mut R,
    t_grid: &[f64],
    y_out: &mut [Vec<f64>],
    rtol: f64,
    atol: f64,
) -> Result<(), IntegrateError> {
    let dim = rhs.dim();
    let tf = *t_grid.last().unwrap();
    let t0 = t_grid[0];

    let mut k = vec![vec![0.0_f64; dim]; 26];
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
            axpy(h_step * A0201, &k[0], &mut y_tmp);
            rhs.eval(t_old + h_step * C1, &y_tmp, &mut k[1]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A0301, &k[0], &mut y_tmp);
            axpy(h_step * A0302, &k[1], &mut y_tmp);
            rhs.eval(t_old + h_step * C2, &y_tmp, &mut k[2]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A0401, &k[0], &mut y_tmp);
            axpy(h_step * A0403, &k[2], &mut y_tmp);
            rhs.eval(t_old + h_step * C3, &y_tmp, &mut k[3]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A0501, &k[0], &mut y_tmp);
            axpy(h_step * A0503, &k[2], &mut y_tmp);
            axpy(h_step * A0504, &k[3], &mut y_tmp);
            rhs.eval(t_old + h_step * C4, &y_tmp, &mut k[4]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A0601, &k[0], &mut y_tmp);
            axpy(h_step * A0604, &k[3], &mut y_tmp);
            axpy(h_step * A0605, &k[4], &mut y_tmp);
            rhs.eval(t_old + h_step * C5, &y_tmp, &mut k[5]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A0701, &k[0], &mut y_tmp);
            axpy(h_step * A0704, &k[3], &mut y_tmp);
            axpy(h_step * A0705, &k[4], &mut y_tmp);
            axpy(h_step * A0706, &k[5], &mut y_tmp);
            rhs.eval(t_old + h_step * C6, &y_tmp, &mut k[6]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A0801, &k[0], &mut y_tmp);
            axpy(h_step * A0806, &k[5], &mut y_tmp);
            axpy(h_step * A0807, &k[6], &mut y_tmp);
            rhs.eval(t_old + h_step * C7, &y_tmp, &mut k[7]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A0901, &k[0], &mut y_tmp);
            axpy(h_step * A0906, &k[5], &mut y_tmp);
            axpy(h_step * A0907, &k[6], &mut y_tmp);
            axpy(h_step * A0908, &k[7], &mut y_tmp);
            rhs.eval(t_old + h_step * C8, &y_tmp, &mut k[8]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A1001, &k[0], &mut y_tmp);
            axpy(h_step * A1006, &k[5], &mut y_tmp);
            axpy(h_step * A1007, &k[6], &mut y_tmp);
            axpy(h_step * A1008, &k[7], &mut y_tmp);
            axpy(h_step * A1009, &k[8], &mut y_tmp);
            rhs.eval(t_old + h_step * C9, &y_tmp, &mut k[9]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A1101, &k[0], &mut y_tmp);
            axpy(h_step * A1106, &k[5], &mut y_tmp);
            axpy(h_step * A1107, &k[6], &mut y_tmp);
            axpy(h_step * A1108, &k[7], &mut y_tmp);
            axpy(h_step * A1109, &k[8], &mut y_tmp);
            axpy(h_step * A1110, &k[9], &mut y_tmp);
            rhs.eval(t_old + h_step * C10, &y_tmp, &mut k[10]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A1201, &k[0], &mut y_tmp);
            axpy(h_step * A1206, &k[5], &mut y_tmp);
            axpy(h_step * A1207, &k[6], &mut y_tmp);
            axpy(h_step * A1208, &k[7], &mut y_tmp);
            axpy(h_step * A1209, &k[8], &mut y_tmp);
            axpy(h_step * A1210, &k[9], &mut y_tmp);
            axpy(h_step * A1211, &k[10], &mut y_tmp);
            rhs.eval(t_old + h_step * C11, &y_tmp, &mut k[11]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A1301, &k[0], &mut y_tmp);
            axpy(h_step * A1306, &k[5], &mut y_tmp);
            axpy(h_step * A1307, &k[6], &mut y_tmp);
            axpy(h_step * A1308, &k[7], &mut y_tmp);
            axpy(h_step * A1309, &k[8], &mut y_tmp);
            axpy(h_step * A1310, &k[9], &mut y_tmp);
            axpy(h_step * A1311, &k[10], &mut y_tmp);
            axpy(h_step * A1312, &k[11], &mut y_tmp);
            rhs.eval(t_old + h_step * C12, &y_tmp, &mut k[12]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A1401, &k[0], &mut y_tmp);
            axpy(h_step * A1406, &k[5], &mut y_tmp);
            axpy(h_step * A1407, &k[6], &mut y_tmp);
            axpy(h_step * A1408, &k[7], &mut y_tmp);
            axpy(h_step * A1409, &k[8], &mut y_tmp);
            axpy(h_step * A1410, &k[9], &mut y_tmp);
            axpy(h_step * A1411, &k[10], &mut y_tmp);
            axpy(h_step * A1412, &k[11], &mut y_tmp);
            axpy(h_step * A1413, &k[12], &mut y_tmp);
            rhs.eval(t_old + h_step * C13, &y_tmp, &mut k[13]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A1501, &k[0], &mut y_tmp);
            axpy(h_step * A1506, &k[5], &mut y_tmp);
            axpy(h_step * A1507, &k[6], &mut y_tmp);
            axpy(h_step * A1508, &k[7], &mut y_tmp);
            axpy(h_step * A1509, &k[8], &mut y_tmp);
            axpy(h_step * A1510, &k[9], &mut y_tmp);
            axpy(h_step * A1511, &k[10], &mut y_tmp);
            axpy(h_step * A1512, &k[11], &mut y_tmp);
            axpy(h_step * A1513, &k[12], &mut y_tmp);
            axpy(h_step * A1514, &k[13], &mut y_tmp);
            rhs.eval(t_old + h_step, &y_tmp, &mut k[14]);

            copy_from(&y_stage_start, &mut y_tmp);
            axpy(h_step * A1601, &k[0], &mut y_tmp);
            axpy(h_step * A1606, &k[5], &mut y_tmp);
            axpy(h_step * A1607, &k[6], &mut y_tmp);
            axpy(h_step * A1608, &k[7], &mut y_tmp);
            axpy(h_step * A1609, &k[8], &mut y_tmp);
            axpy(h_step * A1610, &k[9], &mut y_tmp);
            axpy(h_step * A1611, &k[10], &mut y_tmp);
            axpy(h_step * A1612, &k[11], &mut y_tmp);
            axpy(h_step * A1613, &k[12], &mut y_tmp);
            rhs.eval(t_old + h_step, &y_tmp, &mut k[15]);

            copy_from(&y_stage_start, &mut y_next);
            axpy(h_step * B1, &k[0], &mut y_next);
            axpy(h_step * B8, &k[7], &mut y_next);
            axpy(h_step * B9, &k[8], &mut y_next);
            axpy(h_step * B10, &k[9], &mut y_next);
            axpy(h_step * B11, &k[10], &mut y_next);
            axpy(h_step * B12, &k[11], &mut y_next);
            axpy(h_step * B13, &k[12], &mut y_next);
            axpy(h_step * B14, &k[13], &mut y_next);
            axpy(h_step * B15, &k[14], &mut y_next);

            for i in 0..dim {
                errw[i] = h_step
                    * (BTILDE1 * k[0][i]
                        + BTILDE8 * k[7][i]
                        + BTILDE9 * k[8][i]
                        + BTILDE10 * k[9][i]
                        + BTILDE11 * k[10][i]
                        + BTILDE12 * k[11][i]
                        + BTILDE13 * k[12][i]
                        + BTILDE14 * k[13][i]
                        + BTILDE15 * k[14][i]
                        + BTILDE16 * k[15][i]);
            }

            let mut err2 = 0.0_f64;
            for i in 0..dim {
                let sci = atol + y_stage_start[i].abs().max(y_next[i].abs()) * rtol;
                let ri = errw[i] / sci;
                err2 += ri * ri;
            }
            let err = (err2 / dim as f64).sqrt();

            let (acc_step, h_new) = adapt_step(err, h_step, 8, 0.9, 0.2, 10.0);
            if acc_step {
                if !all_finite(&y_next) {
                    return Err(IntegrateError::Diverged { t: t_old + h_step });
                }

                copy_from(&y_stage_start, &mut y_tmp);
                axpy(h_step * A1701, &k[0], &mut y_tmp);
                axpy(h_step * A1708, &k[7], &mut y_tmp);
                axpy(h_step * A1709, &k[8], &mut y_tmp);
                axpy(h_step * A1710, &k[9], &mut y_tmp);
                axpy(h_step * A1711, &k[10], &mut y_tmp);
                axpy(h_step * A1712, &k[11], &mut y_tmp);
                axpy(h_step * A1713, &k[12], &mut y_tmp);
                axpy(h_step * A1714, &k[13], &mut y_tmp);
                axpy(h_step * A1715, &k[14], &mut y_tmp);
                rhs.eval(t_old + h_step * C17, &y_tmp, &mut k[K_EXTRA_START]);

                copy_from(&y_stage_start, &mut y_tmp);
                axpy(h_step * A1801, &k[0], &mut y_tmp);
                axpy(h_step * A1808, &k[7], &mut y_tmp);
                axpy(h_step * A1809, &k[8], &mut y_tmp);
                axpy(h_step * A1810, &k[9], &mut y_tmp);
                axpy(h_step * A1811, &k[10], &mut y_tmp);
                axpy(h_step * A1812, &k[11], &mut y_tmp);
                axpy(h_step * A1813, &k[12], &mut y_tmp);
                axpy(h_step * A1814, &k[13], &mut y_tmp);
                axpy(h_step * A1815, &k[14], &mut y_tmp);
                axpy(h_step * A1817, &k[K_EXTRA_START], &mut y_tmp);
                rhs.eval(t_old + h_step * C18, &y_tmp, &mut k[17]);

                copy_from(&y_stage_start, &mut y_tmp);
                axpy(h_step * A1901, &k[0], &mut y_tmp);
                axpy(h_step * A1908, &k[7], &mut y_tmp);
                axpy(h_step * A1909, &k[8], &mut y_tmp);
                axpy(h_step * A1910, &k[9], &mut y_tmp);
                axpy(h_step * A1911, &k[10], &mut y_tmp);
                axpy(h_step * A1912, &k[11], &mut y_tmp);
                axpy(h_step * A1913, &k[12], &mut y_tmp);
                axpy(h_step * A1914, &k[13], &mut y_tmp);
                axpy(h_step * A1915, &k[14], &mut y_tmp);
                axpy(h_step * A1917, &k[K_EXTRA_START], &mut y_tmp);
                axpy(h_step * A1918, &k[17], &mut y_tmp);
                rhs.eval(t_old + h_step * C19, &y_tmp, &mut k[18]);

                copy_from(&y_stage_start, &mut y_tmp);
                axpy(h_step * A2001, &k[0], &mut y_tmp);
                axpy(h_step * A2008, &k[7], &mut y_tmp);
                axpy(h_step * A2009, &k[8], &mut y_tmp);
                axpy(h_step * A2010, &k[9], &mut y_tmp);
                axpy(h_step * A2011, &k[10], &mut y_tmp);
                axpy(h_step * A2012, &k[11], &mut y_tmp);
                axpy(h_step * A2013, &k[12], &mut y_tmp);
                axpy(h_step * A2014, &k[13], &mut y_tmp);
                axpy(h_step * A2015, &k[14], &mut y_tmp);
                axpy(h_step * A2017, &k[K_EXTRA_START], &mut y_tmp);
                axpy(h_step * A2018, &k[17], &mut y_tmp);
                axpy(h_step * A2019, &k[18], &mut y_tmp);
                rhs.eval(t_old + h_step * C20, &y_tmp, &mut k[19]);

                copy_from(&y_stage_start, &mut y_tmp);
                axpy(h_step * A2101, &k[0], &mut y_tmp);
                axpy(h_step * A2108, &k[7], &mut y_tmp);
                axpy(h_step * A2109, &k[8], &mut y_tmp);
                axpy(h_step * A2110, &k[9], &mut y_tmp);
                axpy(h_step * A2111, &k[10], &mut y_tmp);
                axpy(h_step * A2112, &k[11], &mut y_tmp);
                axpy(h_step * A2113, &k[12], &mut y_tmp);
                axpy(h_step * A2114, &k[13], &mut y_tmp);
                axpy(h_step * A2115, &k[14], &mut y_tmp);
                axpy(h_step * A2117, &k[K_EXTRA_START], &mut y_tmp);
                axpy(h_step * A2118, &k[17], &mut y_tmp);
                axpy(h_step * A2119, &k[18], &mut y_tmp);
                axpy(h_step * A2120, &k[19], &mut y_tmp);
                rhs.eval(t_old + h_step * C21, &y_tmp, &mut k[20]);

                copy_from(&y_stage_start, &mut y_tmp);
                axpy(h_step * A2201, &k[0], &mut y_tmp);
                axpy(h_step * A2208, &k[7], &mut y_tmp);
                axpy(h_step * A2209, &k[8], &mut y_tmp);
                axpy(h_step * A2210, &k[9], &mut y_tmp);
                axpy(h_step * A2211, &k[10], &mut y_tmp);
                axpy(h_step * A2212, &k[11], &mut y_tmp);
                axpy(h_step * A2213, &k[12], &mut y_tmp);
                axpy(h_step * A2214, &k[13], &mut y_tmp);
                axpy(h_step * A2215, &k[14], &mut y_tmp);
                axpy(h_step * A2217, &k[K_EXTRA_START], &mut y_tmp);
                axpy(h_step * A2218, &k[17], &mut y_tmp);
                axpy(h_step * A2219, &k[18], &mut y_tmp);
                axpy(h_step * A2220, &k[19], &mut y_tmp);
                axpy(h_step * A2221, &k[20], &mut y_tmp);
                rhs.eval(t_old + h_step * C22, &y_tmp, &mut k[21]);

                copy_from(&y_stage_start, &mut y_tmp);
                axpy(h_step * A2301, &k[0], &mut y_tmp);
                axpy(h_step * A2308, &k[7], &mut y_tmp);
                axpy(h_step * A2309, &k[8], &mut y_tmp);
                axpy(h_step * A2310, &k[9], &mut y_tmp);
                axpy(h_step * A2311, &k[10], &mut y_tmp);
                axpy(h_step * A2312, &k[11], &mut y_tmp);
                axpy(h_step * A2313, &k[12], &mut y_tmp);
                axpy(h_step * A2314, &k[13], &mut y_tmp);
                axpy(h_step * A2315, &k[14], &mut y_tmp);
                axpy(h_step * A2317, &k[K_EXTRA_START], &mut y_tmp);
                axpy(h_step * A2318, &k[17], &mut y_tmp);
                axpy(h_step * A2319, &k[18], &mut y_tmp);
                axpy(h_step * A2320, &k[19], &mut y_tmp);
                axpy(h_step * A2321, &k[20], &mut y_tmp);
                rhs.eval(t_old + h_step * C23, &y_tmp, &mut k[22]);

                copy_from(&y_stage_start, &mut y_tmp);
                axpy(h_step * A2401, &k[0], &mut y_tmp);
                axpy(h_step * A2408, &k[7], &mut y_tmp);
                axpy(h_step * A2409, &k[8], &mut y_tmp);
                axpy(h_step * A2410, &k[9], &mut y_tmp);
                axpy(h_step * A2411, &k[10], &mut y_tmp);
                axpy(h_step * A2412, &k[11], &mut y_tmp);
                axpy(h_step * A2413, &k[12], &mut y_tmp);
                axpy(h_step * A2414, &k[13], &mut y_tmp);
                axpy(h_step * A2415, &k[14], &mut y_tmp);
                axpy(h_step * A2417, &k[K_EXTRA_START], &mut y_tmp);
                axpy(h_step * A2418, &k[17], &mut y_tmp);
                axpy(h_step * A2419, &k[18], &mut y_tmp);
                axpy(h_step * A2420, &k[19], &mut y_tmp);
                axpy(h_step * A2421, &k[20], &mut y_tmp);
                rhs.eval(t_old + h_step * C24, &y_tmp, &mut k[23]);

                copy_from(&y_stage_start, &mut y_tmp);
                axpy(h_step * A2501, &k[0], &mut y_tmp);
                axpy(h_step * A2508, &k[7], &mut y_tmp);
                axpy(h_step * A2509, &k[8], &mut y_tmp);
                axpy(h_step * A2510, &k[9], &mut y_tmp);
                axpy(h_step * A2511, &k[10], &mut y_tmp);
                axpy(h_step * A2512, &k[11], &mut y_tmp);
                axpy(h_step * A2513, &k[12], &mut y_tmp);
                axpy(h_step * A2514, &k[13], &mut y_tmp);
                axpy(h_step * A2515, &k[14], &mut y_tmp);
                axpy(h_step * A2517, &k[K_EXTRA_START], &mut y_tmp);
                axpy(h_step * A2518, &k[17], &mut y_tmp);
                axpy(h_step * A2519, &k[18], &mut y_tmp);
                axpy(h_step * A2520, &k[19], &mut y_tmp);
                axpy(h_step * A2521, &k[20], &mut y_tmp);
                rhs.eval(t_old + h_step * C25, &y_tmp, &mut k[24]);

                copy_from(&y_stage_start, &mut y_tmp);
                axpy(h_step * A2601, &k[0], &mut y_tmp);
                axpy(h_step * A2608, &k[7], &mut y_tmp);
                axpy(h_step * A2609, &k[8], &mut y_tmp);
                axpy(h_step * A2610, &k[9], &mut y_tmp);
                axpy(h_step * A2611, &k[10], &mut y_tmp);
                axpy(h_step * A2612, &k[11], &mut y_tmp);
                axpy(h_step * A2613, &k[12], &mut y_tmp);
                axpy(h_step * A2614, &k[13], &mut y_tmp);
                axpy(h_step * A2615, &k[14], &mut y_tmp);
                axpy(h_step * A2617, &k[K_EXTRA_START], &mut y_tmp);
                axpy(h_step * A2618, &k[17], &mut y_tmp);
                axpy(h_step * A2619, &k[18], &mut y_tmp);
                axpy(h_step * A2620, &k[19], &mut y_tmp);
                axpy(h_step * A2621, &k[20], &mut y_tmp);
                rhs.eval(t_old + h_step * C26, &y_tmp, &mut k[25]);

                let t_new = t_old + h_step;
                while i_out < t_grid.len() && t_grid[i_out] <= t_new + 1e-12 {
                    let tg = t_grid[i_out];
                    let theta = (tg - t_old) / h_step;
                    dense_vern9(
                        &y_stage_start,
                        &k,
                        h_step,
                        theta,
                        &mut dense_acc,
                        &mut y_out[i_out],
                    );
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
