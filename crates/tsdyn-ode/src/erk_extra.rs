//! Additional explicit Runge–Kutta drivers (DP8, Tsit5, BS3).

use crate::butcher::{bs3, dop853_err, dopri853, tsit5};
use crate::controller::adapt_step;
use crate::rhs::Rhs;
use crate::util::{axpy, copy_from, dense_bs3_eval, dense_dop853_eval, dense_tsit5_eval};
use crate::{all_finite, h_init, IntegrateError};

fn dp853_error_norm(
    k: &[Vec<f64>],
    h: f64,
    y: &[f64],
    y_next: &[f64],
    rtol: f64,
    atol: f64,
) -> f64 {
    let dim = y.len();
    let mut err5_norm2 = 0.0_f64;
    let mut err3_norm2 = 0.0_f64;
    for i in 0..dim {
        let sci = atol + y[i].abs().max(y_next[i].abs()) * rtol;
        let mut acc5 = 0.0_f64;
        let mut acc3 = 0.0_f64;
        for j in 0..13 {
            acc5 += k[j][i] * dop853_err::E5[j];
            acc3 += k[j][i] * dop853_err::E3[j];
        }
        let r5 = (h * acc5) / sci;
        let r3 = (h * acc3) / sci;
        err5_norm2 += r5 * r5;
        err3_norm2 += r3 * r3;
    }
    if err5_norm2 == 0.0 && err3_norm2 == 0.0 {
        return 0.0;
    }
    let denom = err5_norm2 + 0.01 * err3_norm2;
    h.abs() * err5_norm2 / (denom * dim as f64).sqrt()
}

fn adapt_dp853(err: f64, h: f64, step_rejected: bool) -> (bool, f64) {
    const SAFETY: f64 = 0.9;
    const MIN_FACTOR: f64 = 0.2;
    const MAX_FACTOR: f64 = 10.0;
    let expo = -1.0 / 8.0;
    if err < 1.0 {
        let mut fac = if err == 0.0 {
            MAX_FACTOR
        } else {
            (SAFETY * err.powf(expo)).min(MAX_FACTOR)
        };
        if step_rejected {
            fac = fac.min(1.0);
        }
        (true, h * fac)
    } else {
        let fac = (SAFETY * err.powf(expo)).max(MIN_FACTOR);
        (false, h * fac)
    }
}

pub(super) fn integrate_dp8<R: Rhs + ?Sized>(
    rhs: &mut R,
    t_grid: &[f64],
    y_out: &mut [Vec<f64>],
    rtol: f64,
    atol: f64,
) -> Result<(), IntegrateError> {
    const MAX_STEPS: u32 = 500_000;
    let dim = rhs.dim();
    let tf = *t_grid.last().unwrap();
    let t0 = t_grid[0];

    let mut k = vec![vec![0.0_f64; dim]; 16];
    let mut y_tmp = vec![0.0_f64; dim];
    let mut y_next = vec![0.0_f64; dim];
    let mut f_dense: [Vec<f64>; 7] = std::array::from_fn(|_| vec![0.0_f64; dim]);

    let mut y = y_out[0].clone();
    let mut t = t0;
    rhs.eval(t, &y, &mut k[0]);

    let mut f_tmp = vec![0.0_f64; dim];
    let mut y1 = vec![0.0_f64; dim];
    let mut f1 = vec![0.0_f64; dim];
    let mut h = h_init(rhs, t0, &y, tf, rtol, atol, &mut f_tmp, &mut y1, &mut f1);
    rhs.eval(t, &y, &mut k[0]);

    let mut i_out = 1usize;
    let mut n_step: u32 = 0;
    let mut dp8_prev_rej = false;

    while i_out < t_grid.len() {
        let t_target = t_grid[i_out];
        while t < t_target - 1e-14 {
            if n_step > MAX_STEPS {
                return Err(IntegrateError::MaxSteps);
            }
            n_step += 1;

            let mut last = false;
            let dir = (tf - t0).signum();
            if (t + 1.01 * h - tf) * dir >= 0.0 {
                h = tf - t;
                last = true;
            }

            let t_old = t;
            let y_stage_start = y.clone();
            let h_step = h;

            for s in 1..=11 {
                copy_from(&y_stage_start, &mut y_tmp);
                for j in 0..s {
                    let a = dopri853::a(s + 1, j + 1);
                    if a != 0.0 {
                        axpy(h_step * a, &k[j], &mut y_tmp);
                    }
                }
                rhs.eval(t_old + h_step * dopri853::c(s + 1), &y_tmp, &mut k[s]);
            }

            copy_from(&y_stage_start, &mut y_next);
            for j in 0..12 {
                let bj = dopri853::b(j + 1);
                if bj != 0.0 {
                    axpy(h_step * bj, &k[j], &mut y_next);
                }
            }

            rhs.eval(t_old + h_step, &y_next, &mut k[12]);

            let err = dp853_error_norm(&k, h_step, &y_stage_start, &y_next, rtol, atol);
            let (acc, h_new) = adapt_dp853(err, h_step, dp8_prev_rej);

            if acc {
                dp8_prev_rej = false;
                if !all_finite(&y_next) {
                    return Err(IntegrateError::Diverged {
                        t: t_old + h_step,
                    });
                }

                for si in [13usize, 14, 15] {
                    copy_from(&y_stage_start, &mut y_tmp);
                    for j in 0..si {
                        let a = dopri853::a(si + 1, j + 1);
                        if a != 0.0 {
                            axpy(h_step * a, &k[j], &mut y_tmp);
                        }
                    }
                    rhs.eval(t_old + h_step * dopri853::c(si + 1), &y_tmp, &mut k[si]);
                }

                for i in 0..dim {
                    f_dense[0][i] = y_next[i] - y_stage_start[i];
                }
                for i in 0..dim {
                    f_dense[1][i] = h_step * k[0][i] - f_dense[0][i];
                }
                for i in 0..dim {
                    f_dense[2][i] = 2.0 * f_dense[0][i] - h_step * (k[12][i] + k[0][i]);
                }
                for row in 0..4 {
                    for i in 0..dim {
                        let mut s = 0.0_f64;
                        for j in 0..16 {
                            let c = dopri853::d(row + 4, j + 1);
                            if c != 0.0 {
                                s += c * k[j][i];
                            }
                        }
                        f_dense[3 + row][i] = h_step * s;
                    }
                }

                let t_new = t_old + h_step;
                while i_out < t_grid.len() && t_grid[i_out] <= t_new + 1e-12 {
                    let tg = t_grid[i_out];
                    let theta = (tg - t_old) / h_step;
                    dense_dop853_eval(&y_stage_start, &f_dense, theta, &mut y_out[i_out]);
                    if !all_finite(&y_out[i_out]) {
                        return Err(IntegrateError::Diverged { t: tg });
                    }
                    i_out += 1;
                }

                for i in 0..dim {
                    k[0][i] = k[12][i];
                }
                y.clone_from(&y_next);
                t = t_new;
                h = if last { h_new } else { h_new };
                if (t - tf).abs() < 1e-12 {
                    break;
                }
            } else {
                dp8_prev_rej = true;
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

pub(super) fn integrate_tsit5<R: Rhs + ?Sized>(
    rhs: &mut R,
    t_grid: &[f64],
    y_out: &mut [Vec<f64>],
    rtol: f64,
    atol: f64,
) -> Result<(), IntegrateError> {
    const MAX_STEPS: u32 = 300_000;
    let dim = rhs.dim();
    let tf = *t_grid.last().unwrap();
    let t0 = t_grid[0];

    let mut k = vec![vec![0.0_f64; dim]; 7];
    let mut y_tmp = vec![0.0_f64; dim];
    let mut y_next = vec![0.0_f64; dim];
    let mut errw = vec![0.0_f64; dim];

    let mut y = y_out[0].clone();
    let mut t = t0;
    rhs.eval(t, &y, &mut k[0]);

    let mut f_tmp = vec![0.0_f64; dim];
    let mut y1 = vec![0.0_f64; dim];
    let mut f1 = vec![0.0_f64; dim];
    let mut h = h_init(rhs, t0, &y, tf, rtol, atol, &mut f_tmp, &mut y1, &mut f1);
    rhs.eval(t, &y, &mut k[0]);

    let mut i_out = 1usize;
    let mut n_step: u32 = 0;

    while i_out < t_grid.len() {
        let t_target = t_grid[i_out];
        while t < t_target - 1e-14 {
            if n_step > MAX_STEPS {
                return Err(IntegrateError::MaxSteps);
            }
            n_step += 1;

            let mut last = false;
            let dir = (tf - t0).signum();
            if (t + 1.01 * h - tf) * dir >= 0.0 {
                h = tf - t;
                last = true;
            }

            let t_old = t;
            let y_stage_start = y.clone();
            let h_step = h;

            let v = h_step * tsit5::A21;
            for i in 0..dim {
                y_tmp[i] = y_stage_start[i] + v * k[0][i];
            }
            rhs.eval(t_old + tsit5::C1 * h_step, &y_tmp, &mut k[1]);

            for i in 0..dim {
                y_tmp[i] = y_stage_start[i]
                    + h_step * (tsit5::A31 * k[0][i] + tsit5::A32 * k[1][i]);
            }
            rhs.eval(t_old + tsit5::C2 * h_step, &y_tmp, &mut k[2]);

            for i in 0..dim {
                y_tmp[i] = y_stage_start[i]
                    + h_step
                        * (tsit5::A41 * k[0][i]
                            + tsit5::A42 * k[1][i]
                            + tsit5::A43 * k[2][i]);
            }
            rhs.eval(t_old + tsit5::C3 * h_step, &y_tmp, &mut k[3]);

            for i in 0..dim {
                y_tmp[i] = y_stage_start[i]
                    + h_step
                        * (tsit5::A51 * k[0][i]
                            + tsit5::A52 * k[1][i]
                            + tsit5::A53 * k[2][i]
                            + tsit5::A54 * k[3][i]);
            }
            rhs.eval(t_old + tsit5::C4 * h_step, &y_tmp, &mut k[4]);

            for i in 0..dim {
                y_tmp[i] = y_stage_start[i]
                    + h_step
                        * (tsit5::A61 * k[0][i]
                            + tsit5::A62 * k[1][i]
                            + tsit5::A63 * k[2][i]
                            + tsit5::A64 * k[3][i]
                            + tsit5::A65 * k[4][i]);
            }
            rhs.eval(t_old + h_step, &y_tmp, &mut k[5]);

            for i in 0..dim {
                y_next[i] = y_stage_start[i]
                    + h_step
                        * (tsit5::A71 * k[0][i]
                            + tsit5::A72 * k[1][i]
                            + tsit5::A73 * k[2][i]
                            + tsit5::A74 * k[3][i]
                            + tsit5::A75 * k[4][i]
                            + tsit5::A76 * k[5][i]);
            }
            rhs.eval(t_old + h_step, &y_next, &mut k[6]);

            for i in 0..dim {
                errw[i] = h_step
                    * (tsit5::BT1 * k[0][i]
                        + tsit5::BT2 * k[1][i]
                        + tsit5::BT3 * k[2][i]
                        + tsit5::BT4 * k[3][i]
                        + tsit5::BT5 * k[4][i]
                        + tsit5::BT6 * k[5][i]
                        + tsit5::BT7 * k[6][i]);
            }

            let mut err2 = 0.0_f64;
            for i in 0..dim {
                let sci = atol + y_stage_start[i].abs().max(y_next[i].abs()) * rtol;
                let ri = errw[i] / sci;
                err2 += ri * ri;
            }
            let err = (err2 / dim as f64).sqrt();

            let (acc, h_new) = adapt_step(err, h_step, 4, 0.9, 0.2, 10.0);
            if acc {
                if !all_finite(&y_next) {
                    return Err(IntegrateError::Diverged {
                        t: t_old + h_step,
                    });
                }

                let t_new = t_old + h_step;
                while i_out < t_grid.len() && t_grid[i_out] <= t_new + 1e-12 {
                    let tg = t_grid[i_out];
                    let theta = (tg - t_old) / h_step;
                    dense_tsit5_eval(&y_stage_start, &k, h_step, theta, &mut y_out[i_out]);
                    if !all_finite(&y_out[i_out]) {
                        return Err(IntegrateError::Diverged { t: tg });
                    }
                    i_out += 1;
                }

                for i in 0..dim {
                    k[0][i] = k[6][i];
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

pub(super) fn integrate_bs3<R: Rhs + ?Sized>(
    rhs: &mut R,
    t_grid: &[f64],
    y_out: &mut [Vec<f64>],
    rtol: f64,
    atol: f64,
) -> Result<(), IntegrateError> {
    const MAX_STEPS: u32 = 300_000;
    let dim = rhs.dim();
    let tf = *t_grid.last().unwrap();
    let t0 = t_grid[0];

    let mut k = vec![vec![0.0_f64; dim]; 4];
    let mut y_tmp = vec![0.0_f64; dim];
    let mut y_next = vec![0.0_f64; dim];
    let mut errw = vec![0.0_f64; dim];

    let mut y = y_out[0].clone();
    let mut t = t0;
    rhs.eval(t, &y, &mut k[0]);

    let mut f_tmp = vec![0.0_f64; dim];
    let mut y1 = vec![0.0_f64; dim];
    let mut f1 = vec![0.0_f64; dim];
    let mut h = h_init(rhs, t0, &y, tf, rtol, atol, &mut f_tmp, &mut y1, &mut f1);
    rhs.eval(t, &y, &mut k[0]);

    let mut i_out = 1usize;
    let mut n_step: u32 = 0;

    while i_out < t_grid.len() {
        let t_target = t_grid[i_out];
        while t < t_target - 1e-14 {
            if n_step > MAX_STEPS {
                return Err(IntegrateError::MaxSteps);
            }
            n_step += 1;

            let mut last = false;
            let dir = (tf - t0).signum();
            if (t + 1.01 * h - tf) * dir >= 0.0 {
                h = tf - t;
                last = true;
            }

            let t_old = t;
            let y_stage_start = y.clone();
            let h_step = h;

            for i in 0..dim {
                y_tmp[i] = y_stage_start[i] + h_step * bs3::A21 * k[0][i];
            }
            rhs.eval(t_old + bs3::C2 * h_step, &y_tmp, &mut k[1]);

            for i in 0..dim {
                y_tmp[i] = y_stage_start[i] + h_step * bs3::A32 * k[1][i];
            }
            rhs.eval(t_old + bs3::C3 * h_step, &y_tmp, &mut k[2]);

            for i in 0..dim {
                y_next[i] = y_stage_start[i]
                    + h_step
                        * (bs3::B1 * k[0][i] + bs3::B2 * k[1][i] + bs3::B3 * k[2][i]);
            }
            rhs.eval(t_old + h_step, &y_next, &mut k[3]);

            for i in 0..dim {
                errw[i] = h_step
                    * (bs3::E0 * k[0][i]
                        + bs3::E1 * k[1][i]
                        + bs3::E2 * k[2][i]
                        + bs3::E3 * k[3][i]);
            }

            let mut err2 = 0.0_f64;
            for i in 0..dim {
                let sci = atol + y_stage_start[i].abs().max(y_next[i].abs()) * rtol;
                let ri = errw[i] / sci;
                err2 += ri * ri;
            }
            let err = (err2 / dim as f64).sqrt();

            let (acc, h_new) = adapt_step(err, h_step, 2, 0.9, 0.2, 10.0);
            if acc {
                if !all_finite(&y_next) {
                    return Err(IntegrateError::Diverged {
                        t: t_old + h_step,
                    });
                }

                let t_new = t_old + h_step;
                while i_out < t_grid.len() && t_grid[i_out] <= t_new + 1e-12 {
                    let tg = t_grid[i_out];
                    let theta = (tg - t_old) / h_step;
                    dense_bs3_eval(&y_stage_start, &k, h_step, theta, &mut y_out[i_out]);
                    if !all_finite(&y_out[i_out]) {
                        return Err(IntegrateError::Diverged { t: tg });
                    }
                    i_out += 1;
                }

                for i in 0..dim {
                    k[0][i] = k[3][i];
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
