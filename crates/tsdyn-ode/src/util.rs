//! Small `f64` vector helpers (no BLAS dependency).
#![allow(dead_code)]

#[inline]
pub fn copy_from(src: &[f64], dst: &mut [f64]) {
    dst.copy_from_slice(src);
}

#[inline]
pub fn axpy(a: f64, x: &[f64], y: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len());
    for (xi, yi) in x.iter().zip(y.iter_mut()) {
        *yi += a * xi;
    }
}

#[inline]
pub fn scale_mut(s: f64, y: &mut [f64]) {
    for yi in y.iter_mut() {
        *yi *= s;
    }
}

#[inline]
pub fn weighted_sum<'a>(
    coeffs: impl Iterator<Item = f64> + 'a,
    vecs: impl Iterator<Item = &'a [f64]> + 'a,
    out: &mut [f64],
) {
    out.fill(0.0);
    for (c, v) in coeffs.zip(vecs) {
        if c != 0.0 {
            axpy(c, v, out);
        }
    }
}

/// DP5 / Tsit5-style dense output (Owren–Zenn for DP5 in Hairer I).
#[inline]
pub fn dense_dp5_eval(r: &[Vec<f64>; 5], theta: f64, theta1: f64, out: &mut [f64]) {
    // y = r0 + (r1 + (r2 + (r3 + r4 * theta1) * theta) * theta1) * theta
    let dim = out.len();
    for i in 0..dim {
        let r4 = r[4][i];
        let r3 = r[3][i];
        let r2 = r[2][i];
        let r1 = r[1][i];
        let r0 = r[0][i];
        let t1 = r3 + r4 * theta1;
        let t2 = r2 + t1 * theta;
        let t3 = r1 + t2 * theta1;
        out[i] = r0 + t3 * theta;
    }
}

/// DOP853 dense output (Hairer, `ode_solvers` `Dop853::compute_y_out`).
#[inline]
pub fn dense_dp8_eval(r: &[Vec<f64>; 8], theta: f64, theta1: f64, out: &mut [f64]) {
    let dim = out.len();
    for i in 0..dim {
        let r7 = r[7][i];
        let r6 = r[6][i];
        let r5 = r[5][i];
        let r4 = r[4][i];
        let r3 = r[3][i];
        let r2 = r[2][i];
        let r1 = r[1][i];
        let r0 = r[0][i];
        let t6 = r6 + r7 * theta;
        let t5 = r5 + t6 * theta1;
        let t4 = r4 + t5 * theta;
        let t3 = r3 + t4 * theta1;
        let t2 = r2 + t3 * theta;
        let t1 = r1 + t2 * theta1;
        out[i] = r0 + t1 * theta;
    }
}

/// Horner form: `c[0] + x*(c[1] + x*(c[2] + ...))`.
#[inline]
pub fn eval_poly(c: &[f64], x: f64) -> f64 {
    let mut acc = 0.0_f64;
    for &coef in c.iter().rev() {
        acc = coef + x * acc;
    }
    acc
}

/// SciPy `Dop853DenseOutput` evaluation (`scipy.integrate._ivp.rk`).
#[inline]
pub fn dense_dop853_eval(y_old: &[f64], f_rows: &[Vec<f64>; 7], x: f64, out: &mut [f64]) {
    let dim = out.len();
    for i in 0..dim {
        let mut yi = 0.0_f64;
        for (j, row) in f_rows.iter().rev().enumerate() {
            yi += row[i];
            if j % 2 == 0 {
                yi *= x;
            } else {
                yi *= 1.0 - x;
            }
        }
        out[i] = yi + y_old[i];
    }
}

/// OrdinaryDiffEq.jl Tsit5 `Val{0}` interpolant (`interpolants.jl`).
#[inline]
pub fn dense_tsit5_eval(y0: &[f64], k: &[Vec<f64>], h: f64, theta: f64, out: &mut [f64]) {
    debug_assert!(k.len() >= 7);
    use crate::butcher::tsit5_interp::*;
    let th = theta;
    let th2 = th * th;
    let b1 = th * eval_poly(&[R011, R012, R013, R014], th);
    let b2 = th2 * eval_poly(&[R22, R23, R24], th);
    let b3 = th2 * eval_poly(&[R32, R33, R34], th);
    let b4 = th2 * eval_poly(&[R42, R43, R44], th);
    let b5 = th2 * eval_poly(&[R52, R53, R54], th);
    let b6 = th2 * eval_poly(&[R62, R63, R64], th);
    let b7 = th2 * eval_poly(&[R72, R73, R74], th);
    let dim = out.len();
    for i in 0..dim {
        out[i] = y0[i]
            + h * (k[0][i] * b1 + k[1][i] * b2 + k[2][i] * b3 + k[3][i] * b4
                + k[4][i] * b5 + k[5][i] * b6 + k[6][i] * b7);
    }
}

/// SciPy `RK23` dense output (`RkDenseOutput` with embedded `P`).
#[inline]
pub fn dense_bs3_eval(y_old: &[f64], k: &[Vec<f64>], h: f64, x: f64, out: &mut [f64]) {
    debug_assert!(k.len() >= 4);
    let p1 = x;
    let p2 = x * p1;
    let p3 = x * p2;
    let dim = out.len();
    for i in 0..dim {
        let q0 = k[0][i];
        let q1 = (-4.0 / 3.0) * k[0][i] + k[1][i] + (4.0 / 3.0) * k[2][i] - k[3][i];
        let q2 = (5.0 / 9.0) * k[0][i] + (-2.0 / 3.0) * k[1][i] + (-8.0 / 9.0) * k[2][i]
            + k[3][i];
        out[i] = y_old[i] + h * (q0 * p1 + q1 * p2 + q2 * p3);
    }
}

/// Cubic Hermite with physical step `h`.
#[inline]
pub fn dense_hermite_h(
    y0: &[f64],
    y1: &[f64],
    m0: &[f64],
    m1: &[f64],
    h: f64,
    theta: f64,
    out: &mut [f64],
) {
    let th = theta;
    let t1 = 1.0 - th;
    let h00 = t1 * t1 * (1.0 + 2.0 * th);
    let h10 = th * th * (3.0 - 2.0 * th);
    let h01 = th * t1 * t1 * h;
    let h11 = th * th * (th - 1.0) * h;
    for i in 0..out.len() {
        out[i] = h00 * y0[i] + h10 * y1[i] + h01 * m0[i] + h11 * m1[i];
    }
}
