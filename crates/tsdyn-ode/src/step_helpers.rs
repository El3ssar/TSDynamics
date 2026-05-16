//! Shared helpers used by adaptive explicit and Rosenbrock drivers.

use crate::rhs::Rhs;

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
