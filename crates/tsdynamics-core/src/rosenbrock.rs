//! Stiff ODE integration over a tape VM.
//!
//! An L-stable **linearly-implicit Euler** step (a one-stage Rosenbrock /
//! W-method) with **step-doubling Richardson extrapolation** for error control.
//! Each step freezes the analytic Jacobian `J = ∂f/∂u` the tape provides at the
//! step start and forms `(I − cJ)`; one big step of size `h` and two of `h/2`
//! give an order-1 base solution, a local error estimate, and an extrapolated
//! order-2 solution (which is what we propagate — local extrapolation).
//!
//! Being L-stable, it integrates stiff systems where the explicit RK4/RK45
//! kernels would need vanishingly small steps. The Jacobian is exact (symbolic,
//! evaluated through the same tape), so there is no finite-difference noise and
//! no Newton iteration — just one LU factorization per distinct step size.
//!
//! `∂f/∂t` is neglected (a W-method): for autonomous systems this is exact, and
//! for non-autonomous ones it only affects the order-1 base method, which the
//! Richardson extrapolation and adaptive controller absorb.

use crate::vm::Tape;

const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 5.0;
const ERR_EXPONENT: f64 = -0.5; // order-1 base method ⇒ error estimate ~ h²

// ---- dense LU with partial pivoting (small dim) --------------------------

/// In-place LU factorization with partial pivoting; `false` if singular.
/// `a` is row-major `n × n`; `piv[k]` records the row swapped to position `k`.
fn lu_factor(a: &mut [f64], n: usize, piv: &mut [usize]) -> bool {
    for k in 0..n {
        let mut pivot = k;
        let mut max = a[k * n + k].abs();
        for i in (k + 1)..n {
            let v = a[i * n + k].abs();
            if v > max {
                max = v;
                pivot = i;
            }
        }
        if max == 0.0 || max.is_nan() {
            return false; // singular (or NaN pivot column)
        }
        if pivot != k {
            for j in 0..n {
                a.swap(k * n + j, pivot * n + j);
            }
        }
        piv[k] = pivot;
        let akk = a[k * n + k];
        for i in (k + 1)..n {
            let f = a[i * n + k] / akk;
            a[i * n + k] = f;
            for j in (k + 1)..n {
                a[i * n + j] -= f * a[k * n + j];
            }
        }
    }
    true
}

/// Solve `LU x = b` in place (`b` ← `x`), applying the recorded row swaps.
#[allow(clippy::needless_range_loop)]
fn lu_solve(a: &[f64], n: usize, piv: &[usize], b: &mut [f64]) {
    for k in 0..n {
        let p = piv[k];
        if p != k {
            b.swap(k, p);
        }
    }
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= a[i * n + j] * b[j];
        }
        b[i] = s;
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i * n + j] * b[j];
        }
        b[i] = s / a[i * n + i];
    }
}

/// Build `mat = I − c·J` (row-major `n × n`).
fn build_shifted(mat: &mut [f64], jac: &[f64], n: usize, c: f64) {
    for i in 0..n {
        for j in 0..n {
            mat[i * n + j] = (if i == j { 1.0 } else { 0.0 }) - c * jac[i * n + j];
        }
    }
}

// ---- workspace -----------------------------------------------------------

/// Per-worker scratch for the stiff stepper (one per rayon task).
pub struct StiffWorkspace {
    regs: Vec<f64>,
    f0: Vec<f64>,
    fmid: Vec<f64>,
    pub fnew: Vec<f64>,
    jac: Vec<f64>,
    mat: Vec<f64>,
    piv: Vec<usize>,
    rhs: Vec<f64>,
    u_big: Vec<f64>,
    u_mid: Vec<f64>,
    u_two: Vec<f64>,
    pub u_new: Vec<f64>,
    scale: Vec<f64>,
}

impl StiffWorkspace {
    pub fn for_tape(tape: &Tape) -> Self {
        let d = tape.dim();
        let z = || vec![0.0; d];
        StiffWorkspace {
            regs: vec![0.0; tape.n_reg()],
            f0: z(),
            fmid: z(),
            fnew: z(),
            jac: vec![0.0; d * d],
            mat: vec![0.0; d * d],
            piv: vec![0; d],
            rhs: z(),
            u_big: z(),
            u_mid: z(),
            u_two: z(),
            u_new: z(),
            scale: z(),
        }
    }
}

/// One trial stiff step of size `h` from `(t, u)`. On return `ws.u_new` holds
/// the extrapolated order-2 candidate and `ws.f0` holds `f(t, u)`; the scaled
/// error norm is returned (`+∞` if the shifted matrix is singular ⇒ reject).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
fn try_step(
    tape: &Tape,
    t: f64,
    u: &[f64],
    p: &[f64],
    h: f64,
    ws: &mut StiffWorkspace,
    rtol: f64,
    atol: f64,
) -> f64 {
    let n = tape.dim();
    // f(t,u) and J(t,u) in one tape pass.
    tape.eval_jac(u, p, t, &mut ws.regs, &mut ws.f0, &mut ws.jac);

    // Big step: (I − hJ) k = h f0 ; u_big = u + k.
    build_shifted(&mut ws.mat, &ws.jac, n, h);
    if !lu_factor(&mut ws.mat, n, &mut ws.piv) {
        return f64::INFINITY;
    }
    for i in 0..n {
        ws.rhs[i] = h * ws.f0[i];
    }
    lu_solve(&ws.mat, n, &ws.piv, &mut ws.rhs);
    for i in 0..n {
        ws.u_big[i] = u[i] + ws.rhs[i];
    }

    // Two half steps with (I − (h/2)J), J frozen at u (one factorization).
    let hh = 0.5 * h;
    build_shifted(&mut ws.mat, &ws.jac, n, hh);
    if !lu_factor(&mut ws.mat, n, &mut ws.piv) {
        return f64::INFINITY;
    }
    for i in 0..n {
        ws.rhs[i] = hh * ws.f0[i];
    }
    lu_solve(&ws.mat, n, &ws.piv, &mut ws.rhs);
    for i in 0..n {
        ws.u_mid[i] = u[i] + ws.rhs[i];
    }
    tape.eval(&ws.u_mid, p, t + hh, &mut ws.regs, &mut ws.fmid);
    for i in 0..n {
        ws.rhs[i] = hh * ws.fmid[i];
    }
    lu_solve(&ws.mat, n, &ws.piv, &mut ws.rhs);
    for i in 0..n {
        ws.u_two[i] = ws.u_mid[i] + ws.rhs[i];
    }

    // Local extrapolation to order 2, and the step-doubling error estimate.
    let mut acc = 0.0;
    for i in 0..n {
        ws.u_new[i] = 2.0 * ws.u_two[i] - ws.u_big[i];
        ws.scale[i] = u[i].abs().max(ws.u_two[i].abs());
        let e = (ws.u_two[i] - ws.u_big[i]) / (atol + rtol * ws.scale[i]);
        acc += e * e;
    }
    (acc / n as f64).sqrt()
}

#[inline]
fn step_factor(err: f64) -> f64 {
    if err == 0.0 {
        MAX_FACTOR
    } else {
        (SAFETY * err.powf(ERR_EXPONENT)).clamp(MIN_FACTOR, MAX_FACTOR)
    }
}

#[allow(clippy::needless_range_loop)]
fn initial_step(u: &[f64], f0: &[f64], rtol: f64, atol: f64, span: f64) -> f64 {
    let n = u.len();
    let (mut d0, mut d1) = (0.0, 0.0);
    for i in 0..n {
        let sc = atol + rtol * u[i].abs();
        d0 += (u[i] / sc).powi(2);
        d1 += (f0[i] / sc).powi(2);
    }
    d0 = (d0 / n as f64).sqrt();
    d1 = (d1 / n as f64).sqrt();
    let h0 = if d0 < 1e-5 || d1 < 1e-5 {
        1e-6
    } else {
        0.01 * d0 / d1
    };
    h0.min(span.abs()).max(1e-10)
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn hermite(u0: &[f64], f0: &[f64], u1: &[f64], f1: &[f64], h: f64, theta: f64, out: &mut [f64]) {
    let t2 = theta * theta;
    let t3 = t2 * theta;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + theta;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    for i in 0..u0.len() {
        out[i] = h00 * u0[i] + h * h10 * f0[i] + h01 * u1[i] + h * h11 * f1[i];
    }
}

/// Adaptive stiff integration with cubic-Hermite dense output at each `t_eval`.
/// `t_eval` must be increasing with `t_eval[0]` the initial time. Returns a
/// row-major `(t_eval.len(), dim)` buffer; an incomplete run (blow-up / step
/// collapse) leaves the unfilled tail non-finite.
pub fn integrate_dense(
    tape: &Tape,
    u0: &[f64],
    p: &[f64],
    t_eval: &[f64],
    rtol: f64,
    atol: f64,
) -> Vec<f64> {
    let d = tape.dim();
    let n_t = t_eval.len();
    let mut out = vec![0.0; n_t * d];
    if n_t == 0 {
        return out;
    }
    out[0..d].copy_from_slice(u0);
    if n_t < 2 {
        return out;
    }
    let mut ws = StiffWorkspace::for_tape(tape);
    let mut u = u0.to_vec();
    let mut f_start = vec![0.0; d];
    let t0 = t_eval[0];
    let t_final = t_eval[n_t - 1];
    // derivative at the very start (for the first segment's Hermite slope)
    tape.eval(&u, p, t0, &mut ws.regs, &mut f_start);
    let mut h = initial_step(&u, &f_start, rtol, atol, t_final - t0);
    let mut t = t0;
    let mut next = 1usize;

    while next < n_t && t < t_final {
        if t + h > t_final {
            h = t_final - t;
        }
        let err = try_step(tape, t, &u, p, h, &mut ws, rtol, atol);
        if err <= 1.0 {
            let t_new = t + h;
            // f at the new point (Hermite slope + start slope of next segment)
            tape.eval(&ws.u_new, p, t_new, &mut ws.regs, &mut ws.fnew);
            while next < n_t && t_eval[next] <= t_new + 1e-12 {
                let theta = ((t_eval[next] - t) / h).clamp(0.0, 1.0);
                hermite(
                    &u,
                    &f_start,
                    &ws.u_new,
                    &ws.fnew,
                    h,
                    theta,
                    &mut out[next * d..(next + 1) * d],
                );
                next += 1;
            }
            u.copy_from_slice(&ws.u_new);
            f_start.copy_from_slice(&ws.fnew);
            t = t_new;
            h *= step_factor(err);
        } else {
            // reject; non-finite error (singular shift / blow-up) ⇒ strongest shrink
            h *= if err.is_finite() {
                step_factor(err).min(1.0)
            } else {
                MIN_FACTOR
            };
        }
        if h <= 1e-14 * (1.0 + t.abs()) {
            break;
        }
    }
    while next < n_t {
        out[next * d..(next + 1) * d].fill(f64::NAN);
        next += 1;
    }
    out
}

/// Adaptive stiff integration from `t0` to `t1`, returning only the final
/// state (the ensemble/basin primitive). Returns `NaN` if it did not reach
/// `t1` (diverged or step collapsed).
#[allow(clippy::too_many_arguments)]
pub fn integrate_final(
    tape: &Tape,
    u0: &[f64],
    p: &[f64],
    t0: f64,
    t1: f64,
    rtol: f64,
    atol: f64,
    ws: &mut StiffWorkspace,
) -> Vec<f64> {
    let d = tape.dim();
    let mut u = u0.to_vec();
    let mut f_start = vec![0.0; d];
    tape.eval(&u, p, t0, &mut ws.regs, &mut f_start);
    let mut h = initial_step(&u, &f_start, rtol, atol, t1 - t0);
    let mut t = t0;
    while t < t1 {
        if t + h > t1 {
            h = t1 - t;
        }
        let err = try_step(tape, t, &u, p, h, ws, rtol, atol);
        if err <= 1.0 {
            u.copy_from_slice(&ws.u_new);
            t += h;
            h *= step_factor(err);
        } else {
            h *= if err.is_finite() {
                step_factor(err).min(1.0)
            } else {
                MIN_FACTOR
            };
        }
        if h <= 1e-14 * (1.0 + t.abs()) {
            break;
        }
    }
    if t1 - t > 1e-12 * (1.0 + t1.abs()) {
        return vec![f64::NAN; d];
    }
    u
}
