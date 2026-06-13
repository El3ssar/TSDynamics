//! Adaptive Dormand-Prince 5(4) integrator over a tape [`Tape`].
//!
//! The classic `dopri5` embedded pair: a 5th-order solution with a 4th-order
//! companion for the local-error estimate, an error-controlled step size, and
//! the FSAL property (the last stage is the first stage of the next step, so an
//! accepted step costs six RHS evals, not seven).  Output between accepted
//! steps uses cubic-Hermite dense interpolation (O(h⁴)), so the reported grid
//! is decoupled from the internal step size — the same contract SciPy's `RK45`
//! offers, against which this is cross-validated.
//!
//! Forward integration only (`t` increasing); that is all the library's
//! `integrate`/ensemble paths require.

use crate::vm::Tape;

// ---- Dormand-Prince 5(4) Butcher tableau ---------------------------------
const C2: f64 = 1.0 / 5.0;
const C3: f64 = 3.0 / 10.0;
const C4: f64 = 4.0 / 5.0;
const C5: f64 = 8.0 / 9.0;

const A21: f64 = 1.0 / 5.0;
const A31: f64 = 3.0 / 40.0;
const A32: f64 = 9.0 / 40.0;
const A41: f64 = 44.0 / 45.0;
const A42: f64 = -56.0 / 15.0;
const A43: f64 = 32.0 / 9.0;
const A51: f64 = 19372.0 / 6561.0;
const A52: f64 = -25360.0 / 2187.0;
const A53: f64 = 64448.0 / 6561.0;
const A54: f64 = -212.0 / 729.0;
const A61: f64 = 9017.0 / 3168.0;
const A62: f64 = -355.0 / 33.0;
const A63: f64 = 46732.0 / 5247.0;
const A64: f64 = 49.0 / 176.0;
const A65: f64 = -5103.0 / 18656.0;

// 5th-order solution weights (b2 = b7 = 0); also the 7th stage row (FSAL).
const B1: f64 = 35.0 / 384.0;
const B3: f64 = 500.0 / 1113.0;
const B4: f64 = 125.0 / 192.0;
const B5: f64 = -2187.0 / 6784.0;
const B6: f64 = 11.0 / 84.0;

// Error weights = b (5th order) − b̂ (4th order), per stage.
const E1: f64 = 35.0 / 384.0 - 5179.0 / 57600.0;
const E3: f64 = 500.0 / 1113.0 - 7571.0 / 16695.0;
const E4: f64 = 125.0 / 192.0 - 393.0 / 640.0;
const E5: f64 = -2187.0 / 6784.0 - (-92097.0 / 339200.0);
const E6: f64 = 11.0 / 84.0 - 187.0 / 2100.0;
const E7: f64 = -1.0 / 40.0;

// Step-size controller (matching SciPy's RK45 defaults).
const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 10.0;
const ERR_EXPONENT: f64 = -0.2; // -1 / (error_estimator_order + 1), order = 4

/// Per-worker scratch for the adaptive stepper (one per rayon task).
pub struct DpWorkspace {
    regs: Vec<f64>,
    k1: Vec<f64>,
    k2: Vec<f64>,
    k3: Vec<f64>,
    k4: Vec<f64>,
    k5: Vec<f64>,
    k6: Vec<f64>,
    k7: Vec<f64>,
    utmp: Vec<f64>,
    u5: Vec<f64>,
    scale: Vec<f64>,
}

impl DpWorkspace {
    pub fn for_tape(tape: &Tape) -> Self {
        let d = tape.dim();
        let z = || vec![0.0; d];
        DpWorkspace {
            regs: vec![0.0; tape.n_reg()],
            k1: z(),
            k2: z(),
            k3: z(),
            k4: z(),
            k5: z(),
            k6: z(),
            k7: z(),
            utmp: z(),
            u5: z(),
            scale: z(),
        }
    }
}

/// RMS of `v` scaled by `sc_i = atol + rtol·|u_i|` — SciPy's error norm.
#[inline]
fn scaled_rms(v: &[f64], u: &[f64], rtol: f64, atol: f64) -> f64 {
    let mut acc = 0.0;
    for i in 0..v.len() {
        let sc = atol + rtol * u[i].abs();
        let r = v[i] / sc;
        acc += r * r;
    }
    (acc / v.len() as f64).sqrt()
}

/// One trial DP45 step of size `h` from `(t, u)`, given `k1 = f(t, u)` already
/// in `ws.k1`.  Writes the 5th-order candidate to `ws.u5` and `f(t+h, u5)` to
/// `ws.k7` (FSAL), and returns the scaled error norm (accept if ≤ 1).
#[allow(clippy::needless_range_loop)]
#[allow(clippy::too_many_arguments)]
fn try_step(
    tape: &Tape,
    t: f64,
    u: &[f64],
    p: &[f64],
    h: f64,
    ws: &mut DpWorkspace,
    rtol: f64,
    atol: f64,
) -> f64 {
    let d = u.len();
    for i in 0..d {
        ws.utmp[i] = u[i] + h * A21 * ws.k1[i];
    }
    tape.eval(&ws.utmp, p, t + C2 * h, &mut ws.regs, &mut ws.k2);
    for i in 0..d {
        ws.utmp[i] = u[i] + h * (A31 * ws.k1[i] + A32 * ws.k2[i]);
    }
    tape.eval(&ws.utmp, p, t + C3 * h, &mut ws.regs, &mut ws.k3);
    for i in 0..d {
        ws.utmp[i] = u[i] + h * (A41 * ws.k1[i] + A42 * ws.k2[i] + A43 * ws.k3[i]);
    }
    tape.eval(&ws.utmp, p, t + C4 * h, &mut ws.regs, &mut ws.k4);
    for i in 0..d {
        ws.utmp[i] = u[i] + h * (A51 * ws.k1[i] + A52 * ws.k2[i] + A53 * ws.k3[i] + A54 * ws.k4[i]);
    }
    tape.eval(&ws.utmp, p, t + C5 * h, &mut ws.regs, &mut ws.k5);
    for i in 0..d {
        ws.utmp[i] = u[i]
            + h * (A61 * ws.k1[i]
                + A62 * ws.k2[i]
                + A63 * ws.k3[i]
                + A64 * ws.k4[i]
                + A65 * ws.k5[i]);
    }
    tape.eval(&ws.utmp, p, t + h, &mut ws.regs, &mut ws.k6);
    // 5th-order solution (b2 = b7 = 0).
    for i in 0..d {
        ws.u5[i] = u[i]
            + h * (B1 * ws.k1[i] + B3 * ws.k3[i] + B4 * ws.k4[i] + B5 * ws.k5[i] + B6 * ws.k6[i]);
    }
    tape.eval(&ws.u5, p, t + h, &mut ws.regs, &mut ws.k7); // FSAL
                                                           // Local error estimate e = h·Σ Eᵢ kᵢ, normed against max(|u|, |u5|).
    for i in 0..d {
        ws.utmp[i] = h
            * (E1 * ws.k1[i]
                + E3 * ws.k3[i]
                + E4 * ws.k4[i]
                + E5 * ws.k5[i]
                + E6 * ws.k6[i]
                + E7 * ws.k7[i]);
    }
    // scale uses the larger magnitude of the two endpoints (componentwise).
    // Use a dedicated buffer, not `regs`: `regs` is sized to the instruction
    // count, which can be < dim for a degenerate tape (e.g. two components that
    // are the same bare symbol), so aliasing it would index out of bounds.
    for i in 0..d {
        ws.scale[i] = u[i].abs().max(ws.u5[i].abs());
    }
    scaled_rms(&ws.utmp, &ws.scale, rtol, atol)
}

#[inline]
fn step_factor(err: f64) -> f64 {
    if err == 0.0 {
        MAX_FACTOR
    } else {
        (SAFETY * err.powf(ERR_EXPONENT)).clamp(MIN_FACTOR, MAX_FACTOR)
    }
}

/// Hairer's initial step-size heuristic.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
fn initial_step(
    tape: &Tape,
    t0: f64,
    u: &[f64],
    p: &[f64],
    f0: &[f64],
    rtol: f64,
    atol: f64,
    span: f64,
    ws: &mut DpWorkspace,
) -> f64 {
    let d = u.len();
    let d0 = scaled_rms(u, u, rtol, atol);
    let d1 = scaled_rms(f0, u, rtol, atol);
    let h0 = if d0 < 1e-5 || d1 < 1e-5 {
        1e-6
    } else {
        0.01 * d0 / d1
    };
    for i in 0..d {
        ws.utmp[i] = u[i] + h0 * f0[i];
    }
    tape.eval(&ws.utmp, p, t0 + h0, &mut ws.regs, &mut ws.k2);
    for i in 0..d {
        ws.k2[i] -= f0[i];
    }
    let d2 = scaled_rms(&ws.k2, u, rtol, atol) / h0;
    let h1 = if d1.max(d2) <= 1e-15 {
        (h0 * 1e-3).max(1e-6)
    } else {
        (0.01 / d1.max(d2)).powf(0.2)
    };
    (100.0 * h0).min(h1).min(span.abs()).max(1e-12)
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

/// Adaptive DP45 with cubic-Hermite dense output at every `t_eval` point.
///
/// `t_eval` must be increasing with `t_eval[0]` the initial time.  Returns a
/// row-major `(t_eval.len(), dim)` buffer.
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
        return out; // nothing to fill (guard before the slice below)
    }
    out[0..d].copy_from_slice(u0);
    if n_t < 2 {
        return out;
    }
    let mut ws = DpWorkspace::for_tape(tape);
    let mut u = u0.to_vec();
    let t0 = t_eval[0];
    let t_final = t_eval[n_t - 1];
    tape.eval(&u, p, t0, &mut ws.regs, &mut ws.k1);
    let mut h = initial_step(
        tape,
        t0,
        &u,
        p,
        &ws.k1.clone(),
        rtol,
        atol,
        t_final - t0,
        &mut ws,
    );
    let mut t = t0;
    let mut next = 1usize;

    while next < n_t && t < t_final {
        if t + h > t_final {
            h = t_final - t;
        }
        let err = try_step(tape, t, &u, p, h, &mut ws, rtol, atol);
        if err <= 1.0 {
            let t_new = t + h;
            while next < n_t && t_eval[next] <= t_new + 1e-12 {
                let theta = ((t_eval[next] - t) / h).clamp(0.0, 1.0);
                hermite(
                    &u,
                    &ws.k1,
                    &ws.u5,
                    &ws.k7,
                    h,
                    theta,
                    &mut out[next * d..(next + 1) * d],
                );
                next += 1;
            }
            u.copy_from_slice(&ws.u5);
            ws.k1.copy_from_slice(&ws.k7); // FSAL carry
            t = t_new;
            h *= step_factor(err);
        } else {
            // Reject and shrink. A non-finite error (the trajectory blew up to
            // Inf/NaN) forces the strongest shrink so h collapses to the guard
            // below and we break — otherwise `NaN.min(1.0) == 1.0` would leave
            // h unchanged and spin forever under a released GIL.
            h *= if err.is_finite() {
                step_factor(err).min(1.0)
            } else {
                MIN_FACTOR
            };
        }
        if h <= 1e-14 * (1.0 + t.abs()) {
            break; // step collapsed — diverged or stiff
        }
    }
    // Reaching here with points still unfilled means the integration did not
    // complete (blow-up or step collapse). Mark the tail non-finite so the
    // caller detects the failure rather than reading a plausible flat tail.
    while next < n_t {
        out[next * d..(next + 1) * d].fill(f64::NAN);
        next += 1;
    }
    out
}

/// Adaptive DP45 from `t0` to `t1`, returning only the final state
/// (the ensemble/basin primitive — no dense output).
#[allow(clippy::too_many_arguments)]
pub fn integrate_final(
    tape: &Tape,
    u0: &[f64],
    p: &[f64],
    t0: f64,
    t1: f64,
    rtol: f64,
    atol: f64,
    ws: &mut DpWorkspace,
) -> Vec<f64> {
    let mut u = u0.to_vec();
    tape.eval(&u, p, t0, &mut ws.regs, &mut ws.k1);
    let f0 = ws.k1.clone();
    let mut h = initial_step(tape, t0, &u, p, &f0, rtol, atol, t1 - t0, ws);
    let mut t = t0;
    while t < t1 {
        if t + h > t1 {
            h = t1 - t;
        }
        let err = try_step(tape, t, &u, p, h, ws, rtol, atol);
        if err <= 1.0 {
            u.copy_from_slice(&ws.u5);
            ws.k1.copy_from_slice(&ws.k7);
            t += h;
            h *= step_factor(err);
        } else {
            // See integrate_dense: non-finite error forces the strongest shrink
            // so a blown-up trajectory collapses to the guard instead of looping.
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
    // If we did not essentially reach t1, the trajectory diverged or the step
    // collapsed: return NaN so the ensemble/basin caller can classify this IC
    // as escaped rather than treating a stale state as a real attractor.
    if t1 - t > 1e-12 * (1.0 + t1.abs()) {
        return vec![f64::NAN; tape.dim()];
    }
    u
}
