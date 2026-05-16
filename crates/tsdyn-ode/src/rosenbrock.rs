//! Rosenbrock–Wanner stiff integrators (N2.c).
//!
//! Tableau constants for **Rodas4** and **ROS34PW1a** (`Rosenbrock34`) are transcribed from
//! SciML OrdinaryDiffEq.jl (`lib/OrdinaryDiffEqRosenbrockTableaus/src/rosenbrock_tableaus.jl`, MIT).
//! Dense output on the fixed grid uses linear interpolation between step endpoints (Hermite deferred).

use crate::controller::adapt_step_pi;
use crate::lu::solve_linear_row_major;
use crate::rhs::IrOdeRhs;
use crate::util::copy_from;
use crate::{all_finite, h_init, IntegrateError, Method, Rhs};
use tsdyn_core::ir::CompiledOde;

/// Rosenbrock23 — OrdinaryDiffEq `Rosenbrock23Tableau`.
const RB23_D: f64 = 1.0 / (2.0 + std::f64::consts::SQRT_2);
const RB23_C32: f64 = 6.0 + std::f64::consts::SQRT_2;

/// ROS34PW1a (`Rosenbrock34`), 4 stages.
const ROS34_N: usize = 4;
const ROS34_GAMMA: f64 = 0.435866521508459;
const ROS34_A: [[f64; ROS34_N]; ROS34_N] = [
    [0.0, 0.0, 0.0, 0.0],
    [5.0905205106702045, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [
        4.005173696367865,
        0.19316470237944158,
        1.147140180139521,
        0.0,
    ],
];
const ROS34_C: [[f64; ROS34_N]; ROS34_N] = [
    [0.0, 0.0, 0.0, 0.0],
    [-11.679081231228288, 0.0, 0.0, 0.0],
    [-0.7100952636543062, -0.04165460771675499, 0.0, 0.0],
    [
        -11.979557762226603,
        -0.48054400523894975,
        1.4350493021655284,
        0.0,
    ],
];
const ROS34_C_TIME: [f64; ROS34_N] = [0.0, 2.218787467653286, 0.0, 1.7837037931914073];
const ROS34_B: [f64; ROS34_N] = [
    6.1538321465310215,
    -0.8364233759732359,
    -0.8614792120957679,
    2.294280360279042,
];
const ROS34_BTILDE: [f64; ROS34_N] = [-5.429679341539398, -1.3273810331413745, 0.0, 0.0];

/// Rodas4, 6 stages (`n_stage` = 6). Pad tableau rows to width 6.
const R4_N: usize = 6;
const R4_GAMMA: f64 = 0.25;
type TabR6 = [[f64; R4_N]; R4_N];

const R4_A: TabR6 = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.544, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.9466785280815826, 0.2557011698983284, 0.0, 0.0, 0.0, 0.0],
    [
        3.314825187068521,
        2.896124015972201,
        0.9986419139977817,
        0.0,
        0.0,
        0.0,
    ],
    [
        1.221224509226641,
        6.019134481288629,
        12.53708332932087,
        -0.687886036105895,
        0.0,
        0.0,
    ],
    [
        1.221224509226641,
        6.019134481288629,
        12.53708332932087,
        -0.687886036105895,
        1.0,
        0.0,
    ],
];
const R4_C: TabR6 = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [-5.6688, 0.0, 0.0, 0.0, 0.0, 0.0],
    [-2.430093356833875, -0.2063599157091915, 0.0, 0.0, 0.0, 0.0],
    [
        -0.1073529058151375,
        -9.594562251023355,
        -20.47028614809616,
        0.0,
        0.0,
        0.0,
    ],
    [
        7.496443313967647,
        -10.24680431464352,
        -33.99990352819905,
        11.7089089320616,
        0.0,
        0.0,
    ],
    [
        8.083246795921522,
        -7.981132988064893,
        -31.52159432874371,
        16.31930543123136,
        -6.058818238834054,
        0.0,
    ],
];
const R4_C_TIME: [f64; R4_N] = [0.0, 0.386, 0.21, 0.63, 1.0, 1.0];
const R4_B: [f64; R4_N] = [
    1.221224509226641,
    6.019134481288629,
    12.53708332932087,
    -0.687886036105895,
    1.0,
    1.0,
];
const R4_BTILDE: [f64; R4_N] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0];

#[inline]
fn build_w_matrix(dim: usize, dt_gamma: f64, jac: &[f64], w: &mut [f64]) {
    let n = dim;
    for i in 0..n {
        for j in 0..n {
            let ij = i * n + j;
            let j_elem = jac[ij];
            w[ij] = if i == j { 1.0 } else { 0.0 } - dt_gamma * j_elem;
        }
    }
}

#[inline]
fn scale_vec(s: f64, v: &mut [f64]) {
    for x in v.iter_mut() {
        *x *= s;
    }
}

#[inline]
fn err_rms(dim: usize, errw: &[f64], y: &[f64], y_next: &[f64], rtol: f64, atol: f64) -> f64 {
    let mut err2 = 0.0_f64;
    for i in 0..dim {
        let sci = atol + y[i].abs().max(y_next[i].abs()) * rtol;
        let ri = errw[i] / sci;
        err2 += ri * ri;
    }
    (err2 / dim as f64).sqrt()
}

struct RosenbrockWorkspace {
    jac: Vec<f64>,
    w: Vec<f64>,
    w_fact: Vec<f64>,
    f0: Vec<f64>,
    f1: Vec<f64>,
    f2: Vec<f64>,
    k1: Vec<f64>,
    k2: Vec<f64>,
    k3: Vec<f64>,
    err: Vec<f64>,
    y_tmp: Vec<f64>,
    y_next: Vec<f64>,
    lin_rhs: Vec<f64>,
}

impl RosenbrockWorkspace {
    fn new(dim: usize) -> Self {
        let n2 = dim * dim;
        Self {
            jac: vec![0.0; n2],
            w: vec![0.0; n2],
            w_fact: vec![0.0; n2],
            f0: vec![0.0; dim],
            f1: vec![0.0; dim],
            f2: vec![0.0; dim],
            k1: vec![0.0; dim],
            k2: vec![0.0; dim],
            k3: vec![0.0; dim],
            err: vec![0.0; dim],
            y_tmp: vec![0.0; dim],
            y_next: vec![0.0; dim],
            lin_rhs: vec![0.0; dim],
        }
    }
}

fn step_rosenbrock23(
    rhs: &mut IrOdeRhs<'_>,
    t: f64,
    y: &[f64],
    dt: f64,
    rtol: f64,
    atol: f64,
    work: &mut RosenbrockWorkspace,
) -> Result<(Vec<f64>, f64, Vec<f64>, Vec<f64>), IntegrateError> {
    let dim = rhs.dim();
    let d = RB23_D;
    let dt_gamma = dt * d;

    rhs.eval(t, y, &mut work.f0);
    rhs.ode
        .eval_jacobian(t, y, rhs.params, &mut work.jac, rhs.scratch);

    build_w_matrix(dim, dt_gamma, &work.jac, &mut work.w);

    // Julia builds `W = J - M/(γ dt)`; we factor `W_us = I - γ dt J = -γ dt · W`.
    // If Julia solves `W k = -b`, then `W_us k = γ dt · b`.
    copy_from(&work.f0, &mut work.lin_rhs);
    scale_vec(dt_gamma, &mut work.lin_rhs);
    copy_from(&work.w, &mut work.w_fact);
    solve_linear_row_major(&mut work.w_fact, dim, &mut work.lin_rhs)
        .map_err(|_| IntegrateError::Diverged { t })?;
    scale_vec(-1.0 / dt_gamma, &mut work.lin_rhs);
    copy_from(&work.lin_rhs, &mut work.k1);

    copy_from(y, &mut work.y_tmp);
    crate::util::axpy(0.5 * dt, &work.k1, &mut work.y_tmp);
    rhs.eval(t + 0.5 * dt, &work.y_tmp, &mut work.f1);

    copy_from(&work.f1, &mut work.lin_rhs);
    for i in 0..dim {
        work.lin_rhs[i] -= work.k1[i];
    }
    scale_vec(dt_gamma, &mut work.lin_rhs);
    copy_from(&work.w, &mut work.w_fact);
    solve_linear_row_major(&mut work.w_fact, dim, &mut work.lin_rhs)
        .map_err(|_| IntegrateError::Diverged { t })?;
    for i in 0..dim {
        work.k2[i] = -work.lin_rhs[i] / dt_gamma + work.k1[i];
    }

    copy_from(y, &mut work.y_next);
    crate::util::axpy(dt, &work.k2, &mut work.y_next);

    rhs.eval(t + dt, &work.y_next, &mut work.f2);

    for i in 0..dim {
        work.lin_rhs[i] =
            work.f2[i] - RB23_C32 * (work.k2[i] - work.f1[i]) - 2.0 * (work.k1[i] - work.f0[i]);
    }
    scale_vec(dt_gamma, &mut work.lin_rhs);
    copy_from(&work.w, &mut work.w_fact);
    solve_linear_row_major(&mut work.w_fact, dim, &mut work.lin_rhs)
        .map_err(|_| IntegrateError::Diverged { t })?;
    scale_vec(-1.0 / dt_gamma, &mut work.lin_rhs);
    copy_from(&work.lin_rhs, &mut work.k3);

    for i in 0..dim {
        work.err[i] = (dt / 6.0) * (work.k1[i] - 2.0 * work.k2[i] + work.k3[i]);
    }

    let err = err_rms(dim, &work.err, y, &work.y_next, rtol, atol);

    if !all_finite(&work.y_next) {
        return Err(IntegrateError::Diverged { t: t + dt });
    }

    let y_ret = std::mem::replace(&mut work.y_next, vec![0.0; dim]);
    let f0_ret = std::mem::replace(&mut work.f0, vec![0.0; dim]);
    let f2_ret = work.f2.clone();
    work.y_next = y_ret.clone();
    work.f0 = f0_ret.clone();

    Ok((y_ret, err, f0_ret, f2_ret))
}

struct RodasWorkspace {
    jac: Vec<f64>,
    w: Vec<f64>,
    w_fact: Vec<f64>,
    f0: Vec<f64>,
    f_end: Vec<f64>,
    du: Vec<f64>,
    acc: Vec<f64>,
    u_st: Vec<f64>,
    y_next: Vec<f64>,
    err: Vec<f64>,
    lin_rhs: Vec<f64>,
    ks: Vec<Vec<f64>>,
}

impl RodasWorkspace {
    fn new(dim: usize) -> Self {
        let n2 = dim * dim;
        Self {
            jac: vec![0.0; n2],
            w: vec![0.0; n2],
            w_fact: vec![0.0; n2],
            f0: vec![0.0; dim],
            f_end: vec![0.0; dim],
            du: vec![0.0; dim],
            acc: vec![0.0; dim],
            u_st: vec![0.0; dim],
            y_next: vec![0.0; dim],
            err: vec![0.0; dim],
            lin_rhs: vec![0.0; dim],
            ks: Vec::new(),
        }
    }
}

fn step_rodas_tableau<const N: usize>(
    rhs: &mut IrOdeRhs<'_>,
    t: f64,
    y: &[f64],
    dt: f64,
    n_stage: usize,
    gamma: f64,
    a: &[[f64; N]; N],
    c_mat: &[[f64; N]; N],
    c_time: &[f64; N],
    b: &[f64; N],
    btilde: &[f64; N],
    rtol: f64,
    atol: f64,
    work: &mut RodasWorkspace,
) -> Result<(Vec<f64>, f64, Vec<f64>, Vec<f64>), IntegrateError> {
    let dim = rhs.dim();
    let dt_gamma = dt * gamma;

    rhs.eval(t, y, &mut work.f0);
    rhs.ode
        .eval_jacobian(t, y, rhs.params, &mut work.jac, rhs.scratch);
    build_w_matrix(dim, dt_gamma, &work.jac, &mut work.w);

    work.ks.clear();
    work.ks.resize(n_stage, vec![0.0; dim]);

    for i in 0..dim {
        work.lin_rhs[i] = dt_gamma * work.f0[i];
    }
    copy_from(&work.w, &mut work.w_fact);
    solve_linear_row_major(&mut work.w_fact, dim, &mut work.lin_rhs)
        .map_err(|_| IntegrateError::Diverged { t })?;
    copy_from(&work.lin_rhs, &mut work.ks[0]);

    for st in 1..n_stage {
        copy_from(y, &mut work.u_st);
        for j in 0..st {
            let a_ij = a[st][j];
            if a_ij != 0.0 {
                crate::util::axpy(a_ij, &work.ks[j], &mut work.u_st);
            }
        }

        let tc = t + c_time[st] * dt;
        let skip_f = st > 1
            && (c_time[st] - c_time[st - 1]).abs() < 1e-15
            && (0..st).all(|j| (a[st][j] - a[st - 1][j]).abs() < 1e-15);
        if !skip_f {
            rhs.eval(tc, &work.u_st, &mut work.du);
        }

        for i in 0..dim {
            work.acc[i] = 0.0;
        }
        for j in 0..st {
            let c_ij = c_mat[st][j];
            if c_ij != 0.0 {
                let coef = c_ij / dt;
                crate::util::axpy(coef, &work.ks[j], &mut work.acc);
            }
        }

        for i in 0..dim {
            work.lin_rhs[i] = dt_gamma * (work.du[i] + work.acc[i]);
        }
        copy_from(&work.w, &mut work.w_fact);
        solve_linear_row_major(&mut work.w_fact, dim, &mut work.lin_rhs)
            .map_err(|_| IntegrateError::Diverged { t })?;
        copy_from(&work.lin_rhs, &mut work.ks[st]);
    }

    copy_from(y, &mut work.y_next);
    for i in 0..n_stage {
        if b[i] != 0.0 {
            crate::util::axpy(b[i], &work.ks[i], &mut work.y_next);
        }
    }

    for i in 0..dim {
        work.err[i] = 0.0;
    }
    for i in 0..n_stage {
        if btilde[i] != 0.0 {
            crate::util::axpy(btilde[i], &work.ks[i], &mut work.err);
        }
    }

    let err = err_rms(dim, &work.err, y, &work.y_next, rtol, atol);

    if !all_finite(&work.y_next) {
        return Err(IntegrateError::Diverged { t: t + dt });
    }

    rhs.eval(t + dt, &work.y_next, &mut work.f_end);

    let y_ret = std::mem::replace(&mut work.y_next, vec![0.0; dim]);
    let f0_ret = std::mem::replace(&mut work.f0, vec![0.0; dim]);
    let f_end_ret = std::mem::replace(&mut work.f_end, vec![0.0; dim]);
    work.y_next = y_ret.clone();
    work.f0 = f0_ret.clone();

    Ok((y_ret, err, f0_ret, f_end_ret))
}

fn integrate_rosenbrock_adaptive(
    ode: &CompiledOde,
    params: &[f64],
    t_grid: &[f64],
    y_out: &mut [Vec<f64>],
    rtol: f64,
    atol: f64,
    method: Method,
) -> Result<(), IntegrateError> {
    let dim = ode.dim;
    let tf = *t_grid.last().unwrap();
    let t0 = t_grid[0];

    let mut scratch: Vec<f64> = Vec::with_capacity(64);
    let mut rhs = IrOdeRhs {
        ode,
        params,
        scratch: &mut scratch,
    };

    let mut rb_ws = RosenbrockWorkspace::new(dim);
    let mut rod_ws = RodasWorkspace::new(dim);

    let order_pi = match method {
        // Principal order of the advancing Rosenbrock23 step (embedded pair is 2(3)).
        Method::Rosenbrock23 => 2_u32,
        Method::Rosenbrock34 | Method::Rodas4 => 4_u32,
        _ => unreachable!(),
    };

    let mut y = y_out[0].clone();
    let mut t = t0;

    let mut f_tmp = vec![0.0_f64; dim];
    let mut y1 = vec![0.0_f64; dim];
    let mut f1 = vec![0.0_f64; dim];
    let mut h = h_init(
        &mut rhs, t0, &y, tf, rtol, atol, &mut f_tmp, &mut y1, &mut f1,
    );

    let mut i_out = 1usize;
    let mut err_prev: Option<f64> = None;

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

            let step_res = match method {
                Method::Rosenbrock23 => {
                    step_rosenbrock23(&mut rhs, t, &y, h_step, rtol, atol, &mut rb_ws)
                }
                Method::Rosenbrock34 => step_rodas_tableau(
                    &mut rhs,
                    t,
                    &y,
                    h_step,
                    ROS34_N,
                    ROS34_GAMMA,
                    &ROS34_A,
                    &ROS34_C,
                    &ROS34_C_TIME,
                    &ROS34_B,
                    &ROS34_BTILDE,
                    rtol,
                    atol,
                    &mut rod_ws,
                ),
                Method::Rodas4 => step_rodas_tableau(
                    &mut rhs,
                    t,
                    &y,
                    h_step,
                    R4_N,
                    R4_GAMMA,
                    &R4_A,
                    &R4_C,
                    &R4_C_TIME,
                    &R4_B,
                    &R4_BTILDE,
                    rtol,
                    atol,
                    &mut rod_ws,
                ),
                _ => unreachable!(),
            };

            let (y_new, err, _f_old, _f_new) = step_res?;

            let (acc, h_new) = adapt_step_pi(err, err_prev, h, order_pi, 0.9, 0.2, 10.0);

            if acc {
                err_prev = Some(err.max(1e-16));
                let t_new = t_old + h_step;
                while i_out < t_grid.len() && t_grid[i_out] <= t_new + 1e-12 {
                    let tg = t_grid[i_out];
                    let theta = ((tg - t_old) / h_step).clamp(0.0, 1.0);
                    let row = &mut y_out[i_out];
                    let d = row.len();
                    for j in 0..d {
                        row[j] = (1.0 - theta) * y[j] + theta * y_new[j];
                    }
                    if !all_finite(row) {
                        return Err(IntegrateError::Diverged { t: tg });
                    }
                    i_out += 1;
                }

                y = y_new;
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

/// Entry from [`crate::integrate_ode`] for stiff Rosenbrock methods.
pub fn integrate_stiff(
    ode: &CompiledOde,
    params: &[f64],
    t_grid: &[f64],
    y_out: &mut [Vec<f64>],
    method: Method,
    rtol: f64,
    atol: f64,
) -> Result<(), IntegrateError> {
    if ode.jacobian.is_none() {
        return Err(IntegrateError::MissingJacobian {
            method: format!("{method:?}"),
        });
    }
    integrate_rosenbrock_adaptive(ode, params, t_grid, y_out, rtol, atol, method)
}
