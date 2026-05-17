//! Variational ODE Lyapunov exponents (**N3**): augmented state `(x, Δ_1, …, Δ_k)`,
//! QR renormalisation of the tangent block, time-weighted `log |R_ii|` accumulator.

use std::str::FromStr;

use nalgebra::DMatrix;
use tsdyn_core::ir::CompiledOde;

use crate::driver::integrate_segment_explicit;
use crate::error::IntegrateError;
use crate::method::Method;
use crate::rhs::{IrOdeRhs, Rhs};

/// Augmented RHS: `ẋ = f(x)`, `Δ̇_k = J(x) Δ_k`.
pub struct VariationalRhs<'a> {
    pub ode: &'a CompiledOde,
    pub params: &'a [f64],
    pub n_exp: usize,
    pub scratch: &'a mut Vec<f64>,
    pub jac: &'a mut Vec<f64>,
}

impl Rhs for VariationalRhs<'_> {
    fn dim(&self) -> usize {
        self.ode.dim * (self.n_exp + 1)
    }

    fn eval(&mut self, t: f64, y: &[f64], dy: &mut [f64]) {
        let d = self.ode.dim;
        let x = &y[0..d];
        self.ode
            .eval_rhs(t, x, self.params, &mut dy[0..d], self.scratch);
        self.ode
            .eval_jacobian(t, x, self.params, self.jac, self.scratch);

        for k in 0..self.n_exp {
            let delta = &y[d + k * d..d + (k + 1) * d];
            let out = &mut dy[d + k * d..d + (k + 1) * d];
            for i in 0..d {
                let mut acc = 0.0_f64;
                let row_off = i * d;
                for j in 0..d {
                    acc += self.jac[row_off + j] * delta[j];
                }
                out[i] = acc;
            }
        }
    }
}

fn qr_accumulate_logs_into_state(
    y: &mut [f64],
    d: usize,
    n_exp: usize,
    log_acc: &mut [f64],
) -> Result<(), IntegrateError> {
    let mat = DMatrix::from_fn(d, n_exp, |i, k| y[d + k * d + i]);
    let qr = mat.qr();
    let q = qr.q();
    let r = qr.r();
    for i in 0..n_exp {
        let rii = r[(i, i)].abs();
        if !rii.is_finite() || rii == 0.0 {
            return Err(IntegrateError::Diverged { t: f64::NAN });
        }
        log_acc[i] += rii.ln();
    }
    for k in 0..n_exp {
        for i in 0..d {
            y[d + k * d + i] = q[(i, k)];
        }
    }
    Ok(())
}

/// Benettin-style spectrum on the IR-encoded ODE with automatic variational equations.
///
/// * `ic` — initial state at `t = 0`
/// * `burn_in` — integrate `ẋ = f(x)` only on `[0, burn_in]`
/// * production window `[burn_in, burn_in + final_time]` on the augmented system
/// * every `dt_reortho` time units: advance augmented state, then QR the tangent block
///
/// Exponents are sorted **descending** (largest first).
pub fn lyapunov_spectrum_ode(
    ode: &CompiledOde,
    params: &[f64],
    ic: &[f64],
    burn_in: f64,
    final_time: f64,
    dt_reortho: f64,
    n_exp: usize,
    method: Method,
    rtol: f64,
    atol: f64,
) -> Result<Vec<f64>, IntegrateError> {
    if params.len() != ode.n_params {
        return Err(IntegrateError::ParamsLen {
            expected: ode.n_params,
            got: params.len(),
        });
    }
    if ic.len() != ode.dim {
        return Err(IntegrateError::BadBytecode(format!(
            "ic length {} != dim {}",
            ic.len(),
            ode.dim
        )));
    }
    if ode.jacobian.is_none() {
        return Err(IntegrateError::LyapunovConfig(
            "IR bytecode has no Jacobian (has_jacobian=0); Lyapunov spectrum requires \
             a symbolic Jacobian — add `_jacobian` or use an RHS that differentiates cleanly"
                .into(),
        ));
    }
    let d = ode.dim;
    if n_exp == 0 || n_exp > d {
        return Err(IntegrateError::LyapunovConfig(format!(
            "n_exp must be in 1..=dim ({d}), got {n_exp}"
        )));
    }
    if !burn_in.is_finite() || !final_time.is_finite() || burn_in < 0.0 || final_time < 0.0 {
        return Err(IntegrateError::LyapunovConfig(
            "burn_in and final_time must be finite and non-negative".into(),
        ));
    }
    if dt_reortho <= 0.0 || !dt_reortho.is_finite() {
        return Err(IntegrateError::LyapunovConfig(
            "dt_reortho (QR interval) must be positive and finite".into(),
        ));
    }

    let mut scratch: Vec<f64> = Vec::with_capacity(64);
    let mut jac: Vec<f64> = vec![0.0; d * d];

    // --- Burn-in on the base system ---
    let mut x = ic.to_vec();
    let mut t = 0.0_f64;
    while t < burn_in - 1e-14 {
        let tn = (t + dt_reortho).min(burn_in);
        {
            let mut rhs = IrOdeRhs {
                ode,
                params,
                scratch: &mut scratch,
            };
            x = integrate_segment_explicit(&mut rhs, t, tn, &x, method, rtol, atol)?;
        }
        t = tn;
    }

    // --- Augmented initial condition ---
    let aug_dim = d * (n_exp + 1);
    let mut y = vec![0.0_f64; aug_dim];
    y[0..d].copy_from_slice(&x);
    for k in 0..n_exp {
        y[d + k * d + k] = 1.0;
    }

    let mut log_sum = vec![0.0_f64; n_exp];
    let mut time_sum = 0.0_f64;

    let t_end = burn_in + final_time;
    t = burn_in;

    while t < t_end - 1e-14 {
        let tn = (t + dt_reortho).min(t_end);
        let dt_seg = tn - t;
        if dt_seg <= 0.0 {
            break;
        }
        {
            let mut v_rhs = VariationalRhs {
                ode,
                params,
                n_exp,
                scratch: &mut scratch,
                jac: &mut jac,
            };
            y = integrate_segment_explicit(&mut v_rhs, t, tn, &y, method, rtol, atol)?;
        }
        qr_accumulate_logs_into_state(&mut y, d, n_exp, &mut log_sum)?;
        time_sum += dt_seg;
        t = tn;
    }

    if time_sum <= 0.0 {
        return Err(IntegrateError::LyapunovConfig(
            "zero production window after burn-in — increase final_time or check tolerances".into(),
        ));
    }

    let mut exps: Vec<f64> = log_sum.iter().map(|s| s / time_sum).collect();
    exps.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    Ok(exps)
}

/// Decode bytecode then [`lyapunov_spectrum_ode`].
pub fn lyapunov_spectrum_ode_bytes(
    bytecode: &[u8],
    params: &[f64],
    ic: &[f64],
    burn_in: f64,
    final_time: f64,
    dt_reortho: f64,
    n_exp: usize,
    method: &str,
    rtol: f64,
    atol: f64,
) -> Result<Vec<f64>, IntegrateError> {
    let ode = CompiledOde::from_bytes(bytecode)
        .map_err(|e| IntegrateError::BadBytecode(e.to_string()))?;
    let m = Method::from_str(method.trim())?;
    lyapunov_spectrum_ode(
        &ode,
        params,
        ic,
        burn_in,
        final_time,
        dt_reortho,
        n_exp,
        m,
        rtol,
        atol,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tsdyn_core::ir::Expr;

    fn linear_scalar_ode(le: f64) -> CompiledOde {
        // ẋ = le * x
        let rhs = vec![vec![Expr::Const(le), Expr::Var(0), Expr::Mul]];
        let jac = Some(vec![vec![vec![Expr::Const(le)]]]);
        CompiledOde {
            dim: 1,
            n_params: 0,
            rhs,
            jacobian: jac,
        }
    }

    #[test]
    fn one_d_linear_matches_parameter() {
        let lam = 0.37;
        let ode = linear_scalar_ode(lam);
        let exps = lyapunov_spectrum_ode(
            &ode,
            &[],
            &[1.0],
            0.0,
            50.0,
            0.2,
            1,
            Method::Dp8,
            1e-10,
            1e-12,
        )
        .unwrap();
        assert_eq!(exps.len(), 1);
        assert!((exps[0] - lam).abs() < 0.02, "got {}", exps[0]);
    }
}
