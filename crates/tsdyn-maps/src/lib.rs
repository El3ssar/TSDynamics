//! Rust kernels for discrete-map iteration and Lyapunov spectrum.
//!
//! The hot loop runs entirely on the Rust side: state updates, NaN
//! detection, and QR reorthonormalisation all happen here. Python only
//! pays the FFI cost once per `iterate` / `lyapunov_spectrum` call.

use nalgebra::{DMatrix, DVector};
use tsdyn_core::ir::CompiledMap;

#[derive(Debug)]
pub enum KernelError {
    Divergence { step: usize },
    BadShape(String),
}

impl std::fmt::Display for KernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KernelError::Divergence { step } => {
                write!(f, "divergence detected at step {step}")
            }
            KernelError::BadShape(s) => write!(f, "shape error: {s}"),
        }
    }
}

impl std::error::Error for KernelError {}

fn eval_step(
    map: &CompiledMap,
    state: &[f64],
    params: &[f64],
    out: &mut [f64],
    scratch: &mut Vec<f64>,
) {
    // Maps don't carry a time component; pass t = 0.0 so the shared
    // evaluator signature (introduced for ODE Time-op support in N2)
    // stays uniform.
    for (i, prog) in map.step.iter().enumerate() {
        out[i] = CompiledMap::eval(prog, 0.0, state, params, scratch);
    }
}

fn eval_jacobian(
    map: &CompiledMap,
    state: &[f64],
    params: &[f64],
    out: &mut DMatrix<f64>,
    scratch: &mut Vec<f64>,
) {
    for i in 0..map.dim {
        for j in 0..map.dim {
            out[(i, j)] = CompiledMap::eval(&map.jacobian[i][j], 0.0, state, params, scratch);
        }
    }
}

fn all_finite(slice: &[f64]) -> bool {
    slice.iter().all(|x| x.is_finite())
}

/// Iterate the map for `steps` steps starting from `ic`. Returns a flat
/// row-major `Vec<f64>` of length `steps * dim`; row i is the state
/// **after** step i+1 (so the first row corresponds to f(ic), matching
/// the existing Python convention).
pub fn iterate(
    map: &CompiledMap,
    ic: &[f64],
    params: &[f64],
    steps: usize,
) -> Result<Vec<f64>, KernelError> {
    let dim = map.dim;
    let mut state = ic.to_vec();
    let mut next = vec![0.0; dim];
    let mut scratch: Vec<f64> = Vec::with_capacity(32);
    let mut out = vec![0.0; steps * dim];

    for step in 0..steps {
        eval_step(map, &state, params, &mut next, &mut scratch);
        if !all_finite(&next) {
            return Err(KernelError::Divergence { step });
        }
        let row = &mut out[step * dim..(step + 1) * dim];
        row.copy_from_slice(&next);
        state.copy_from_slice(&next);
    }
    Ok(out)
}

/// QR-based Lyapunov spectrum.
///
/// Mirrors the Python implementation: at each step (a) evaluate the next
/// state, (b) evaluate the Jacobian at the new state, (c) propagate
/// `Q = J @ Q`, and (d) every `reortho_interval` steps reorthonormalise
/// via QR, accumulating `log|diag(R)|` into the running sum. Final
/// exponents are `sum / (intervals * reortho_interval)`, descending.
pub fn lyapunov_spectrum(
    map: &CompiledMap,
    ic: &[f64],
    params: &[f64],
    steps: usize,
    n_exp: usize,
    reortho_interval: usize,
) -> Result<Vec<f64>, KernelError> {
    let dim = map.dim;
    if n_exp == 0 || n_exp > dim {
        return Err(KernelError::BadShape(format!(
            "n_exp {n_exp} must be in 1..={dim}"
        )));
    }
    let reortho_interval = reortho_interval.max(1);

    let mut state = ic.to_vec();
    let mut next = vec![0.0; dim];
    let mut scratch: Vec<f64> = Vec::with_capacity(32);

    // Tangent bundle as (dim, n_exp) — columns are tangent vectors.
    // Initialise to the first n_exp columns of the identity.
    let mut q = DMatrix::<f64>::zeros(dim, n_exp);
    for k in 0..n_exp {
        q[(k, k)] = 1.0;
    }

    let mut jac = DMatrix::<f64>::zeros(dim, dim);
    let mut lyap_sums = DVector::<f64>::zeros(n_exp);
    let mut intervals: usize = 0;

    for step in 0..steps {
        eval_step(map, &state, params, &mut next, &mut scratch);
        if !all_finite(&next) {
            return Err(KernelError::Divergence { step });
        }
        state.copy_from_slice(&next);

        eval_jacobian(map, &state, params, &mut jac, &mut scratch);
        for c in 0..n_exp {
            for r in 0..dim {
                if !jac[(r, c)].is_finite() {
                    return Err(KernelError::Divergence { step });
                }
            }
        }

        q = &jac * &q;
        if q.iter().any(|v| !v.is_finite()) {
            return Err(KernelError::Divergence { step });
        }

        if (step + 1) % reortho_interval == 0 {
            // Modified Gram-Schmidt on the columns of q.
            // For tall-thin (dim >= n_exp) this is simple and matches
            // numpy.linalg.qr(reduced) up to column-sign convention; we
            // accumulate log|diag(R)|, so signs are irrelevant.
            let tiny = f64::MIN_POSITIVE;
            for k in 0..n_exp {
                // Project out previous columns.
                for j in 0..k {
                    let mut dot = 0.0;
                    for r in 0..dim {
                        dot += q[(r, j)] * q[(r, k)];
                    }
                    for r in 0..dim {
                        q[(r, k)] -= dot * q[(r, j)];
                    }
                }
                let mut norm_sq = 0.0;
                for r in 0..dim {
                    norm_sq += q[(r, k)] * q[(r, k)];
                }
                let norm = norm_sq.sqrt();
                let safe = if norm < tiny { tiny } else { norm };
                lyap_sums[k] += safe.ln();
                let inv = 1.0 / safe;
                for r in 0..dim {
                    q[(r, k)] *= inv;
                }
            }
            intervals += 1;
        }
    }

    if intervals == 0 {
        return Err(KernelError::BadShape(
            "no reorthonormalisation interval completed".into(),
        ));
    }
    let denom = (intervals * reortho_interval) as f64;
    Ok(lyap_sums.iter().map(|s| s / denom).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tsdyn_core::ir::Expr;

    fn henon_compiled() -> CompiledMap {
        // xp = 1 - a*x^2 + y, yp = b*x
        let xp = vec![
            Expr::Const(1.0),
            Expr::Param(0),
            Expr::Var(0),
            Expr::Pow(2),
            Expr::Mul,
            Expr::Sub,
            Expr::Var(1),
            Expr::Add,
        ];
        let yp = vec![Expr::Param(1), Expr::Var(0), Expr::Mul];
        // Jacobian:
        // J = [[-2 a x, 1], [b, 0]]
        let j00 = vec![
            Expr::Const(-2.0),
            Expr::Param(0),
            Expr::Mul,
            Expr::Var(0),
            Expr::Mul,
        ];
        let j01 = vec![Expr::Const(1.0)];
        let j10 = vec![Expr::Param(1)];
        let j11 = vec![Expr::Const(0.0)];
        CompiledMap {
            dim: 2,
            n_params: 2,
            step: vec![xp, yp],
            jacobian: vec![vec![j00, j01], vec![j10, j11]],
        }
    }

    #[test]
    fn henon_iterate_matches_known_value() {
        let map = henon_compiled();
        // From the golden file: Henon at (a=1.4, b=0.3, ic=[0.1, 0.1])
        // After 1 step: xp = 1 - 1.4*0.01 + 0.1 = 1.086; yp = 0.03
        let out = iterate(&map, &[0.1, 0.1], &[1.4, 0.3], 1).unwrap();
        assert!((out[0] - 1.086).abs() < 1e-12);
        assert!((out[1] - 0.03).abs() < 1e-12);
    }

    #[test]
    fn henon_lyapunov_converges() {
        let map = henon_compiled();
        let exps = lyapunov_spectrum(&map, &[0.1, 0.1], &[1.4, 0.3], 10_000, 2, 1).unwrap();
        // Henon canonical exponents: 0.42, -1.62
        assert!((exps[0] - 0.42).abs() < 0.05, "got {}", exps[0]);
        assert!((exps[1] - (-1.62)).abs() < 0.05, "got {}", exps[1]);
    }
}
