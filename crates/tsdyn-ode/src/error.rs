//! Errors returned by the ODE driver.

use std::fmt;

#[derive(Debug, Clone)]
pub enum IntegrateError {
    BadBytecode(String),
    BadMethod(String),
    Diverged { t: f64 },
    ParamsLen { expected: usize, got: usize },
    MissingJacobian { method: String },
    /// Rosenbrock / Rodas methods need an analytic augmented Jacobian — use an explicit method.
    VariationalNeedsExplicit(String),
    LyapunovConfig(String),
}

impl fmt::Display for IntegrateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadBytecode(s) => write!(f, "bad ODE bytecode: {s}"),
            Self::BadMethod(s) => write!(f, "unknown ODE method: {s}"),
            Self::Diverged { t } => write!(f, "non-finite state at t={t}"),
            Self::ParamsLen { expected, got } => {
                write!(f, "params length mismatch: expected {expected}, got {got}")
            }
            Self::MissingJacobian { method } => write!(
                f,
                "method {method} requires a Jacobian in the IR bytecode (has_jacobian=false)"
            ),
            Self::VariationalNeedsExplicit(m) => write!(
                f,
                "variational Lyapunov integration does not support stiff method {m}; use an explicit RK method (DP5, DP8, TSIT5, VERN6–VERN9, BS3, RK4)"
            ),
            Self::LyapunovConfig(s) => write!(f, "Lyapunov configuration error: {s}"),
        }
    }
}

impl std::error::Error for IntegrateError {}
