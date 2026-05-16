//! Errors returned by the ODE driver.

use std::fmt;

#[derive(Debug, Clone)]
pub enum IntegrateError {
    BadBytecode(String),
    BadMethod(String),
    Diverged { t: f64 },
    ParamsLen { expected: usize, got: usize },
    MissingJacobian { method: String },
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
        }
    }
}

impl std::error::Error for IntegrateError {}
