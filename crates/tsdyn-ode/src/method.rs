//! Catalogue of integrators accepted by [`crate::integrate_ode`] (string-sync with Python).

use std::str::FromStr;

use crate::error::IntegrateError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Method {
    Dp5,
    Dp8,
    Tsit5,
    Vern9,
    Bs3,
    Rk4,
    Rosenbrock23,
    Rosenbrock34,
    Rodas4,
}

impl FromStr for Method {
    type Err = IntegrateError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_uppercase().as_str() {
            "DP5" => Ok(Self::Dp5),
            "DP8" => Ok(Self::Dp8),
            "TSIT5" => Ok(Self::Tsit5),
            "VERN9" => Ok(Self::Vern9),
            "BS3" => Ok(Self::Bs3),
            "RK4" => Ok(Self::Rk4),
            "ROSENBROCK23" => Ok(Self::Rosenbrock23),
            "ROSENBROCK34" => Ok(Self::Rosenbrock34),
            "RODAS4" => Ok(Self::Rodas4),
            _ => Err(IntegrateError::BadMethod(s.to_string())),
        }
    }
}
