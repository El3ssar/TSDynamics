//! Compiled ODE IR: RHS + optional Jacobian.
//!
//! Six built-in systems (MultiChua, AnishchenkoAstakhov, StickSlipOscillator,
//! CellularNeuralNetwork, Colpitts, FluidTrampoline) use `Abs` / `sign`
//! without an explicit `_jacobian`, and SymEngine returns unevaluated
//! `Derivative` nodes for those — for them the Python lowerer sets
//! `has_jacobian = false` so we never see a malformed Jacobian here.
//!
//! N5 will add `CompiledDde` beside this file.

use super::{decode_program, DecodeError, Expr, Reader};
use crate::ir::map::CompiledMap;

/// One compiled ODE: the RHS (`dim` postfix programs) plus an optional
/// Jacobian (`dim × dim` postfix programs).
#[derive(Debug, Clone)]
pub struct CompiledOde {
    pub dim: usize,
    pub n_params: usize,
    pub rhs: Vec<Vec<Expr>>,
    pub jacobian: Option<Vec<Vec<Vec<Expr>>>>,
}

impl CompiledOde {
    /// Decode the wire format produced by `tsdynamics.base._ir.serialize_ode`.
    ///
    /// Layout (all little-endian):
    /// ```text
    /// u32 dim
    /// u32 n_params
    /// u32 n_rhs  (== dim)
    /// for each rhs expr: u32 n_ops + ops
    /// u8  has_jacobian  (0 or 1)
    /// if has_jacobian:
    ///     u32 n_jac_rows  (== dim)
    ///     for each row: u32 n_cols (== dim)
    ///         for each cell: u32 n_ops + ops
    /// ```
    pub fn from_bytes(buf: &[u8]) -> Result<Self, DecodeError> {
        let mut r = Reader::new(buf);
        let dim = r.read_u32()? as usize;
        let n_params = r.read_u32()? as usize;

        let n_rhs = r.read_u32()? as usize;
        if n_rhs != dim {
            return Err(DecodeError::BadShape);
        }
        let mut rhs = Vec::with_capacity(dim);
        for _ in 0..dim {
            rhs.push(decode_program(&mut r)?);
        }

        let has_jac = r.read_u8()? != 0;
        let jacobian = if has_jac {
            let n_jac_rows = r.read_u32()? as usize;
            if n_jac_rows != dim {
                return Err(DecodeError::BadShape);
            }
            let mut jac = Vec::with_capacity(dim);
            for _ in 0..dim {
                let n_cols = r.read_u32()? as usize;
                if n_cols != dim {
                    return Err(DecodeError::BadShape);
                }
                let mut row = Vec::with_capacity(dim);
                for _ in 0..dim {
                    row.push(decode_program(&mut r)?);
                }
                jac.push(row);
            }
            Some(jac)
        } else {
            None
        };

        if !r.done() {
            return Err(DecodeError::BadShape);
        }
        Ok(Self {
            dim,
            n_params,
            rhs,
            jacobian,
        })
    }

    /// Evaluate the RHS at `(t, state, params)`, writing each component
    /// into `out_dy`. `scratch` is the per-call f64 stack — caller-owned
    /// so the stepper loop can amortise allocation across thousands of
    /// RHS calls.
    pub fn eval_rhs(
        &self,
        t: f64,
        state: &[f64],
        params: &[f64],
        out_dy: &mut [f64],
        scratch: &mut Vec<f64>,
    ) {
        assert_eq!(state.len(), self.dim, "state.len() != dim");
        assert_eq!(params.len(), self.n_params, "params.len() != n_params");
        assert_eq!(out_dy.len(), self.dim, "out_dy.len() != dim");
        for (i, program) in self.rhs.iter().enumerate() {
            out_dy[i] = CompiledMap::eval(program, t, state, params, scratch);
        }
    }

    /// Evaluate the Jacobian \(J_{ij} = \partial f_i / \partial y_j\) at `(t, state)`,
    /// writing **row-major** `dim × dim` entries into `out_jac`.
    ///
    /// Requires [`CompiledOde::jacobian`](Self::jacobian); panics in debug builds if absent.
    pub fn eval_jacobian(
        &self,
        t: f64,
        state: &[f64],
        params: &[f64],
        out_jac: &mut [f64],
        scratch: &mut Vec<f64>,
    ) {
        let jac = self
            .jacobian
            .as_ref()
            .expect("eval_jacobian requires has_jacobian bytecode");
        assert_eq!(state.len(), self.dim, "state.len() != dim");
        assert_eq!(params.len(), self.n_params, "params.len() != n_params");
        assert_eq!(out_jac.len(), self.dim * self.dim, "out_jac.len() != dim^2");
        let dim = self.dim;
        for i in 0..dim {
            for j in 0..dim {
                out_jac[i * dim + j] = CompiledMap::eval(&jac[i][j], t, state, params, scratch);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::opcodes;

    #[test]
    fn eval_compiled_ode_roundtrip() {
        // RHS: dx/dt = -a * x + sin(x);  dim=1, params=[a]
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&1u32.to_le_bytes()); // dim
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_params
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_rhs
        buf.extend_from_slice(&7u32.to_le_bytes()); // 7 ops
        buf.push(opcodes::PARAM);
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.push(opcodes::VAR);
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.push(opcodes::MUL);
        buf.push(opcodes::NEG);
        buf.push(opcodes::VAR);
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.push(opcodes::SIN);
        buf.push(opcodes::ADD);
        buf.push(0); // has_jacobian = false

        let ode = CompiledOde::from_bytes(&buf).unwrap();
        assert_eq!(ode.dim, 1);
        assert_eq!(ode.n_params, 1);
        assert!(ode.jacobian.is_none());

        let mut out = [0.0; 1];
        let mut scratch = Vec::new();
        ode.eval_rhs(0.0, &[0.3], &[2.0], &mut out, &mut scratch);
        let expected = -2.0 * 0.3 + (0.3_f64).sin();
        assert!((out[0] - expected).abs() < 1e-14);
    }
}
