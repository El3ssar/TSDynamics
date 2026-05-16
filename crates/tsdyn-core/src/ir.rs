//! Intermediate representation for compiled symbolic expressions.
//!
//! N1 lowers `DiscreteMap._step` / `_jacobian` from Python into a
//! stack-machine bytecode that this module decodes into [`CompiledMap`].
//! The op set is sized for the 26 built-in maps (see the N1 milestone
//! doc); ODE / DDE milestones extend it but never break it.
//!
//! Booleans live in the same f64 stack as numeric values: comparisons
//! push `1.0` for true / `0.0` for false. `Where` selects on
//! `cond != 0.0`. `And` is `min(a, b)` of two booleans (clean for both
//! the IR walker and the cranelift JIT we'll add in N4).

use std::fmt;

/// A single IR opcode.
///
/// `Pow` carries an `i32` exponent — `f**non_int` cannot reach the IR
/// (the Python tracer rejects it). Numeric immediates are inlined for
/// `Const`; `Var` / `Param` carry an index into the state / params
/// slice.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Const(f64),
    Var(u32),
    Param(u32),
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Pow(i32),
    Mod,
    Sin,
    Cos,
    Exp,
    Log,
    Abs,
    Sqrt,
    Arccos,
    Sign,
    Where,
    Lt,
    Le,
    Gt,
    Ge,
    And,
}

/// One compiled map: the step function (dim postfix programs) plus the
/// Jacobian (`dim × dim` postfix programs). Held entirely on the Rust
/// side; Python passes the bytecode and we decode once per
/// `_compile_ir` cache miss.
#[derive(Debug, Clone)]
pub struct CompiledMap {
    pub dim: usize,
    pub n_params: usize,
    pub step: Vec<Vec<Expr>>,
    pub jacobian: Vec<Vec<Vec<Expr>>>,
}

#[derive(Debug)]
pub enum DecodeError {
    UnexpectedEof,
    UnknownOpcode(u8),
    BadShape,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::UnexpectedEof => write!(f, "unexpected end of bytecode"),
            DecodeError::UnknownOpcode(b) => write!(f, "unknown opcode 0x{b:02x}"),
            DecodeError::BadShape => write!(f, "shape mismatch in encoded program"),
        }
    }
}

impl std::error::Error for DecodeError {}

/// Opcode byte values. The Python side must use the same constants.
mod opcodes {
    pub const CONST: u8 = 0x00;
    pub const VAR: u8 = 0x01;
    pub const PARAM: u8 = 0x02;
    pub const ADD: u8 = 0x10;
    pub const SUB: u8 = 0x11;
    pub const MUL: u8 = 0x12;
    pub const DIV: u8 = 0x13;
    pub const NEG: u8 = 0x14;
    pub const POW: u8 = 0x15;
    pub const MOD: u8 = 0x16;
    pub const SIN: u8 = 0x20;
    pub const COS: u8 = 0x21;
    pub const EXP: u8 = 0x22;
    pub const LOG: u8 = 0x23;
    pub const ABS: u8 = 0x24;
    pub const SQRT: u8 = 0x25;
    pub const ARCCOS: u8 = 0x26;
    pub const SIGN: u8 = 0x27;
    pub const WHERE: u8 = 0x30;
    pub const LT: u8 = 0x31;
    pub const LE: u8 = 0x32;
    pub const GT: u8 = 0x33;
    pub const GE: u8 = 0x34;
    pub const AND: u8 = 0x35;
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn read_u8(&mut self) -> Result<u8, DecodeError> {
        let b = *self.buf.get(self.pos).ok_or(DecodeError::UnexpectedEof)?;
        self.pos += 1;
        Ok(b)
    }

    fn read_u32(&mut self) -> Result<u32, DecodeError> {
        let end = self.pos + 4;
        let slice = self.buf.get(self.pos..end).ok_or(DecodeError::UnexpectedEof)?;
        self.pos = end;
        Ok(u32::from_le_bytes(slice.try_into().unwrap()))
    }

    fn read_i32(&mut self) -> Result<i32, DecodeError> {
        let end = self.pos + 4;
        let slice = self.buf.get(self.pos..end).ok_or(DecodeError::UnexpectedEof)?;
        self.pos = end;
        Ok(i32::from_le_bytes(slice.try_into().unwrap()))
    }

    fn read_f64(&mut self) -> Result<f64, DecodeError> {
        let end = self.pos + 8;
        let slice = self.buf.get(self.pos..end).ok_or(DecodeError::UnexpectedEof)?;
        self.pos = end;
        Ok(f64::from_le_bytes(slice.try_into().unwrap()))
    }

    fn done(&self) -> bool {
        self.pos == self.buf.len()
    }
}

fn decode_program(reader: &mut Reader<'_>) -> Result<Vec<Expr>, DecodeError> {
    use opcodes::*;
    let n_ops = reader.read_u32()? as usize;
    let mut out = Vec::with_capacity(n_ops);
    for _ in 0..n_ops {
        let op = reader.read_u8()?;
        let expr = match op {
            CONST => Expr::Const(reader.read_f64()?),
            VAR => Expr::Var(reader.read_u32()?),
            PARAM => Expr::Param(reader.read_u32()?),
            ADD => Expr::Add,
            SUB => Expr::Sub,
            MUL => Expr::Mul,
            DIV => Expr::Div,
            NEG => Expr::Neg,
            POW => Expr::Pow(reader.read_i32()?),
            MOD => Expr::Mod,
            SIN => Expr::Sin,
            COS => Expr::Cos,
            EXP => Expr::Exp,
            LOG => Expr::Log,
            ABS => Expr::Abs,
            SQRT => Expr::Sqrt,
            ARCCOS => Expr::Arccos,
            SIGN => Expr::Sign,
            WHERE => Expr::Where,
            LT => Expr::Lt,
            LE => Expr::Le,
            GT => Expr::Gt,
            GE => Expr::Ge,
            AND => Expr::And,
            other => return Err(DecodeError::UnknownOpcode(other)),
        };
        out.push(expr);
    }
    Ok(out)
}

impl CompiledMap {
    /// Decode the wire format produced by `tsdynamics.base._ir.serialize`.
    ///
    /// Layout (all little-endian):
    /// ```text
    /// u32 dim
    /// u32 n_params
    /// u32 n_step  (== dim)
    /// for each step expr: u32 n_ops + ops
    /// u32 n_jac_rows  (== dim)
    /// for each row: u32 n_cols (== dim)
    ///   for each cell: u32 n_ops + ops
    /// ```
    pub fn from_bytes(buf: &[u8]) -> Result<Self, DecodeError> {
        let mut r = Reader::new(buf);
        let dim = r.read_u32()? as usize;
        let n_params = r.read_u32()? as usize;

        let n_step = r.read_u32()? as usize;
        if n_step != dim {
            return Err(DecodeError::BadShape);
        }
        let mut step = Vec::with_capacity(dim);
        for _ in 0..dim {
            step.push(decode_program(&mut r)?);
        }

        let n_jac_rows = r.read_u32()? as usize;
        if n_jac_rows != dim {
            return Err(DecodeError::BadShape);
        }
        let mut jacobian = Vec::with_capacity(dim);
        for _ in 0..dim {
            let n_cols = r.read_u32()? as usize;
            if n_cols != dim {
                return Err(DecodeError::BadShape);
            }
            let mut row = Vec::with_capacity(dim);
            for _ in 0..dim {
                row.push(decode_program(&mut r)?);
            }
            jacobian.push(row);
        }

        if !r.done() {
            return Err(DecodeError::BadShape);
        }
        Ok(Self { dim, n_params, step, jacobian })
    }

    /// Evaluate one postfix program against (`state`, `params`).
    ///
    /// `scratch` is the stack; it's caller-owned so the iterate / Lyapunov
    /// loops can amortise allocation. The scratch is truncated on entry.
    pub fn eval(
        program: &[Expr],
        state: &[f64],
        params: &[f64],
        scratch: &mut Vec<f64>,
    ) -> f64 {
        scratch.clear();
        for op in program {
            match op {
                Expr::Const(c) => scratch.push(*c),
                Expr::Var(i) => scratch.push(state[*i as usize]),
                Expr::Param(i) => scratch.push(params[*i as usize]),
                Expr::Add => {
                    let b = scratch.pop().unwrap();
                    let a = scratch.pop().unwrap();
                    scratch.push(a + b);
                }
                Expr::Sub => {
                    let b = scratch.pop().unwrap();
                    let a = scratch.pop().unwrap();
                    scratch.push(a - b);
                }
                Expr::Mul => {
                    let b = scratch.pop().unwrap();
                    let a = scratch.pop().unwrap();
                    scratch.push(a * b);
                }
                Expr::Div => {
                    let b = scratch.pop().unwrap();
                    let a = scratch.pop().unwrap();
                    scratch.push(a / b);
                }
                Expr::Neg => {
                    let a = scratch.pop().unwrap();
                    scratch.push(-a);
                }
                Expr::Pow(k) => {
                    let a = scratch.pop().unwrap();
                    scratch.push(a.powi(*k));
                }
                Expr::Mod => {
                    let b = scratch.pop().unwrap();
                    let a = scratch.pop().unwrap();
                    // Match Python's % (floor-modulo), not fmod.
                    scratch.push(a - (a / b).floor() * b);
                }
                Expr::Sin => {
                    let a = scratch.pop().unwrap();
                    scratch.push(a.sin());
                }
                Expr::Cos => {
                    let a = scratch.pop().unwrap();
                    scratch.push(a.cos());
                }
                Expr::Exp => {
                    let a = scratch.pop().unwrap();
                    scratch.push(a.exp());
                }
                Expr::Log => {
                    let a = scratch.pop().unwrap();
                    scratch.push(a.ln());
                }
                Expr::Abs => {
                    let a = scratch.pop().unwrap();
                    scratch.push(a.abs());
                }
                Expr::Sqrt => {
                    let a = scratch.pop().unwrap();
                    scratch.push(a.sqrt());
                }
                Expr::Arccos => {
                    let a = scratch.pop().unwrap();
                    scratch.push(a.acos());
                }
                Expr::Sign => {
                    let a = scratch.pop().unwrap();
                    scratch.push(if a > 0.0 {
                        1.0
                    } else if a < 0.0 {
                        -1.0
                    } else {
                        0.0
                    });
                }
                Expr::Where => {
                    let f = scratch.pop().unwrap();
                    let t = scratch.pop().unwrap();
                    let c = scratch.pop().unwrap();
                    scratch.push(if c != 0.0 { t } else { f });
                }
                Expr::Lt => {
                    let b = scratch.pop().unwrap();
                    let a = scratch.pop().unwrap();
                    scratch.push(if a < b { 1.0 } else { 0.0 });
                }
                Expr::Le => {
                    let b = scratch.pop().unwrap();
                    let a = scratch.pop().unwrap();
                    scratch.push(if a <= b { 1.0 } else { 0.0 });
                }
                Expr::Gt => {
                    let b = scratch.pop().unwrap();
                    let a = scratch.pop().unwrap();
                    scratch.push(if a > b { 1.0 } else { 0.0 });
                }
                Expr::Ge => {
                    let b = scratch.pop().unwrap();
                    let a = scratch.pop().unwrap();
                    scratch.push(if a >= b { 1.0 } else { 0.0 });
                }
                Expr::And => {
                    let b = scratch.pop().unwrap();
                    let a = scratch.pop().unwrap();
                    scratch.push(if a != 0.0 && b != 0.0 { 1.0 } else { 0.0 });
                }
            }
        }
        scratch.pop().expect("empty program stack on exit")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_simple_program() {
        // 3 + 4 * 2 = 11
        let program = vec![
            Expr::Const(3.0),
            Expr::Const(4.0),
            Expr::Const(2.0),
            Expr::Mul,
            Expr::Add,
        ];
        let mut scratch = Vec::new();
        let v = CompiledMap::eval(&program, &[], &[], &mut scratch);
        assert_eq!(v, 11.0);
    }

    #[test]
    fn eval_henon_step() {
        // Henon: xp = 1 - a*x^2 + y, yp = b*x
        // Encode xp(x=0.1, y=0.1, a=1.4, b=0.3):
        // 1 - 1.4 * 0.01 + 0.1 = 1.086
        let program = vec![
            Expr::Const(1.0),
            Expr::Param(0),     // a
            Expr::Var(0),       // x
            Expr::Pow(2),
            Expr::Mul,
            Expr::Sub,          // 1 - a*x^2
            Expr::Var(1),       // y
            Expr::Add,
        ];
        let mut scratch = Vec::new();
        let v = CompiledMap::eval(&program, &[0.1, 0.1], &[1.4, 0.3], &mut scratch);
        assert!((v - 1.086).abs() < 1e-12);
    }
}
