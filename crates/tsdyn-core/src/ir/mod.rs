//! Intermediate representation for compiled symbolic expressions.
//!
//! The `Expr` enum and bytecode format are stable interfaces — Python emits
//! the bytecode, Rust decodes once per system, and every kernel evaluates
//! against the same op set.
//!
//! Booleans live in the same f64 stack as numeric values: comparisons
//! push `1.0` for true / `0.0` for false. `Where` selects on
//! `cond != 0.0`. `And` is `min(a, b)` of two booleans (clean for both
//! the IR walker and the cranelift JIT we'll add in N4).
//!
//! # Module layout
//!
//! - [`mod.rs`] (this file) — shared: `Expr`, opcodes, bytecode reader/decoder.
//! - [`map`] — `CompiledMap` for discrete-map kernels.
//! - [`ode`] — `CompiledOde` for ODE integration.
//! - (N5 will add `dde` alongside `ode`.)

pub mod map;
pub mod ode;

pub use map::CompiledMap;
pub use ode::CompiledOde;

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
    /// N2: push the per-call simulation time `t`. Maps evaluate with
    /// `t = 0.0`; ODE RHS evaluation passes the real time.
    Time,
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Pow(i32),
    Mod,
    /// N2: general two-operand power. Pops `exp` then `base`, pushes
    /// `base.powf(exp)`. Used when the exponent is itself an
    /// expression (e.g. CircadianRhythm's `x ** n`, YuWang's
    /// `x ** (y * z)`). Integer-exponent powers continue to use
    /// `Pow(i32)` for the cheaper `powi` codegen.
    Pow2,
    Sin,
    Cos,
    Exp,
    Log,
    Abs,
    Sqrt,
    Arccos,
    Sign,
    /// N2: hyperbolic tangent — used by coupled / climate / physical systems.
    Tanh,
    /// N2: hyperbolic sine.
    Sinh,
    /// N2: hyperbolic cosine.
    Cosh,
    /// N2: fractional / non-integer power, evaluated as `base.powf(exp)`.
    PowF(f64),
    Where,
    Lt,
    Le,
    Gt,
    Ge,
    And,
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
pub(super) mod opcodes {
    pub const CONST: u8 = 0x00;
    pub const VAR: u8 = 0x01;
    pub const PARAM: u8 = 0x02;
    pub const TIME: u8 = 0x03;
    pub const ADD: u8 = 0x10;
    pub const SUB: u8 = 0x11;
    pub const MUL: u8 = 0x12;
    pub const DIV: u8 = 0x13;
    pub const NEG: u8 = 0x14;
    pub const POW: u8 = 0x15;
    pub const MOD: u8 = 0x16;
    pub const POW2: u8 = 0x17;
    pub const SIN: u8 = 0x20;
    pub const COS: u8 = 0x21;
    pub const EXP: u8 = 0x22;
    pub const LOG: u8 = 0x23;
    pub const ABS: u8 = 0x24;
    pub const SQRT: u8 = 0x25;
    pub const ARCCOS: u8 = 0x26;
    pub const SIGN: u8 = 0x27;
    pub const TANH: u8 = 0x28;
    pub const SINH: u8 = 0x29;
    pub const COSH: u8 = 0x2A;
    pub const WHERE: u8 = 0x30;
    pub const LT: u8 = 0x31;
    pub const LE: u8 = 0x32;
    pub const GT: u8 = 0x33;
    pub const GE: u8 = 0x34;
    pub const AND: u8 = 0x35;
    pub const POWF: u8 = 0x40;
}

pub(super) struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub(super) fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    pub(super) fn read_u8(&mut self) -> Result<u8, DecodeError> {
        let b = *self.buf.get(self.pos).ok_or(DecodeError::UnexpectedEof)?;
        self.pos += 1;
        Ok(b)
    }

    pub(super) fn read_u32(&mut self) -> Result<u32, DecodeError> {
        let end = self.pos + 4;
        let slice = self
            .buf
            .get(self.pos..end)
            .ok_or(DecodeError::UnexpectedEof)?;
        self.pos = end;
        Ok(u32::from_le_bytes(slice.try_into().unwrap()))
    }

    pub(super) fn read_i32(&mut self) -> Result<i32, DecodeError> {
        let end = self.pos + 4;
        let slice = self
            .buf
            .get(self.pos..end)
            .ok_or(DecodeError::UnexpectedEof)?;
        self.pos = end;
        Ok(i32::from_le_bytes(slice.try_into().unwrap()))
    }

    pub(super) fn read_f64(&mut self) -> Result<f64, DecodeError> {
        let end = self.pos + 8;
        let slice = self
            .buf
            .get(self.pos..end)
            .ok_or(DecodeError::UnexpectedEof)?;
        self.pos = end;
        Ok(f64::from_le_bytes(slice.try_into().unwrap()))
    }

    pub(super) fn done(&self) -> bool {
        self.pos == self.buf.len()
    }
}

pub(super) fn decode_program(reader: &mut Reader<'_>) -> Result<Vec<Expr>, DecodeError> {
    use opcodes::*;
    let n_ops = reader.read_u32()? as usize;
    let mut out = Vec::with_capacity(n_ops);
    for _ in 0..n_ops {
        let op = reader.read_u8()?;
        let expr = match op {
            CONST => Expr::Const(reader.read_f64()?),
            VAR => Expr::Var(reader.read_u32()?),
            PARAM => Expr::Param(reader.read_u32()?),
            TIME => Expr::Time,
            ADD => Expr::Add,
            SUB => Expr::Sub,
            MUL => Expr::Mul,
            DIV => Expr::Div,
            NEG => Expr::Neg,
            POW => Expr::Pow(reader.read_i32()?),
            MOD => Expr::Mod,
            POW2 => Expr::Pow2,
            SIN => Expr::Sin,
            COS => Expr::Cos,
            EXP => Expr::Exp,
            LOG => Expr::Log,
            ABS => Expr::Abs,
            SQRT => Expr::Sqrt,
            ARCCOS => Expr::Arccos,
            SIGN => Expr::Sign,
            TANH => Expr::Tanh,
            SINH => Expr::Sinh,
            COSH => Expr::Cosh,
            POWF => Expr::PowF(reader.read_f64()?),
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
