//! [`TapeBuilder`] — construct a [`Tape`] instruction-by-instruction with the
//! single-static-assignment invariants enforced by construction.
//!
//! This is the Rust counterpart of the Python tape emitter: each `push`
//! appends one instruction and returns the [`Reg`] it writes, which later
//! instructions consume.  Because instructions are emitted in order, every
//! returned register necessarily refers to an earlier instruction, so the
//! forward-reference invariant holds automatically; [`finish`](TapeBuilder::finish)
//! runs full validation for the index/shape invariants.
//!
//! The builder does **not** deduplicate common subexpressions — the symbolic
//! lowering in the Python layer already CSE's before emitting.  It exists for
//! hand-written tapes (tests, fixtures, and any in-Rust lowering) where a typed
//! API beats juggling raw parallel arrays.

use crate::op::Op;
use crate::tape::{IrError, Tape};

/// A register: the index an instruction writes (equal to its position on the
/// tape).  A newtype so a register reference can't be confused with a leaf's
/// input index (`State`/`Param` take a plain `usize`, not a `Reg`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Reg(pub u32);

impl Reg {
    #[inline]
    fn as_i32(self) -> i32 {
        self.0 as i32
    }
}

/// Builds a [`Tape`].  See the module docs.
#[derive(Clone, Debug, Default)]
pub struct TapeBuilder {
    ops: Vec<Op>,
    a: Vec<i32>,
    b: Vec<i32>,
    imm: Vec<f64>,
}

impl TapeBuilder {
    /// A fresh, empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of instructions emitted so far.
    #[inline]
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Whether no instruction has been emitted yet.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    #[inline]
    fn push(&mut self, op: Op, a: i32, b: i32, imm: f64) -> Reg {
        let idx = self.ops.len() as u32;
        self.ops.push(op);
        self.a.push(a);
        self.b.push(b);
        self.imm.push(imm);
        Reg(idx)
    }

    // ---- leaves ----

    /// Emit a constant immediate.
    pub fn constant(&mut self, value: f64) -> Reg {
        self.push(Op::Const, 0, 0, value)
    }

    /// Emit a read of state component `u[index]`.
    pub fn state(&mut self, index: usize) -> Reg {
        self.push(Op::State, index as i32, 0, 0.0)
    }

    /// Emit a read of parameter `p[index]`.
    pub fn param(&mut self, index: usize) -> Reg {
        self.push(Op::Param, index as i32, 0, 0.0)
    }

    /// Emit a read of the independent variable `t`.
    pub fn time(&mut self) -> Reg {
        self.push(Op::Time, 0, 0, 0.0)
    }

    // ---- binary ----

    /// `x + y`.
    pub fn add(&mut self, x: Reg, y: Reg) -> Reg {
        self.push(Op::Add, x.as_i32(), y.as_i32(), 0.0)
    }

    /// `x - y`.
    pub fn sub(&mut self, x: Reg, y: Reg) -> Reg {
        self.push(Op::Sub, x.as_i32(), y.as_i32(), 0.0)
    }

    /// `x * y`.
    pub fn mul(&mut self, x: Reg, y: Reg) -> Reg {
        self.push(Op::Mul, x.as_i32(), y.as_i32(), 0.0)
    }

    /// `x / y`.
    pub fn div(&mut self, x: Reg, y: Reg) -> Reg {
        self.push(Op::Div, x.as_i32(), y.as_i32(), 0.0)
    }

    /// `x.powf(y)` — register base, register exponent.
    pub fn pow(&mut self, x: Reg, y: Reg) -> Reg {
        self.push(Op::Pow, x.as_i32(), y.as_i32(), 0.0)
    }

    /// `x.powi(exp)` — register base, integer exponent stored inline.
    pub fn powi(&mut self, x: Reg, exp: i32) -> Reg {
        self.push(Op::Powi, x.as_i32(), exp, 0.0)
    }

    // ---- unary ----

    /// `-x`.
    pub fn neg(&mut self, x: Reg) -> Reg {
        self.push(Op::Neg, x.as_i32(), 0, 0.0)
    }

    /// `1 / x`.
    pub fn recip(&mut self, x: Reg) -> Reg {
        self.push(Op::Recip, x.as_i32(), 0, 0.0)
    }

    /// Apply any unary opcode (the elementary functions, `Neg`, `Recip`) to `x`.
    ///
    /// Panics if `op` is not a unary op (a programming error — use the binary or
    /// leaf constructors for those).  Convenience wrappers ([`sin`](Self::sin)
    /// etc.) cover the common functions.
    pub fn unary(&mut self, op: Op, x: Reg) -> Reg {
        assert!(
            matches!(op.kind(), crate::op::OpKind::Unary),
            "TapeBuilder::unary called with non-unary op {:?}",
            op
        );
        self.push(op, x.as_i32(), 0, 0.0)
    }

    // ---- elementary function shorthands (`builder.sin(x)` etc.) ----

    /// `x.sin()`.
    pub fn sin(&mut self, x: Reg) -> Reg {
        self.unary(Op::Sin, x)
    }
    /// `x.cos()`.
    pub fn cos(&mut self, x: Reg) -> Reg {
        self.unary(Op::Cos, x)
    }
    /// `x.tan()`.
    pub fn tan(&mut self, x: Reg) -> Reg {
        self.unary(Op::Tan, x)
    }
    /// `x.exp()`.
    pub fn exp(&mut self, x: Reg) -> Reg {
        self.unary(Op::Exp, x)
    }
    /// `x.ln()` — natural logarithm.
    pub fn ln(&mut self, x: Reg) -> Reg {
        self.unary(Op::Log, x)
    }
    /// `x.sqrt()`.
    pub fn sqrt(&mut self, x: Reg) -> Reg {
        self.unary(Op::Sqrt, x)
    }
    /// `x.abs()`.
    pub fn abs(&mut self, x: Reg) -> Reg {
        self.unary(Op::Abs, x)
    }
    /// `sign(x)`, with `sign(0) = 0`.
    pub fn sign(&mut self, x: Reg) -> Reg {
        self.unary(Op::Sign, x)
    }
    /// `x.sinh()`.
    pub fn sinh(&mut self, x: Reg) -> Reg {
        self.unary(Op::Sinh, x)
    }
    /// `x.cosh()`.
    pub fn cosh(&mut self, x: Reg) -> Reg {
        self.unary(Op::Cosh, x)
    }
    /// `x.tanh()`.
    pub fn tanh(&mut self, x: Reg) -> Reg {
        self.unary(Op::Tanh, x)
    }
    /// `x.asin()`.
    pub fn asin(&mut self, x: Reg) -> Reg {
        self.unary(Op::Asin, x)
    }
    /// `x.acos()`.
    pub fn acos(&mut self, x: Reg) -> Reg {
        self.unary(Op::Acos, x)
    }
    /// `x.atan()`.
    pub fn atan(&mut self, x: Reg) -> Reg {
        self.unary(Op::Atan, x)
    }
    /// `x.asinh()`.
    pub fn asinh(&mut self, x: Reg) -> Reg {
        self.unary(Op::Asinh, x)
    }
    /// `x.acosh()`.
    pub fn acosh(&mut self, x: Reg) -> Reg {
        self.unary(Op::Acosh, x)
    }
    /// `x.atanh()`.
    pub fn atanh(&mut self, x: Reg) -> Reg {
        self.unary(Op::Atanh, x)
    }

    // ---- non-smooth / piecewise block (stream E-OPS) ----

    /// Apply any binary opcode (the arithmetic binaries, the comparisons,
    /// `Min`/`Max`, `Mod`/`Rem`) to `x` and `y`.
    ///
    /// Panics if `op` is not a binary op (a programming error — use the unary,
    /// leaf or `powi` constructors for those).
    pub fn binary(&mut self, op: Op, x: Reg, y: Reg) -> Reg {
        assert!(
            matches!(op.kind(), crate::op::OpKind::Binary),
            "TapeBuilder::binary called with non-binary op {:?}",
            op
        );
        self.push(op, x.as_i32(), y.as_i32(), 0.0)
    }

    /// `(x < y) as f64` (1.0 / 0.0).
    pub fn lt(&mut self, x: Reg, y: Reg) -> Reg {
        self.binary(Op::Lt, x, y)
    }
    /// `(x <= y) as f64`.
    pub fn le(&mut self, x: Reg, y: Reg) -> Reg {
        self.binary(Op::Le, x, y)
    }
    /// `(x > y) as f64`.
    pub fn gt(&mut self, x: Reg, y: Reg) -> Reg {
        self.binary(Op::Gt, x, y)
    }
    /// `(x >= y) as f64`.
    pub fn ge(&mut self, x: Reg, y: Reg) -> Reg {
        self.binary(Op::Ge, x, y)
    }
    /// `(x == y) as f64`.
    pub fn eq(&mut self, x: Reg, y: Reg) -> Reg {
        self.binary(Op::Eq, x, y)
    }
    /// `(x != y) as f64`.
    pub fn ne(&mut self, x: Reg, y: Reg) -> Reg {
        self.binary(Op::Ne, x, y)
    }
    /// `x.min(y)` (`f64::min`).
    pub fn min(&mut self, x: Reg, y: Reg) -> Reg {
        self.binary(Op::Min, x, y)
    }
    /// `x.max(y)` (`f64::max`).
    pub fn max(&mut self, x: Reg, y: Reg) -> Reg {
        self.binary(Op::Max, x, y)
    }
    /// `x.floor()`.
    pub fn floor(&mut self, x: Reg) -> Reg {
        self.unary(Op::Floor, x)
    }
    /// `x.ceil()`.
    pub fn ceil(&mut self, x: Reg) -> Reg {
        self.unary(Op::Ceil, x)
    }
    /// Floored modulo `x - y * (x / y).floor()`.
    pub fn modulo(&mut self, x: Reg, y: Reg) -> Reg {
        self.binary(Op::Mod, x, y)
    }
    /// Truncated remainder `x % y` (C `fmod`).
    pub fn rem(&mut self, x: Reg, y: Reg) -> Reg {
        self.binary(Op::Rem, x, y)
    }

    /// Finish into a validated [`Tape`].
    ///
    /// `outputs` are the registers holding each derivative component;
    /// `jac_outputs` is either empty or the row-major `dim × dim` Jacobian
    /// registers (`dim = outputs.len()`).  `n_state`/`n_param` are the declared
    /// input widths used to bound leaf indices.
    pub fn finish(
        self,
        outputs: &[Reg],
        jac_outputs: &[Reg],
        n_state: usize,
        n_param: usize,
    ) -> Result<Tape, IrError> {
        Tape::from_parts(
            self.ops,
            self.a,
            self.b,
            self.imm,
            outputs.iter().map(|r| r.as_i32()).collect(),
            jac_outputs.iter().map(|r| r.as_i32()).collect(),
            n_state,
            n_param,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_lorenz_dx_component() {
        // dx/dt = sigma * (y - x), with sigma = p0, x = u0, y = u1.
        let mut b = TapeBuilder::new();
        let sigma = b.param(0);
        let x = b.state(0);
        let y = b.state(1);
        let ymx = b.sub(y, x);
        let dx = b.mul(sigma, ymx);
        let tape = b.finish(&[dx], &[], 2, 1).unwrap();
        assert_eq!(tape.n_reg(), 5);
        assert_eq!(tape.dim(), 1);
        assert_eq!(tape.outputs(), &[4]);
        assert_eq!(tape.ops_i32(), vec![2, 1, 1, 11, 12]);
    }

    #[test]
    fn finish_validates_forward_reference_free_by_construction() {
        // Every returned Reg is necessarily earlier, so a well-used builder
        // always finishes; the registers are 0,1,2 in order.
        let mut b = TapeBuilder::new();
        let c = b.constant(2.0);
        let s = b.state(0);
        let _ = b.pow(s, c);
        assert_eq!(b.len(), 3);
        assert!(b.finish(&[Reg(2)], &[], 1, 0).is_ok());
    }

    #[test]
    fn finish_rejects_out_of_range_output() {
        let mut b = TapeBuilder::new();
        let _ = b.state(0);
        // Reg(7) was never emitted.
        let err = b.finish(&[Reg(7)], &[], 1, 0).unwrap_err();
        assert!(matches!(err, IrError::OutputIndexOutOfRange { .. }));
    }

    #[test]
    #[should_panic(expected = "non-unary op")]
    fn unary_rejects_binary_op() {
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let _ = b.unary(Op::Add, x);
    }
}
