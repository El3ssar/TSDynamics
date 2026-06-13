//! The opcode set — the frozen vocabulary of the instruction tape.
//!
//! Each [`Op`] is one primitive operation a tape instruction performs.  The
//! discriminants are the **wire values**: they are exactly the integers the
//! Python tape emitter writes into the `ops` array and the values the v2
//! `tsdynamics-core` VM matched on, so a tape produced by the existing emitter
//! decodes here unchanged.  **Never renumber a variant** — the numbering is part
//! of the FFI contract (a round-trip test pins it).
//!
//! How an instruction reads its operands depends on the op's [`OpKind`]:
//!
//! | Kind        | Reads                                   | Ops |
//! |-------------|-----------------------------------------|-----|
//! | [`Leaf`]    | an input (`u`/`p`/`t`) or an immediate  | `Const`, `State`, `Param`, `Time` |
//! | [`Unary`]   | register `a`                            | `Neg`, `Recip`, and the transcendental functions |
//! | [`Binary`]  | registers `a` **and** `b`               | `Add`, `Sub`, `Mul`, `Div`, `Pow` |
//! | [`Powi`]    | register `a`, integer exponent in `b`   | `Powi` |
//!
//! For the per-instruction operand layout (which of `a`/`b`/`imm` each kind
//! reads) see [`crate::Tape`].
//!
//! [`Leaf`]: OpKind::Leaf
//! [`Unary`]: OpKind::Unary
//! [`Binary`]: OpKind::Binary
//! [`Powi`]: OpKind::Powi

use crate::tape::IrError;

/// A single tape opcode.  The `#[repr(i32)]` discriminant is the wire value.
///
/// The set mirrors the functions the symbolic `_equations` lower to: the four
/// leaves, the arithmetic binaries, the two power forms (`Pow` for a runtime or
/// non-integer exponent, `Powi` for an integer exponent baked into `b`), and the
/// elementary unary functions.  Subtraction, division and negation have explicit
/// opcodes even though the current SymPy/SymEngine emitter normalises them into
/// `Add`/`Mul`/`Pow` forms — they are valid instructions that map, DDE and SDE
/// lowerings (and direct [`TapeBuilder`](crate::TapeBuilder) users) may emit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum Op {
    // ---- leaves (no register operands) ----
    /// Push the immediate `imm[i]` (the constant stored at this instruction's
    /// own index).
    Const = 0,
    /// Push the state component `u[a]`.
    State = 1,
    /// Push the parameter `p[a]`.
    Param = 2,
    /// Push the independent variable `t`.
    Time = 3,

    // ---- binary (read registers `a` and `b`) ----
    /// `regs[a] + regs[b]`.
    Add = 10,
    /// `regs[a] - regs[b]`.
    Sub = 11,
    /// `regs[a] * regs[b]`.
    Mul = 12,
    /// `regs[a] / regs[b]`.
    Div = 13,
    /// `regs[a].powf(regs[b])` — base in register `a`, exponent in register `b`.
    Pow = 14,
    /// `regs[a].powi(b)` — base in register `a`, **integer exponent is `b`
    /// itself** (not a register index).  See [`OpKind::Powi`].
    Powi = 15,

    // ---- unary (read register `a`) ----
    /// `-regs[a]`.
    Neg = 20,
    /// `1.0 / regs[a]`.
    Recip = 21,

    // ---- elementary functions (read register `a`) ----
    /// `regs[a].sin()`.
    Sin = 30,
    /// `regs[a].cos()`.
    Cos = 31,
    /// `regs[a].tan()`.
    Tan = 32,
    /// `regs[a].exp()`.
    Exp = 33,
    /// `regs[a].ln()` — natural logarithm.
    Log = 34,
    /// `regs[a].sqrt()`.
    Sqrt = 35,
    /// `regs[a].abs()`.
    Abs = 36,
    /// Sign with `sign(0) = 0`: `regs[a].signum() * ((regs[a] != 0.0) as f64)`.
    /// Matches the a.e. convention of the symbolic `sign` (so `d|x|/dx = sign x`
    /// and `d sign/dx = 0` lower consistently).
    Sign = 37,
    /// `regs[a].sinh()`.
    Sinh = 38,
    /// `regs[a].cosh()`.
    Cosh = 39,
    /// `regs[a].tanh()`.
    Tanh = 40,
    /// `regs[a].asin()`.
    Asin = 41,
    /// `regs[a].acos()`.
    Acos = 42,
    /// `regs[a].atan()`.
    Atan = 43,
    /// `regs[a].asinh()`.
    Asinh = 44,
    /// `regs[a].acosh()`.
    Acosh = 45,
    /// `regs[a].atanh()`.
    Atanh = 46,
}

/// How an [`Op`] reads its operands — drives tape validation and documents arity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpKind {
    /// Reads an input (`u`/`p`/`t`) or an immediate; no register operands.
    Leaf,
    /// Reads register `a`.
    Unary,
    /// Reads registers `a` and `b`.
    Binary,
    /// Reads register `a`; `b` is the literal integer exponent.
    Powi,
}

impl Op {
    /// Every opcode, in ascending wire-value order.  Handy for exhaustive
    /// iteration in tests and tooling.
    pub const ALL: [Op; 29] = [
        Op::Const,
        Op::State,
        Op::Param,
        Op::Time,
        Op::Add,
        Op::Sub,
        Op::Mul,
        Op::Div,
        Op::Pow,
        Op::Powi,
        Op::Neg,
        Op::Recip,
        Op::Sin,
        Op::Cos,
        Op::Tan,
        Op::Exp,
        Op::Log,
        Op::Sqrt,
        Op::Abs,
        Op::Sign,
        Op::Sinh,
        Op::Cosh,
        Op::Tanh,
        Op::Asin,
        Op::Acos,
        Op::Atan,
        Op::Asinh,
        Op::Acosh,
        Op::Atanh,
    ];

    /// The wire value (the `#[repr(i32)]` discriminant).
    #[inline]
    pub fn to_i32(self) -> i32 {
        self as i32
    }

    /// Decode a wire value, rejecting unknown opcodes (the v2 VM silently
    /// produced `NaN` for these; the contract layer makes it an error).
    #[inline]
    pub fn from_i32(v: i32) -> Result<Op, IrError> {
        let op = match v {
            0 => Op::Const,
            1 => Op::State,
            2 => Op::Param,
            3 => Op::Time,
            10 => Op::Add,
            11 => Op::Sub,
            12 => Op::Mul,
            13 => Op::Div,
            14 => Op::Pow,
            15 => Op::Powi,
            20 => Op::Neg,
            21 => Op::Recip,
            30 => Op::Sin,
            31 => Op::Cos,
            32 => Op::Tan,
            33 => Op::Exp,
            34 => Op::Log,
            35 => Op::Sqrt,
            36 => Op::Abs,
            37 => Op::Sign,
            38 => Op::Sinh,
            39 => Op::Cosh,
            40 => Op::Tanh,
            41 => Op::Asin,
            42 => Op::Acos,
            43 => Op::Atan,
            44 => Op::Asinh,
            45 => Op::Acosh,
            46 => Op::Atanh,
            other => return Err(IrError::UnknownOpcode(other)),
        };
        Ok(op)
    }

    /// How this opcode reads its operands.
    #[inline]
    pub fn kind(self) -> OpKind {
        use Op::*;
        match self {
            Const | State | Param | Time => OpKind::Leaf,
            Add | Sub | Mul | Div | Pow => OpKind::Binary,
            Powi => OpKind::Powi,
            // everything else is a register-`a` unary
            _ => OpKind::Unary,
        }
    }

    /// A short stable name (matches the SymEngine/emitter spelling for the
    /// functions; uppercase mnemonic for the structural ops).
    pub fn name(self) -> &'static str {
        use Op::*;
        match self {
            Const => "CONST",
            State => "STATE",
            Param => "PARAM",
            Time => "TIME",
            Add => "ADD",
            Sub => "SUB",
            Mul => "MUL",
            Div => "DIV",
            Pow => "POW",
            Powi => "POWI",
            Neg => "NEG",
            Recip => "RECIP",
            Sin => "SIN",
            Cos => "COS",
            Tan => "TAN",
            Exp => "EXP",
            Log => "LOG",
            Sqrt => "SQRT",
            Abs => "ABS",
            Sign => "SIGN",
            Sinh => "SINH",
            Cosh => "COSH",
            Tanh => "TANH",
            Asin => "ASIN",
            Acos => "ACOS",
            Atan => "ATAN",
            Asinh => "ASINH",
            Acosh => "ACOSH",
            Atanh => "ATANH",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_values_match_the_v2_contract() {
        // These integers are the FFI contract: the Python emitter writes them
        // and the v2 VM read them.  Pin every one.
        assert_eq!(Op::Const.to_i32(), 0);
        assert_eq!(Op::State.to_i32(), 1);
        assert_eq!(Op::Param.to_i32(), 2);
        assert_eq!(Op::Time.to_i32(), 3);
        assert_eq!(Op::Add.to_i32(), 10);
        assert_eq!(Op::Sub.to_i32(), 11);
        assert_eq!(Op::Mul.to_i32(), 12);
        assert_eq!(Op::Div.to_i32(), 13);
        assert_eq!(Op::Pow.to_i32(), 14);
        assert_eq!(Op::Powi.to_i32(), 15);
        assert_eq!(Op::Neg.to_i32(), 20);
        assert_eq!(Op::Recip.to_i32(), 21);
        assert_eq!(Op::Sin.to_i32(), 30);
        assert_eq!(Op::Atanh.to_i32(), 46);
    }

    #[test]
    fn from_i32_round_trips_every_op() {
        for &op in &Op::ALL {
            assert_eq!(Op::from_i32(op.to_i32()).unwrap(), op);
        }
    }

    #[test]
    fn all_is_complete_and_ordered() {
        assert_eq!(Op::ALL.len(), 29);
        for win in Op::ALL.windows(2) {
            assert!(win[0].to_i32() < win[1].to_i32(), "ALL must be ascending");
        }
    }

    #[test]
    fn unknown_opcode_is_rejected() {
        for bad in [-1, 4, 9, 16, 22, 29, 47, 1000] {
            assert!(matches!(Op::from_i32(bad), Err(IrError::UnknownOpcode(v)) if v == bad));
        }
    }

    #[test]
    fn kinds_are_assigned_correctly() {
        assert_eq!(Op::Const.kind(), OpKind::Leaf);
        assert_eq!(Op::Time.kind(), OpKind::Leaf);
        assert_eq!(Op::Add.kind(), OpKind::Binary);
        assert_eq!(Op::Pow.kind(), OpKind::Binary);
        assert_eq!(Op::Powi.kind(), OpKind::Powi);
        assert_eq!(Op::Neg.kind(), OpKind::Unary);
        assert_eq!(Op::Recip.kind(), OpKind::Unary);
        assert_eq!(Op::Tanh.kind(), OpKind::Unary);
        assert_eq!(Op::Atanh.kind(), OpKind::Unary);
    }
}
