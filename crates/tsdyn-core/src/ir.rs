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

    /// Evaluate one postfix program against (`t`, `state`, `params`).
    ///
    /// `scratch` is the stack; it's caller-owned so the iterate / Lyapunov
    /// loops can amortise allocation. The scratch is truncated on entry.
    /// `t` is only consulted by ODE programs that contain the `Time`
    /// opcode — discrete-map callers pass `0.0`.
    pub fn eval(
        program: &[Expr],
        t: f64,
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
                Expr::Time => scratch.push(t),
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
                Expr::Pow2 => {
                    let b = scratch.pop().unwrap();
                    let a = scratch.pop().unwrap();
                    scratch.push(a.powf(b));
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
                Expr::Tanh => {
                    let a = scratch.pop().unwrap();
                    scratch.push(a.tanh());
                }
                Expr::Sinh => {
                    let a = scratch.pop().unwrap();
                    scratch.push(a.sinh());
                }
                Expr::Cosh => {
                    let a = scratch.pop().unwrap();
                    scratch.push(a.cosh());
                }
                Expr::PowF(k) => {
                    let a = scratch.pop().unwrap();
                    scratch.push(a.powf(*k));
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

/// One compiled ODE: the RHS (`dim` postfix programs) plus an optional
/// Jacobian (`dim × dim` postfix programs). N2.a ships RHS evaluation
/// only — stiff methods that consume the Jacobian arrive in N2.c.
///
/// Six built-in systems (MultiChua, AnishchenkoAstakhov, StickSlipOscillator,
/// CellularNeuralNetwork, Colpitts, FluidTrampoline) use `Abs` / `sign`
/// without an explicit `_jacobian`, and SymEngine returns unevaluated
/// `Derivative` nodes for those — for them the Python lowerer sets
/// `has_jacobian = false` so we never see a malformed Jacobian here.
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
        Ok(Self { dim, n_params, rhs, jacobian })
    }

    /// Evaluate the RHS at `(t, state, params)`, writing each component
    /// into `out_dy`. `scratch` is the per-call f64 stack — caller-owned
    /// so the stepper loop can amortise allocation across thousands of
    /// RHS calls.
    ///
    /// The state vector is exposed to the IR via `Expr::Var(i)`. There
    /// is no implicit time variable yet — autonomous systems only; if a
    /// non-autonomous system surfaces, lift `t` into an extra state
    /// component the way JiTCODE does.
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
        let v = CompiledMap::eval(&program, 0.0, &[], &[], &mut scratch);
        assert_eq!(v, 11.0);
    }

    #[test]
    fn eval_tanh_sinh_cosh_powf() {
        // tanh(x) at x = 0.5
        let mut scratch = Vec::new();
        let prog = vec![Expr::Var(0), Expr::Tanh];
        let v = CompiledMap::eval(&prog, 0.0, &[0.5], &[], &mut scratch);
        assert!((v - (0.5_f64).tanh()).abs() < 1e-15);

        // sinh(x) at x = -0.7
        let prog = vec![Expr::Var(0), Expr::Sinh];
        let v = CompiledMap::eval(&prog, 0.0, &[-0.7], &[], &mut scratch);
        assert!((v - (-0.7_f64).sinh()).abs() < 1e-15);

        // cosh(x) at x = 1.3
        let prog = vec![Expr::Var(0), Expr::Cosh];
        let v = CompiledMap::eval(&prog, 0.0, &[1.3], &[], &mut scratch);
        assert!((v - (1.3_f64).cosh()).abs() < 1e-15);

        // x ** 1.25 at x = 4.0  →  4 ** 1.25 = 4 * 4 ** 0.25 ≈ 5.6568542...
        let prog = vec![Expr::Var(0), Expr::PowF(1.25)];
        let v = CompiledMap::eval(&prog, 0.0, &[4.0], &[], &mut scratch);
        assert!((v - (4.0_f64).powf(1.25)).abs() < 1e-13);

        // x ** 0.5 == sqrt(x) at x = 9.0
        let prog = vec![Expr::Var(0), Expr::PowF(0.5)];
        let v = CompiledMap::eval(&prog, 0.0, &[9.0], &[], &mut scratch);
        assert!((v - 3.0).abs() < 1e-13);
    }

    #[test]
    fn eval_time_and_pow2() {
        let mut scratch = Vec::new();

        // sin(t) at t = 0.5
        let prog = vec![Expr::Time, Expr::Sin];
        let v = CompiledMap::eval(&prog, 0.5, &[], &[], &mut scratch);
        assert!((v - (0.5_f64).sin()).abs() < 1e-15);

        // x ** y at x=2.5, y=3.7  →  2.5_f64.powf(3.7)
        let prog = vec![Expr::Var(0), Expr::Var(1), Expr::Pow2];
        let v = CompiledMap::eval(&prog, 0.0, &[2.5, 3.7], &[], &mut scratch);
        assert!((v - (2.5_f64).powf(3.7)).abs() < 1e-13);

        // Var(0) ** Param(0) at x=4, n=2.5 → 4 ** 2.5 = 32
        let prog = vec![Expr::Var(0), Expr::Param(0), Expr::Pow2];
        let v = CompiledMap::eval(&prog, 0.0, &[4.0], &[2.5], &mut scratch);
        assert!((v - 32.0).abs() < 1e-12);
    }

    #[test]
    fn eval_compiled_ode_roundtrip() {
        // RHS: dx/dt = -a * x + sin(x);  dim=1, params=[a]
        // Wire format: dim=1 n_params=1 n_rhs=1
        //              expr len=6 ops: [Param(0), Var(0), Mul, Neg, Var(0) sin, Add]
        // Then has_jacobian=0.
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&1u32.to_le_bytes()); // dim
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_params
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_rhs
                                                    // program: a * x  ->  Neg  ->  + sin(x)
                                                    //   Param(0), Var(0), Mul, Neg, Var(0), Sin, Add  = 7 ops
        buf.extend_from_slice(&7u32.to_le_bytes());
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
        // -2*0.3 + sin(0.3) ≈ -0.6 + 0.29552...
        let expected = -2.0 * 0.3 + (0.3_f64).sin();
        assert!((out[0] - expected).abs() < 1e-14);
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
        let v = CompiledMap::eval(&program, 0.0, &[0.1, 0.1], &[1.4, 0.3], &mut scratch);
        assert!((v - 1.086).abs() < 1e-12);
    }
}
