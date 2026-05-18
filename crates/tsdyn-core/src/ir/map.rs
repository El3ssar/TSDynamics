//! Compiled discrete-map IR: step function + Jacobian.

use super::{decode_program, DecodeError, Expr, Reader};

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
        Ok(Self {
            dim,
            n_params,
            step,
            jacobian,
        })
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
        let mut scratch = Vec::new();
        let prog = vec![Expr::Var(0), Expr::Tanh];
        let v = CompiledMap::eval(&prog, 0.0, &[0.5], &[], &mut scratch);
        assert!((v - (0.5_f64).tanh()).abs() < 1e-15);

        let prog = vec![Expr::Var(0), Expr::Sinh];
        let v = CompiledMap::eval(&prog, 0.0, &[-0.7], &[], &mut scratch);
        assert!((v - (-0.7_f64).sinh()).abs() < 1e-15);

        let prog = vec![Expr::Var(0), Expr::Cosh];
        let v = CompiledMap::eval(&prog, 0.0, &[1.3], &[], &mut scratch);
        assert!((v - (1.3_f64).cosh()).abs() < 1e-15);

        // x ** 1.25 at x = 4.0  →  4 * 4 ** 0.25 ≈ 5.6568542...
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

        // x ** y at x=2.5, y=3.7
        let prog = vec![Expr::Var(0), Expr::Var(1), Expr::Pow2];
        let v = CompiledMap::eval(&prog, 0.0, &[2.5, 3.7], &[], &mut scratch);
        assert!((v - (2.5_f64).powf(3.7)).abs() < 1e-13);

        // Var(0) ** Param(0) at x=4, n=2.5 → 4 ** 2.5 = 32
        let prog = vec![Expr::Var(0), Expr::Param(0), Expr::Pow2];
        let v = CompiledMap::eval(&prog, 0.0, &[4.0], &[2.5], &mut scratch);
        assert!((v - 32.0).abs() < 1e-12);
    }

    #[test]
    fn eval_henon_step() {
        // Henon: xp = 1 - a*x^2 + y
        // 1 - 1.4 * 0.01 + 0.1 = 1.086
        let program = vec![
            Expr::Const(1.0),
            Expr::Param(0), // a
            Expr::Var(0),   // x
            Expr::Pow(2),
            Expr::Mul,
            Expr::Sub,    // 1 - a*x^2
            Expr::Var(1), // y
            Expr::Add,
        ];
        let mut scratch = Vec::new();
        let v = CompiledMap::eval(&program, 0.0, &[0.1, 0.1], &[1.4, 0.3], &mut scratch);
        assert!((v - 1.086).abs() < 1e-12);
    }
}
