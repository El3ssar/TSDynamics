//! A **reference** tape evaluator — the canonical operational semantics of the
//! IR, behind the off-by-default `reference` feature.
//!
//! This is the executable specification of what every opcode *means*: a direct,
//! unoptimised port of the v2 `tsdynamics-core` stack machine, matching on
//! [`Op`] so the compiler proves the opcode set is handled exhaustively.  It
//! serves two purposes:
//!
//! 1. it lets this crate prove the migrated tape still reproduces the symbolic
//!    right-hand side to machine precision (the golden-fixture tests);
//! 2. it is a dependency-free in-workspace oracle the production evaluators in
//!    `tsdyn-vm` (stream E1) and `tsdyn-jit` (stream E2) can validate against.
//!
//! It deliberately does **not** implement an `Evaluator` trait — that seam
//! belongs to stream F2 — and it is not on the production build path.  Use the
//! real evaluators for integration; use this only as a reference/oracle.

use crate::op::Op;
use crate::tape::Tape;

/// Run the instruction tape, filling the register file `regs` (length
/// [`Tape::n_reg`]).  One linear pass; no allocation.
pub fn run(tape: &Tape, u: &[f64], p: &[f64], t: f64, regs: &mut [f64]) {
    let ops = tape.ops();
    let a = tape.a();
    let b = tape.b();
    let imm = tape.imm();
    for i in 0..ops.len() {
        let ai = a[i] as usize;
        let r = match ops[i] {
            Op::Const => imm[i],
            Op::State => u[ai],
            Op::Param => p[ai],
            Op::Time => t,
            Op::Add => regs[ai] + regs[b[i] as usize],
            Op::Sub => regs[ai] - regs[b[i] as usize],
            Op::Mul => regs[ai] * regs[b[i] as usize],
            Op::Div => regs[ai] / regs[b[i] as usize],
            Op::Pow => regs[ai].powf(regs[b[i] as usize]),
            Op::Powi => regs[ai].powi(b[i]),
            Op::Neg => -regs[ai],
            Op::Recip => 1.0 / regs[ai],
            Op::Sin => regs[ai].sin(),
            Op::Cos => regs[ai].cos(),
            Op::Tan => regs[ai].tan(),
            Op::Exp => regs[ai].exp(),
            Op::Log => regs[ai].ln(),
            Op::Sqrt => regs[ai].sqrt(),
            Op::Abs => regs[ai].abs(),
            // sign(0) = 0, matching the symbolic `sign` a.e. convention.
            Op::Sign => regs[ai].signum() * ((regs[ai] != 0.0) as i32 as f64),
            Op::Sinh => regs[ai].sinh(),
            Op::Cosh => regs[ai].cosh(),
            Op::Tanh => regs[ai].tanh(),
            Op::Asin => regs[ai].asin(),
            Op::Acos => regs[ai].acos(),
            Op::Atan => regs[ai].atan(),
            Op::Asinh => regs[ai].asinh(),
            Op::Acosh => regs[ai].acosh(),
            Op::Atanh => regs[ai].atanh(),
            // Comparisons: 1.0 if the relation holds, else 0.0 (NaN compares
            // false, so `Ne` is the only one true on a NaN operand).
            Op::Lt => (regs[ai] < regs[b[i] as usize]) as i32 as f64,
            Op::Le => (regs[ai] <= regs[b[i] as usize]) as i32 as f64,
            Op::Gt => (regs[ai] > regs[b[i] as usize]) as i32 as f64,
            Op::Ge => (regs[ai] >= regs[b[i] as usize]) as i32 as f64,
            Op::Eq => (regs[ai] == regs[b[i] as usize]) as i32 as f64,
            Op::Ne => (regs[ai] != regs[b[i] as usize]) as i32 as f64,
            Op::Min => regs[ai].min(regs[b[i] as usize]),
            Op::Max => regs[ai].max(regs[b[i] as usize]),
            Op::Floor => regs[ai].floor(),
            Op::Ceil => regs[ai].ceil(),
            // Floored modulo (Python `%` / `np.mod`): result takes the sign of
            // the divisor; `mod(a, 0)` is NaN (`0 * inf`).
            Op::Mod => {
                let (x, y) = (regs[ai], regs[b[i] as usize]);
                x - y * (x / y).floor()
            }
            // Truncated remainder (Rust `%` / C `fmod`): sign of the dividend.
            Op::Rem => regs[ai] % regs[b[i] as usize],
        };
        regs[i] = r;
    }
}

/// Evaluate `du/dt` at `(u, p, t)` into `deriv` (length [`Tape::dim`]), using
/// `regs` (length [`Tape::n_reg`]) as caller-owned scratch.
pub fn eval(tape: &Tape, u: &[f64], p: &[f64], t: f64, regs: &mut [f64], deriv: &mut [f64]) {
    run(tape, u, p, t, regs);
    for (k, &slot) in tape.outputs().iter().enumerate() {
        deriv[k] = regs[slot as usize];
    }
}

/// Evaluate `du/dt` (into `deriv`) and the row-major `dim × dim` Jacobian
/// `∂f/∂u` (into `jac`) in one tape pass.  Requires a tape carrying
/// `jac_outputs` (see [`Tape::has_jacobian`]).
pub fn eval_jac(
    tape: &Tape,
    u: &[f64],
    p: &[f64],
    t: f64,
    regs: &mut [f64],
    deriv: &mut [f64],
    jac: &mut [f64],
) {
    run(tape, u, p, t, regs);
    for (k, &slot) in tape.outputs().iter().enumerate() {
        deriv[k] = regs[slot as usize];
    }
    for (m, &slot) in tape.jac_outputs().iter().enumerate() {
        jac[m] = regs[slot as usize];
    }
}

/// Convenience wrapper that allocates its own scratch and returns `du/dt`.
pub fn eval_alloc(tape: &Tape, u: &[f64], p: &[f64], t: f64) -> Vec<f64> {
    let mut regs = vec![0.0; tape.n_reg()];
    let mut deriv = vec![0.0; tape.dim()];
    eval(tape, u, p, t, &mut regs, &mut deriv);
    deriv
}

/// Convenience wrapper that allocates its own scratch and returns
/// `(du/dt, Jacobian)` with the Jacobian row-major `dim × dim`.
pub fn eval_jac_alloc(tape: &Tape, u: &[f64], p: &[f64], t: f64) -> (Vec<f64>, Vec<f64>) {
    let dim = tape.dim();
    let mut regs = vec![0.0; tape.n_reg()];
    let mut deriv = vec![0.0; dim];
    let mut jac = vec![0.0; dim * dim];
    eval_jac(tape, u, p, t, &mut regs, &mut deriv, &mut jac);
    (deriv, jac)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::{Reg, TapeBuilder};

    #[test]
    fn evaluates_lorenz_rhs() {
        // dx = sigma (y - x); dy = x (rho - z) - y; dz = x y - beta z
        let mut bld = TapeBuilder::new();
        let sigma = bld.param(0);
        let rho = bld.param(1);
        let beta = bld.param(2);
        let x = bld.state(0);
        let y = bld.state(1);
        let z = bld.state(2);
        let ymx = bld.sub(y, x);
        let dx = bld.mul(sigma, ymx);
        let rmz = bld.sub(rho, z);
        let xrmz = bld.mul(x, rmz);
        let dy = bld.sub(xrmz, y);
        let xy = bld.mul(x, y);
        let bz = bld.mul(beta, z);
        let dz = bld.sub(xy, bz);
        let tape = bld.finish(&[dx, dy, dz], &[], 3, 3).unwrap();

        let u = [1.0, 2.0, 3.0];
        let p = [10.0, 28.0, 8.0 / 3.0];
        let got = eval_alloc(&tape, &u, &p, 0.0);
        let want = [
            10.0 * (2.0 - 1.0),
            1.0 * (28.0 - 3.0) - 2.0,
            1.0 * 2.0 - (8.0 / 3.0) * 3.0,
        ];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-15, "got {g}, want {w}");
        }
    }

    #[test]
    fn sign_is_zero_at_zero() {
        let mut bld = TapeBuilder::new();
        let x = bld.state(0);
        let s = bld.sign(x);
        let tape = bld.finish(&[s], &[], 1, 0).unwrap();
        assert_eq!(eval_alloc(&tape, &[0.0], &[], 0.0)[0], 0.0);
        assert_eq!(eval_alloc(&tape, &[3.5], &[], 0.0)[0], 1.0);
        assert_eq!(eval_alloc(&tape, &[-3.5], &[], 0.0)[0], -1.0);
    }

    #[test]
    fn jacobian_of_a_quadratic() {
        // f(u) = u0^2 ; ∂f/∂u0 = 2 u0 (built as 2*u0).
        let mut bld = TapeBuilder::new();
        let x = bld.state(0);
        let f = bld.powi(x, 2);
        let two = bld.constant(2.0);
        let df = bld.mul(two, x);
        let tape = bld.finish(&[f], &[df], 1, 0).unwrap();
        assert!(tape.has_jacobian());
        let (d, j) = eval_jac_alloc(&tape, &[3.0], &[], 0.0);
        assert!((d[0] - 9.0).abs() < 1e-15);
        assert!((j[0] - 6.0).abs() < 1e-15);
    }

    #[test]
    fn reused_scratch_gives_identical_results() {
        let mut bld = TapeBuilder::new();
        let x = bld.state(0);
        let s = bld.sin(x);
        let tape = bld.finish(&[s], &[], 1, 0).unwrap();
        let mut regs = vec![0.0; tape.n_reg()];
        let mut d1 = vec![0.0; 1];
        let mut d2 = vec![0.0; 1];
        eval(&tape, &[0.7], &[], 0.0, &mut regs, &mut d1);
        eval(&tape, &[0.7], &[], 0.0, &mut regs, &mut d2);
        assert_eq!(d1, d2);
        assert!((d1[0] - 0.7_f64.sin()).abs() < 1e-15);
        let _ = Reg(0); // keep the Reg import exercised
    }
}
