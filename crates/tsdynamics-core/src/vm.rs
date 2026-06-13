//! A tiny SSA virtual machine that evaluates a system right-hand side compiled
//! to a flat instruction tape.
//!
//! The Python layer walks a system's symbolic `_equations` (the same symbolic
//! core that feeds the DiffSL backend) and emits a tape: a list of primitive
//! instructions in single-static-assignment form.  Instruction `i` writes
//! register `i`; later instructions read earlier registers by index.  Common
//! subexpressions are shared at emit time, so the tape is already CSE'd.
//!
//! Evaluating a tape is a single linear pass over the instruction arrays with a
//! reusable register scratch buffer — allocation-free, no Python callbacks, and
//! GIL-free, so many trajectories can run concurrently under rayon.
//!
//! Opcodes are defined here and mirrored in `tsdynamics`' Python tape emitter;
//! the two MUST stay in sync (a round-trip test guards the contract).

// ---- opcodes (keep in sync with the Python emitter) ----------------------
// leaves
const CONST: i32 = 0; // imm[i]
const STATE: i32 = 1; // u[a]
const PARAM: i32 = 2; // p[a]
const TIME: i32 = 3; // t

// binary (read regs a, b)
const ADD: i32 = 10;
const SUB: i32 = 11;
const MUL: i32 = 12;
const DIV: i32 = 13;
const POW: i32 = 14; // regs[a] ^ regs[b]
const POWI: i32 = 15; // regs[a] ^ b   (b is the integer exponent itself)

// unary (read reg a)
const NEG: i32 = 20;
const RECIP: i32 = 21;
const SIN: i32 = 30;
const COS: i32 = 31;
const TAN: i32 = 32;
const EXP: i32 = 33;
const LOG: i32 = 34;
const SQRT: i32 = 35;
const ABS: i32 = 36;
const SIGN: i32 = 37;
const SINH: i32 = 38;
const COSH: i32 = 39;
const TANH: i32 = 40;
const ASIN: i32 = 41;
const ACOS: i32 = 42;
const ATAN: i32 = 43;
const ASINH: i32 = 44;
const ACOSH: i32 = 45;
const ATANH: i32 = 46;

/// A compiled right-hand side: `du/dt = f(u, p, t)`.
pub struct Tape {
    pub ops: Vec<i32>,
    pub a: Vec<i32>,
    pub b: Vec<i32>,
    pub imm: Vec<f64>,
    /// Register index holding each derivative component (length = `dim`).
    pub outputs: Vec<i32>,
    /// Register index holding each Jacobian entry, row-major `dim × dim`
    /// (`jac_outputs[k*dim + j] = ∂f_k/∂u_j`).  Empty for the explicit kernels
    /// that need no Jacobian; populated for the stiff solver.
    pub jac_outputs: Vec<i32>,
    // Declared lengths from the Python side; carried for FFI symmetry and
    // future validation, not read by the evaluator (indices are baked in).
    #[allow(dead_code)]
    pub n_state: usize,
    #[allow(dead_code)]
    pub n_param: usize,
}

impl Tape {
    #[inline]
    pub fn dim(&self) -> usize {
        self.outputs.len()
    }

    #[inline]
    pub fn n_reg(&self) -> usize {
        self.ops.len()
    }

    /// Run the instruction tape, filling the register file `regs`.
    #[inline]
    fn run(&self, u: &[f64], p: &[f64], t: f64, regs: &mut [f64]) {
        for i in 0..self.ops.len() {
            let a = self.a[i] as usize;
            let r = match self.ops[i] {
                CONST => self.imm[i],
                STATE => u[a],
                PARAM => p[a],
                TIME => t,
                ADD => regs[a] + regs[self.b[i] as usize],
                SUB => regs[a] - regs[self.b[i] as usize],
                MUL => regs[a] * regs[self.b[i] as usize],
                DIV => regs[a] / regs[self.b[i] as usize],
                POW => regs[a].powf(regs[self.b[i] as usize]),
                POWI => regs[a].powi(self.b[i]),
                NEG => -regs[a],
                RECIP => 1.0 / regs[a],
                SIN => regs[a].sin(),
                COS => regs[a].cos(),
                TAN => regs[a].tan(),
                EXP => regs[a].exp(),
                LOG => regs[a].ln(),
                SQRT => regs[a].sqrt(),
                ABS => regs[a].abs(),
                SIGN => regs[a].signum() * ((regs[a] != 0.0) as i32 as f64),
                SINH => regs[a].sinh(),
                COSH => regs[a].cosh(),
                TANH => regs[a].tanh(),
                ASIN => regs[a].asin(),
                ACOS => regs[a].acos(),
                ATAN => regs[a].atan(),
                ASINH => regs[a].asinh(),
                ACOSH => regs[a].acosh(),
                ATANH => regs[a].atanh(),
                _ => f64::NAN,
            };
            regs[i] = r;
        }
    }

    /// Evaluate `du/dt` at `(u, p, t)` into `deriv`, using `regs` as scratch.
    ///
    /// `regs` must have length `n_reg()` and `deriv` length `dim()`; both are
    /// caller-owned so a hot loop reuses them across steps with no allocation.
    #[inline]
    pub fn eval(&self, u: &[f64], p: &[f64], t: f64, regs: &mut [f64], deriv: &mut [f64]) {
        self.run(u, p, t, regs);
        for (k, &slot) in self.outputs.iter().enumerate() {
            deriv[k] = regs[slot as usize];
        }
    }

    /// Evaluate both `du/dt` (into `deriv`) and the Jacobian `∂f/∂u` (into
    /// `jac`, row-major `dim × dim`) in a single tape pass.  Requires a tape
    /// built with `jac_outputs`.
    #[inline]
    pub fn eval_jac(
        &self,
        u: &[f64],
        p: &[f64],
        t: f64,
        regs: &mut [f64],
        deriv: &mut [f64],
        jac: &mut [f64],
    ) {
        self.run(u, p, t, regs);
        for (k, &slot) in self.outputs.iter().enumerate() {
            deriv[k] = regs[slot as usize];
        }
        for (m, &slot) in self.jac_outputs.iter().enumerate() {
            jac[m] = regs[slot as usize];
        }
    }
}

/// Scratch buffers for one integrating worker (one per rayon task).
pub struct Workspace {
    regs: Vec<f64>,
    k1: Vec<f64>,
    k2: Vec<f64>,
    k3: Vec<f64>,
    k4: Vec<f64>,
    tmp: Vec<f64>,
}

impl Workspace {
    pub fn for_tape(tape: &Tape) -> Self {
        let d = tape.dim();
        Workspace {
            regs: vec![0.0; tape.n_reg()],
            k1: vec![0.0; d],
            k2: vec![0.0; d],
            k3: vec![0.0; d],
            k4: vec![0.0; d],
            tmp: vec![0.0; d],
        }
    }
}

/// One classical RK4 step of size `h`, advancing `u` in place at time `t`.
//  The four-stage combination touches several parallel buffers by the same
//  index; the explicit indexed form is clearer than zipping five iterators.
#[allow(clippy::needless_range_loop)]
#[inline]
pub fn rk4_step(tape: &Tape, u: &mut [f64], p: &[f64], t: f64, h: f64, ws: &mut Workspace) {
    let d = u.len();
    tape.eval(u, p, t, &mut ws.regs, &mut ws.k1);
    for i in 0..d {
        ws.tmp[i] = u[i] + 0.5 * h * ws.k1[i];
    }
    tape.eval(&ws.tmp, p, t + 0.5 * h, &mut ws.regs, &mut ws.k2);
    for i in 0..d {
        ws.tmp[i] = u[i] + 0.5 * h * ws.k2[i];
    }
    tape.eval(&ws.tmp, p, t + 0.5 * h, &mut ws.regs, &mut ws.k3);
    for i in 0..d {
        ws.tmp[i] = u[i] + h * ws.k3[i];
    }
    tape.eval(&ws.tmp, p, t + h, &mut ws.regs, &mut ws.k4);
    for i in 0..d {
        u[i] += (h / 6.0) * (ws.k1[i] + 2.0 * ws.k2[i] + 2.0 * ws.k3[i] + ws.k4[i]);
    }
}

/// Integrate from `u0` and write the state at each `t_eval` point.
///
/// Each interval `[t_eval[j-1], t_eval[j]]` is subdivided into `ceil(span / h)`
/// RK4 substeps so accuracy is governed by `h`, not by the output spacing.
/// Returns a row-major `(t_eval.len(), dim)` buffer.
pub fn integrate_dense(tape: &Tape, u0: &[f64], p: &[f64], t_eval: &[f64], h: f64) -> Vec<f64> {
    let d = tape.dim();
    let n_t = t_eval.len();
    let mut out = vec![0.0; n_t * d];
    if n_t == 0 {
        return out; // nothing to fill (guard before the slice below)
    }
    let mut u = u0.to_vec();
    let mut ws = Workspace::for_tape(tape);
    out[0..d].copy_from_slice(&u);
    for j in 1..n_t {
        let ta = t_eval[j - 1];
        let span = t_eval[j] - ta;
        let nsub = ((span.abs() / h).ceil() as usize).max(1);
        let hh = span / nsub as f64;
        let mut t = ta;
        for _ in 0..nsub {
            rk4_step(tape, &mut u, p, t, hh, &mut ws);
            t += hh;
        }
        out[j * d..(j + 1) * d].copy_from_slice(&u);
    }
    out
}

/// Integrate a single trajectory from `t0` to `t1`, returning only the final
/// state (the basin/ensemble primitive — no dense output to store).
pub fn integrate_final(
    tape: &Tape,
    u0: &[f64],
    p: &[f64],
    t0: f64,
    t1: f64,
    h: f64,
    ws: &mut Workspace,
) -> Vec<f64> {
    let span = t1 - t0;
    let nsub = ((span.abs() / h).ceil() as usize).max(1);
    let hh = span / nsub as f64;
    let mut u = u0.to_vec();
    let mut t = t0;
    for _ in 0..nsub {
        rk4_step(tape, &mut u, p, t, hh, ws);
        t += hh;
    }
    u
}
