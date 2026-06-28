//! The [`Interpreter`] — the stack-machine evaluator over an IR [`Tape`].
//!
//! An interpreter walks the tape in one linear pass: instruction `i` reads the
//! registers earlier instructions wrote and stores its own result in register
//! `i` (single static assignment — the [`Tape`] guarantees the references only
//! point backwards). A caller-owned **scratch** slice is that register file, so
//! the hot loop allocates nothing. There is no callback into Python and no
//! shared mutable state, so an `&Interpreter` is [`Sync`]: an ensemble builds one
//! interpreter and shares it across rayon workers, each owning its own scratch
//! buffer — exactly the v2 engine's build-once / per-worker-`Workspace` pattern
//! (ROADMAP §4c).
//!
//! # Dead-register elimination on the RHS-only path
//!
//! A Jacobian-bearing tape holds registers that feed *only* the Jacobian
//! outputs. The RHS-only [`eval`](Interpreter::eval) does not need them, so the
//! interpreter precomputes (once, at [`Interpreter::new`]) a liveness mask —
//! the registers some RHS output transitively depends on — and the RHS-only
//! pass skips every op not in it. The mask comes from the shared
//! [`Tape::reachable_from`](tsdyn_ir::Tape::reachable_from)`(false)` pass that
//! the Cranelift JIT (`tsdyn-jit/src/codegen.rs`) also calls for its
//! `tsdyn_eval` body, so the interpreter-executed register set and the
//! JIT-emitted register set are one algorithm, not two parallel copies. The
//! arithmetic in each *executed* arm is unchanged, so the live registers — and
//! hence the RHS outputs — stay **bit-for-bit** identical to a full pass.
//! [`eval_jac`] deliberately ignores the mask and runs the *whole* tape: the
//! DCE here is **RHS-only**, never a Jacobian-liveness mask.
//!
//! # Surface mirrors the engine `Evaluator` seam
//!
//! The method surface here — [`dim`], [`n_param`], [`n_scratch`],
//! [`has_jacobian`], [`eval`], [`eval_jac`] — deliberately matches the
//! object-safe `Evaluator` trait the engine and solvers dispatch through
//! (`&dyn Evaluator`, frozen by stream F2 in `tsdyn-ir`). The interpreter is the
//! register-file evaluator, so `n_scratch()` is the tape's register count; the
//! JIT (stream E2) will report `0`. Once the F2 trait lands on `main`, wiring it
//! up is a one-line `impl Evaluator for Interpreter` that forwards to these
//! inherent methods — E1 builds only against the frozen F1 IR and does not take
//! that dependency itself.
//!
//! [`dim`]: Interpreter::dim
//! [`n_param`]: Interpreter::n_param
//! [`n_scratch`]: Interpreter::n_scratch
//! [`has_jacobian`]: Interpreter::has_jacobian
//! [`eval`]: Interpreter::eval
//! [`eval_jac`]: Interpreter::eval_jac
//!
//! # Relationship to the reference evaluator
//!
//! `tsdyn-ir`'s off-by-default `reference` evaluator is the *canonical
//! operational semantics* of the IR — a direct port of the v2 stack machine.
//! This interpreter is the *production* path and reproduces that semantics
//! **bit-for-bit**: every arithmetic expression below is identical to the
//! reference, instruction for instruction, so the two agree to the last ULP
//! (the `oracle_equivalence` integration test pins this with [`f64::to_bits`]
//! equality). Matching the reference exactly is what satisfies stream E1's
//! contract — the interpreter equals the v2 tape semantics for *every* tape, not
//! merely to a tolerance on a fixed system list.

use tsdyn_ir::{Op, Tape};

/// A tape evaluator that interprets the instruction stream directly.
///
/// Construct one from a validated [`Tape`]; it owns the tape and exposes the two
/// evaluation entry points the engine and solvers need:
///
/// - [`eval`](Interpreter::eval) — `du/dt` only, the explicit-solver path;
/// - [`eval_jac`](Interpreter::eval_jac) — `du/dt` and the analytic Jacobian in
///   one pass, the implicit/stiff-solver path.
///
/// Both take a caller-owned `scratch` register file of length
/// [`n_scratch`](Interpreter::n_scratch) and write into caller-owned output
/// slices, so a hot loop reuses every buffer. The [`eval_alloc`] /
/// [`eval_jac_alloc`] convenience methods allocate their own scratch for one-off
/// calls and tests.
///
/// [`eval_alloc`]: Interpreter::eval_alloc
/// [`eval_jac_alloc`]: Interpreter::eval_jac_alloc
#[derive(Clone, Debug, PartialEq)]
pub struct Interpreter {
    tape: Tape,
    /// RHS liveness mask, one bool per tape register: `rhs_live[i]` is true iff
    /// some RHS output (`tape.outputs()`) transitively depends on register `i`.
    /// Computed once at construction by the shared
    /// [`Tape::reachable_from`](tsdyn_ir::Tape::reachable_from)`(false)` pass (it
    /// depends only on the frozen tape) and reused by every
    /// [`eval`](Interpreter::eval) call so the RHS-only pass skips Jacobian-only
    /// subexpressions — the *same* pass the Cranelift JIT calls for its
    /// `tsdyn_eval` function. The Jacobian path
    /// ([`eval_jac`](Interpreter::eval_jac)) ignores this mask and runs the full
    /// tape, so the Jacobian outputs are unaffected.
    rhs_live: Vec<bool>,
}

impl Interpreter {
    /// Wrap a validated [`Tape`] in an interpreter.
    ///
    /// Precomputes the RHS-only liveness mask via the shared
    /// [`Tape::reachable_from`]`(false)` — the same dead-register-elimination
    /// backward pass the Cranelift JIT calls for its `tsdyn_eval` body — so the
    /// interpreter-executed and JIT-emitted register sets are structurally one
    /// algorithm, not two parallel copies.
    #[inline]
    pub fn new(tape: Tape) -> Self {
        let rhs_live = tape.reachable_from(false);
        Interpreter { tape, rhs_live }
    }

    /// The tape being interpreted.
    #[inline]
    pub fn tape(&self) -> &Tape {
        &self.tape
    }

    /// System dimension — the length of `u`, `deriv`, and `√(jac.len())`.
    #[inline]
    pub fn dim(&self) -> usize {
        self.tape.dim()
    }

    /// Declared parameter width — the expected length of `p`.
    #[inline]
    pub fn n_param(&self) -> usize {
        self.tape.n_param()
    }

    /// Length of the `scratch` register file [`eval`](Interpreter::eval)
    /// requires — the tape's instruction (register) count. Size one buffer per
    /// worker from this and reuse it across every step.
    #[inline]
    pub fn n_scratch(&self) -> usize {
        self.tape.n_reg()
    }

    /// Whether the tape carries an analytic Jacobian (so
    /// [`eval_jac`](Interpreter::eval_jac) is meaningful).
    #[inline]
    pub fn has_jacobian(&self) -> bool {
        self.tape.has_jacobian()
    }

    /// Walk the tape, filling the register file `regs`.
    ///
    /// One linear pass, no allocation. Every match arm is identical to the
    /// `tsdyn-ir` reference evaluator so results are bit-for-bit equal.
    ///
    /// When `live` is `Some(mask)`, an instruction whose result register is not
    /// marked live is skipped (dead-register elimination): the RHS-only path
    /// passes the precomputed [`rhs_live`](Interpreter::rhs_live) mask so it
    /// never computes Jacobian-only subexpressions. Skipping is safe because the
    /// tape is strict SSA (every operand index < the op that writes it) and the
    /// mask is closed under operands — any register a live op reads is itself
    /// live, hence already computed. When `live` is `None` every op runs (the
    /// Jacobian path). The arithmetic in each executed arm is byte-for-byte
    /// unchanged, so the live registers hold identical values either way.
    #[inline]
    fn run(&self, u: &[f64], p: &[f64], t: f64, regs: &mut [f64], live: Option<&[bool]>) {
        let ops = self.tape.ops();
        let a = self.tape.a();
        let b = self.tape.b();
        let imm = self.tape.imm();
        for i in 0..ops.len() {
            // Dead-register elimination: skip an op whose result feeds no live
            // output. A skipped register's slot is left untouched; reachability
            // guarantees no live op ever reads it.
            if let Some(mask) = live {
                if !mask[i] {
                    continue;
                }
            }
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
                // Non-smooth / piecewise ops (E-OPS) — identical expressions to
                // the `tsdyn-ir` reference, so the two agree to the last bit.
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
                Op::Mod => {
                    let (x, y) = (regs[ai], regs[b[i] as usize]);
                    x - y * (x / y).floor()
                }
                Op::Rem => regs[ai] % regs[b[i] as usize],
            };
            regs[i] = r;
        }
    }

    /// Evaluate `du/dt` at `(u, p, t)` into `deriv`, using `scratch` as the
    /// working register file.
    ///
    /// `u` must be at least [`dim`](Interpreter::dim) long, `p` at least
    /// [`n_param`](Interpreter::n_param), `deriv` at least `dim`, and `scratch`
    /// at least [`n_scratch`](Interpreter::n_scratch). The scratch contents on
    /// entry are irrelevant and on exit unspecified. All buffers are
    /// caller-owned so a hot loop reuses them with no allocation.
    #[inline]
    pub fn eval(&self, u: &[f64], p: &[f64], t: f64, scratch: &mut [f64], deriv: &mut [f64]) {
        self.debug_check_buffers(u, p, scratch, deriv);
        // RHS-only: run with the precomputed liveness mask so Jacobian-only
        // subexpressions are never evaluated. The live registers — every one a
        // RHS output reads — are computed by the unchanged arithmetic, so the
        // outputs are bit-for-bit identical to a full pass.
        self.run(u, p, t, scratch, Some(&self.rhs_live));
        for (k, &slot) in self.tape.outputs().iter().enumerate() {
            deriv[k] = scratch[slot as usize];
        }
    }

    /// Evaluate `du/dt` (into `deriv`) and the row-major `dim × dim` Jacobian
    /// `∂f_k/∂u_j` at `jac[k * dim + j]` in a single tape pass.
    ///
    /// Requires a tape carrying a Jacobian ([`has_jacobian`]); for a tape
    /// without one, `jac` is left untouched (the tape exposes no Jacobian
    /// registers). `jac` must hold at least `dim * dim` elements; the other
    /// buffers follow [`eval`](Interpreter::eval)'s contract.
    ///
    /// [`has_jacobian`]: Interpreter::has_jacobian
    #[inline]
    pub fn eval_jac(
        &self,
        u: &[f64],
        p: &[f64],
        t: f64,
        scratch: &mut [f64],
        deriv: &mut [f64],
        jac: &mut [f64],
    ) {
        self.debug_check_buffers(u, p, scratch, deriv);
        debug_assert!(
            jac.len() >= self.tape.jac_outputs().len(),
            "jac too small: {} < {} Jacobian entries",
            jac.len(),
            self.tape.jac_outputs().len()
        );
        // Jacobian path: run the FULL tape (`None`) — the RHS-only liveness
        // mask does not cover Jacobian outputs, so everything must be computed,
        // exactly as before.
        self.run(u, p, t, scratch, None);
        for (k, &slot) in self.tape.outputs().iter().enumerate() {
            deriv[k] = scratch[slot as usize];
        }
        for (m, &slot) in self.tape.jac_outputs().iter().enumerate() {
            jac[m] = scratch[slot as usize];
        }
    }

    /// Allocate scratch and return `du/dt` — convenience for one-off calls.
    pub fn eval_alloc(&self, u: &[f64], p: &[f64], t: f64) -> Vec<f64> {
        let mut scratch = vec![0.0; self.n_scratch()];
        let mut deriv = vec![0.0; self.dim()];
        self.eval(u, p, t, &mut scratch, &mut deriv);
        deriv
    }

    /// Allocate scratch and return `(du/dt, Jacobian)` with the Jacobian
    /// row-major `dim × dim` — convenience for one-off calls.
    pub fn eval_jac_alloc(&self, u: &[f64], p: &[f64], t: f64) -> (Vec<f64>, Vec<f64>) {
        let dim = self.dim();
        let mut scratch = vec![0.0; self.n_scratch()];
        let mut deriv = vec![0.0; dim];
        let mut jac = vec![0.0; dim * dim];
        self.eval_jac(u, p, t, &mut scratch, &mut deriv, &mut jac);
        (deriv, jac)
    }

    /// Debug-only precondition checks shared by [`eval`](Interpreter::eval) and
    /// [`eval_jac`](Interpreter::eval_jac): surface a wrong-sized buffer with a
    /// clear message in dev/test builds (the trait contract invites this).
    /// Compiled out of release builds, so the hot loop pays nothing.
    #[inline]
    fn debug_check_buffers(&self, u: &[f64], p: &[f64], scratch: &[f64], deriv: &[f64]) {
        debug_assert!(
            scratch.len() >= self.n_scratch(),
            "scratch too small: {} < {} registers (use Interpreter::n_scratch)",
            scratch.len(),
            self.n_scratch()
        );
        debug_assert!(
            deriv.len() >= self.dim(),
            "deriv too small: {} < dim {}",
            deriv.len(),
            self.dim()
        );
        debug_assert!(
            u.len() >= self.tape.n_state(),
            "state slice too small: {} < n_state {}",
            u.len(),
            self.tape.n_state()
        );
        debug_assert!(
            p.len() >= self.n_param(),
            "param slice too small: {} < n_param {}",
            p.len(),
            self.n_param()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tsdyn_ir::TapeBuilder;

    /// Build the Lorenz right-hand side with its analytic Jacobian.
    ///
    /// dx = σ(y − x); dy = x(ρ − z) − y; dz = xy − βz, with σ=p0, ρ=p1, β=p2.
    fn lorenz() -> Tape {
        let mut b = TapeBuilder::new();
        let sigma = b.param(0);
        let rho = b.param(1);
        let beta = b.param(2);
        let x = b.state(0);
        let y = b.state(1);
        let z = b.state(2);

        // RHS
        let ymx = b.sub(y, x);
        let dx = b.mul(sigma, ymx);
        let rmz = b.sub(rho, z);
        let xrmz = b.mul(x, rmz);
        let dy = b.sub(xrmz, y);
        let xy = b.mul(x, y);
        let bz = b.mul(beta, z);
        let dz = b.sub(xy, bz);

        // Jacobian rows (∂f_k/∂u_j), built explicitly:
        //   [ -σ,  σ,  0 ]
        //   [ ρ−z, -1, -x ]
        //   [  y,   x, -β ]
        let neg_sigma = b.neg(sigma);
        let zero = b.constant(0.0);
        let neg_one = b.constant(-1.0);
        let neg_x = b.neg(x);
        let neg_beta = b.neg(beta);
        // rmz (= ρ−z) is already a register; reuse it for ∂(dy)/∂x.
        b.finish(
            &[dx, dy, dz],
            &[
                neg_sigma, sigma, zero, // row 0
                rmz, neg_one, neg_x, // row 1
                y, x, neg_beta, // row 2
            ],
            3,
            3,
        )
        .unwrap()
    }

    const LORENZ_P: [f64; 3] = [10.0, 28.0, 8.0 / 3.0];

    #[test]
    fn evaluates_lorenz_rhs_to_machine_precision() {
        let interp = Interpreter::new(lorenz());
        let u = [1.0, 2.0, 3.0];
        let got = interp.eval_alloc(&u, &LORENZ_P, 0.0);
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
    fn evaluates_lorenz_jacobian_to_machine_precision() {
        let interp = Interpreter::new(lorenz());
        assert!(interp.has_jacobian());
        let u = [1.0, 2.0, 3.0];
        let (deriv, jac) = interp.eval_jac_alloc(&u, &LORENZ_P, 0.0);
        // derivative still correct alongside the Jacobian
        assert!((deriv[1] - (1.0 * (28.0 - 3.0) - 2.0)).abs() < 1e-15);
        let [sigma, _rho, beta] = LORENZ_P;
        let (x, y, z) = (u[0], u[1], u[2]);
        let want = [
            -sigma,
            sigma,
            0.0, // row 0
            28.0 - z,
            -1.0,
            -x, // row 1: ρ−z, −1, −x
            y,
            x,
            -beta, // row 2
        ];
        for (k, (g, w)) in jac.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-15, "jac[{k}]: got {g}, want {w}");
        }
    }

    #[test]
    fn covers_every_opcode_against_hand_values() {
        // One tape that emits all 29 opcodes at least once, checked against a
        // hand-written expression. This is deliberately independent of the
        // `reference` oracle: a shared misreading of the contract by both
        // evaluators would slip past oracle-equivalence but not past this.
        //
        // `frac` (= 0.25) feeds the domain-restricted inverse functions
        // (asin/acos/atanh ∈ (−1, 1)); `c` (= 2.0) feeds acosh (∈ [1, ∞)).
        let mut b = TapeBuilder::new();
        let x = b.state(0); // State
        let y = b.state(1);
        let k = b.param(0); // Param
        let tt = b.time(); // Time
        let c = b.constant(2.0); // Const
        let frac = b.constant(0.25); // Const, an in-domain inverse-fn argument

        let s_add = b.add(x, y); // Add
        let s_sub = b.sub(x, y); // Sub
        let s_mul = b.mul(x, c); // Mul
        let s_div = b.div(x, c); // Div  (4/2 = 2)
        let s_pow = b.pow(x, c); // Pow  (x^2 via powf)
        let s_powi = b.powi(y, 3); // Powi (y^3)
        let s_neg = b.neg(x); // Neg  (consumed by Abs below)
        let s_recip = b.recip(c); // Recip
        let s_sin = b.sin(x); // Sin
        let s_cos = b.cos(x); // Cos
        let s_tan = b.tan(x); // Tan
        let s_exp = b.exp(frac); // Exp
        let s_log = b.ln(k); // Log
        let s_sqrt = b.sqrt(x); // Sqrt
        let s_abs = b.abs(s_neg); // Abs  (covers Neg as its input)
        let s_sign = b.sign(s_sub); // Sign
        let s_sinh = b.sinh(frac); // Sinh
        let s_cosh = b.cosh(frac); // Cosh
        let s_tanh = b.tanh(frac); // Tanh
        let s_asin = b.asin(frac); // Asin
        let s_acos = b.acos(frac); // Acos
        let s_atan = b.atan(x); // Atan
        let s_asinh = b.asinh(x); // Asinh
        let s_acosh = b.acosh(c); // Acosh
        let s_atanh = b.atanh(frac); // Atanh

        // Fold every covered register into one output so nothing is dead.
        let acc = b.add(s_add, s_sub);
        let acc = b.add(acc, s_mul);
        let acc = b.add(acc, s_div);
        let acc = b.add(acc, s_pow);
        let acc = b.add(acc, s_powi);
        let acc = b.add(acc, s_recip);
        let acc = b.add(acc, s_sin);
        let acc = b.add(acc, s_cos);
        let acc = b.add(acc, s_tan);
        let acc = b.add(acc, s_exp);
        let acc = b.add(acc, s_log);
        let acc = b.add(acc, s_sqrt);
        let acc = b.add(acc, s_abs);
        let acc = b.add(acc, s_sign);
        let acc = b.add(acc, s_sinh);
        let acc = b.add(acc, s_cosh);
        let acc = b.add(acc, s_tanh);
        let acc = b.add(acc, s_asin);
        let acc = b.add(acc, s_acos);
        let acc = b.add(acc, s_atan);
        let acc = b.add(acc, s_asinh);
        let acc = b.add(acc, s_acosh);
        let acc = b.add(acc, s_atanh);
        let acc = b.add(acc, tt); // fold Time in too
        let tape = b.finish(&[acc], &[], 2, 1).unwrap();

        let interp = Interpreter::new(tape);
        let (x, y, k, t) = (4.0_f64, 2.0_f64, 5.0_f64, 0.5_f64);
        let got = interp.eval_alloc(&[x, y], &[k], t)[0];

        let q = 0.25_f64;
        let want = (x + y)
            + (x - y)
            + (x * 2.0)
            + (x / 2.0)
            + x.powf(2.0)
            + y.powi(3)
            + (1.0 / 2.0)
            + x.sin()
            + x.cos()
            + x.tan()
            + q.exp()
            + k.ln()
            + x.sqrt()
            + (-x).abs()
            + ((x - y).signum() * (((x - y) != 0.0) as i32 as f64))
            + q.sinh()
            + q.cosh()
            + q.tanh()
            + q.asin()
            + q.acos()
            + x.atan()
            + x.asinh()
            + 2.0_f64.acosh()
            + q.atanh()
            + t;
        assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
    }

    #[test]
    fn sign_is_zero_at_zero() {
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let s = b.sign(x);
        let interp = Interpreter::new(b.finish(&[s], &[], 1, 0).unwrap());
        assert_eq!(interp.eval_alloc(&[0.0], &[], 0.0)[0], 0.0);
        assert_eq!(interp.eval_alloc(&[3.5], &[], 0.0)[0], 1.0);
        assert_eq!(interp.eval_alloc(&[-3.5], &[], 0.0)[0], -1.0);
    }

    #[test]
    fn reused_scratch_is_deterministic() {
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let s = b.sin(x);
        let interp = Interpreter::new(b.finish(&[s], &[], 1, 0).unwrap());
        let mut scratch = vec![0.0; interp.n_scratch()];
        let mut d1 = [0.0];
        let mut d2 = [0.0];
        interp.eval(&[0.7], &[], 0.0, &mut scratch, &mut d1);
        // Re-run on the *dirty* scratch: every register is overwritten first, so
        // the result must be identical to the cold-buffer run.
        interp.eval(&[0.7], &[], 0.0, &mut scratch, &mut d2);
        assert_eq!(d1, d2);
        assert!((d1[0] - 0.7_f64.sin()).abs() < 1e-15);
    }

    #[test]
    fn n_scratch_is_the_register_count() {
        let interp = Interpreter::new(lorenz());
        assert_eq!(interp.n_scratch(), interp.tape().n_reg());
        assert!(interp.n_scratch() > 0);
        assert_eq!(interp.dim(), 3);
        assert_eq!(interp.n_param(), 3);
    }

    #[test]
    #[should_panic(expected = "scratch too small")]
    fn eval_rejects_a_too_small_scratch_in_debug() {
        // An undersized scratch is the classic misuse the debug precondition
        // check catches early (debug builds only).
        let interp = Interpreter::new(lorenz());
        let mut tiny = vec![0.0; 1]; // lorenz needs many more registers
        let mut deriv = vec![0.0; interp.dim()];
        interp.eval(&[1.0, 2.0, 3.0], &LORENZ_P, 0.0, &mut tiny, &mut deriv);
    }

    #[test]
    fn eval_jac_leaves_jac_untouched_without_a_jacobian() {
        // A tape with no jac_outputs writes no Jacobian entries.
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let interp = Interpreter::new(b.finish(&[x], &[], 1, 0).unwrap());
        assert!(!interp.has_jacobian());
        let mut scratch = vec![0.0; interp.n_scratch()];
        let mut deriv = [0.0];
        let mut jac = [f64::NAN]; // sentinel; must remain untouched
        interp.eval_jac(&[5.0], &[], 0.0, &mut scratch, &mut deriv, &mut jac);
        assert_eq!(deriv[0], 5.0);
        assert!(jac[0].is_nan(), "jac should be left untouched");
    }

    #[test]
    fn rhs_only_eval_skips_jacobian_only_registers() {
        // Lorenz carries Jacobian outputs whose registers (neg_sigma, zero,
        // neg_one, neg_x, neg_beta) feed ONLY the Jacobian, not the RHS. The
        // RHS-only liveness mask must exclude them.
        let interp = Interpreter::new(lorenz());

        // The RHS outputs and everything they read are live; at least one
        // register (a Jacobian-only one) must be dead, proving DCE has teeth.
        for &slot in interp.tape().outputs() {
            assert!(interp.rhs_live[slot as usize], "an RHS output must be live");
        }
        assert!(
            interp.rhs_live.iter().any(|&b| !b),
            "Lorenz has Jacobian-only registers, so some register must be dead"
        );
    }

    #[test]
    fn rhs_is_bit_identical_with_and_without_dce() {
        // The whole bar: DCE must not perturb the RHS by a single bit. Compare
        // the RHS-only `eval` (mask-driven) against the full-tape pass `eval_jac`
        // runs (its `deriv` is computed with NO mask).
        let interp = Interpreter::new(lorenz());
        let u = [1.3, -2.7, 4.1];
        let p = [10.0, 28.0, 8.0 / 3.0];

        let deriv_dce = interp.eval_alloc(&u, &p, 0.7); // mask-driven RHS-only
        let (deriv_full, _jac) = interp.eval_jac_alloc(&u, &p, 0.7); // full pass

        for (k, (&d, &f)) in deriv_dce.iter().zip(deriv_full.iter()).enumerate() {
            assert_eq!(
                d.to_bits(),
                f.to_bits(),
                "deriv[{k}] differs bit-for-bit: dce {d} vs full {f}"
            );
        }
    }

    #[test]
    fn dead_register_left_untouched_does_not_corrupt_rhs() {
        // A skipped op never writes its slot; a stale value there must not leak
        // into the RHS. Run on deliberately dirty scratch (poisoned with NaN):
        // the live registers are all recomputed, so the RHS stays correct.
        let interp = Interpreter::new(lorenz());
        let u = [1.0, 2.0, 3.0];
        let p = [10.0, 28.0, 8.0 / 3.0];
        let mut scratch = vec![f64::NAN; interp.n_scratch()];
        let mut deriv = [0.0; 3];
        interp.eval(&u, &p, 0.0, &mut scratch, &mut deriv);
        let want = [
            10.0 * (2.0 - 1.0),
            1.0 * (28.0 - 3.0) - 2.0,
            1.0 * 2.0 - (8.0 / 3.0) * 3.0,
        ];
        for (k, (&g, &w)) in deriv.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-15, "deriv[{k}]: got {g}, want {w}");
        }
    }

    #[test]
    fn no_jacobian_tape_keeps_every_rhs_register_live() {
        // With no Jacobian outputs there is nothing dead to drop, but the mask
        // must still be correct: every register an RHS output depends on is live.
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let y = b.state(1);
        let s = b.add(x, y);
        let interp = Interpreter::new(b.finish(&[s], &[], 2, 0).unwrap());
        assert!(interp.rhs_live.iter().all(|&b| b), "no dead registers here");
        assert_eq!(interp.eval_alloc(&[2.0, 5.0], &[], 0.0)[0], 7.0);
    }

    #[test]
    fn rhs_live_mask_is_the_shared_reachable_from_pass() {
        // The interpreter's RHS-only mask must BE `Tape::reachable_from(false)` —
        // the same shared DCE pass the Cranelift JIT calls for `tsdyn_eval`. This
        // pins the structural invariant that the two backends share one liveness
        // computation rather than two parallel copies that could drift.
        let tape = lorenz();
        let interp = Interpreter::new(tape.clone());
        assert_eq!(interp.rhs_live, tape.reachable_from(false));
        // And the RHS-only mask is a strict subset of the with-Jacobian mask:
        // every RHS-live register is also live when the Jacobian is included.
        let with_jac = tape.reachable_from(true);
        for (i, &live) in interp.rhs_live.iter().enumerate() {
            if live {
                assert!(
                    with_jac[i],
                    "register {i} live for RHS but not with Jacobian"
                );
            }
        }
    }

    #[test]
    fn interpreter_is_send_and_sync() {
        // The ensemble path shares one `&Interpreter` across rayon workers, each
        // owning its own scratch buffer; lock the bounds in at compile time.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Interpreter>();
    }
}
