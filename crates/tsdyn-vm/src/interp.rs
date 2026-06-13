//! The [`Interpreter`] — the stack-machine `Evaluator` over an IR [`Tape`],
//! plus its reusable [`Scratch`] register file.
//!
//! An interpreter walks the tape in one linear pass: instruction `i` reads the
//! registers earlier instructions wrote and stores its own result in register
//! `i` (single static assignment — the [`Tape`] guarantees the references only
//! point backwards). The register file is the only working memory, so the hot
//! loop allocates nothing once a [`Scratch`] exists. There is no callback into
//! Python and no shared mutable state, so an `&Interpreter` is [`Sync`]: an
//! ensemble shares one interpreter across rayon workers, each with its own
//! [`Scratch`] (ROADMAP §4c).
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

/// The register file an [`Interpreter`] writes as it walks a tape — one `f64`
/// per instruction.
///
/// Sized once with [`Scratch::for_tape`] and reused across steps and across
/// trajectories: every register is overwritten before it is read within a single
/// evaluation, so a dirty buffer left over from a previous call yields identical
/// results (the `reused_scratch_is_deterministic` test pins this). Keeping the
/// buffer caller-owned is what lets [`Interpreter::eval`] take `&self` and run
/// allocation-free inside a parallel ensemble.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Scratch {
    regs: Vec<f64>,
}

impl Scratch {
    /// A register file sized for `tape` (one slot per instruction), zero-filled.
    #[inline]
    pub fn for_tape(tape: &Tape) -> Self {
        Scratch {
            regs: vec![0.0; tape.n_reg()],
        }
    }

    /// Number of registers.
    #[inline]
    pub fn len(&self) -> usize {
        self.regs.len()
    }

    /// Whether the register file is empty (a tape with no instructions).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.regs.is_empty()
    }

    /// Grow or shrink the register file to fit `tape`, reusing the allocation.
    ///
    /// Lets one `Scratch` be retargeted when a worker moves between tapes of
    /// different sizes without freeing and reallocating the buffer.
    #[inline]
    pub fn resize_for(&mut self, tape: &Tape) {
        self.regs.resize(tape.n_reg(), 0.0);
    }
}

/// A tape evaluator that interprets the instruction stream directly.
///
/// Construct one from a validated [`Tape`]; it owns the tape and exposes the two
/// evaluation entry points the engine and solvers need:
///
/// - [`eval`](Interpreter::eval) — `du/dt` only, the explicit-solver path;
/// - [`eval_jac`](Interpreter::eval_jac) — `du/dt` and the analytic Jacobian in
///   one pass, the implicit/stiff-solver path.
///
/// Both take a caller-owned [`Scratch`] and write into caller-owned output
/// slices, so a hot loop reuses every buffer. The [`eval_alloc`] /
/// [`eval_jac_alloc`] convenience methods allocate their own scratch for one-off
/// calls and tests.
///
/// [`eval_alloc`]: Interpreter::eval_alloc
/// [`eval_jac_alloc`]: Interpreter::eval_jac_alloc
#[derive(Clone, Debug, PartialEq)]
pub struct Interpreter {
    tape: Tape,
}

impl Interpreter {
    /// Wrap a validated [`Tape`] in an interpreter.
    #[inline]
    pub fn new(tape: Tape) -> Self {
        Interpreter { tape }
    }

    /// The tape being interpreted.
    #[inline]
    pub fn tape(&self) -> &Tape {
        &self.tape
    }

    /// System dimension (number of derivative outputs).
    #[inline]
    pub fn dim(&self) -> usize {
        self.tape.dim()
    }

    /// Whether the tape carries an analytic Jacobian (see
    /// [`Interpreter::eval_jac`]).
    #[inline]
    pub fn has_jacobian(&self) -> bool {
        self.tape.has_jacobian()
    }

    /// A fresh [`Scratch`] sized for this interpreter's tape.
    #[inline]
    pub fn scratch(&self) -> Scratch {
        Scratch::for_tape(&self.tape)
    }

    /// Walk the tape, filling the register file `regs`.
    ///
    /// One linear pass, no allocation. Every match arm is identical to the
    /// `tsdyn-ir` reference evaluator so results are bit-for-bit equal.
    #[inline]
    fn run(&self, u: &[f64], p: &[f64], t: f64, regs: &mut [f64]) {
        let ops = self.tape.ops();
        let a = self.tape.a();
        let b = self.tape.b();
        let imm = self.tape.imm();
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
            };
            regs[i] = r;
        }
    }

    /// Evaluate `du/dt` at `(u, p, t)` into `deriv`, using `scratch` as the
    /// working register file.
    ///
    /// `u` must have at least the tape's declared state width, `p` its parameter
    /// width, `deriv` exactly [`dim`](Interpreter::dim), and `scratch` must be
    /// sized for this tape (use [`scratch`](Interpreter::scratch)). All are
    /// caller-owned so a hot loop reuses them with no allocation.
    #[inline]
    pub fn eval(&self, u: &[f64], p: &[f64], t: f64, scratch: &mut Scratch, deriv: &mut [f64]) {
        self.debug_check_buffers(u, p, scratch, deriv);
        self.run(u, p, t, &mut scratch.regs);
        for (k, &slot) in self.tape.outputs().iter().enumerate() {
            deriv[k] = scratch.regs[slot as usize];
        }
    }

    /// Evaluate `du/dt` (into `deriv`) and the row-major `dim × dim` Jacobian
    /// `∂f_k/∂u_j` (into `jac`) in a single tape pass.
    ///
    /// Requires a tape carrying a Jacobian ([`has_jacobian`]); for a tape
    /// without one, `jac` is left untouched (the tape exposes no Jacobian
    /// registers). `jac` must hold `dim * dim` elements.
    ///
    /// [`has_jacobian`]: Interpreter::has_jacobian
    #[inline]
    pub fn eval_jac(
        &self,
        u: &[f64],
        p: &[f64],
        t: f64,
        scratch: &mut Scratch,
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
        self.run(u, p, t, &mut scratch.regs);
        for (k, &slot) in self.tape.outputs().iter().enumerate() {
            deriv[k] = scratch.regs[slot as usize];
        }
        for (m, &slot) in self.tape.jac_outputs().iter().enumerate() {
            jac[m] = scratch.regs[slot as usize];
        }
    }

    /// Debug-only precondition checks shared by [`eval`](Interpreter::eval) and
    /// [`eval_jac`](Interpreter::eval_jac): surface a wrong-sized buffer with a
    /// clear message in dev/test builds. Compiled out of release builds, so the
    /// hot loop pays nothing.
    #[inline]
    fn debug_check_buffers(&self, u: &[f64], p: &[f64], scratch: &Scratch, deriv: &[f64]) {
        debug_assert!(
            scratch.len() == self.tape.n_reg(),
            "scratch sized {} but tape needs {} registers (use Interpreter::scratch)",
            scratch.len(),
            self.tape.n_reg()
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
            p.len() >= self.tape.n_param(),
            "param slice too small: {} < n_param {}",
            p.len(),
            self.tape.n_param()
        );
    }

    /// Allocate scratch and return `du/dt` — convenience for one-off calls.
    pub fn eval_alloc(&self, u: &[f64], p: &[f64], t: f64) -> Vec<f64> {
        let mut scratch = self.scratch();
        let mut deriv = vec![0.0; self.dim()];
        self.eval(u, p, t, &mut scratch, &mut deriv);
        deriv
    }

    /// Allocate scratch and return `(du/dt, Jacobian)` with the Jacobian
    /// row-major `dim × dim` — convenience for one-off calls.
    pub fn eval_jac_alloc(&self, u: &[f64], p: &[f64], t: f64) -> (Vec<f64>, Vec<f64>) {
        let dim = self.dim();
        let mut scratch = self.scratch();
        let mut deriv = vec![0.0; dim];
        let mut jac = vec![0.0; dim * dim];
        self.eval_jac(u, p, t, &mut scratch, &mut deriv, &mut jac);
        (deriv, jac)
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
        let mut scratch = interp.scratch();
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
    fn scratch_is_sized_to_n_reg() {
        let interp = Interpreter::new(lorenz());
        let scratch = interp.scratch();
        assert_eq!(scratch.len(), interp.tape().n_reg());
        assert!(!scratch.is_empty());
    }

    #[test]
    fn resize_for_retargets_a_reused_scratch() {
        // A worker carrying one Scratch across tapes of different sizes resizes
        // in place, then evaluates the new tape correctly.
        let small = {
            let mut b = TapeBuilder::new();
            let x = b.state(0);
            Interpreter::new(b.finish(&[x], &[], 1, 0).unwrap()) // n_reg = 1
        };
        let big = Interpreter::new(lorenz()); // n_reg > 1

        let mut scratch = small.scratch();
        assert_eq!(scratch.len(), small.tape().n_reg());

        scratch.resize_for(big.tape());
        assert_eq!(scratch.len(), big.tape().n_reg());
        let mut deriv = vec![0.0; big.dim()];
        big.eval(&[1.0, 2.0, 3.0], &LORENZ_P, 0.0, &mut scratch, &mut deriv);
        assert!((deriv[0] - 10.0 * (2.0 - 1.0)).abs() < 1e-15);

        // …and back down for the smaller tape.
        scratch.resize_for(small.tape());
        assert_eq!(scratch.len(), small.tape().n_reg());
    }

    #[test]
    #[should_panic(expected = "scratch sized")]
    fn eval_rejects_a_wrongly_sized_scratch_in_debug() {
        // A scratch from a *different* tape is the classic misuse the debug
        // precondition check catches early (debug builds only).
        let interp = Interpreter::new(lorenz());
        let other = Interpreter::new({
            let mut b = TapeBuilder::new();
            let x = b.state(0);
            b.finish(&[x], &[], 1, 0).unwrap()
        });
        let mut wrong = other.scratch(); // sized 1, not lorenz's n_reg
        let mut deriv = vec![0.0; interp.dim()];
        interp.eval(&[1.0, 2.0, 3.0], &LORENZ_P, 0.0, &mut wrong, &mut deriv);
    }

    #[test]
    fn eval_jac_leaves_jac_untouched_without_a_jacobian() {
        // A tape with no jac_outputs writes no Jacobian entries.
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let interp = Interpreter::new(b.finish(&[x], &[], 1, 0).unwrap());
        assert!(!interp.has_jacobian());
        let mut scratch = interp.scratch();
        let mut deriv = [0.0];
        let mut jac = [f64::NAN]; // sentinel; must remain untouched
        interp.eval_jac(&[5.0], &[], 0.0, &mut scratch, &mut deriv, &mut jac);
        assert_eq!(deriv[0], 5.0);
        assert!(jac[0].is_nan(), "jac should be left untouched");
    }

    #[test]
    fn interpreter_and_scratch_are_send_and_sync() {
        // The ensemble path shares one `&Interpreter` across rayon workers, each
        // owning its own `Scratch`; lock those bounds in at compile time.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Interpreter>();
        assert_send_sync::<Scratch>();
    }
}
