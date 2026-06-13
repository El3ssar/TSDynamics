//! The interpreter must reproduce the IR's canonical operational semantics
//! **bit-for-bit**.
//!
//! `tsdyn-ir`'s `reference` evaluator is a direct port of the v2
//! `tsdynamics-core` stack machine — the executable specification of what each
//! opcode means. This test fuzzes a large population of randomly-generated valid
//! tapes, evaluates each one with both the production [`Interpreter`] and the
//! reference, and asserts the results are *identical to the last bit*
//! ([`f64::to_bits`] equality, which is exact even for `NaN`/`±∞`).
//!
//! That is a stronger statement than "matches to 1e-15 on the catalogue": the
//! interpreter equals the v2 tape semantics for *every* tape the contract can
//! express, so any system that lowers to a tape is reproduced exactly. (The
//! end-to-end 118-ODE trajectory sweep against the live v2 backends is the
//! Python cross-validation harness's job; this pins the Rust evaluation kernel.)
//!
//! The `reference` evaluator is pulled in via tsdyn-vm's dev-dependency on
//! `tsdyn-ir` (which enables its `reference` feature), so this test always runs.

use tsdyn_ir::{reference, Op, OpKind, Tape};
use tsdyn_vm::Interpreter;

/// SplitMix64 — a tiny, dependency-free, fully deterministic PRNG so the fuzz
/// corpus is identical on every run and machine.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` (n > 0).
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Uniform `f64` in `[lo, hi)` from 53 random bits.
    fn f64_in(&mut self, lo: f64, hi: f64) -> f64 {
        let unit = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + unit * (hi - lo)
    }

    /// Uniform integer in `[lo, hi]`.
    fn i32_in(&mut self, lo: i32, hi: i32) -> i32 {
        lo + self.below((hi - lo + 1) as usize) as i32
    }

    fn choice<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len())]
    }
}

const LEAVES: [Op; 4] = [Op::Const, Op::State, Op::Param, Op::Time];

/// Build a random *valid* tape: leaves seed the register file, later
/// instructions only reference strictly-earlier registers, and all leaf indices
/// stay in range — so [`Tape::from_arrays`] accepts it by construction.
fn random_tape(rng: &mut Rng) -> Tape {
    let n_state = rng.i32_in(1, 4) as usize;
    let n_param = rng.i32_in(1, 4) as usize; // ≥1 so `Param` is always emittable
    let n = rng.i32_in(3, 24) as usize;

    let mut ops = Vec::with_capacity(n);
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    let mut imm = Vec::with_capacity(n);

    for i in 0..n {
        // Instruction 0 has no earlier registers, so it must be a leaf.
        let op = if i == 0 {
            *rng.choice(&LEAVES)
        } else {
            *rng.choice(&Op::ALL)
        };
        let (ai, bi, immi) = match op.kind() {
            OpKind::Leaf => match op {
                Op::Const => (0, 0, rng.f64_in(-3.0, 3.0)),
                Op::State => (rng.below(n_state) as i32, 0, 0.0),
                Op::Param => (rng.below(n_param) as i32, 0, 0.0),
                Op::Time => (0, 0, 0.0),
                _ => unreachable!("non-leaf op classified as Leaf"),
            },
            OpKind::Unary => (rng.below(i) as i32, 0, 0.0),
            OpKind::Binary => (rng.below(i) as i32, rng.below(i) as i32, 0.0),
            // `b` is the literal integer exponent, not a register.
            OpKind::Powi => (rng.below(i) as i32, rng.i32_in(-4, 5), 0.0),
        };
        ops.push(op.to_i32());
        a.push(ai);
        b.push(bi);
        imm.push(immi);
    }

    let dim = n_state;
    let outputs: Vec<i32> = (0..dim).map(|_| rng.below(n) as i32).collect();
    // Carry a Jacobian on roughly half the tapes to exercise `eval_jac`.
    let jac_outputs: Vec<i32> = if rng.next_u64() & 1 == 0 {
        (0..dim * dim).map(|_| rng.below(n) as i32).collect()
    } else {
        Vec::new()
    };

    Tape::from_arrays(&ops, &a, &b, &imm, &outputs, &jac_outputs, n_state, n_param)
        .expect("generator must only build valid tapes")
}

#[track_caller]
fn assert_bits_eq(label: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{label}[{i}]: interpreter gave {g:?}, reference gave {w:?}"
        );
    }
}

#[test]
fn interpreter_matches_reference_bit_for_bit() {
    let mut rng = Rng(0x7053_D1CE_2024_0601);
    let n_tapes = 400;
    let n_points = 8;
    let mut jacobian_tapes = 0;

    for _ in 0..n_tapes {
        let tape = random_tape(&mut rng);
        let n_state = tape.n_state();
        let n_param = tape.n_param();
        let interp = Interpreter::new(tape.clone());

        for _ in 0..n_points {
            let u: Vec<f64> = (0..n_state).map(|_| rng.f64_in(-3.0, 3.0)).collect();
            let p: Vec<f64> = (0..n_param).map(|_| rng.f64_in(-3.0, 3.0)).collect();
            let t = rng.f64_in(-3.0, 3.0);

            // RHS-only path.
            let mut scratch = interp.scratch();
            let mut deriv = vec![0.0; interp.dim()];
            interp.eval(&u, &p, t, &mut scratch, &mut deriv);
            let want = reference::eval_alloc(&tape, &u, &p, t);
            assert_bits_eq("eval", &deriv, &want);

            // Combined RHS + Jacobian path.
            if tape.has_jacobian() {
                let (dj, jj) = interp.eval_jac_alloc(&u, &p, t);
                let (wd, wj) = reference::eval_jac_alloc(&tape, &u, &p, t);
                assert_bits_eq("eval_jac deriv", &dj, &wd);
                assert_bits_eq("eval_jac jac", &jj, &wj);
            }
        }
        if tape.has_jacobian() {
            jacobian_tapes += 1;
        }
    }

    // Guard the corpus actually exercised both code paths (a degenerate RNG that
    // never produced a Jacobian tape would silently skip `eval_jac`).
    assert!(
        jacobian_tapes > n_tapes / 4,
        "expected a healthy share of Jacobian tapes, got {jacobian_tapes}/{n_tapes}"
    );
}

/// A reused, dirty register file must not change results: every register is
/// overwritten before it is read, so back-to-back evaluations on one `Scratch`
/// agree bit-for-bit with the reference (the ensemble-reuse invariant).
#[test]
fn reused_scratch_stays_bit_identical_to_reference() {
    let mut rng = Rng(0x0FF1_CED0_5EED_1234);
    for _ in 0..100 {
        let tape = random_tape(&mut rng);
        let interp = Interpreter::new(tape.clone());
        let mut scratch = interp.scratch();
        let mut deriv = vec![0.0; interp.dim()];
        for _ in 0..5 {
            let u: Vec<f64> = (0..tape.n_state()).map(|_| rng.f64_in(-2.0, 2.0)).collect();
            let p: Vec<f64> = (0..tape.n_param()).map(|_| rng.f64_in(-2.0, 2.0)).collect();
            let t = rng.f64_in(-2.0, 2.0);
            interp.eval(&u, &p, t, &mut scratch, &mut deriv);
            let want = reference::eval_alloc(&tape, &u, &p, t);
            assert_bits_eq("reused-scratch eval", &deriv, &want);
        }
    }
}
