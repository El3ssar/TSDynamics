//! The JIT must reproduce the interpreter's results — stream E2's central
//! acceptance ("JIT eval == interpreter eval to ~1e-12 across the catalogue").
//!
//! We prove the stronger statement: **bit-for-bit** equality on a large fuzz
//! corpus of randomly-generated valid tapes. The JIT lowers arithmetic, `Sqrt`
//! and `Abs` to native IEEE-754 instructions (identical to the Rust operators the
//! interpreter uses) and every transcendental / `Pow` / `Powi` / `Sign` to a host
//! call into the same `std` function the interpreter calls, so the two paths must
//! agree to the last bit — including `NaN`/`±∞` payloads ([`f64::to_bits`] is
//! exact). That subsumes the `~1e-12` bar for every tape the IR can express, not
//! just a fixed system list. We cross-check against the canonical `reference`
//! evaluator too (the v2 stack-machine port), so a shared misreading of the
//! contract could not slip past.
//!
//! The generator is the same one `tsdyn-vm`'s `oracle_equivalence` test uses, so
//! both evaluators face an identical corpus.

use tsdyn_ir::{reference, Op, OpKind, Tape};
use tsdyn_jit::JitEvaluator;
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

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    fn f64_in(&mut self, lo: f64, hi: f64) -> f64 {
        let unit = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + unit * (hi - lo)
    }

    fn i32_in(&mut self, lo: i32, hi: i32) -> i32 {
        lo + self.below((hi - lo + 1) as usize) as i32
    }

    fn choice<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len())]
    }
}

const LEAVES: [Op; 4] = [Op::Const, Op::State, Op::Param, Op::Time];

/// Build a random *valid* tape: leaves seed the register file, later instructions
/// only reference strictly-earlier registers, and all leaf indices stay in range
/// — so [`Tape::from_arrays`] accepts it by construction.
fn random_tape(rng: &mut Rng) -> Tape {
    let n_state = rng.i32_in(1, 4) as usize;
    let n_param = rng.i32_in(1, 4) as usize;
    let n = rng.i32_in(3, 24) as usize;

    let mut ops = Vec::with_capacity(n);
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    let mut imm = Vec::with_capacity(n);

    for i in 0..n {
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
            OpKind::Powi => (rng.below(i) as i32, rng.i32_in(-4, 5), 0.0),
        };
        ops.push(op.to_i32());
        a.push(ai);
        b.push(bi);
        imm.push(immi);
    }

    let dim = n_state;
    let outputs: Vec<i32> = (0..dim).map(|_| rng.below(n) as i32).collect();
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
            "{label}[{i}]: JIT gave {g:?}, reference gave {w:?}"
        );
    }
}

#[test]
fn jit_matches_interpreter_and_reference_bit_for_bit() {
    let mut rng = Rng(0x7053_D1CE_2024_0601);
    let n_tapes = 250;
    let n_points = 8;
    let mut jacobian_tapes = 0;

    for _ in 0..n_tapes {
        let tape = random_tape(&mut rng);
        let n_state = tape.n_state();
        let n_param = tape.n_param();
        let jit = JitEvaluator::new(&tape).expect("tape compiles");
        let interp = Interpreter::new(tape.clone());

        for _ in 0..n_points {
            let u: Vec<f64> = (0..n_state).map(|_| rng.f64_in(-3.0, 3.0)).collect();
            let p: Vec<f64> = (0..n_param).map(|_| rng.f64_in(-3.0, 3.0)).collect();
            let t = rng.f64_in(-3.0, 3.0);

            // RHS-only path: JIT == interpreter == reference, bit-for-bit.
            let jit_d = jit.eval_alloc(&u, &p, t);
            let int_d = interp.eval_alloc(&u, &p, t);
            let ref_d = reference::eval_alloc(&tape, &u, &p, t);
            assert_bits_eq("eval vs interpreter", &jit_d, &int_d);
            assert_bits_eq("eval vs reference", &jit_d, &ref_d);

            // Combined RHS + Jacobian path.
            if tape.has_jacobian() {
                let (jit_dd, jit_jj) = jit.eval_jac_alloc(&u, &p, t);
                let (int_dd, int_jj) = interp.eval_jac_alloc(&u, &p, t);
                let (ref_dd, ref_jj) = reference::eval_jac_alloc(&tape, &u, &p, t);
                assert_bits_eq("eval_jac deriv vs interpreter", &jit_dd, &int_dd);
                assert_bits_eq("eval_jac jac vs interpreter", &jit_jj, &int_jj);
                assert_bits_eq("eval_jac deriv vs reference", &jit_dd, &ref_dd);
                assert_bits_eq("eval_jac jac vs reference", &jit_jj, &ref_jj);
            }
        }
        if tape.has_jacobian() {
            jacobian_tapes += 1;
        }
    }

    assert!(
        jacobian_tapes > n_tapes / 4,
        "expected a healthy share of Jacobian tapes, got {jacobian_tapes}/{n_tapes}"
    );
}

/// The JIT shares one compiled function across calls; back-to-back evaluations
/// on the same evaluator must stay bit-identical to the interpreter (the
/// ensemble-reuse invariant — many rayon workers call one `&JitEvaluator`).
#[test]
fn repeated_calls_stay_bit_identical() {
    let mut rng = Rng(0x0FF1_CED0_5EED_1234);
    for _ in 0..120 {
        let tape = random_tape(&mut rng);
        let jit = JitEvaluator::new(&tape).expect("tape compiles");
        let interp = Interpreter::new(tape.clone());
        for _ in 0..5 {
            let u: Vec<f64> = (0..tape.n_state()).map(|_| rng.f64_in(-2.0, 2.0)).collect();
            let p: Vec<f64> = (0..tape.n_param()).map(|_| rng.f64_in(-2.0, 2.0)).collect();
            let t = rng.f64_in(-2.0, 2.0);
            assert_bits_eq(
                "repeated eval",
                &jit.eval_alloc(&u, &p, t),
                &interp.eval_alloc(&u, &p, t),
            );
        }
    }
}
