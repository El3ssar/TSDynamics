//! Compile-latency benchmark for the JIT — stream E2's "compile latency
//! benchmarked" acceptance.
//!
//! Cranelift JIT compilation is a one-time, per-system cost (unlike the
//! zero-warmup interpreter), so the engine pays it once and shares the result
//! across every ensemble worker (D2). This test measures and reports that cost
//! over tapes of growing size and guards it against a (generous) regression
//! ceiling. Run with `--nocapture` to see the numbers:
//!
//! ```text
//! cargo test -p tsdyn-jit --test compile_latency -- --nocapture
//! ```

use std::time::Instant;

use tsdyn_ir::{Tape, TapeBuilder};
use tsdyn_jit::JitEvaluator;

/// A single-state tape of roughly `n_ops` instructions, alternating native-lowered
/// ops (`mul`, `add`) with host-call ops (`sin`) so the timing reflects a mix of
/// both codegen paths.
fn chain_tape(n_ops: usize) -> Tape {
    let mut b = TapeBuilder::new();
    let x = b.state(0);
    let mut acc = x;
    for i in 0..n_ops {
        acc = match i % 4 {
            0 => b.sin(acc),
            1 => {
                let c = b.constant(1.000_1);
                b.mul(acc, c)
            }
            2 => b.add(acc, x),
            _ => {
                let c = b.constant(0.5);
                b.add(acc, c)
            }
        };
    }
    b.finish(&[acc], &[], 1, 0).unwrap()
}

#[test]
fn compile_latency_is_reported_and_bounded() {
    // Cold-start one compile so the first measurement below isn't dominated by
    // one-time process initialization (ISA setup, page mapping).
    let _ = JitEvaluator::new(&chain_tape(4)).expect("warmup compiles");

    for &n in &[10_usize, 50, 200, 800] {
        let tape = chain_tape(n);
        let n_reg = tape.n_reg();

        let start = Instant::now();
        let jit = JitEvaluator::new(&tape).expect("tape compiles");
        let elapsed = start.elapsed();

        eprintln!(
            "tsdyn-jit compile latency: {n_reg:>5} registers -> {:>8.3} ms",
            elapsed.as_secs_f64() * 1e3
        );

        // Sanity: the compiled function actually runs and is finite at a generic
        // point — guards against "fast because it compiled nothing".
        let out = jit.eval_alloc(&[0.5], &[], 0.0);
        assert!(
            out[0].is_finite(),
            "compiled chain ({n_reg} regs) gave {out:?}"
        );

        // Generous regression ceiling: even an ~800-instruction tape compiles in
        // well under a second; 5 s leaves ample slack for a slow/contended CI box
        // while still catching a pathological blow-up.
        assert!(
            elapsed.as_secs_f64() < 5.0,
            "compile latency for {n_reg} registers was {:.3} ms (> 5 s ceiling)",
            elapsed.as_secs_f64() * 1e3
        );
    }
}
