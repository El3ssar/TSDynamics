//! Wire round-trip: a tape survives `Tape → arrays → Tape` unchanged.
//!
//! This is the in-Rust half of the Python↔Rust contract.  The Python emitter
//! produces the integer arrays; [`Tape::from_arrays`] ingests them and the slice
//! accessors (plus [`Tape::ops_i32`]) re-emit them.  A tape reconstructed from
//! its own re-emitted arrays must compare equal — so nothing is lost or
//! reinterpreted at the boundary.  (The Python end of the loop is pinned by
//! `tests/test_ir_contract.py` and the committed golden fixtures.)

use tsdyn_ir::{Tape, TapeBuilder};

/// Build a small RHS *with* a Jacobian: f(u) = [p0 * u0^2, sin(u1)].
fn sample_tape() -> Tape {
    let mut b = TapeBuilder::new();
    let p0 = b.param(0);
    let u0 = b.state(0);
    let u1 = b.state(1);
    let u0sq = b.powi(u0, 2);
    let f0 = b.mul(p0, u0sq);
    let f1 = b.sin(u1);
    // Jacobian, row-major 2x2:
    //   ∂f0/∂u0 = 2 p0 u0   ∂f0/∂u1 = 0
    //   ∂f1/∂u0 = 0         ∂f1/∂u1 = cos(u1)
    let two = b.constant(2.0);
    let twop = b.mul(two, p0);
    let j00 = b.mul(twop, u0);
    let zero = b.constant(0.0);
    let j11 = b.cos(u1);
    b.finish(&[f0, f1], &[j00, zero, zero, j11], 2, 1).unwrap()
}

#[test]
fn tape_survives_arrays_round_trip() {
    let tape = sample_tape();
    let rebuilt = Tape::from_arrays(
        &tape.ops_i32(),
        tape.a(),
        tape.b(),
        tape.imm(),
        tape.outputs(),
        tape.jac_outputs(),
        tape.n_state(),
        tape.n_param(),
    )
    .unwrap();
    assert_eq!(tape, rebuilt);
}

#[test]
fn accessors_expose_consistent_shapes() {
    let tape = sample_tape();
    let n = tape.n_reg();
    assert_eq!(tape.ops().len(), n);
    assert_eq!(tape.a().len(), n);
    assert_eq!(tape.b().len(), n);
    assert_eq!(tape.imm().len(), n);
    assert_eq!(tape.ops_i32().len(), n);
    assert_eq!(tape.dim(), 2);
    assert!(tape.has_jacobian());
    assert_eq!(tape.jac_outputs().len(), tape.dim() * tape.dim());
    assert_eq!(tape.n_state(), 2);
    assert_eq!(tape.n_param(), 1);
}

#[test]
fn malformed_arrays_are_rejected_not_panicked() {
    // ops references register 9 that doesn't exist (forward/out-of-range).
    let err = Tape::from_arrays(&[1, 12], &[0, 9], &[0, 0], &[0.0, 0.0], &[1], &[], 1, 0);
    assert!(err.is_err());
}
