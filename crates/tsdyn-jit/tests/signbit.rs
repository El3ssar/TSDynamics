//! Signbit is a bit observation, not a floating-point comparison or reciprocal.

use tsdyn_ir::{reference, Op, TapeBuilder};
use tsdyn_jit::JitEvaluator;
use tsdyn_vm::Interpreter;

#[test]
fn signs_survive_every_leaf_and_both_evaluation_entry_points() {
    // Include subnormals, infinities, quiet/signaling NaNs and distinct payloads.
    let magnitudes = [
        0x0000_0000_0000_0000,
        0x0000_0000_0000_0001,
        0x000f_ffff_ffff_ffff,
        0x0010_0000_0000_0000,
        0x3ff0_0000_0000_0000,
        0x7fef_ffff_ffff_ffff,
        0x7ff0_0000_0000_0000,
        0x7ff0_0000_0000_0001,
        0x7ff8_0000_0000_0000,
        0x7ffa_bcde_f123_4567,
        0x7fff_ffff_ffff_ffff,
    ];
    for magnitude in magnitudes {
        for (mask, expected) in [(0, 0.0_f64), (1u64 << 63, 1.0_f64)] {
            let value = f64::from_bits(magnitude | mask);
            for source in [Op::State, Op::Param, Op::Time, Op::Const] {
                for jacobian_only in [false, true] {
                    let mut b = TapeBuilder::new();
                    let input = match source {
                        Op::State => b.state(0),
                        Op::Param => b.param(0),
                        Op::Time => b.time(),
                        Op::Const => b.constant(value),
                        _ => unreachable!(),
                    };
                    let sign = b.signbit(input);
                    let zero = b.constant(0.0);
                    // These are selected wire outputs, not a claimed derivative
                    // of signbit. Exercise RHS liveness and the fused J body.
                    let (rhs, jac) = if jacobian_only {
                        (zero, sign)
                    } else {
                        (sign, zero)
                    };
                    let tape = b.finish(&[rhs], &[jac], 1, 1).unwrap();
                    let jit = JitEvaluator::new(&tape).unwrap();
                    let vm = Interpreter::new(tape.clone());
                    let expected_rhs = if jacobian_only { 0.0 } else { expected };
                    let expected_jac = if jacobian_only { expected } else { 0.0 };
                    for result in [
                        reference::eval_alloc(&tape, &[value], &[value], value),
                        vm.eval_alloc(&[value], &[value], value),
                        jit.eval_alloc(&[value], &[value], value),
                    ] {
                        assert_eq!(result[0].to_bits(), expected_rhs.to_bits());
                    }
                    for (rhs_values, jac_values) in [
                        reference::eval_jac_alloc(&tape, &[value], &[value], value),
                        vm.eval_jac_alloc(&[value], &[value], value),
                        jit.eval_jac_alloc(&[value], &[value], value),
                    ] {
                        assert_eq!(rhs_values[0].to_bits(), expected_rhs.to_bits());
                        assert_eq!(jac_values[0].to_bits(), expected_jac.to_bits());
                    }
                }
            }
        }
    }
}

#[test]
fn computed_nan_sign_is_observed_and_reaches_finite_arithmetic() {
    let mut b = TapeBuilder::new();
    let x = b.state(0);
    let nan = b.sqrt(x);
    let sign = b.signbit(nan);
    let ten = b.constant(10.0);
    let three = b.constant(3.0);
    let scaled = b.mul(sign, ten);
    let finite = b.sub(scaled, three);
    let tape = b.finish(&[nan, sign, finite], &[], 1, 0).unwrap();
    let jit = JitEvaluator::new(&tape).unwrap();
    let vm = Interpreter::new(tape.clone());
    for result in [
        reference::eval_alloc(&tape, &[-1.0], &[], 0.0),
        vm.eval_alloc(&[-1.0], &[], 0.0),
        jit.eval_alloc(&[-1.0], &[], 0.0),
    ] {
        assert!(result[0].is_nan());
        // Generated NaN signs may be platform-specific; observe the actual
        // produced value instead of assuming a particular libm NaN sign.
        let expected = (result[0].to_bits() >> 63) as f64;
        assert_eq!(result[1].to_bits(), expected.to_bits());
        assert_eq!(result[2], 10.0 * expected - 3.0);
    }
}
