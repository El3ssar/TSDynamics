//! True selection retains the chosen bits and tolerates an invalid unused arm.

use tsdyn_ir::{reference, TapeBuilder};
use tsdyn_jit::JitEvaluator;
use tsdyn_vm::Interpreter;

#[test]
fn truth_and_selected_bits_match_on_rhs_and_fused_paths() {
    let conditions = [
        (0x0000_0000_0000_0000, false),
        (0x8000_0000_0000_0000, false),
        (0x0000_0000_0000_0001, true),
        (0x8000_0000_0000_0001, true),
        (0x3ff0_0000_0000_0000, true),
        (0xc000_0000_0000_0000, true),
        (0x7ff0_0000_0000_0000, true),
        (0xfff0_0000_0000_0000, true),
        (0x7ff8_0000_0000_1234, true),
        (0xfff0_0000_0000_0001, true),
    ];
    let arms = [
        0x0000_0000_0000_0000,
        0x8000_0000_0000_0000,
        0x0000_0000_0000_0001,
        0x3ff0_0000_0000_0000,
        0xc000_0000_0000_0000,
        0x7ff0_0000_0000_0000,
        0xfff0_0000_0000_0000,
        0x7ff8_0000_0000_1234,
        0xfff8_0000_0000_5678,
        0x7ff0_0000_0000_0001,
        0xfff0_0000_0000_0001,
    ];
    for jacobian_only in [false, true] {
        let mut b = TapeBuilder::new();
        let condition = b.state(0);
        let left = b.param(0);
        let right = b.param(1);
        let chosen = b.select(condition, left, right);
        let zero = b.constant(0.0);
        // Jacobian entries here are declared wire outputs to test liveness,
        // not automatic derivatives of this deliberately nonsmooth function.
        let (rhs, jac) = if jacobian_only {
            (zero, chosen)
        } else {
            (chosen, zero)
        };
        let tape = b.finish(&[rhs], &[jac], 1, 2).unwrap();
        let jit = JitEvaluator::new(&tape).unwrap();
        let vm = Interpreter::new(tape.clone());
        for (condition_bits, truth) in conditions {
            let condition = f64::from_bits(condition_bits);
            for left_bits in arms {
                for right_bits in arms {
                    let parameters = [f64::from_bits(left_bits), f64::from_bits(right_bits)];
                    let wanted = if truth { left_bits } else { right_bits };
                    let rhs_bits = if jacobian_only { 0 } else { wanted };
                    let jac_bits = if jacobian_only { wanted } else { 0 };
                    for value in [
                        reference::eval_alloc(&tape, &[condition], &parameters, 0.0),
                        vm.eval_alloc(&[condition], &parameters, 0.0),
                        jit.eval_alloc(&[condition], &parameters, 0.0),
                    ] {
                        assert_eq!(value[0].to_bits(), rhs_bits);
                    }
                    for (value, derivative) in [
                        reference::eval_jac_alloc(&tape, &[condition], &parameters, 0.0),
                        vm.eval_jac_alloc(&[condition], &parameters, 0.0),
                        jit.eval_jac_alloc(&[condition], &parameters, 0.0),
                    ] {
                        assert_eq!(value[0].to_bits(), rhs_bits);
                        assert_eq!(derivative[0].to_bits(), jac_bits);
                    }
                }
            }
        }
    }
}

#[test]
fn unused_nan_arm_cannot_change_a_later_finite_minimum() {
    let mut b = TapeBuilder::new();
    let x = b.state(0);
    let zero = b.constant(0.0);
    let one = b.constant(1.0);
    let two = b.constant(2.0);
    let condition = b.gt(x, zero);
    let negative = b.neg(x);
    let alternative = b.sqrt(negative);
    let chosen = b.select(condition, one, alternative);
    let clipped = b.min(chosen, two);
    let tape = b.finish(&[chosen, clipped], &[], 1, 0).unwrap();
    let jit = JitEvaluator::new(&tape).unwrap();
    let vm = Interpreter::new(tape.clone());
    for (x, wanted) in [(1.0, [1.0, 1.0]), (-9.0, [3.0, 2.0])] {
        for actual in [
            reference::eval_alloc(&tape, &[x], &[], 0.0),
            vm.eval_alloc(&[x], &[], 0.0),
            jit.eval_alloc(&[x], &[], 0.0),
        ] {
            assert_eq!(actual, wanted);
        }
    }
}
