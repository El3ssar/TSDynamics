//! Exhaustive opcode coverage: a single hand-built tape that exercises **every**
//! [`Op`] once, checked against closed-form values to machine precision.
//!
//! The golden-system fixtures only cover the opcodes the symbolic emitter
//! actually produces (sympy normalises `a-b`, `a/b`, `-a` away, so `Sub`, `Div`,
//! `Neg` — and `Tan`, `Log`, `Sqrt`, the inverse trig — never appear in the ODE
//! catalogue).  Those opcodes are still part of the frozen contract: maps, DDEs,
//! SDEs and direct [`TapeBuilder`] users emit them.  This test pins their
//! semantics independently of any system.
#![cfg(feature = "reference")]

use tsdyn_ir::{reference, Op, Reg, TapeBuilder};

/// Evaluate a one-output tape whose single output is `reg`, with the shared
/// inputs `u = [0.3, 1.5, 2.0]`, `p = [3.0]`, `t = 0.7`.
fn eval_one(build: impl FnOnce(&mut TapeBuilder) -> Reg) -> f64 {
    let mut b = TapeBuilder::new();
    let out = build(&mut b);
    let tape = b.finish(&[out], &[], 3, 1).unwrap();
    reference::eval_alloc(&tape, &[0.3, 1.5, 2.0], &[3.0], 0.7)[0]
}

/// Assert two `f64`s agree to a tight mixed tolerance (these are byte-identical
/// `f64` operations, so they agree to the ULP; 1e-15 leaves margin).
#[track_caller]
fn close(got: f64, want: f64) {
    let tol = 1e-15 * want.abs().max(1.0);
    assert!((got - want).abs() <= tol, "got {got}, want {want}");
}

#[test]
fn leaves() {
    close(eval_one(|b| b.constant(7.5)), 7.5);
    close(eval_one(|b| b.state(2)), 2.0);
    close(eval_one(|b| b.param(0)), 3.0);
    close(eval_one(|b| b.time()), 0.7);
}

#[test]
fn arithmetic_binaries() {
    close(
        eval_one(|b| {
            let x = b.state(2);
            let p = b.param(0);
            b.add(x, p)
        }),
        5.0,
    );
    close(
        eval_one(|b| {
            let x = b.state(2);
            let p = b.param(0);
            b.sub(x, p)
        }),
        -1.0,
    );
    close(
        eval_one(|b| {
            let x = b.state(2);
            let p = b.param(0);
            b.mul(x, p)
        }),
        6.0,
    );
    close(
        eval_one(|b| {
            let x = b.state(2);
            let p = b.param(0);
            b.div(x, p)
        }),
        2.0 / 3.0,
    );
}

#[test]
fn powers_and_negation() {
    // Pow: register base, register exponent → 2^3 = 8.
    close(
        eval_one(|b| {
            let x = b.state(2);
            let p = b.param(0);
            b.pow(x, p)
        }),
        8.0,
    );
    // Powi: register base, inline integer exponent → 2^3 = 8.
    close(
        eval_one(|b| {
            let x = b.state(2);
            b.powi(x, 3)
        }),
        2.0_f64.powi(3),
    );
    // Powi with a negative exponent → 2^-2 = 0.25.
    close(
        eval_one(|b| {
            let x = b.state(2);
            b.powi(x, -2)
        }),
        0.25,
    );
    close(
        eval_one(|b| {
            let x = b.state(2);
            b.neg(x)
        }),
        -2.0,
    );
    close(
        eval_one(|b| {
            let x = b.state(2);
            b.recip(x)
        }),
        0.5,
    );
}

#[test]
fn elementary_functions() {
    let s = 0.3_f64; // u0 — in every relevant function's domain
    close(
        eval_one(|b| {
            let x = b.state(0);
            b.sin(x)
        }),
        s.sin(),
    );
    close(
        eval_one(|b| {
            let x = b.state(0);
            b.cos(x)
        }),
        s.cos(),
    );
    close(
        eval_one(|b| {
            let x = b.state(0);
            b.tan(x)
        }),
        s.tan(),
    );
    close(
        eval_one(|b| {
            let x = b.state(0);
            b.exp(x)
        }),
        s.exp(),
    );
    close(
        eval_one(|b| {
            let x = b.state(2);
            b.ln(x)
        }),
        2.0_f64.ln(),
    );
    close(
        eval_one(|b| {
            let x = b.state(2);
            b.sqrt(x)
        }),
        2.0_f64.sqrt(),
    );
    close(
        eval_one(|b| {
            let x = b.state(0);
            b.sinh(x)
        }),
        s.sinh(),
    );
    close(
        eval_one(|b| {
            let x = b.state(0);
            b.cosh(x)
        }),
        s.cosh(),
    );
    close(
        eval_one(|b| {
            let x = b.state(0);
            b.tanh(x)
        }),
        s.tanh(),
    );
    close(
        eval_one(|b| {
            let x = b.state(0);
            b.asin(x)
        }),
        s.asin(),
    );
    close(
        eval_one(|b| {
            let x = b.state(0);
            b.acos(x)
        }),
        s.acos(),
    );
    close(
        eval_one(|b| {
            let x = b.state(0);
            b.atan(x)
        }),
        s.atan(),
    );
    close(
        eval_one(|b| {
            let x = b.state(0);
            b.asinh(x)
        }),
        s.asinh(),
    );
    close(
        eval_one(|b| {
            let x = b.state(1);
            b.acosh(x)
        }),
        1.5_f64.acosh(),
    );
    close(
        eval_one(|b| {
            let x = b.state(0);
            b.atanh(x)
        }),
        s.atanh(),
    );
}

#[test]
fn abs_and_sign() {
    // |-2| = 2
    close(
        eval_one(|b| {
            let x = b.state(2);
            let nx = b.neg(x);
            b.abs(nx)
        }),
        2.0,
    );
    // sign(-2) = -1, sign(+0.3) = 1, sign(0) = 0
    close(
        eval_one(|b| {
            let x = b.state(2);
            let nx = b.neg(x);
            b.sign(nx)
        }),
        -1.0,
    );
    close(
        eval_one(|b| {
            let x = b.state(0);
            b.sign(x)
        }),
        1.0,
    );
    close(
        eval_one(|b| {
            let z = b.constant(0.0);
            b.sign(z)
        }),
        0.0,
    );
}

#[test]
fn nonsmooth_ops() {
    // A NaN built once for the "NaN compares false / returns the other" checks.
    let nan = |b: &mut TapeBuilder| {
        let z = b.constant(0.0);
        b.div(z, z)
    };

    // Comparisons yield 1.0 (true) / 0.0 (false). u0 = 0.3, u2 = 2.0.
    close(
        eval_one(|b| {
            let (x, y) = (b.state(0), b.state(2));
            b.lt(x, y)
        }),
        1.0,
    ); // 0.3 < 2.0
    close(
        eval_one(|b| {
            let (x, y) = (b.state(2), b.state(0));
            b.lt(x, y)
        }),
        0.0,
    ); // 2.0 < 0.3
    close(
        eval_one(|b| {
            let (x, y) = (b.state(0), b.state(0));
            b.le(x, y)
        }),
        1.0,
    ); // 0.3 <= 0.3
    close(
        eval_one(|b| {
            let (x, y) = (b.state(2), b.state(0));
            b.gt(x, y)
        }),
        1.0,
    ); // 2.0 > 0.3
    close(
        eval_one(|b| {
            let (x, y) = (b.state(0), b.state(2));
            b.ge(x, y)
        }),
        0.0,
    ); // 0.3 >= 2.0
    close(
        eval_one(|b| {
            let (x, y) = (b.state(0), b.state(0));
            b.eq(x, y)
        }),
        1.0,
    ); // 0.3 == 0.3
    close(
        eval_one(|b| {
            let (x, y) = (b.state(0), b.state(2));
            b.eq(x, y)
        }),
        0.0,
    ); // 0.3 == 2.0
    close(
        eval_one(|b| {
            let (x, y) = (b.state(0), b.state(2));
            b.ne(x, y)
        }),
        1.0,
    ); // 0.3 != 2.0
       // A NaN operand compares false everywhere except `!=`.
    close(
        eval_one(|b| {
            let (n, x) = (nan(b), b.state(0));
            b.lt(n, x)
        }),
        0.0,
    );
    close(
        eval_one(|b| {
            let (n, x) = (nan(b), b.state(0));
            b.ne(n, x)
        }),
        1.0,
    );

    // min / max — u0 = 0.3, u2 = 2.0.
    close(
        eval_one(|b| {
            let (x, y) = (b.state(0), b.state(2));
            b.min(x, y)
        }),
        0.3,
    );
    close(
        eval_one(|b| {
            let (x, y) = (b.state(0), b.state(2));
            b.max(x, y)
        }),
        2.0,
    );
    // f64::min: a NaN operand returns the other.
    close(
        eval_one(|b| {
            let (x, n) = (b.state(0), nan(b));
            b.min(x, n)
        }),
        0.3,
    );

    // floor / ceil — u1 = 1.5; also a negative argument.
    close(
        eval_one(|b| {
            let x = b.state(1);
            b.floor(x)
        }),
        1.0,
    );
    close(
        eval_one(|b| {
            let x = b.state(1);
            let nx = b.neg(x);
            b.floor(nx)
        }),
        -2.0,
    );
    close(
        eval_one(|b| {
            let x = b.state(1);
            b.ceil(x)
        }),
        2.0,
    );
    close(
        eval_one(|b| {
            let x = b.state(1);
            let nx = b.neg(x);
            b.ceil(nx)
        }),
        -1.0,
    );

    // Floored modulo (sign of divisor) vs truncated remainder (sign of dividend),
    // shown by a negative dividend: mod(-1.5, 2) = 0.5 but rem(-1.5, 2) = -1.5.
    close(
        eval_one(|b| {
            let (x, y) = (b.state(1), b.state(2));
            b.modulo(x, y)
        }),
        1.5,
    );
    close(
        eval_one(|b| {
            let x = b.state(1);
            let nx = b.neg(x);
            let y = b.state(2);
            b.modulo(nx, y)
        }),
        0.5,
    );
    close(
        eval_one(|b| {
            let (x, y) = (b.state(1), b.state(2));
            b.rem(x, y)
        }),
        1.5,
    );
    close(
        eval_one(|b| {
            let x = b.state(1);
            let nx = b.neg(x);
            let y = b.state(2);
            b.rem(nx, y)
        }),
        -1.5,
    );
}

/// Belt-and-braces: confirm the tests above collectively touch all 41 opcodes
/// (a single tape that uses each exactly once still evaluates).
#[test]
fn every_opcode_is_reachable_in_one_tape() {
    let mut b = TapeBuilder::new();
    let u0 = b.state(0); // 0.3
    let u1 = b.state(1); // 1.5
    let u2 = b.state(2); // 2.0
    let p0 = b.param(0); // 3.0
    let t = b.time(); // 0.7
    let c = b.constant(0.0);

    let mut outs = vec![
        b.add(u2, p0),
        b.sub(u2, p0),
        b.mul(u2, p0),
        b.div(u2, p0),
        b.pow(u2, p0),
        b.powi(u2, 3),
        b.neg(u2),
        b.recip(u2),
        b.sin(u0),
        b.cos(u0),
        b.tan(u0),
        b.exp(u0),
        b.ln(u2),
        b.sqrt(u2),
        b.abs(u2),
        b.sign(c),
        b.sinh(u0),
        b.cosh(u0),
        b.tanh(u0),
        b.asin(u0),
        b.acos(u0),
        b.atan(u0),
        b.asinh(u0),
        b.acosh(u1),
        b.atanh(u0),
        // non-smooth / piecewise block (E-OPS)
        b.lt(u0, u2),
        b.le(u0, u2),
        b.gt(u0, u2),
        b.ge(u0, u2),
        b.eq(u0, u2),
        b.ne(u0, u2),
        b.min(u0, u2),
        b.max(u0, u2),
        b.floor(u1),
        b.ceil(u1),
        b.modulo(u1, u2),
        b.rem(u1, u2),
    ];
    // also surface the leaves so Const/State/Param/Time are outputs too
    outs.extend([c, u0, p0, t]);

    let tape = b.finish(&outs, &[], 3, 1).unwrap();

    // Confirm the tape body uses every opcode at least once.
    let mut seen = std::collections::HashSet::new();
    for &op in tape.ops() {
        seen.insert(op);
    }
    for op in Op::ALL {
        assert!(seen.contains(&op), "opcode {:?} never appears", op);
    }

    // And it evaluates without panicking, with finite results everywhere.
    let d = reference::eval_alloc(&tape, &[0.3, 1.5, 2.0], &[3.0], 0.7);
    assert!(d.iter().all(|v| v.is_finite()), "non-finite output: {d:?}");
}
