//! Host math shims — the functions the JIT *calls* for opcodes Cranelift has no
//! native instruction for (transcendentals, `Pow`, `Powi`, `Sign`).
//!
//! Each shim is a thin `extern "C"` wrapper around the exact `std` (`f64`)
//! method the interpreter and the IR `reference` evaluator use. Routing these
//! opcodes through the *same* function — rather than re-deriving them in
//! Cranelift IR — is what makes the JIT bit-for-bit identical to the interpreter:
//! `JIT(sin x)` is literally `f64::sin(x)`, same rounding, same NaN payloads.
//!
//! They are registered with the JIT by address (`fn as *const u8`), not by linker
//! name, so there is no `#[no_mangle]` and no risk of a global-symbol clash. The
//! ABI is the platform C convention, matching the imported signatures the codegen
//! declares (`(f64) -> f64`, `(f64, f64) -> f64`, `(f64, i32) -> f64`). None of
//! them panic or unwind, so the `extern "C"` boundary is sound.

/// Define an `extern "C"` unary shim `name(x) = x.method()`.
macro_rules! unary_shim {
    ($name:ident, $method:ident) => {
        pub(crate) extern "C" fn $name(x: f64) -> f64 {
            x.$method()
        }
    };
}

unary_shim!(sin, sin);
unary_shim!(cos, cos);
unary_shim!(tan, tan);
unary_shim!(exp, exp);
unary_shim!(log, ln); // Op::Log is the natural logarithm
unary_shim!(sinh, sinh);
unary_shim!(cosh, cosh);
unary_shim!(tanh, tanh);
unary_shim!(asin, asin);
unary_shim!(acos, acos);
unary_shim!(atan, atan);
unary_shim!(asinh, asinh);
unary_shim!(acosh, acosh);
unary_shim!(atanh, atanh);

/// `base.powf(exp)` — register base, register exponent (`Op::Pow`).
pub(crate) extern "C" fn pow(base: f64, exp: f64) -> f64 {
    base.powf(exp)
}

/// `base.powi(exp)` — register base, integer exponent (`Op::Powi`).
pub(crate) extern "C" fn powi(base: f64, exp: i32) -> f64 {
    base.powi(exp)
}

/// `sign(x)` with `sign(0) = 0` — the a.e. convention of the symbolic `sign`,
/// identical to the interpreter/reference (`Op::Sign`).
pub(crate) extern "C" fn sign(x: f64) -> f64 {
    x.signum() * ((x != 0.0) as i32 as f64)
}
