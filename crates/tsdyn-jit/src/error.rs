//! [`JitError`] — the failure type for JIT compilation.

use cranelift_module::ModuleError;

/// Why building a [`JitEvaluator`](crate::JitEvaluator) failed.
///
/// Every failure path inside Cranelift — host-ISA construction, function
/// declaration, codegen/verification, and finalization — surfaces as a
/// [`ModuleError`]. It is boxed so `Result<_, JitError>` stays small (a
/// `ModuleError` is large; boxing keeps the `Ok` path cheap and quiets
/// `clippy::result_large_err`).
#[derive(Debug)]
pub enum JitError {
    /// A Cranelift codegen / module-building step failed.
    Compile(Box<ModuleError>),
}

impl core::fmt::Display for JitError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            JitError::Compile(e) => write!(f, "JIT compilation failed: {e}"),
        }
    }
}

impl std::error::Error for JitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            JitError::Compile(e) => Some(e),
        }
    }
}

impl From<ModuleError> for JitError {
    fn from(e: ModuleError) -> Self {
        JitError::Compile(Box::new(e))
    }
}
