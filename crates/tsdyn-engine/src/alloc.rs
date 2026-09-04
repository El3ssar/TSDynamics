//! Fallible allocation of the engine's output buffers (stream v6 WP2-safety).
//!
//! Every family sizes at least one output buffer from a number the *caller*
//! chooses — `steps` for a map orbit, `t_eval.len()` for a dense trajectory —
//! multiplied by the system dimension. Written the obvious way, as
//! `vec![0.0; rows * cols]`, that is two uncatchable process-level failures for
//! the price of one:
//!
//! - `rows * cols` overflows `usize` and `RawVec` panics with `capacity
//!   overflow` (a panic across the FFI seam, which pyo3 can only translate into
//!   a `PanicException` — no useful diagnosis, and the unwind has to cross a
//!   `catch_unwind`);
//! - the product is representable but far larger than the machine can serve, so
//!   the allocator returns null and Rust's allocation-error handler calls
//!   [`std::alloc::handle_alloc_error`], which **aborts the process**. That is
//!   `SIGABRT` inside the user's interpreter: no traceback, no `except`, and
//!   every unsaved notebook cell gone.
//!
//! A library must never do either. [`try_zeroed`] does the multiplication with
//! [`usize::checked_mul`] and the reservation with
//! [`Vec::try_reserve_exact`], so both failures become an ordinary [`Err`] that
//! each family lifts into its own error enum and the binding layer maps to a
//! Python `MemoryError`.
//!
//! Note this is about the *output* buffers only. The engine's working buffers
//! (`scratch`, `next`, a `dim`-length state) are sized by the tape, which the
//! bridge has already validated, and are small; they keep the infallible
//! `vec![]` form so the hot paths stay unchanged.

/// An output buffer could not be allocated.
///
/// Carries the requested shape rather than the element count so the message can
/// name what the caller actually asked for (`steps` × `dim`), which is the thing
/// they have to make smaller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllocFailed {
    /// Requested row count (map steps, grid points, ensemble members).
    pub rows: usize,
    /// Requested row width (the system dimension).
    pub cols: usize,
}

impl AllocFailed {
    /// The requested buffer size in bytes, widened so the multiplication that
    /// overflowed `usize` can still be *reported*.
    pub fn bytes(&self) -> u128 {
        (self.rows as u128) * (self.cols as u128) * (core::mem::size_of::<f64>() as u128)
    }
}

impl core::fmt::Display for AllocFailed {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "cannot allocate the {} x {} output buffer ({} bytes): request fewer \
             steps/output times, or a smaller ensemble",
            self.rows,
            self.cols,
            self.bytes()
        )
    }
}

impl std::error::Error for AllocFailed {}

/// Allocate a zeroed row-major `(rows, cols)` `f64` buffer, or fail cleanly.
///
/// Returns [`AllocFailed`] instead of panicking on `rows * cols` overflow and
/// instead of aborting when the allocator cannot serve the request.
///
/// The reservation is exact (`try_reserve_exact`) because these buffers are
/// filled once to a known length and never pushed to, so the amortised-growth
/// slack a plain `try_reserve` may add is pure waste on the very allocations
/// most likely to be near the machine's limit.
pub fn try_zeroed(rows: usize, cols: usize) -> Result<Vec<f64>, AllocFailed> {
    let fail = AllocFailed { rows, cols };
    let len = rows.checked_mul(cols).ok_or(fail)?;
    let mut buf: Vec<f64> = Vec::new();
    buf.try_reserve_exact(len).map_err(|_| fail)?;
    // Cannot reallocate: the capacity reserved above is exactly `len`.
    buf.resize(len, 0.0);
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_the_requested_shape() {
        let v = try_zeroed(3, 4).expect("a small buffer allocates");
        assert_eq!(v.len(), 12);
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn zero_sized_requests_are_fine() {
        assert_eq!(try_zeroed(0, 5).expect("no rows").len(), 0);
        assert_eq!(try_zeroed(5, 0).expect("no cols").len(), 0);
    }

    /// `rows * cols` overflowing `usize` must be an `Err`, not the `capacity
    /// overflow` panic `vec![0.0; rows * cols]` raises.
    #[test]
    fn element_count_overflow_is_an_error_not_a_panic() {
        let err = try_zeroed(usize::MAX, 2).expect_err("the product overflows usize");
        assert_eq!(err.rows, usize::MAX);
        assert_eq!(err.cols, 2);
        assert!(err.to_string().contains("cannot allocate"));
    }

    /// A representable but unservable request must be an `Err`, not the
    /// `handle_alloc_error` abort `vec![0.0; n]` triggers.
    #[test]
    fn unservable_request_is_an_error_not_an_abort() {
        // 2^50 rows x 2 cols = 2^51 f64 = 16 PiB: representable, never servable.
        let err = try_zeroed(1usize << 50, 2).expect_err("16 PiB is not servable");
        assert_eq!(err.bytes(), (1u128 << 51) * 8);
    }
}
