//! Small dense linear algebra for the implicit kernels: an in-place LU
//! factorization with partial pivoting and the matching triangular solve, plus a
//! builder for the shifted iteration matrix `I − c·J`.
//!
//! Implicit methods spend their per-step cost solving a linear system whose
//! matrix is `I − c·h·J` (a Rosenbrock stage, a Newton correction). The systems
//! here are *dense and small* — the dimension is the number of state components,
//! a handful for the catalogue's stiff benchmarks — so a textbook dense LU is the
//! right tool: no sparsity bookkeeping, cache-friendly, and exact up to rounding.
//! For a large stiff system a sparse/banded factorization would win, but that is
//! a later specialization, not the contract this stream owns.
//!
//! The routines are deliberately the same ones proven in the v2 accelerator's
//! stiff kernel (cross-validated against reference stiff integrators), lifted
//! here verbatim in spirit so the migrated engine inherits the same numerics.

/// In-place LU factorization of the row-major `n × n` matrix `a` with partial
/// (row) pivoting.
///
/// On success `a` holds `L` (unit lower, implicit unit diagonal) and `U` (upper)
/// packed together, `piv[k]` records the row swapped into position `k`, and the
/// return is `true`. Returns `false` — leaving `a` partially modified — when a
/// pivot column is entirely zero or `NaN`, i.e. the matrix is singular; the
/// caller treats that as a failed step to retry with a smaller `h` (a singular
/// `I − c·h·J` is almost always a too-large step rather than a true singularity).
pub(crate) fn lu_factor(a: &mut [f64], n: usize, piv: &mut [usize]) -> bool {
    debug_assert_eq!(a.len(), n * n);
    debug_assert_eq!(piv.len(), n);
    for k in 0..n {
        // Choose the largest-magnitude pivot in column k (partial pivoting keeps
        // the multipliers ≤ 1 and the factorization numerically stable).
        let mut pivot = k;
        let mut max = a[k * n + k].abs();
        for i in (k + 1)..n {
            let v = a[i * n + k].abs();
            if v > max {
                max = v;
                pivot = i;
            }
        }
        if max == 0.0 || max.is_nan() {
            return false; // singular (or NaN pivot column)
        }
        if pivot != k {
            for j in 0..n {
                a.swap(k * n + j, pivot * n + j);
            }
        }
        piv[k] = pivot;
        let akk = a[k * n + k];
        for i in (k + 1)..n {
            let f = a[i * n + k] / akk;
            a[i * n + k] = f;
            for j in (k + 1)..n {
                a[i * n + j] -= f * a[k * n + j];
            }
        }
    }
    true
}

/// Solve `A x = b` in place (`b` ← `x`) given the LU factors produced by
/// [`lu_factor`] for the *same* `a`, `n`, and `piv`.
///
/// Applies the recorded row swaps to `b`, then forward- and back-substitutes.
#[allow(clippy::needless_range_loop)]
pub(crate) fn lu_solve(a: &[f64], n: usize, piv: &[usize], b: &mut [f64]) {
    debug_assert_eq!(a.len(), n * n);
    debug_assert_eq!(b.len(), n);
    for k in 0..n {
        let p = piv[k];
        if p != k {
            b.swap(k, p);
        }
    }
    // Forward substitution (unit lower triangle).
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= a[i * n + j] * b[j];
        }
        b[i] = s;
    }
    // Back substitution (upper triangle).
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i * n + j] * b[j];
        }
        b[i] = s / a[i * n + i];
    }
}

/// Write the shifted iteration matrix `mat = I − c·J` (row-major `n × n`), the
/// matrix every implicit stage factorizes. `jac` is the row-major Jacobian
/// `∂f/∂u`; `c` folds in the step size and the method's stage coefficient (e.g.
/// `c = h` for linearly-implicit Euler, `c = γ·h/2` for a TR-BDF2 Newton stage).
pub(crate) fn build_shifted(mat: &mut [f64], jac: &[f64], n: usize, c: f64) {
    debug_assert_eq!(mat.len(), n * n);
    debug_assert_eq!(jac.len(), n * n);
    for i in 0..n {
        for j in 0..n {
            mat[i * n + j] = (if i == j { 1.0 } else { 0.0 }) - c * jac[i * n + j];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Solve a 2×2 system with a known answer.
    #[test]
    fn lu_solves_a_known_system() {
        // [[2, 1], [1, 3]] x = [3, 5]  ⇒  x = [0.8, 1.4]
        let mut a = vec![2.0, 1.0, 1.0, 3.0];
        let mut piv = [0usize; 2];
        assert!(lu_factor(&mut a, 2, &mut piv));
        let mut b = vec![3.0, 5.0];
        lu_solve(&a, 2, &piv, &mut b);
        assert!((b[0] - 0.8).abs() < 1e-14, "x0 = {}", b[0]);
        assert!((b[1] - 1.4).abs() < 1e-14, "x1 = {}", b[1]);
    }

    /// Pivoting must handle a zero leading pivot.
    #[test]
    fn lu_pivots_through_a_zero_leading_entry() {
        // [[0, 1], [1, 0]] x = [2, 3]  ⇒  x = [3, 2]
        let mut a = vec![0.0, 1.0, 1.0, 0.0];
        let mut piv = [0usize; 2];
        assert!(lu_factor(&mut a, 2, &mut piv));
        let mut b = vec![2.0, 3.0];
        lu_solve(&a, 2, &piv, &mut b);
        assert!((b[0] - 3.0).abs() < 1e-14, "x0 = {}", b[0]);
        assert!((b[1] - 2.0).abs() < 1e-14, "x1 = {}", b[1]);
    }

    /// A singular matrix is reported, not silently mis-factored.
    #[test]
    fn lu_reports_singular() {
        let mut a = vec![1.0, 2.0, 2.0, 4.0]; // rank 1
        let mut piv = [0usize; 2];
        assert!(!lu_factor(&mut a, 2, &mut piv));
    }

    /// `I − c·J` has the identity on the diagonal and `−c·J` off it.
    #[test]
    fn shifted_matrix_layout() {
        let jac = vec![1.0, 2.0, 3.0, 4.0];
        let mut mat = vec![0.0; 4];
        build_shifted(&mut mat, &jac, 2, 0.5);
        assert_eq!(mat, vec![1.0 - 0.5, -1.0, -1.5, 1.0 - 2.0]);
    }

    /// Larger random-ish system: factor then solve reproduces the RHS when
    /// multiplied back (a round-trip residual check).
    #[test]
    fn lu_round_trip_3x3() {
        let a0 = [4.0, 3.0, 2.0, 1.0, 5.0, 3.0, 2.0, 1.0, 6.0];
        let x_true = [1.0, -2.0, 0.5];
        // b = A x_true
        let n = 3;
        let mut b = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                b[i] += a0[i * n + j] * x_true[j];
            }
        }
        let mut a = a0.to_vec();
        let mut piv = [0usize; 3];
        assert!(lu_factor(&mut a, n, &mut piv));
        lu_solve(&a, n, &piv, &mut b);
        for i in 0..n {
            assert!((b[i] - x_true[i]).abs() < 1e-12, "x[{i}] = {}", b[i]);
        }
    }
}
