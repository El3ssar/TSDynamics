//! Dense linear solves for Rosenbrock `W` matrices (small `n`, no BLAS).

/// Gauss–Jordan elimination with partial pivoting.
/// Overwrites the `n × n` row-major matrix `mat` and rhs vector `rhs` (`mat x = rhs`).
pub fn solve_linear_row_major(mat: &mut [f64], n: usize, rhs: &mut [f64]) -> Result<(), ()> {
    debug_assert!(mat.len() >= n * n);
    debug_assert!(rhs.len() >= n);

    const THRESH: f64 = 1e-18;

    for k in 0..n {
        let mut piv = k;
        let mut maxv = mat[k * n + k].abs();
        for i in k..n {
            let v = mat[i * n + k].abs();
            if v > maxv {
                maxv = v;
                piv = i;
            }
        }
        if maxv < THRESH {
            return Err(());
        }
        if piv != k {
            for j in 0..n {
                mat.swap(k * n + j, piv * n + j);
            }
            rhs.swap(k, piv);
        }

        let diag = mat[k * n + k];
        for j in 0..n {
            mat[k * n + j] /= diag;
        }
        rhs[k] /= diag;

        for i in 0..n {
            if i == k {
                continue;
            }
            let f = mat[i * n + k];
            if f != 0.0 {
                for j in 0..n {
                    mat[i * n + j] -= f * mat[k * n + j];
                }
                rhs[i] -= f * rhs[k];
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_2x2() {
        let mut a = vec![2.0_f64, 1.0, 1.0, 3.0];
        let mut b = vec![1.0_f64, 2.0];
        solve_linear_row_major(&mut a, 2, &mut b).unwrap();
        // x ≈ 0.2, y ≈ 0.6
        assert!((b[0] - 0.2).abs() < 1e-12);
        assert!((b[1] - 0.6).abs() < 1e-12);
    }

    #[test]
    fn solve_identity_3x3() {
        let base = vec![
            1.0_f64, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0,
        ];
        let mut a = base.clone();
        let mut b = vec![5.0_f64, -3.0, 2.5];
        solve_linear_row_major(&mut a, 3, &mut b).unwrap();
        assert!((b[0] - 5.0).abs() < 1e-12);
        assert!((b[1] + 3.0).abs() < 1e-12);
        assert!((b[2] - 2.5).abs() < 1e-12);
    }
}
