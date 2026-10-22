//! Deterministic, mixed initial tangent frames (mirrored in Python tangent.py).

use crate::rng::SplitMix64;

/// Column-major orthonormal frame with no privileged coordinate direction.
///
/// Identity columns can remain forever inside a contracting invariant subspace,
/// missing a maximal exponent in another component. A fixed, local SplitMix64
/// stream supplies a generic dense frame without changing global RNG state or
/// introducing run-to-run variation. Two-pass modified Gram--Schmidt controls
/// orthogonality even when many directions are requested.
pub(crate) fn initial_frame(dim: usize, k: usize) -> Vec<f64> {
    let mut rng = SplitMix64::new(0x5441_4e47_454e_5453);
    let mut w = vec![0.0; dim * k];
    for j in 0..k {
        for d in 0..dim {
            w[j * dim + d] = 2.0 * rng.next_f64() - 1.0;
        }
        let mut norm = orthogonalise(&mut w, dim, j);
        if norm <= f64::EPSILON {
            // A numerically dependent draw: choose the coordinate least covered
            // by the existing columns, then project it into their complement.
            let mut best = 0;
            let mut least = f64::INFINITY;
            for d in 0..dim {
                let used: f64 = (0..j).map(|i| w[i * dim + d].powi(2)).sum();
                if used < least {
                    least = used;
                    best = d;
                }
            }
            w[j * dim..(j + 1) * dim].fill(0.0);
            w[j * dim + best] = 1.0;
            norm = orthogonalise(&mut w, dim, j);
        }
        for d in 0..dim {
            w[j * dim + d] /= norm;
        }
    }
    w
}

fn orthogonalise(w: &mut [f64], dim: usize, j: usize) -> f64 {
    for _ in 0..2 {
        for i in 0..j {
            let mut dot = 0.0;
            for d in 0..dim {
                dot += w[i * dim + d] * w[j * dim + d];
            }
            for d in 0..dim {
                w[j * dim + d] -= dot * w[i * dim + d];
            }
        }
    }
    let mut norm2 = 0.0;
    for d in 0..dim {
        norm2 += w[j * dim + d] * w[j * dim + d];
    }
    norm2.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_frame_is_reproducible_and_orthonormal() {
        for dim in [1, 2, 3, 10, 64] {
            let w = initial_frame(dim, dim);
            assert_eq!(w, initial_frame(dim, dim));
            assert_eq!(&w[..dim], initial_frame(dim, 1).as_slice());
            for i in 0..dim {
                for j in 0..dim {
                    let dot: f64 = (0..dim).map(|d| w[i * dim + d] * w[j * dim + d]).sum();
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!((dot - expected).abs() < 1e-13);
                }
            }
            assert!(w[..dim].iter().all(|value| *value != 0.0));
        }
    }
}
