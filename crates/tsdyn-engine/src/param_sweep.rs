//! Map orbit-diagram parameter sweep — the entire sweep in one engine call
//! (stream `perf/param-sweep-kernel`).
//!
//! [`map_orbit_sweep`] runs the whole `orbit_diagram` parameter sweep of a
//! discrete map — for every value of one control parameter, iterate `transient`
//! steps then record `n` asymptotic states — inside a single engine call, instead
//! of the released path's one `iterate` FFI round-trip *per parameter value* (a
//! 1000-value logistic sweep is ~1000 round-trips, the named bottleneck this
//! stream removes).
//!
//! # How the swept parameter reaches the kernel
//!
//! A lowered map normally folds *all* its parameters into the tape as constants
//! (`n_param == 0`). To vary one parameter without re-lowering, the Python wiring
//! lowers the map **once** keeping the swept parameter as the tape's single
//! runtime `Param` input (`n_param == 1`); the other parameters stay folded. This
//! kernel then sets `p[sweep_index] = value` for each swept value and drives the
//! shared evaluator — so the per-iterate numerics are byte-for-byte the released
//! [`iterate_dense`](crate::map::iterate_dense) path (a `Param` leaf reads the
//! same `f64` a folded `Const` would, then feeds the same op).
//!
//! # Semantics (mirroring the Python per-value loop exactly)
//!
//! * **Output convention.** As [`iterate_dense`](crate::map::iterate_dense), each
//!   recorded iterate is *after* the initial condition: a value runs
//!   `transient + n` applications of `f` and records the last `n`.
//! * **`carry_state`.** With `carry_state = true`, value `k + 1` starts from the
//!   final state value `k` reached (the attractor-following sweep); with `false`,
//!   every value restarts from the base initial condition `ic`. A value that
//!   diverged resets the carry to `ic` for the next value — exactly the Python
//!   `state = None` reset.
//! * **Divergence.** A non-finite iterate at any point in a value's
//!   `transient + n` run marks that value [`SweepStatus::Diverged`]; its recorded
//!   rows are left zero (the binding turns the status into an empty point set and
//!   a `RuntimeWarning`, as the per-value path does) and the next value restarts
//!   from `ic`.
//!
//! # Determinism / single-threaded
//!
//! The sweep is strictly sequential: with `carry_state` a value depends on the
//! previous value's final state, so it cannot be parallelised, and even without
//! it the per-value work is tiny (the win is eliminating the FFI/Python per value,
//! not threads). A repeat on the same inputs returns byte-identical output.

use tsdyn_ir::Evaluator;

/// Why a parameter sweep could not be set up (a shape validation failure). The
/// sweep itself never errors — a diverged value is a [`SweepStatus::Diverged`]
/// marker, not an error — so this only covers caller-side mistakes the binding
/// maps to `ValueError`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SweepError {
    /// A buffer length / index invariant disagrees with the tape or itself.
    BadShape(String),
}

impl core::fmt::Display for SweepError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SweepError::BadShape(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for SweepError {}

/// The fate of one parameter value in a sweep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SweepStatus {
    /// The value ran `transient + n` finite iterates; its `n` recorded rows are
    /// the asymptotic states.
    Ok,
    /// A non-finite iterate occurred during this value's run; its recorded rows
    /// are zero (the binding drops them to an empty set) and the carry state was
    /// reset to `ic`.
    Diverged,
}

impl SweepStatus {
    /// Whether the value completed its full `transient + n` run finitely.
    pub fn is_ok(&self) -> bool {
        matches!(self, SweepStatus::Ok)
    }
}

/// The recorded asymptotic states + per-value fate a [`map_orbit_sweep`] returns.
///
/// `points` is a row-major `(n_values * n, n_components)` buffer: the block for
/// value `k` is rows `k*n .. (k+1)*n`, and column `j` is recorded component
/// `components[j]`. A diverged value's block is all-zero; `status[k]` tells the
/// binding which blocks to keep.
#[derive(Clone, Debug)]
pub struct SweepOutcome {
    /// Row-major `(n_values * n, n_components)` recorded states.
    pub points: Vec<f64>,
    /// Per-value fate, length `n_values`, in sweep order.
    pub status: Vec<SweepStatus>,
    /// Recorded rows per value (`n`) — the block stride.
    pub n_record: usize,
    /// Recorded components per row.
    pub n_components: usize,
}

/// Run the whole map orbit-diagram parameter sweep in one call.
///
/// `ev` is the lowered map evaluator whose single runtime `Param` (`n_param == 1`)
/// is the swept parameter; `base_params` is the full runtime parameter vector
/// (`base_params.len() == ev.n_param()`), of which `sweep_index` is overwritten
/// with each swept value. `values` are the parameter values in sweep order; `ic`
/// the base initial condition; `components` the (validated) state-component
/// indices to record; `transient` / `n_record` the discard / record counts; and
/// `carry_state` whether each value resumes from the previous value's final
/// state.
///
/// Returns a [`SweepOutcome`]: the recorded asymptotic states for every value plus
/// each value's fate. A diverged value records a zero block and resets the carry
/// to `ic`, never aborting the sweep — the per-value `RuntimeWarning` contract is
/// reproduced by the binding from the status.
#[allow(clippy::too_many_arguments)]
pub fn map_orbit_sweep(
    ev: &dyn Evaluator,
    base_params: &[f64],
    sweep_index: usize,
    values: &[f64],
    ic: &[f64],
    components: &[usize],
    transient: usize,
    n_record: usize,
    carry_state: bool,
) -> Result<SweepOutcome, SweepError> {
    let dim = ev.dim();
    if dim == 0 {
        return Err(SweepError::BadShape("system dimension is zero".to_string()));
    }
    if ic.len() < dim {
        return Err(SweepError::BadShape(format!(
            "initial condition has length {}, need dim = {dim}",
            ic.len()
        )));
    }
    if base_params.len() != ev.n_param() {
        return Err(SweepError::BadShape(format!(
            "base parameter vector has length {}, need n_param = {}",
            base_params.len(),
            ev.n_param()
        )));
    }
    if sweep_index >= ev.n_param() {
        return Err(SweepError::BadShape(format!(
            "swept parameter index {sweep_index} is out of range for n_param = {}",
            ev.n_param()
        )));
    }
    for &c in components {
        if c >= dim {
            return Err(SweepError::BadShape(format!(
                "recorded component index {c} is out of range for dim = {dim}"
            )));
        }
    }

    let n_values = values.len();
    let n_components = components.len();
    let mut points = vec![0.0; n_values * n_record * n_components];
    let mut status = vec![SweepStatus::Ok; n_values];

    // Reusable buffers (the loop allocates nothing per value/iterate).
    let mut p = base_params.to_vec();
    let mut scratch = vec![0.0; ev.n_scratch()];
    let mut cur = vec![0.0; dim];
    let mut next = vec![0.0; dim];

    // The carry-over state across values (the previous value's final state); seeded
    // from `ic` and reset to `ic` after a divergence — the Python `state`/`None`.
    let mut carry = ic[..dim].to_vec();

    for (k, &value) in values.iter().enumerate() {
        p[sweep_index] = value;
        // Start: the carried state (carry_state) or always the base ic.
        if carry_state {
            cur.copy_from_slice(&carry);
        } else {
            cur.copy_from_slice(&ic[..dim]);
        }

        let block = &mut points[k * n_record * n_components..(k + 1) * n_record * n_components];
        let mut diverged = false;
        let total = transient + n_record;
        for it in 0..total {
            // One map application; t = 0.0 (maps are autonomous in the IR), matching
            // the engine map loop and the reference path.
            ev.eval(&cur, &p, 0.0, &mut scratch, &mut next);
            if !next.iter().all(|x| x.is_finite()) {
                diverged = true;
                break;
            }
            cur.copy_from_slice(&next);
            // Record the post-transient iterates into this value's block.
            if it >= transient {
                let row = it - transient;
                let dst = &mut block[row * n_components..(row + 1) * n_components];
                for (j, &comp) in components.iter().enumerate() {
                    dst[j] = cur[comp];
                }
            }
        }

        if diverged {
            status[k] = SweepStatus::Diverged;
            // The diverged value's block stays zero (the binding drops it to an
            // empty set); the next value restarts from `ic`.
            carry.copy_from_slice(&ic[..dim]);
        } else if carry_state {
            // Carry the final state forward. With `transient + n == 0` no iterate
            // ran, so the final state is the (unchanged) starting state — exactly
            // the Python loop's `current.state()` after taking no steps.
            carry.copy_from_slice(&cur);
        }
    }

    Ok(SweepOutcome {
        points,
        status,
        n_record,
        n_components,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::iterate_dense;
    use crate::testkit::VmEval;
    use tsdyn_ir::TapeBuilder;
    use tsdyn_vm::Interpreter;

    /// Logistic map `x ← r x (1 - x)` with `r` kept as the single runtime `Param`
    /// (`n_param == 1`) — the sweep-lowering convention.
    fn logistic_param() -> Interpreter {
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let r = b.param(0);
        let one = b.constant(1.0);
        let omx = b.sub(one, x);
        let rx = b.mul(r, x);
        let nx = b.mul(rx, omx);
        Interpreter::new(b.finish(&[nx], &[], 1, 1).unwrap())
    }

    /// Logistic map with `r` folded in as a constant (the released per-value
    /// lowering, `n_param == 0`) — the byte-identity oracle.
    fn logistic_const(r: f64) -> Interpreter {
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let rc = b.constant(r);
        let one = b.constant(1.0);
        let omx = b.sub(one, x);
        let rx = b.mul(rc, x);
        let nx = b.mul(rx, omx);
        Interpreter::new(b.finish(&[nx], &[], 1, 0).unwrap())
    }

    #[test]
    fn sweep_matches_per_value_iterate_byte_for_bit_no_carry() {
        // With carry_state = false every value restarts from `ic`, so each value's
        // recorded block must equal a per-value `iterate_dense` (with r folded in)
        // sliced to the recorded tail — bit-for-bit.
        let ev = VmEval::new(logistic_param());
        let values: Vec<f64> = (0..50).map(|i| 2.5 + 0.03 * i as f64).collect();
        let (transient, n) = (200usize, 64usize);
        let ic = [0.3];
        let out = map_orbit_sweep(&ev, &[0.0], 0, &values, &ic, &[0], transient, n, false).unwrap();
        assert_eq!(out.points.len(), values.len() * n);
        for (k, &r) in values.iter().enumerate() {
            let oracle = VmEval::new(logistic_const(r));
            let dense = iterate_dense(&oracle, &ic, &[], transient + n).unwrap();
            let tail = &dense[transient..transient + n];
            let block = &out.points[k * n..(k + 1) * n];
            for (i, (&a, &b)) in block.iter().zip(tail.iter()).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "value {k} row {i}: {a} != {b}");
            }
        }
    }

    #[test]
    fn sweep_carry_state_matches_per_value_iterate_byte_for_bit() {
        // With carry_state = true value k+1 starts from value k's final state. The
        // oracle threads that final state forward by hand, folding r in per value.
        let ev = VmEval::new(logistic_param());
        let values: Vec<f64> = (0..40).map(|i| 2.8 + 0.03 * i as f64).collect();
        let (transient, n) = (150usize, 48usize);
        let ic = [0.123];
        let out = map_orbit_sweep(&ev, &[0.0], 0, &values, &ic, &[0], transient, n, true).unwrap();

        let mut carry = ic.to_vec();
        for (k, &r) in values.iter().enumerate() {
            let oracle = VmEval::new(logistic_const(r));
            let dense = iterate_dense(&oracle, &carry, &[], transient + n).unwrap();
            let tail = &dense[transient..transient + n];
            let block = &out.points[k * n..(k + 1) * n];
            for (i, (&a, &b)) in block.iter().zip(tail.iter()).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "value {k} row {i}: {a} != {b}");
            }
            // Final state of this value carries to the next.
            carry = dense[(transient + n - 1)..(transient + n)].to_vec();
            assert!(out.status[k].is_ok());
        }
    }

    #[test]
    fn divergent_value_is_marked_and_resets_carry() {
        // r = 4.5 escapes [0, 1] from x0 = 0.5 → diverges; the surrounding finite
        // values record normally, and the diverged block stays zero.
        let ev = VmEval::new(logistic_param());
        let values = [3.7, 4.5, 3.2];
        let (transient, n) = (80usize, 40usize);
        let ic = [0.5];
        let out = map_orbit_sweep(&ev, &[0.0], 0, &values, &ic, &[0], transient, n, true).unwrap();
        assert!(out.status[0].is_ok());
        assert_eq!(out.status[1], SweepStatus::Diverged);
        assert!(out.status[2].is_ok());
        // The diverged value's block is left zero.
        let block1 = &out.points[n..2 * n];
        assert!(block1.iter().all(|&x| x == 0.0));
        // Value 2 restarts from ic (carry reset), so it equals a fresh per-value run.
        let oracle = VmEval::new(logistic_const(3.2));
        let dense = iterate_dense(&oracle, &ic, &[], transient + n).unwrap();
        let tail = &dense[transient..transient + n];
        let block2 = &out.points[2 * n..3 * n];
        for (&a, &b) in block2.iter().zip(tail.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn zero_transient_and_n_records_empty_blocks() {
        // transient + n == 0: no iterate runs; each value records a zero-row block.
        let ev = VmEval::new(logistic_param());
        let values = [3.2, 3.5, 3.8];
        let out = map_orbit_sweep(&ev, &[0.0], 0, &values, &[0.3], &[0], 0, 0, true).unwrap();
        assert!(out.points.is_empty());
        assert!(out.status.iter().all(|s| s.is_ok()));
    }

    #[test]
    fn rejects_out_of_range_sweep_index_and_component() {
        let ev = VmEval::new(logistic_param());
        let bad_idx =
            map_orbit_sweep(&ev, &[0.0], 5, &[3.5], &[0.3], &[0], 10, 10, true).unwrap_err();
        assert!(matches!(bad_idx, SweepError::BadShape(_)));
        let bad_comp =
            map_orbit_sweep(&ev, &[0.0], 0, &[3.5], &[0.3], &[7], 10, 10, true).unwrap_err();
        assert!(matches!(bad_comp, SweepError::BadShape(_)));
    }

    #[test]
    fn multi_component_records_selected_columns() {
        // A 2-D map: record both components, swapped order, and check each column.
        // (x, y) ← (a x, b y) with a kept runtime (param 0), b folded in.
        let mut bld = TapeBuilder::new();
        let x = bld.state(0);
        let y = bld.state(1);
        let a = bld.param(0);
        let bc = bld.constant(0.5);
        let nx = bld.mul(a, x);
        let ny = bld.mul(bc, y);
        let ev = VmEval::new(Interpreter::new(bld.finish(&[nx, ny], &[], 2, 1).unwrap()));
        // a < 1 so x contracts to 0; record (y, x) order to check column mapping.
        let out =
            map_orbit_sweep(&ev, &[0.0], 0, &[0.5], &[1.0, 1.0], &[1, 0], 0, 3, false).unwrap();
        assert_eq!(out.n_components, 2);
        // Row r: x = 0.5^{r+1}, y = 0.5^{r+1}; recorded as (y, x).
        for r in 0..3 {
            let want = 0.5_f64.powi(r as i32 + 1);
            let yv = out.points[r * 2];
            let xv = out.points[r * 2 + 1];
            assert!((yv - want).abs() < 1e-15, "row {r} y: {yv} vs {want}");
            assert!((xv - want).abs() < 1e-15, "row {r} x: {xv} vs {want}");
        }
    }
}
