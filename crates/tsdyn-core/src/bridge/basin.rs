//! Basin / attractor recurrence-FSM entry points (stream `perf/basin-march`).
//!
//! Thin bridge over the engine's [`tsdyn_engine::basin`] kernel: build the
//! evaluator (interpreter or JIT) + resolve the solver here (the cranelift edge
//! lives in this crate, not the engine), then drive the sequential FSM over every
//! seed in one call. The numerics per cell-check are byte-for-byte the per-`dt`
//! `ContinuousSystem.step()` (flows) / one map iteration (maps), so the located
//! labels are bit-identical to the Python `_AttractorMapper`.
//!
//! Shared plumbing (evaluator/solver builders, the `EngineError` enum) lives in
//! [`super::marshal`]; the FSM, the cell grid and the validation live in
//! [`tsdyn_engine::basin`].

use tsdyn_engine::basin::{
    basin_march_flow, basin_march_map, BasinError, BasinMarchOutcome, MarchConfig,
};
use tsdyn_ir::Tape;
use tsdyn_solvers::Solver;

use super::marshal::{
    build_evaluator, build_solver, guard_continuous, require_jacobian_if_needed, resolve_solver,
    EngineError,
};

/// Map a [`BasinError`] (a kernel-side shape/grid failure) to the bridge's
/// [`EngineError`] so the binding layer routes it to the same Python exception the
/// other entry points use (`ValueError`).
fn to_engine_err(e: BasinError) -> EngineError {
    match e {
        BasinError::BadShape(m) => EngineError::BadShape(m),
    }
}

/// Run the basin recurrence FSM over a **flow** (ODE) tape.
///
/// Builds the evaluator over `tape`, resolves `method` (an implicit kernel needs a
/// Jacobian-carrying tape, rejected up front exactly as the integrate path does),
/// and marches every seed sequentially advancing one `dt` per cell check. `cfg`
/// carries the six FSM thresholds. Returns the per-seed labels + the accumulated
/// `att_cells`/`bas_cells`/`att_points` the Python wiring rebuilds the mapper from.
#[allow(clippy::too_many_arguments)]
pub fn basin_march_flow_bridge(
    tape: Tape,
    p: &[f64],
    method: &str,
    rtol: f64,
    atol: f64,
    dt: f64,
    grid_lo: &[f64],
    grid_hi: &[f64],
    grid_counts: &[i64],
    seeds: &[f64],
    cfg: MarchConfig,
    jit: bool,
) -> Result<BasinMarchOutcome, EngineError> {
    // An ODE-shaped tape only (a DDE/SDE tape has its own family path; the Python
    // wiring already excludes DDE/SDE, but guard at the boundary too).
    guard_continuous(&tape)?;
    let name = resolve_solver(method)?;
    require_jacobian_if_needed(&tape, name)?;
    let ev = build_evaluator(tape, jit)?;
    // A fresh kernel per `dt` segment (mirrors `OdeStepper::advance`); the name is
    // a registry name, so `build_solver` always succeeds.
    let factory = move || -> Box<dyn Solver> { build_solver(name, rtol, atol) };
    basin_march_flow(
        &*ev,
        factory,
        p,
        dt,
        grid_lo,
        grid_hi,
        grid_counts,
        seeds,
        cfg,
    )
    .map_err(to_engine_err)
}

/// Run the basin recurrence FSM over a **map** tape.
///
/// As [`basin_march_flow_bridge`], but each cell check applies the lowered `_step`
/// map once (no solver, no `dt`). The map tape folds its parameters in, so `p` is
/// empty.
#[allow(clippy::too_many_arguments)]
pub fn basin_march_map_bridge(
    tape: Tape,
    p: &[f64],
    grid_lo: &[f64],
    grid_hi: &[f64],
    grid_counts: &[i64],
    seeds: &[f64],
    cfg: MarchConfig,
    jit: bool,
) -> Result<BasinMarchOutcome, EngineError> {
    let ev = build_evaluator(tape, jit)?;
    basin_march_map(&*ev, p, grid_lo, grid_hi, grid_counts, seeds, cfg).map_err(to_engine_err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tsdyn_ir::TapeBuilder;

    /// `x ← a x − x³`, `a` folded in (two fixed points at `±√(a−1)` for `1<a<2`).
    fn cubic_map(a: f64) -> Tape {
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let ac = b.constant(a);
        let ax = b.mul(ac, x);
        let x2 = b.mul(x, x);
        let x3 = b.mul(x2, x);
        let nx = b.sub(ax, x3);
        b.finish(&[nx], &[], 1, 0).unwrap()
    }

    fn cfg() -> MarchConfig {
        MarchConfig {
            max_steps: 500,
            mx_fnd: 8,
            mx_loc: 5,
            mx_att: 2,
            mx_bas: 10,
            mx_lost: 20,
        }
    }

    #[test]
    fn map_bridge_interp_equals_jit_bit_for_bit() {
        // The bridge drives both evaluators; the FSM is deterministic and the
        // per-step numerics are bit-identical, so interp == jit to the bit.
        let seeds: Vec<f64> = (0..30).map(|i| -1.0 + 0.06 * i as f64).collect();
        let run = |jit| {
            basin_march_map_bridge(
                cubic_map(1.5),
                &[],
                &[-2.0],
                &[2.0],
                &[300],
                &seeds,
                cfg(),
                jit,
            )
            .unwrap()
            .labels
        };
        assert_eq!(run(false), run(true));
    }

    #[test]
    fn flow_bridge_decays_to_origin_and_interp_equals_jit() {
        // dx=-x, dv=-v spirals to the origin; one attractor region. interp == jit.
        let build = || {
            let mut b = TapeBuilder::new();
            let x = b.state(0);
            let v = b.state(1);
            let dx = b.neg(x);
            let dv = b.neg(v);
            b.finish(&[dx, dv], &[], 2, 0).unwrap()
        };
        let seeds = vec![0.5, 0.5, -0.5, -0.5, 0.3, -0.4];
        let mcfg = MarchConfig {
            max_steps: 2000,
            mx_fnd: 20,
            mx_loc: 10,
            mx_att: 2,
            mx_bas: 10,
            mx_lost: 20,
        };
        let run = |jit| {
            basin_march_flow_bridge(
                build(),
                &[],
                "rk45",
                1e-6,
                1e-9,
                0.2,
                &[-1.0, -1.0],
                &[1.0, 1.0],
                &[50, 50],
                &seeds,
                mcfg,
                jit,
            )
            .unwrap()
            .labels
        };
        let interp = run(false);
        let jit = run(true);
        assert_eq!(interp, jit, "interp vs jit basin labels differ");
        // No seed diverged.
        assert!(
            interp.iter().all(|&l| l >= 1),
            "a seed diverged: {interp:?}"
        );
    }

    #[test]
    fn flow_bridge_rejects_implicit_without_jacobian() {
        // An implicit kernel over a tape with no Jacobian is rejected at the
        // boundary (BadShape → ValueError), exactly as the integrate path does.
        let build = || {
            let mut b = TapeBuilder::new();
            let x = b.state(0);
            let dx = b.neg(x);
            b.finish(&[dx], &[], 1, 0).unwrap()
        };
        let err = basin_march_flow_bridge(
            build(),
            &[],
            "bdf",
            1e-6,
            1e-9,
            0.1,
            &[-1.0],
            &[1.0],
            &[10],
            &[0.5],
            cfg(),
            false,
        )
        .unwrap_err();
        assert!(matches!(err, EngineError::BadShape(_)), "got {err:?}");
    }
}
