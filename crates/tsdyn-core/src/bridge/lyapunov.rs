//! ODE Lyapunov-spectrum entry point (stream `perf/ode-lyapunov-engine`).
//!
//! Thin bridge over the engine's [`tsdyn_engine::lyapunov`] kernel: build the
//! evaluator (interpreter or JIT) over the **extended** variational tape + resolve
//! the solver here (the cranelift edge lives in this crate, not the engine), then
//! run the whole burn-in + averaging Benettin loop in one call. The per-`dt`
//! integration is byte-for-byte the released `TangentSystem` ODE numerics, so
//! `interp == jit` and the spectrum matches the Python path to floating-point
//! tolerance.
//!
//! Shared plumbing (evaluator/solver builders, the `EngineError` enum) lives in
//! [`super::marshal`]; the renormalisation loop and the QR live in
//! [`tsdyn_engine::lyapunov`].

use tsdyn_engine::lyapunov::{lyapunov_spectrum_ode, LyapunovError, LyapunovOutcome};
use tsdyn_ir::Tape;
use tsdyn_solvers::Solver;

use super::marshal::{
    build_evaluator, build_solver, require_jacobian_if_needed, resolve_solver, EngineError,
};

/// Map a [`LyapunovError`] (a kernel-side shape / divergence failure) to the
/// bridge's [`EngineError`] so the binding routes it to the matching Python
/// exception (`ValueError` / `RuntimeError`).
fn to_engine_err(e: LyapunovError) -> EngineError {
    match e {
        LyapunovError::BadShape(m) => EngineError::BadShape(m),
        LyapunovError::Diverged(m) => EngineError::Diverged(m),
    }
}

/// Run the burn-in + averaging Benettin Lyapunov-spectrum loop over an
/// **extended variational** ODE tape.
///
/// The `tape` is the `dim*(k+1)`-wide extended variational system (base RHS ⊕ the
/// `k` tangent-vector RHS blocks) built by the Python
/// `tsdynamics.derived._variational.build_variational_tape`. `method` resolves
/// through the solver registry; an implicit kernel needs a Jacobian-carrying tape
/// (the extended tape carries one — rejected up front exactly as the integrate
/// path does, for the defence-in-depth message). `dt` is the renormalisation
/// interval, `burn_in` / `final_time` the two window lengths. `p` is the live
/// control-parameter vector. Returns the spectrum plus the final extended state.
#[allow(clippy::too_many_arguments)]
pub fn lyapunov_spectrum_ode_bridge(
    tape: Tape,
    p: &[f64],
    method: &str,
    rtol: f64,
    atol: f64,
    dim: usize,
    k: usize,
    z0: &[f64],
    t0: f64,
    dt: f64,
    burn_in: f64,
    final_time: f64,
    jit: bool,
) -> Result<LyapunovOutcome, EngineError> {
    // The extended tape is `n_state == ev.dim()` (a plain ODE-shaped tape over the
    // stacked state), validated by the kernel against `dim*(k+1)`. Resolve the
    // method + the Jacobian guard exactly like the integrate / basin paths.
    let name = resolve_solver(method)?;
    require_jacobian_if_needed(&tape, name)?;
    let ev = build_evaluator(tape, jit)?;
    // A fresh kernel per `dt` chunk (mirrors `OdeStepper::advance` / the basin
    // march); the name is a registry name, so `build_solver` always succeeds.
    let factory = move || -> Box<dyn Solver> { build_solver(name, rtol, atol) };
    lyapunov_spectrum_ode(
        &*ev, factory, p, dim, k, z0, t0, dt, burn_in, final_time,
    )
    .map_err(to_engine_err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tsdyn_ir::TapeBuilder;

    /// Extended variational tape of `dx = a x, dy = b y` with `k` tangents (the
    /// Jacobian is the constant `diag(a, b)`). Spectrum is `[max(a,b), min(a,b)]`.
    fn linear_extended(a: f64, b: f64, k: usize) -> Tape {
        let dim = 2;
        let mut bld = TapeBuilder::new();
        let x = bld.state(0);
        let y = bld.state(1);
        let ac = bld.constant(a);
        let bc = bld.constant(b);
        let dx = bld.mul(ac, x);
        let dy = bld.mul(bc, y);
        let mut outs = vec![dx, dy];
        for i in 0..k {
            let base = dim + i * dim;
            let w0 = bld.state(base);
            let w1 = bld.state(base + 1);
            let dw0 = bld.mul(ac, w0);
            let dw1 = bld.mul(bc, w1);
            outs.push(dw0);
            outs.push(dw1);
        }
        let n = dim * (k + 1);
        bld.finish(&outs, &[], n, 0).unwrap()
    }

    #[test]
    fn bridge_spectrum_matches_analytic_and_interp_equals_jit() {
        let (a, b, k) = (0.5, -2.0, 2);
        let z0 = vec![1.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let run = |jit| {
            lyapunov_spectrum_ode_bridge(
                linear_extended(a, b, k),
                &[],
                "rk45",
                1e-9,
                1e-11,
                2,
                k,
                &z0,
                0.0,
                0.1,
                5.0,
                50.0,
                jit,
            )
            .unwrap()
        };
        let interp = run(false);
        let jit = run(true);
        // interp == jit bit-for-bit (both drive the same &dyn Evaluator).
        assert_eq!(interp.spectrum.len(), jit.spectrum.len());
        for (i, (x, y)) in interp.spectrum.iter().zip(jit.spectrum.iter()).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "exponent {i}: jit {y} != interp {x}");
        }
        // and matches the analytic spectrum.
        assert!((interp.spectrum[0] - 0.5).abs() < 1e-4);
        assert!((interp.spectrum[1] - (-2.0)).abs() < 1e-4);
    }

    #[test]
    fn bridge_rejects_unknown_method() {
        let z0 = vec![1.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let err = lyapunov_spectrum_ode_bridge(
            linear_extended(0.5, -2.0, 2),
            &[],
            "no-such",
            1e-6,
            1e-9,
            2,
            2,
            &z0,
            0.0,
            0.1,
            1.0,
            1.0,
            false,
        )
        .unwrap_err();
        assert!(matches!(err, EngineError::UnknownMethod(_)), "got {err:?}");
    }
}
