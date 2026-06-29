//! Discrete-map Lyapunov-spectrum entry point (stream `perf/map-lyapunov-kernel`).
//!
//! Thin bridge over the engine's [`tsdyn_engine::map_lyapunov`] kernel: build the
//! evaluator (interpreter or JIT) here (the cranelift edge lives in this crate, not
//! the engine), then run the whole QR tangent-map iteration in one call. The map
//! tape must carry its Jacobian (`with_jacobian=True`) — the tangent map needs
//! `∂f/∂u`; a Jacobian-less tape is rejected as the engine error
//! [`MapLyapunovError::NoJacobian`].
//!
//! Driven over the same lowered IR tape, the interpreter and JIT agree bit-for-bit;
//! against the released Python `TangentSystem._accumulate_map` the result differs
//! only by the lowered-IR vs pure-Python `_step`/`_jacobian` float order (the
//! WS-MAPITER caveat) — the same attractor, the same spectrum to tolerance.

use tsdyn_engine::map_lyapunov::{map_lyapunov, MapLyapunovError, MapLyapunovOutcome};
use tsdyn_ir::Tape;

use super::marshal::{build_evaluator, EngineError};

/// Map a [`MapLyapunovError`] (a kernel-side shape / Jacobian / divergence failure)
/// to the bridge's [`EngineError`] so the binding layer routes it to the same
/// Python exception the other entry points use (`ValueError` / `RuntimeError`). The
/// Jacobian-less case carries the actionable "recompile with a Jacobian" message.
fn to_engine_err(e: MapLyapunovError) -> EngineError {
    match e {
        MapLyapunovError::BadShape(m) => EngineError::BadShape(m),
        MapLyapunovError::NoJacobian => EngineError::BadShape(
            "map Lyapunov needs the step Jacobian, but the map tape was compiled without one \
             (with_jacobian=False). Recompile the map problem with a Jacobian."
                .to_string(),
        ),
        MapLyapunovError::Diverged(m) => EngineError::Diverged(m),
    }
}

/// Run the discrete-map QR tangent-map Lyapunov spectrum over a **map** tape.
///
/// Builds the evaluator over `tape` (which must carry its step Jacobian), seeds the
/// deviation frame to the leading `k` identity columns, iterates `steps` map steps
/// propagating the frame through `J(x_n)` (pre-image convention) with a QR
/// reorthonormalisation every `reortho_interval` steps, and returns the `k`
/// time-averaged exponents (largest first) plus the completed-interval count. The
/// map tape folds its parameters in, so `p` is empty. `jit` selects the Cranelift
/// evaluator (numerically identical to the interpreter).
#[allow(clippy::too_many_arguments)]
pub fn map_lyapunov_bridge(
    tape: Tape,
    p: &[f64],
    ic: &[f64],
    steps: usize,
    k: usize,
    reortho_interval: usize,
    jit: bool,
) -> Result<MapLyapunovOutcome, EngineError> {
    let ev = build_evaluator(tape, jit)?;
    map_lyapunov(&*ev, p, ic, steps, k, reortho_interval).map_err(to_engine_err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tsdyn_ir::TapeBuilder;

    /// Hénon `(x, y) ← (1 - a x² + y, b x)` carrying its analytic Jacobian.
    fn henon_jac(a: f64, bcoef: f64) -> Tape {
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let y = b.state(1);
        let one = b.constant(1.0);
        let ac = b.constant(a);
        let bc = b.constant(bcoef);
        let xx = b.mul(x, x);
        let axx = b.mul(ac, xx);
        let omaxx = b.sub(one, axx);
        let nx = b.add(omaxx, y);
        let ny = b.mul(bc, x);
        let two = b.constant(2.0);
        let twoa = b.mul(two, ac);
        let twoax = b.mul(twoa, x);
        let neg_twoax = b.neg(twoax);
        let zero = b.constant(0.0);
        b.finish(&[nx, ny], &[neg_twoax, one, bc, zero], 2, 0)
            .unwrap()
    }

    /// A Jacobian-less map tape (the rejected case).
    fn henon_no_jac(a: f64, bcoef: f64) -> Tape {
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let y = b.state(1);
        let one = b.constant(1.0);
        let ac = b.constant(a);
        let bc = b.constant(bcoef);
        let xx = b.mul(x, x);
        let axx = b.mul(ac, xx);
        let omaxx = b.sub(one, axx);
        let nx = b.add(omaxx, y);
        let ny = b.mul(bc, x);
        b.finish(&[nx, ny], &[], 2, 0).unwrap()
    }

    #[test]
    fn bridge_interp_equals_jit_bit_for_bit() {
        // The bridge drives both evaluators over the same lowered tape; the
        // iteration is deterministic and eval/eval_jac are bit-identical between
        // interpreter and JIT, so the exponents agree to the last bit.
        let run = |jit| {
            map_lyapunov_bridge(henon_jac(1.4, 0.3), &[], &[0.1, 0.1], 5000, 2, 1, jit)
                .unwrap()
                .exponents
        };
        let interp = run(false);
        let jit = run(true);
        assert_eq!(interp.len(), jit.len());
        for (a, b) in interp.iter().zip(jit.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "interp {a} != jit {b}");
        }
    }

    #[test]
    fn bridge_reproduces_henon_literature() {
        let out = map_lyapunov_bridge(henon_jac(1.4, 0.3), &[], &[0.1, 0.1], 10_000, 2, 1, false)
            .unwrap();
        assert!(
            (out.exponents[0] - 0.419).abs() < 0.05,
            "{:?}",
            out.exponents
        );
        assert!(
            (out.exponents[1] - (-1.623)).abs() < 0.05,
            "{:?}",
            out.exponents
        );
    }

    #[test]
    fn bridge_rejects_jacobianless_tape() {
        let err = map_lyapunov_bridge(henon_no_jac(1.4, 0.3), &[], &[0.1, 0.1], 100, 2, 1, false)
            .unwrap_err();
        assert!(matches!(err, EngineError::BadShape(_)), "got {err:?}");
    }
}
