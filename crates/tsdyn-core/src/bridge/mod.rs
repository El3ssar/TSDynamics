//! The pure-Rust engine bridge — everything `tsdynamics._rust` does, expressed
//! over plain Rust types so it is unit-testable with `cargo test` (no Python).
//!
//! The thin PyO3 layer in [`crate`] (`lib.rs`) only marshals NumPy arrays to/from
//! slices, releases the GIL, and forwards to the functions here. Keeping the
//! numerics Python-free means a transcription error in the integrate dispatch or
//! the solver/tolerance resolution fails a fast Rust unit test rather than only a
//! slower end-to-end pytest.
//!
//! # Module layout (bridge-split reorg)
//!
//! This was a single 2900-line `bridge.rs`; it is now split by concern into
//! submodules, with **no behaviour change** — every function and the
//! [`OdeStepper`] handle moved verbatim and is re-exported here, so `lib.rs` (and
//! the test suite) see exactly the same `bridge::*` surface as before:
//!
//! - [`marshal`] — the shared FFI plumbing: the [`EngineError`] enum, the
//!   [`VmEvaluator`] adapter + evaluator builders, [`build_tape`] ingestion,
//!   [`resolve_solver`] / [`build_solver`], the input/grid/shape guards and the
//!   divergence-message helpers. Every family entry point is built on it.
//! - [`ode`] — [`eval_rhs`] / [`eval_jac`] / [`integrate_dense`] /
//!   [`ensemble_final`].
//! - [`dde`] — [`integrate_dde_dense`] (method of steps).
//! - [`sde`] — [`sde_integrate_dense`] / [`sde_ensemble_final`] (diagonal-Itô,
//!   drift + diffusion tapes).
//! - [`map`] — [`iterate_map`] / [`map_ensemble_final`].
//! - [`events`] — [`integrate_events_dense`] (crossings of one event over a span).
//! - [`stepper`] — the resumable [`OdeStepper`] handle (stream WS-STEPPER).
//!
//! Errors surface as [`EngineError`], a Python-free enum the binding layer maps
//! onto the right Python exception type.

pub mod dde;
pub mod events;
pub mod map;
pub mod marshal;
pub mod ode;
pub mod sde;
pub mod stepper;

// Re-export the full bridge surface at `crate::bridge::*`, exactly as the
// pre-split single-file module exposed it. `lib.rs` and the unit tests below
// depend on this flat surface; keeping it here means the file split is invisible
// to every caller.
pub use dde::integrate_dde_dense;
pub use events::integrate_events_dense;
pub use map::{iterate_map, map_ensemble_final};
pub use marshal::{build_tape, EngineError};
pub use ode::{ensemble_final, eval_jac, eval_rhs, integrate_dense};
pub use sde::{sde_ensemble_final, sde_integrate_dense};
pub use stepper::OdeStepper;

#[cfg(test)]
mod tests {
    use super::*;
    // build_solver / resolve_solver are consumed directly from marshal by the
    // family submodules (not via the crate::bridge re-export), so the tests
    // import them straight from marshal too rather than widening the public
    // re-export with test-only names.
    use super::marshal::{build_solver, resolve_solver};
    use tsdyn_ir::Tape;
    use tsdyn_ir::TapeBuilder;

    /// dx/dt = -k x ⇒ x(t) = x0 e^{-k t}; one parameter, one state. No Jacobian
    /// (jac_outputs empty), so `has_jacobian()` is false — the case an implicit
    /// kernel must refuse.
    fn decay_tape() -> Tape {
        let mut b = TapeBuilder::new();
        let k = b.param(0);
        let x = b.state(0);
        let kx = b.mul(k, x);
        let dx = b.neg(kx);
        b.finish(&[dx], &[], 1, 1).unwrap()
    }

    /// The same decay system carrying its analytic Jacobian ∂(dx)/∂x = −k, so the
    /// implicit kernels have the `∂f/∂u` they require.
    fn decay_tape_jac() -> Tape {
        let mut b = TapeBuilder::new();
        let k = b.param(0);
        let x = b.state(0);
        let kx = b.mul(k, x);
        let dx = b.neg(kx);
        let neg_k = b.neg(k);
        b.finish(&[dx], &[neg_k], 1, 1).unwrap()
    }

    /// Undamped oscillator dx=v, dv=-x ⇒ (cos t, -sin t) from (1, 0).
    fn oscillator_tape() -> Tape {
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let v = b.state(1);
        let dv = b.neg(x);
        b.finish(&[v, dv], &[], 2, 0).unwrap()
    }

    /// dx/dt = x² ⇒ finite-time blow-up at t = 1 from x0 = 1.
    fn blowup_tape() -> Tape {
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let dx = b.mul(x, x);
        b.finish(&[dx], &[], 1, 0).unwrap()
    }

    /// Lorenz with its analytic Jacobian (for eval_jac).
    fn lorenz_tape() -> Tape {
        let mut b = TapeBuilder::new();
        let (sg, rho, bt) = (b.param(0), b.param(1), b.param(2));
        let (x, y, z) = (b.state(0), b.state(1), b.state(2));
        let ymx = b.sub(y, x);
        let dx = b.mul(sg, ymx);
        let rmz = b.sub(rho, z);
        let xrmz = b.mul(x, rmz);
        let dy = b.sub(xrmz, y);
        let xy = b.mul(x, y);
        let bz = b.mul(bt, z);
        let dz = b.sub(xy, bz);
        let neg_sg = b.neg(sg);
        let zero = b.constant(0.0);
        let neg_one = b.constant(-1.0);
        let neg_x = b.neg(x);
        let neg_bt = b.neg(bt);
        b.finish(
            &[dx, dy, dz],
            &[neg_sg, sg, zero, rmz, neg_one, neg_x, y, x, neg_bt],
            3,
            3,
        )
        .unwrap()
    }

    /// Hénon map next-state: x' = 1 - a x² + y, y' = b x, with a=1.4, b=0.3
    /// folded in as constants (n_param = 0, the map lowering convention).
    fn henon_tape() -> Tape {
        let mut bld = TapeBuilder::new();
        let x = bld.state(0);
        let y = bld.state(1);
        let one = bld.constant(1.0);
        let a = bld.constant(1.4);
        let bb = bld.constant(0.3);
        let x2 = bld.mul(x, x);
        let ax2 = bld.mul(a, x2);
        let one_m = bld.sub(one, ax2);
        let nx = bld.add(one_m, y);
        let ny = bld.mul(bb, x);
        bld.finish(&[nx, ny], &[], 2, 0).unwrap()
    }

    #[test]
    fn eval_rhs_matches_hand_value() {
        let got = eval_rhs(decay_tape(), &[2.0], &[3.0], 0.0).unwrap();
        assert!((got[0] - (-6.0)).abs() < 1e-15);
    }

    #[test]
    fn eval_rhs_rejects_short_state() {
        let err = eval_rhs(lorenz_tape(), &[1.0, 2.0], &[10.0, 28.0, 2.0], 0.0).unwrap_err();
        assert!(matches!(err, EngineError::BadShape(_)));
    }

    #[test]
    fn eval_jac_returns_rhs_and_jacobian() {
        let p = [10.0, 28.0, 8.0 / 3.0];
        let (d, j) = eval_jac(lorenz_tape(), &[1.0, 2.0, 3.0], &p, 0.0).unwrap();
        assert_eq!(d.len(), 3);
        assert_eq!(j.len(), 9);
        // Row 0 of J is [-sigma, sigma, 0].
        assert!((j[0] - (-10.0)).abs() < 1e-15);
        assert!((j[1] - 10.0).abs() < 1e-15);
        assert!((j[2]).abs() < 1e-15);
    }

    #[test]
    fn eval_jac_without_jacobian_is_an_error() {
        let err = eval_jac(decay_tape(), &[1.0], &[1.0], 0.0).unwrap_err();
        assert!(matches!(err, EngineError::BadShape(_)));
    }

    #[test]
    fn resolve_solver_is_case_insensitive() {
        assert_eq!(resolve_solver("rk45").unwrap(), "rk45");
        assert_eq!(resolve_solver("RK45").unwrap(), "rk45");
        assert_eq!(resolve_solver("DOP853").unwrap(), "dop853");
        assert!(matches!(
            resolve_solver("nope").unwrap_err(),
            EngineError::UnknownMethod(_)
        ));
    }

    #[test]
    fn build_solver_honours_tolerances_and_names() {
        // Smoke: every documented name builds a kernel reporting that name.
        for name in [
            "rk4",
            "rk45",
            "tsit5",
            "dop853",
            "rosenbrock",
            "trbdf2",
            "bdf",
        ] {
            let resolved = resolve_solver(name).unwrap();
            let s = build_solver(resolved, 1e-9, 1e-12);
            assert_eq!(s.name(), name);
        }
    }

    #[test]
    fn integrate_dense_matches_closed_form_decay() {
        let t_eval: Vec<f64> = (0..=10).map(|i| i as f64 * 0.3).collect();
        let y = integrate_dense(
            decay_tape(),
            &[1.0],
            &[2.0],
            &t_eval,
            "rk45",
            1e-10,
            1e-12,
            false,
        )
        .unwrap();
        assert_eq!(y.len(), t_eval.len());
        assert_eq!(y[0], 1.0); // first row is the IC
        for (i, &t) in t_eval.iter().enumerate() {
            let want = (-2.0 * t).exp();
            assert!((y[i] - want).abs() < 1e-7, "row {i}: {} vs {want}", y[i]);
        }
    }

    #[test]
    fn integrate_dense_tracks_oscillator() {
        let t_eval: Vec<f64> = (0..=8)
            .map(|i| i as f64 * core::f64::consts::FRAC_PI_4)
            .collect();
        let y = integrate_dense(
            oscillator_tape(),
            &[1.0, 0.0],
            &[],
            &t_eval,
            "rk45",
            1e-11,
            1e-13,
            false,
        )
        .unwrap();
        for (k, &t) in t_eval.iter().enumerate() {
            let (x, v) = (y[2 * k], y[2 * k + 1]);
            assert!((x - t.cos()).abs() < 1e-7, "x at t={t}: {x}");
            assert!((v + t.sin()).abs() < 1e-7, "v at t={t}: {v}");
        }
    }

    #[test]
    fn integrate_dense_reports_divergence() {
        let t_eval = [0.0, 0.5, 1.5, 2.0];
        let err = integrate_dense(
            blowup_tape(),
            &[1.0],
            &[],
            &t_eval,
            "rk45",
            1e-8,
            1e-10,
            false,
        )
        .unwrap_err();
        assert!(matches!(err, EngineError::Diverged(_)), "got {err:?}");
    }

    #[test]
    fn integrate_dense_rejects_unknown_method() {
        let t_eval = [0.0, 1.0];
        assert!(matches!(
            integrate_dense(
                decay_tape(),
                &[1.0],
                &[1.0],
                &t_eval,
                "no-such",
                1e-6,
                1e-9,
                false
            )
            .unwrap_err(),
            EngineError::UnknownMethod(_)
        ));
    }

    #[test]
    fn integrate_dense_jit_matches_interp_bit_for_bit() {
        // The JIT bridge: the same problem integrated with jit=true vs jit=false
        // must agree bit-for-bit. The JIT evaluator is bit-identical to the
        // interpreter on `eval`, and the adaptive loop drives both through the
        // same `&dyn Evaluator` code, so every step decision is identical.
        let t_eval: Vec<f64> = (0..=12).map(|i| i as f64 * 0.25).collect();
        let run = |jit| {
            integrate_dense(
                lorenz_tape(),
                &[1.0, 2.0, 3.0],
                &[10.0, 28.0, 8.0 / 3.0],
                &t_eval,
                "dop853",
                1e-9,
                1e-11,
                jit,
            )
            .unwrap()
        };
        let interp = run(false);
        let jit = run(true);
        assert_eq!(interp.len(), jit.len());
        for (i, (a, b)) in interp.iter().zip(jit.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "row {i}: jit {b} != interp {a}");
        }
    }

    #[test]
    fn ensemble_jit_matches_interp_bit_for_bit() {
        let ics: Vec<f64> = (0..8).map(|i| 0.5 + i as f64).collect();
        let run = |jit| {
            ensemble_final(
                decay_tape(),
                &ics,
                &[2.0],
                0.0,
                1.5,
                0.015,
                "rk45",
                1e-10,
                1e-12,
                jit,
            )
            .unwrap()
        };
        let interp = run(false);
        let jit = run(true);
        for (a, b) in interp.iter().zip(jit.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn ensemble_matches_a_serial_dense_loop() {
        let ics: Vec<f64> = (0..16).map(|i| 0.5 + i as f64).collect();
        let states = ensemble_final(
            decay_tape(),
            &ics,
            &[2.0],
            0.0,
            1.5,
            0.015,
            "rk45",
            1e-10,
            1e-12,
            false,
        )
        .unwrap();
        assert_eq!(states.len(), ics.len());
        let factor = (-2.0_f64 * 1.5).exp();
        for (i, &x0) in ics.iter().enumerate() {
            assert!(
                (states[i] - x0 * factor).abs() < 1e-7,
                "traj {i}: {} vs {}",
                states[i],
                x0 * factor
            );
        }
    }

    #[test]
    fn rk4_dense_and_ensemble_take_the_same_fixed_step() {
        // The fix: a fixed-step method must reach the same final state through the
        // single-trajectory (dense) and the parallel (ensemble) paths when handed
        // the same cadence. `rk4`'s step is fixed — the first step IS the step for
        // the whole run — so the dense path (which derives it from the output grid)
        // and the ensemble (handed it as `first_step`) must agree. dt = 0.25 is
        // binary-exact and divides the span, so the two take identical steps; the
        // oscillator is autonomous, so the final state is bit-for-bit equal.
        let dt = 0.25_f64;
        let (t0, t1) = (0.0, 2.0);
        let ic = [1.0, 0.0];
        // Dense over a grid spaced exactly `dt`, so each segment is one rk4 step and
        // `first_step_from_grid` recovers `dt`; the last (x, v) row is the final state.
        let n = ((t1 - t0) / dt).round() as usize; // 8 segments
        let t_eval: Vec<f64> = (0..=n).map(|i| t0 + i as f64 * dt).collect();
        let dense = integrate_dense(
            oscillator_tape(),
            &ic,
            &[],
            &t_eval,
            "rk4",
            1e-6,
            1e-9,
            false,
        )
        .unwrap();
        let dense_final = &dense[dense.len() - 2..];
        // Ensemble of the single IC with the matching cadence.
        let ens = ensemble_final(
            oscillator_tape(),
            &ic,
            &[],
            t0,
            t1,
            dt,
            "rk4",
            1e-6,
            1e-9,
            false,
        )
        .unwrap();
        assert_eq!(ens.len(), 2);
        assert_eq!(
            dense_final[0].to_bits(),
            ens[0].to_bits(),
            "x: dense {} vs ensemble {}",
            dense_final[0],
            ens[0]
        );
        assert_eq!(
            dense_final[1].to_bits(),
            ens[1].to_bits(),
            "v: dense {} vs ensemble {}",
            dense_final[1],
            ens[1]
        );
        // The cadence is load-bearing for a fixed-step kernel: the retired
        // `span/100` guess (0.02 here) takes different steps and lands elsewhere, so
        // the two paths would have disagreed without this knob.
        let ens_old_guess = ensemble_final(
            oscillator_tape(),
            &ic,
            &[],
            t0,
            t1,
            (t1 - t0) / 100.0,
            "rk4",
            1e-6,
            1e-9,
            false,
        )
        .unwrap();
        assert!(
            (ens_old_guess[0] - ens[0]).abs() > 1e-9,
            "a different fixed step should land elsewhere: {} ~ {}",
            ens_old_guess[0],
            ens[0]
        );
    }

    #[test]
    fn ensemble_rejects_non_positive_cadence() {
        // The first_step validation: a non-finite or non-positive cadence is a clean
        // BadShape (→ ValueError), not a PanicException from the engine's hard assert.
        for bad in [0.0, -0.1, f64::NAN, f64::INFINITY] {
            let err = ensemble_final(
                decay_tape(),
                &[1.0, 2.0],
                &[2.0],
                0.0,
                1.0,
                bad,
                "rk45",
                1e-6,
                1e-9,
                false,
            )
            .unwrap_err();
            assert!(
                matches!(err, EngineError::BadShape(_)),
                "cadence {bad}: got {err:?}"
            );
        }
    }

    #[test]
    fn ensemble_isolates_a_diverging_trajectory() {
        // x0 = 1 blows up before t = 2; x0 = -1 decays and stays finite.
        let states = ensemble_final(
            blowup_tape(),
            &[1.0, -1.0],
            &[],
            0.0,
            2.0,
            0.02,
            "rk45",
            1e-8,
            1e-10,
            false,
        )
        .unwrap();
        assert!(states[0].is_nan(), "diverged row should be NaN");
        assert!(states[1].is_finite());
    }

    #[test]
    fn ensemble_rejects_ragged_batch() {
        // 5 is not a multiple of dim = 2.
        let err = ensemble_final(
            oscillator_tape(),
            &[1.0, 0.0, 2.0, 0.0, 3.0],
            &[],
            0.0,
            1.0,
            0.01,
            "rk45",
            1e-6,
            1e-9,
            false,
        )
        .unwrap_err();
        assert!(matches!(err, EngineError::BadShape(_)));
    }

    #[test]
    fn iterate_map_reproduces_henon_orbit() {
        let out = iterate_map(henon_tape(), &[0.1, 0.1], 3, false).unwrap();
        assert_eq!(out.len(), 6);
        // Hand-roll three Hénon steps (a=1.4, b=0.3).
        let (mut x, mut y) = (0.1_f64, 0.1_f64);
        for i in 0..3 {
            let nx = 1.0 - 1.4 * x * x + y;
            let ny = 0.3 * x;
            x = nx;
            y = ny;
            assert!((out[2 * i] - x).abs() < 1e-14, "step {i} x");
            assert!((out[2 * i + 1] - y).abs() < 1e-14, "step {i} y");
        }
    }

    #[test]
    fn iterate_map_diverges_loudly() {
        // The Hénon map from a far-outside initial condition escapes to infinity;
        // delegation to the engine loop must raise at the first non-finite iterate
        // instead of returning a buffer of inf/NaN.
        let err = iterate_map(henon_tape(), &[10.0, 10.0], 200, false).unwrap_err();
        assert!(matches!(err, EngineError::Diverged(_)), "got {err:?}");
    }

    #[test]
    fn build_tape_rejects_unknown_opcode() {
        let err = build_tape(
            &[1, 2, 99],
            &[0, 0, 0],
            &[0, 0, 1],
            &[0.0; 3],
            &[2],
            &[],
            1,
            1,
        )
        .unwrap_err();
        assert!(matches!(err, EngineError::BadTape(_)));
    }

    #[test]
    fn implicit_kernels_honour_requested_tolerances() {
        // The build_solver fix: rosenbrock/trbdf2/bdf must integrate at the
        // requested tolerance, not the kernel default — a tight run on the smooth
        // decay must hit the closed form to well under the loose default accuracy.
        let t_eval: Vec<f64> = (0..=6).map(|i| i as f64 * 0.4).collect();
        for method in ["rosenbrock", "trbdf2", "bdf"] {
            let y = integrate_dense(
                decay_tape_jac(),
                &[1.0],
                &[2.0],
                &t_eval,
                method,
                1e-10,
                1e-12,
                false,
            )
            .unwrap_or_else(|e| panic!("{method} failed: {e}"));
            for (i, &t) in t_eval.iter().enumerate() {
                let want = (-2.0 * t).exp();
                assert!(
                    (y[i] - want).abs() < 1e-7,
                    "{method} row {i}: {} vs {want}",
                    y[i]
                );
            }
        }
    }

    #[test]
    fn integrate_dense_rejects_non_finite_or_descending_grid() {
        // Non-finite grid time → clean BadShape (not a PanicException via the
        // engine's hard first-step assert).
        let bad_inf = integrate_dense(
            decay_tape(),
            &[1.0],
            &[2.0],
            &[0.0, f64::INFINITY],
            "rk45",
            1e-6,
            1e-9,
            false,
        )
        .unwrap_err();
        assert!(
            matches!(bad_inf, EngineError::BadShape(_)),
            "got {bad_inf:?}"
        );
        // Descending grid → clean BadShape (not a silently-stale row in release).
        let bad_desc = integrate_dense(
            decay_tape(),
            &[1.0],
            &[2.0],
            &[0.0, 1.0, 0.5],
            "rk45",
            1e-6,
            1e-9,
            false,
        )
        .unwrap_err();
        assert!(
            matches!(bad_desc, EngineError::BadShape(_)),
            "got {bad_desc:?}"
        );
    }

    #[test]
    fn ensemble_rejects_non_finite_times() {
        let err = ensemble_final(
            decay_tape(),
            &[1.0, 2.0],
            &[2.0],
            0.0,
            f64::INFINITY,
            0.01,
            "rk45",
            1e-6,
            1e-9,
            false,
        )
        .unwrap_err();
        assert!(matches!(err, EngineError::BadShape(_)), "got {err:?}");
    }

    #[test]
    fn iterate_map_rejects_mismatched_tape() {
        // A tape with a Param op (n_param > 0) is not a valid map tape; iteration
        // must reject it cleanly rather than index an empty parameter slice.
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let k = b.param(0);
        let nx = b.add(x, k);
        let tape = b.finish(&[nx], &[], 1, 1).unwrap();
        let err = iterate_map(tape, &[0.1], 3, false).unwrap_err();
        assert!(matches!(err, EngineError::BadShape(_)), "got {err:?}");
    }

    /// The lowered DDE tape for `y'(t) = −y(t − 1)`: one output `−u1` over two
    /// inputs (`u0 = y(t)`, `u1 = y(t − 1)`, the single delay slot), no params.
    fn neg_delay_dde_tape() -> Tape {
        let mut b = TapeBuilder::new();
        let _y = b.state(0);
        let y_tau = b.state(1);
        let dy = b.neg(y_tau);
        b.finish(&[dy], &[], 2, 0).unwrap()
    }

    #[test]
    fn integrate_dde_matches_method_of_steps_closed_form() {
        // Constant past 1; analytic y(1) = 0, y(2) = −0.5 (method of steps).
        let t_eval: Vec<f64> = (0..=20).map(|i| i as f64 * 0.1).collect();
        let y = integrate_dde_dense(
            neg_delay_dde_tape(),
            &[0],   // slot 0 → component 0
            &[1.0], // delay 1
            &[1.0], // ic
            &[0.0], // past times (single sample → constant past)
            &[1.0], // past values
            &t_eval,
            "rk45",
            1e-9,
            1e-11,
            false,
        )
        .unwrap();
        assert_eq!(y.len(), t_eval.len());
        assert_eq!(y[0], 1.0); // first row is the IC
        assert!((y[10] - 0.0).abs() < 1e-6, "y(1) = {}", y[10]);
        assert!((y[20] + 0.5).abs() < 1e-6, "y(2) = {}", y[20]);
    }

    #[test]
    fn integrate_dde_rejects_implicit_method() {
        let err = integrate_dde_dense(
            neg_delay_dde_tape(),
            &[0],
            &[1.0],
            &[1.0],
            &[0.0],
            &[1.0],
            &[0.0, 1.0],
            "rosenbrock",
            1e-6,
            1e-9,
            false,
        )
        .unwrap_err();
        assert!(matches!(err, EngineError::Unsupported(_)), "got {err:?}");
    }

    #[test]
    fn integrate_dde_rejects_shape_mismatches() {
        // n_state (2) != dim (1) + slots (2): the tape has only one delay input.
        let bad_slots = integrate_dde_dense(
            neg_delay_dde_tape(),
            &[0, 0],
            &[1.0, 2.0],
            &[1.0],
            &[0.0],
            &[1.0],
            &[0.0, 1.0],
            "rk45",
            1e-6,
            1e-9,
            false,
        )
        .unwrap_err();
        assert!(
            matches!(bad_slots, EngineError::BadShape(_)),
            "got {bad_slots:?}"
        );

        // A delay slot referencing a non-existent component.
        let bad_comp = integrate_dde_dense(
            neg_delay_dde_tape(),
            &[5],
            &[1.0],
            &[1.0],
            &[0.0],
            &[1.0],
            &[0.0, 1.0],
            "rk45",
            1e-6,
            1e-9,
            false,
        )
        .unwrap_err();
        assert!(
            matches!(bad_comp, EngineError::BadShape(_)),
            "got {bad_comp:?}"
        );

        // A non-positive delay.
        let bad_delay = integrate_dde_dense(
            neg_delay_dde_tape(),
            &[0],
            &[0.0],
            &[1.0],
            &[0.0],
            &[1.0],
            &[0.0, 1.0],
            "rk45",
            1e-6,
            1e-9,
            false,
        )
        .unwrap_err();
        assert!(
            matches!(bad_delay, EngineError::BadShape(_)),
            "got {bad_delay:?}"
        );
    }

    #[test]
    fn implicit_methods_refuse_a_tape_without_a_jacobian() {
        // The critical guard: rosenbrock/trbdf2/bdf freeze ∂f/∂u each step. On a
        // tape compiled without a Jacobian the iteration matrix would collapse to I
        // and the implicit step would silently degrade to forward Euler. The engine
        // must reject this loudly (BadShape → ValueError), not integrate.
        let t_eval = [0.0, 0.5, 1.0];
        for method in ["rosenbrock", "trbdf2", "bdf"] {
            let err = integrate_dense(
                decay_tape(),
                &[1.0],
                &[2.0],
                &t_eval,
                method,
                1e-6,
                1e-9,
                false,
            )
            .unwrap_err();
            assert!(
                matches!(err, EngineError::BadShape(_)),
                "{method} (no Jacobian): got {err:?}"
            );
            // The ensemble path guards identically.
            let err = ensemble_final(
                decay_tape(),
                &[1.0, 2.0],
                &[2.0],
                0.0,
                1.0,
                0.01,
                method,
                1e-6,
                1e-9,
                false,
            )
            .unwrap_err();
            assert!(
                matches!(err, EngineError::BadShape(_)),
                "{method} ensemble (no Jacobian): got {err:?}"
            );
        }
        // With the Jacobian present, the same implicit method integrates fine.
        let y = integrate_dense(
            decay_tape_jac(),
            &[1.0],
            &[2.0],
            &t_eval,
            "rosenbrock",
            1e-9,
            1e-11,
            false,
        );
        assert!(
            y.is_ok(),
            "rosenbrock with a Jacobian should integrate: {y:?}"
        );
        // Explicit methods never need a Jacobian, so the no-Jacobian tape is fine.
        assert!(integrate_dense(
            decay_tape(),
            &[1.0],
            &[2.0],
            &t_eval,
            "rk45",
            1e-6,
            1e-9,
            false
        )
        .is_ok());
    }

    #[test]
    fn ensemble_rejects_backward_integration() {
        // t1 < t0 would never enter the forward step loop and silently return the
        // unchanged ICs as a successful batch; reject it instead.
        let err = ensemble_final(
            decay_tape(),
            &[1.0, 2.0],
            &[2.0],
            1.0,
            0.0,
            0.01,
            "rk45",
            1e-6,
            1e-9,
            false,
        )
        .unwrap_err();
        assert!(matches!(err, EngineError::BadShape(_)), "got {err:?}");
    }

    #[test]
    fn integrate_dde_jit_matches_interp_bit_for_bit() {
        // The JIT bridge reaches the DDE method-of-steps engine too: the lowered
        // DDE tape compiles to native code, and the same explicit kernel over the
        // same history must reproduce the interpreter run bit-for-bit.
        let t_eval: Vec<f64> = (0..=20).map(|i| i as f64 * 0.1).collect();
        let run = |jit| {
            integrate_dde_dense(
                neg_delay_dde_tape(),
                &[0],
                &[1.0],
                &[1.0],
                &[0.0],
                &[1.0],
                &t_eval,
                "rk45",
                1e-9,
                1e-11,
                jit,
            )
            .unwrap()
        };
        let interp = run(false);
        let jit = run(true);
        assert_eq!(interp.len(), jit.len());
        for (i, (a, b)) in interp.iter().zip(jit.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "row {i}: jit {b} != interp {a}");
        }
    }

    // -- SDE bridge ----------------------------------------------------------

    /// GBM drift `f(x) = μx` (no parameters; μ folded in as a constant).
    fn gbm_drift_tape(mu: f64) -> Tape {
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let m = b.constant(mu);
        let f = b.mul(m, x);
        b.finish(&[f], &[], 1, 0).unwrap()
    }

    /// GBM diffusion `g(x) = σx` carrying its diagonal Jacobian `∂g/∂x = σ`
    /// (so the Milstein kernel has the `∂g/∂u` it reads).
    fn gbm_diff_tape_jac(sigma: f64) -> Tape {
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let s = b.constant(sigma);
        let g = b.mul(s, x);
        b.finish(&[g], &[s], 1, 0).unwrap()
    }

    /// GBM diffusion `g(x) = σx` *without* a Jacobian (the Euler–Maruyama case,
    /// and the tape a Milstein run must refuse).
    fn gbm_diff_tape(sigma: f64) -> Tape {
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let s = b.constant(sigma);
        let g = b.mul(s, x);
        b.finish(&[g], &[], 1, 0).unwrap()
    }

    #[test]
    fn sde_integrate_dense_first_row_is_ic_and_stays_finite() {
        let t_eval: Vec<f64> = (0..=10).map(|i| i as f64 * 0.1).collect();
        let y = sde_integrate_dense(
            gbm_drift_tape(0.1),
            gbm_diff_tape(0.3),
            &[1.0],
            &[],
            &t_eval,
            "euler_maruyama",
            0.01,
            42,
            false,
        )
        .unwrap();
        assert_eq!(y.len(), t_eval.len());
        assert_eq!(y[0], 1.0, "first row must be the initial condition");
        assert!(y.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn sde_ensemble_is_parallel_equals_serial() {
        // The headline determinism contract through the bridge: ensemble row i
        // (seeded by seed_for(base, i) inside the engine) equals a single
        // trajectory seeded the same way and integrated [t0, t1] in one segment.
        use tsdyn_engine::rng::seed_for;
        let (base, t0, t1, dt) = (0xC0FFEE_u64, 0.0, 1.0, 0.01);
        let ics: Vec<f64> = (0..6).map(|i| 1.0 + 0.1 * i as f64).collect();
        let batch = sde_ensemble_final(
            gbm_drift_tape(0.15),
            gbm_diff_tape_jac(0.3),
            &ics,
            &[],
            t0,
            t1,
            "milstein",
            dt,
            base,
            false,
        )
        .unwrap();
        assert_eq!(batch.len(), ics.len());
        for (i, &x0) in ics.iter().enumerate() {
            let single = sde_integrate_dense(
                gbm_drift_tape(0.15),
                gbm_diff_tape_jac(0.3),
                &[x0],
                &[],
                &[t0, t1],
                "milstein",
                dt,
                seed_for(base, i as u64),
                false,
            )
            .unwrap();
            assert_eq!(
                batch[i].to_bits(),
                single[1].to_bits(),
                "trajectory {i}: ensemble != seeded single"
            );
        }
    }

    #[test]
    fn sde_jit_matches_interp_bit_for_bit() {
        // jit==interp for an SDE: the noise stream is seed-driven (identical), and
        // both evaluators are bit-identical, so the whole path matches bit-for-bit.
        let t_eval: Vec<f64> = (0..=15).map(|i| i as f64 * 0.05).collect();
        let run = |jit| {
            sde_integrate_dense(
                gbm_drift_tape(0.2),
                gbm_diff_tape_jac(0.4),
                &[1.0],
                &[],
                &t_eval,
                "milstein",
                0.005,
                7,
                jit,
            )
            .unwrap()
        };
        let interp = run(false);
        let jit = run(true);
        for (a, b) in interp.iter().zip(jit.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn sde_rejects_unknown_method() {
        let err = sde_integrate_dense(
            gbm_drift_tape(0.1),
            gbm_diff_tape(0.3),
            &[1.0],
            &[],
            &[0.0, 0.1],
            "no-such-scheme",
            0.01,
            0,
            false,
        )
        .unwrap_err();
        assert!(matches!(err, EngineError::UnknownMethod(_)), "got {err:?}");
    }

    #[test]
    fn sde_milstein_requires_a_diffusion_jacobian() {
        // Milstein reads ∂g/∂u; a diffusion tape lowered without one must be
        // rejected loudly (BadShape → ValueError), not run with a missing term.
        let err = sde_integrate_dense(
            gbm_drift_tape(0.1),
            gbm_diff_tape(0.3), // no Jacobian
            &[1.0],
            &[],
            &[0.0, 0.1],
            "milstein",
            0.01,
            0,
            false,
        )
        .unwrap_err();
        assert!(matches!(err, EngineError::BadShape(_)), "got {err:?}");
        // Euler–Maruyama needs no diffusion Jacobian, so the same tape is fine.
        assert!(sde_integrate_dense(
            gbm_drift_tape(0.1),
            gbm_diff_tape(0.3),
            &[1.0],
            &[],
            &[0.0, 0.1],
            "euler_maruyama",
            0.01,
            0,
            false,
        )
        .is_ok());
    }

    #[test]
    fn sde_ensemble_isolates_a_diverged_trajectory() {
        // Drift f(x)=x² (g=0) blows up from x0=1 before t=2 but decays from -1;
        // the diverged row must be NaN, the healthy one finite.
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let f = b.mul(x, x);
        let drift = b.finish(&[f], &[], 1, 0).unwrap();
        let zero_diff = {
            let mut b = TapeBuilder::new();
            let _x = b.state(0);
            let z = b.constant(0.0);
            b.finish(&[z], &[], 1, 0).unwrap()
        };
        let states = sde_ensemble_final(
            drift,
            zero_diff,
            &[1.0, -1.0],
            &[],
            0.0,
            2.0,
            "euler_maruyama",
            0.01,
            0,
            false,
        )
        .unwrap();
        assert!(states[0].is_nan(), "diverged row should be NaN");
        assert!(states[1].is_finite());
    }

    #[test]
    fn sde_rejects_non_positive_dt_and_backward_time() {
        assert!(matches!(
            sde_integrate_dense(
                gbm_drift_tape(0.1),
                gbm_diff_tape(0.3),
                &[1.0],
                &[],
                &[0.0, 0.1],
                "euler_maruyama",
                0.0, // bad dt
                0,
                false,
            )
            .unwrap_err(),
            EngineError::BadShape(_)
        ));
        assert!(matches!(
            sde_ensemble_final(
                gbm_drift_tape(0.1),
                gbm_diff_tape(0.3),
                &[1.0],
                &[],
                1.0,
                0.0, // backward
                "euler_maruyama",
                0.01,
                0,
                false,
            )
            .unwrap_err(),
            EngineError::BadShape(_)
        ));
    }

    // -- Map ensemble --------------------------------------------------------

    #[test]
    fn map_ensemble_matches_a_serial_iterate_loop() {
        // The ensemble final states must equal iterating each IC with the single
        // map loop — bit-for-bit (maps carry no randomness).
        let ics = [0.1, 0.1, -0.2, 0.05, 0.3, -0.1];
        let steps = 25;
        let n_ic = ics.len() / 2;
        let batch = map_ensemble_final(henon_tape(), &ics, steps, false).unwrap();
        assert_eq!(batch.len(), ics.len());
        for i in 0..n_ic {
            let dense = iterate_map(henon_tape(), &ics[2 * i..2 * i + 2], steps, false).unwrap();
            let last = &dense[(steps - 1) * 2..]; // final iterate row
            assert_eq!(batch[2 * i].to_bits(), last[0].to_bits(), "traj {i} x");
            assert_eq!(batch[2 * i + 1].to_bits(), last[1].to_bits(), "traj {i} y");
        }
    }

    #[test]
    fn map_ensemble_isolates_a_diverged_trajectory() {
        // Hénon from far outside the basin escapes to infinity → a NaN row; an
        // on-attractor IC stays finite. The batch is not aborted.
        let states = map_ensemble_final(henon_tape(), &[10.0, 10.0, 0.1, 0.1], 200, false).unwrap();
        assert!(
            states[0].is_nan() || states[1].is_nan(),
            "diverged row should be NaN"
        );
        assert!(states[2].is_finite() && states[3].is_finite());
    }

    #[test]
    fn map_ensemble_rejects_ragged_batch() {
        // 3 is not a multiple of dim = 2.
        let err = map_ensemble_final(henon_tape(), &[0.1, 0.1, 0.2], 5, false).unwrap_err();
        assert!(matches!(err, EngineError::BadShape(_)), "got {err:?}");
    }

    #[test]
    fn iterate_map_jit_equals_interpreter_bit_for_bit() {
        // The map JIT path (jit=true) must reproduce the interpreter path
        // (jit=false) to the last bit — the E2 equality invariant, applied to the
        // map loop the same way it holds for the continuous path. Cover a 2-D map
        // (Hénon) and a 1-D map (x' = x², contracting from x0 < 1).
        let steps = 64;
        for ic in [[0.1, 0.1], [-0.2, 0.05], [0.31, -0.12]] {
            let interp = iterate_map(henon_tape(), &ic, steps, false).unwrap();
            let jit = iterate_map(henon_tape(), &ic, steps, true).unwrap();
            assert_eq!(interp.len(), jit.len());
            for (k, (i, j)) in interp.iter().zip(jit.iter()).enumerate() {
                assert_eq!(
                    i.to_bits(),
                    j.to_bits(),
                    "henon iterate {k}: interp {i} != jit {j}"
                );
            }
        }
        // 1-D map under the JIT.
        let interp = iterate_map(blowup_tape(), &[0.5], 20, false).unwrap();
        let jit = iterate_map(blowup_tape(), &[0.5], 20, true).unwrap();
        for (k, (i, j)) in interp.iter().zip(jit.iter()).enumerate() {
            assert_eq!(
                i.to_bits(),
                j.to_bits(),
                "1-D iterate {k}: interp {i} != jit {j}"
            );
        }
    }

    #[test]
    fn iterate_map_jit_diverges_loudly() {
        // The diverge-loudly contract must hold on the JIT path too: x' = x² from
        // x0 = 10 escapes to +inf, and the engine must raise at the first
        // non-finite iterate rather than return a poisoned buffer.
        let err = iterate_map(blowup_tape(), &[10.0], 2000, true).unwrap_err();
        assert!(matches!(err, EngineError::Diverged(_)), "got {err:?}");
    }

    #[test]
    fn map_ensemble_jit_isolates_a_diverged_trajectory() {
        // Mixed-fate batch under the JIT: x0 = 10 blows up (NaN row), x0 = 0.5
        // stays finite — the NaN must stay isolated to the diverged row.
        let states = map_ensemble_final(blowup_tape(), &[10.0, 0.5], 2000, true).unwrap();
        assert!(
            states[0].is_nan(),
            "diverged row should be NaN, got {}",
            states[0]
        );
        assert!(
            states[1].is_finite(),
            "finite row should survive, got {}",
            states[1]
        );
    }

    #[test]
    fn map_ensemble_jit_equals_interpreter_bit_for_bit() {
        // Same invariant for the parallel map ensemble path.
        let ics = [0.1, 0.1, -0.2, 0.05, 0.3, -0.1, 0.0, 0.2];
        let steps = 40;
        let interp = map_ensemble_final(henon_tape(), &ics, steps, false).unwrap();
        let jit = map_ensemble_final(henon_tape(), &ics, steps, true).unwrap();
        assert_eq!(interp.len(), jit.len());
        for (k, (i, j)) in interp.iter().zip(jit.iter()).enumerate() {
            assert_eq!(
                i.to_bits(),
                j.to_bits(),
                "row elem {k}: interp {i} != jit {j}"
            );
        }
    }

    // -- Resumable ODE stepper handle (WS-STEPPER) ---------------------------

    /// Event `g(u) = u[component] - c` as a one-output tape over `dim` inputs.
    fn plane_tape(dim: usize, component: usize, c: f64) -> Tape {
        let mut b = TapeBuilder::new();
        let x = b.state(component);
        // Touch every state input so n_state == dim (the engine's event contract
        // reads the full state); add 0·u_i for the untouched components.
        let off = b.constant(c);
        let mut g = b.sub(x, off);
        for i in 0..dim {
            if i != component {
                let ui = b.state(i);
                let z = b.constant(0.0);
                let zui = b.mul(z, ui);
                g = b.add(g, zui);
            }
        }
        b.finish(&[g], &[], dim, 0).unwrap()
    }

    #[test]
    fn stepper_advance_reproduces_per_dt_integrate_dense_bit_for_bit() {
        // The byte-identity contract the handle must hold: `advance(dt)` reproduces
        // the released `ContinuousSystem.step()` numerics, which integrate a *fresh*
        // two-node grid `[t, t+dt]` per step (the adaptive controller is re-seeded
        // each `dt`, by design — that is exactly why the WS-STEPBUF batch-ahead was
        // wrong). So the reference is a *sequence of two-node `integrate_dense`
        // calls*, each re-built from the previous final state — NOT one long
        // multi-node grid (which would carry the controller across nodes and differ
        // by ~1 ULP). The handle amortises the build/marshalling, not the numerics.
        let dt = 0.01_f64;
        let n = 50usize;
        let p = [10.0, 28.0, 8.0 / 3.0];
        let dim = 3;

        let mut ref_state = vec![1.0, 1.0, 1.0];
        let mut t = 0.0_f64;
        let mut stepper = OdeStepper::new(
            lorenz_tape(),
            &[1.0, 1.0, 1.0],
            0.0,
            "rk45",
            1e-6,
            1e-9,
            false,
        )
        .unwrap();
        for k in 1..=n {
            // The per-`dt` batch reference: a fresh two-node `integrate_dense` from
            // the live state, exactly what `_step_continuous` runs.
            let tf = t + dt;
            let seg = integrate_dense(
                lorenz_tape(),
                &ref_state,
                &p,
                &[t, tf],
                "rk45",
                1e-6,
                1e-9,
                false,
            )
            .unwrap();
            ref_state = seg[dim..2 * dim].to_vec();
            t = tf;

            let u = stepper.advance(dt, &p).unwrap();
            for (d, (a, b)) in u.iter().zip(ref_state.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "step {k} comp {d}: stepper {a} != per-dt integrate_dense {b}"
                );
            }
        }
    }

    #[test]
    fn stepper_advance_over_segments_matches_one_longer_integrate() {
        // Advancing several `dt` segments lands (to tolerance) where a single longer
        // integrate of the same span does — the headline resumability acceptance.
        let dt = 0.05_f64;
        let n = 40usize;
        let (t0, t1) = (0.0, n as f64 * dt);
        // One longer integrate over `[t0, t1]` (two-node grid → final state).
        let one = integrate_dense(
            oscillator_tape(),
            &[1.0, 0.0],
            &[],
            &[t0, t1],
            "dop853",
            1e-10,
            1e-12,
            false,
        )
        .unwrap();
        let one_final = &one[2..4];

        let mut stepper = OdeStepper::new(
            oscillator_tape(),
            &[1.0, 0.0],
            t0,
            "dop853",
            1e-10,
            1e-12,
            false,
        )
        .unwrap();
        let mut last = vec![0.0; 2];
        for _ in 0..n {
            last = stepper.advance(dt, &[]).unwrap();
        }
        assert!(
            (stepper.time() - t1).abs() < 1e-12,
            "t = {}",
            stepper.time()
        );
        // The two march different paths (n re-seeded segments vs one shot) but track
        // the same smooth solution (cos t, -sin t) to well within step accuracy.
        assert!(
            (last[0] - t1.cos()).abs() < 1e-6,
            "x: {} vs {}",
            last[0],
            t1.cos()
        );
        assert!(
            (last[1] + t1.sin()).abs() < 1e-6,
            "v: {} vs {}",
            last[1],
            -t1.sin()
        );
        assert!(
            (last[0] - one_final[0]).abs() < 1e-6 && (last[1] - one_final[1]).abs() < 1e-6,
            "segmented vs one-shot disagree: {last:?} vs {one_final:?}"
        );
    }

    #[test]
    fn stepper_advance_jit_matches_interp_bit_for_bit() {
        // The handle drives both evaluators; jit==interp bit-for-bit, the same
        // invariant the batch paths hold.
        let dt = 0.02_f64;
        let p = [10.0, 28.0, 8.0 / 3.0];
        let run = |jit| {
            let mut s = OdeStepper::new(
                lorenz_tape(),
                &[1.0, 2.0, 3.0],
                0.0,
                "dop853",
                1e-9,
                1e-11,
                jit,
            )
            .unwrap();
            let mut out = Vec::new();
            for _ in 0..30 {
                out.extend(s.advance(dt, &p).unwrap());
            }
            out
        };
        let interp = run(false);
        let jit = run(true);
        assert_eq!(interp.len(), jit.len());
        for (i, (a, b)) in interp.iter().zip(jit.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "elem {i}: jit {b} != interp {a}");
        }
    }

    #[test]
    fn stepper_advance_to_event_lands_on_a_known_crossing() {
        // x(t) = cos t falls through x = 0 at t = π/2 (v = -1 there). The handle's
        // event search must land on that crossing (rk4 + Hermite, refined).
        let g = plane_tape(2, 0, 0.0);
        let mut s = OdeStepper::new(
            oscillator_tape(),
            &[1.0, 0.0],
            0.0,
            "rk4",
            1e-6,
            1e-9,
            false,
        )
        .unwrap();
        let (found, t_cross, u_cross, dir) = s.advance_to_event(g, 3.0, 0.01, -1, &[]).unwrap();
        assert!(found, "a falling crossing exists in [0, 3]");
        assert_eq!(dir, -1);
        assert!(
            (t_cross - core::f64::consts::FRAC_PI_2).abs() < 1e-6,
            "t = {t_cross}"
        );
        assert!(u_cross[0].abs() < 1e-6, "x at crossing = {}", u_cross[0]);
        assert!(
            (u_cross[1] + 1.0).abs() < 1e-6,
            "v at crossing = {}",
            u_cross[1]
        );
        // The live point advanced strictly past the crossing.
        assert!(s.time() > t_cross, "state resumes past the crossing");
    }

    #[test]
    fn stepper_advance_to_event_marches_successive_crossings() {
        // Repeated event calls collect successive crossings of x = cos t through 0
        // without re-finding the same one — the Poincaré-map use case.
        let mut s = OdeStepper::new(
            oscillator_tape(),
            &[1.0, 0.0],
            0.0,
            "rk4",
            1e-6,
            1e-9,
            false,
        )
        .unwrap();
        let mut times = Vec::new();
        for _ in 0..4 {
            let (found, t_cross, _u, _d) = s
                .advance_to_event(plane_tape(2, 0, 0.0), 10.0, 0.01, 0, &[])
                .unwrap();
            assert!(found);
            times.push(t_cross);
        }
        // π/2, 3π/2, 5π/2, 7π/2 — strictly increasing, spaced ~π.
        for w in times.windows(2) {
            assert!(w[1] > w[0], "crossings must be increasing: {times:?}");
            assert!(
                (w[1] - w[0] - core::f64::consts::PI).abs() < 1e-3,
                "spacing ~ π: {times:?}"
            );
        }
    }

    #[test]
    fn stepper_advance_to_event_reports_no_crossing() {
        // The plane x = 5 is never reached → found = false, the live point advances
        // to t + max_span (ready to keep searching).
        let g = plane_tape(2, 0, 5.0);
        let mut s = OdeStepper::new(
            oscillator_tape(),
            &[1.0, 0.0],
            0.0,
            "rk4",
            1e-6,
            1e-9,
            false,
        )
        .unwrap();
        let (found, _t, _u, _d) = s.advance_to_event(g, 1.0, 0.01, 0, &[]).unwrap();
        assert!(!found, "no crossing expected");
        assert!(
            (s.time() - 1.0).abs() < 1e-12,
            "advanced to t1, got {}",
            s.time()
        );
    }

    #[test]
    fn stepper_advance_diverges_loudly() {
        // dx/dt = x² blows up at t = 1 from x0 = 1; advancing past it must raise.
        let mut s =
            OdeStepper::new(blowup_tape(), &[1.0], 0.0, "rk45", 1e-8, 1e-10, false).unwrap();
        // March up to the singularity; one of these segments must report divergence.
        let mut diverged = false;
        for _ in 0..200 {
            match s.advance(0.01, &[]) {
                Ok(_) => {}
                Err(EngineError::Diverged(_)) => {
                    diverged = true;
                    break;
                }
                Err(e) => panic!("unexpected error: {e:?}"),
            }
        }
        assert!(diverged, "the blow-up must surface as a Diverged error");
    }

    #[test]
    fn stepper_reads_live_parameters_each_advance() {
        // Changing the parameter between advances must take effect on the next step,
        // mirroring the live-stepper semantics of ContinuousSystem.step().
        // dx/dt = -k x: a larger k decays faster.
        let mut slow =
            OdeStepper::new(decay_tape(), &[1.0], 0.0, "rk45", 1e-9, 1e-11, false).unwrap();
        let a = slow.advance(0.5, &[1.0]).unwrap()[0]; // k = 1 for this segment
        let b = slow.advance(0.5, &[5.0]).unwrap()[0]; // k = 5 for the next segment
                                                       // After [0,0.5] at k=1 then [0.5,1.0] at k=5: x = e^{-0.5} · e^{-2.5}.
        let want = (-0.5_f64).exp() * (-2.5_f64).exp();
        assert!((b - want).abs() < 1e-7, "live-param decay: {b} vs {want}");
        assert!(a > b, "second (faster) segment decays further: {a} -> {b}");
    }

    #[test]
    fn stepper_new_rejects_implicit_without_jacobian() {
        // The same guard the batch path applies: an implicit kernel over a tape with
        // no Jacobian is rejected at construction (not mid-march). `OdeStepper` is
        // not `Debug` (it boxes an evaluator), so match the error rather than
        // `unwrap_err`.
        match OdeStepper::new(decay_tape(), &[1.0], 0.0, "bdf", 1e-6, 1e-9, false) {
            Err(EngineError::BadShape(_)) => {}
            other => panic!("expected BadShape, got {:?}", other.err()),
        }
        // With the Jacobian present, the implicit stepper builds and advances.
        let mut ok =
            OdeStepper::new(decay_tape_jac(), &[1.0], 0.0, "bdf", 1e-9, 1e-11, false).unwrap();
        let x = ok.advance(0.4, &[2.0]).unwrap()[0];
        assert!((x - (-0.8_f64).exp()).abs() < 1e-6, "bdf decay: {x}");
    }

    #[test]
    fn stepper_set_state_reseats_the_live_point() {
        let mut s = OdeStepper::new(
            oscillator_tape(),
            &[1.0, 0.0],
            0.0,
            "rk45",
            1e-9,
            1e-11,
            false,
        )
        .unwrap();
        s.advance(0.3, &[]).unwrap();
        s.set_state(&[0.5, -0.5], 2.0).unwrap();
        assert_eq!(s.state(), vec![0.5, -0.5]);
        assert_eq!(s.time(), 2.0);
        // Rejects a non-finite reseat.
        assert!(matches!(
            s.set_state(&[f64::NAN, 0.0], 0.0).unwrap_err(),
            EngineError::BadShape(_)
        ));
    }
}
