//! Ensemble integration — the rayon fan-out over many initial conditions
//! (ROADMAP §4c). This is the basin/attractor primitive: thousands of
//! independent trajectories, each a GIL-free integration on its own worker.
//!
//! # The determinism contract
//!
//! An ensemble must give **bit-for-bit identical** results no matter how many
//! threads run it. Two design choices guarantee that:
//!
//! 1. **One shared evaluator, one solver per trajectory.** The [`Evaluator`] is
//!    `Sync` (built once, shared by reference); each trajectory gets its own
//!    `Box<dyn Solver>` from the caller's factory, so no mutable state is ever
//!    shared between workers — the v2 build-once / per-worker pattern.
//! 2. **Randomness keyed by trajectory index, not thread.** The factory takes
//!    the trajectory index, so a stochastic kernel seeds itself with
//!    [`seed_for`](crate::rng::seed_for)`(base, i)` — its entire random stream
//!    depends only on `i`. Results are collected in index order, so the output
//!    is independent of scheduling.
//!
//! Because the frozen [`Solver::step`] signature carries no RNG, this
//! index-keyed factory *is* the seam through which stream E-SDE's stochastic
//! kernels become reproducible under parallelism.

use rayon::prelude::*;
use tsdyn_ir::Evaluator;
use tsdyn_solvers::Solver;

use crate::integrate::{integrate_final, IntegrateConfig, IntegrateError};

/// The fate of one ensemble trajectory.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TrajStatus {
    /// Reached the target time; its row in [`EnsembleFinal::states`] is the
    /// final state.
    Ok,
    /// Diverged or stalled; its row is filled with `NaN` and the reason is
    /// carried here (the v2 contract: a failed trajectory is never silently
    /// passed off as data).
    Failed(IntegrateError),
}

impl TrajStatus {
    /// Whether the trajectory completed successfully.
    pub fn is_ok(&self) -> bool {
        matches!(self, TrajStatus::Ok)
    }
}

/// The result of an ensemble-to-final-time integration.
///
/// `states` is a row-major `(n_ic, dim)` buffer of final states (one row per
/// initial condition, in input order); failed rows are all-`NaN`. `status[i]`
/// records each trajectory's fate.
#[derive(Clone, Debug)]
pub struct EnsembleFinal {
    /// System dimension (row width of [`states`](EnsembleFinal::states)).
    pub dim: usize,
    /// Row-major `(n_ic, dim)` final states; `NaN` rows for failed trajectories.
    pub states: Vec<f64>,
    /// Per-trajectory fate, length `n_ic`, in input order.
    pub status: Vec<TrajStatus>,
}

impl EnsembleFinal {
    /// Number of initial conditions integrated.
    pub fn n_ic(&self) -> usize {
        self.status.len()
    }

    /// The final state of trajectory `i` (a `NaN`-filled slice if it failed).
    ///
    /// Panics if `i >= ` [`n_ic`](EnsembleFinal::n_ic).
    pub fn row(&self, i: usize) -> &[f64] {
        &self.states[i * self.dim..(i + 1) * self.dim]
    }

    /// How many trajectories failed to reach the target time.
    pub fn n_failed(&self) -> usize {
        self.status.iter().filter(|s| !s.is_ok()).count()
    }
}

/// Integrate a batch of initial conditions to `t1` in parallel, returning each
/// final state.
///
/// `u0_batch` is a row-major `(n_ic, dim)` buffer; `dim` is taken from `ev`.
/// `make_solver(i)` builds a fresh kernel for trajectory `i` — for a
/// deterministic problem it can ignore `i` (`|_| Box::new(Rk4::new())`); for a
/// stochastic one it seeds from `i` (`|i| Box::new(EM::new(seed_for(base, i)))`)
/// to make the run reproducible (see the [module docs](self)). A diverging
/// trajectory yields a `NaN` row and a [`TrajStatus::Failed`] rather than
/// aborting the whole batch.
pub fn ensemble_final<F>(
    ev: &dyn Evaluator,
    make_solver: F,
    u0_batch: &[f64],
    p: &[f64],
    t0: f64,
    t1: f64,
    cfg: &IntegrateConfig,
) -> EnsembleFinal
where
    F: Fn(usize) -> Box<dyn Solver> + Sync,
{
    let dim = ev.dim();
    // Hard asserts (validated once here, on the calling thread, before the rayon
    // fan-out): a bad config or a ragged batch is caller error, and silently
    // flooring `n_ic` would drop trajectories without a trace.
    assert!(dim > 0, "evaluator dimension must be positive");
    assert!(
        cfg.first_step.is_finite() && cfg.first_step > 0.0,
        "first step must be finite and positive, got {}",
        cfg.first_step
    );
    assert_eq!(
        u0_batch.len() % dim,
        0,
        "u0_batch length {} is not a multiple of dim {dim}",
        u0_batch.len()
    );
    let n_ic = u0_batch.len() / dim;

    // Preallocate the contiguous output and write each trajectory DIRECTLY into
    // its own row slice in parallel. Zipping `states.par_chunks_mut(dim)` with
    // `status.par_iter_mut()` partitions the buffers into disjoint, index-aligned
    // pieces, so worker `i` only ever touches row `i`. The win is avoiding a
    // GLOBAL collect-then-scatter: there is no intermediate `Vec<Vec<f64>>` of
    // per-trajectory heap allocations to gather and copy back into the contiguous
    // buffer. Each trajectory still copies its own final state once — `integrate_final`
    // returns an owned final-state `Vec`, which is copied into its row. The
    // partition is positional, so the output is still in input order regardless of
    // how rayon schedules the workers (parallel == serial bit-for-bit, the
    // determinism contract above).
    let mut states = vec![0.0; n_ic * dim];
    let mut status = vec![TrajStatus::Ok; n_ic];
    states
        .par_chunks_mut(dim)
        .zip(status.par_iter_mut())
        .enumerate()
        .for_each(|(i, (row, st))| {
            let u0 = &u0_batch[i * dim..(i + 1) * dim];
            let mut solver = make_solver(i);
            match integrate_final(ev, &mut *solver, u0, p, t0, t1, cfg) {
                Ok(uf) => {
                    row.copy_from_slice(&uf);
                    *st = TrajStatus::Ok;
                }
                Err(e) => {
                    row.fill(f64::NAN);
                    *st = TrajStatus::Failed(e);
                }
            }
        });
    EnsembleFinal {
        dim,
        states,
        status,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::seed_for;
    use crate::testkit::{NoisyEuler, Rk4, VmEval};
    use tsdyn_ir::TapeBuilder;
    use tsdyn_vm::Interpreter;

    /// dx/dt = -x ⇒ x(t) = x0 e^{-t}.
    fn decay() -> Interpreter {
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let dx = b.neg(x);
        Interpreter::new(b.finish(&[dx], &[], 1, 0).unwrap())
    }

    /// dx/dt = x² (finite-time blow-up for x0 > 0, decay for x0 < 0).
    fn blowup() -> Interpreter {
        let mut b = TapeBuilder::new();
        let x = b.state(0);
        let dx = b.mul(x, x);
        Interpreter::new(b.finish(&[dx], &[], 1, 0).unwrap())
    }

    #[test]
    fn deterministic_ensemble_matches_a_serial_loop() {
        let ev = VmEval::new(decay());
        let cfg = IntegrateConfig::new(0.001);
        let u0: Vec<f64> = (0..64).map(|i| 0.1 + i as f64).collect();

        let ens = ensemble_final(&ev, |_| Box::new(Rk4::new()), &u0, &[], 0.0, 2.0, &cfg);

        // Serial reference: integrate each IC by hand, same kernel.
        for (i, &x0) in u0.iter().enumerate() {
            let mut s = Rk4::new();
            let want = integrate_final(&ev, &mut s, &[x0], &[], 0.0, 2.0, &cfg).unwrap();
            assert!(ens.status[i].is_ok());
            assert_eq!(ens.row(i), want.as_slice(), "trajectory {i} differs");
        }
        assert_eq!(ens.n_failed(), 0);
    }

    #[test]
    fn seeded_stochastic_ensemble_is_parallel_equals_serial() {
        // The headline guarantee: a stochastic ensemble seeded per-index gives
        // bit-for-bit the same states as a serial loop building the same seeds.
        let ev = VmEval::new(decay());
        let cfg = IntegrateConfig::new(0.01);
        let base = 0xC0FFEE;
        let sigma = 0.3;
        let n_ic = 200;
        let u0 = vec![1.0; n_ic];

        let factory = |i: usize| -> Box<dyn Solver> {
            Box::new(NoisyEuler::new(sigma, seed_for(base, i as u64)))
        };

        // Force a multi-thread pool so this is a genuine concurrency test even on
        // a single-core runner: if scheduling could perturb the result, eight
        // workers racing over 200 trajectories would expose it.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        let parallel = pool.install(|| ensemble_final(&ev, factory, &u0, &[], 0.0, 1.0, &cfg));

        // Serial reference, identical per-index construction, plain loop.
        let mut serial = vec![0.0; n_ic];
        for (i, slot) in serial.iter_mut().enumerate() {
            let mut s = NoisyEuler::new(sigma, seed_for(base, i as u64));
            *slot = integrate_final(&ev, &mut s, &[1.0], &[], 0.0, 1.0, &cfg).unwrap()[0];
        }

        for (i, &serial_x) in serial.iter().enumerate() {
            let parallel_x = parallel.row(i)[0];
            assert_eq!(
                parallel_x.to_bits(),
                serial_x.to_bits(),
                "trajectory {i}: parallel {parallel_x} != serial {serial_x}",
            );
        }
    }

    #[test]
    fn seeded_ensemble_is_reproducible_across_runs() {
        let ev = VmEval::new(decay());
        let cfg = IntegrateConfig::new(0.01);
        let factory = |i: usize| -> Box<dyn Solver> {
            Box::new(NoisyEuler::new(0.5, seed_for(42, i as u64)))
        };
        let u0 = vec![1.0; 128];
        let a = ensemble_final(&ev, factory, &u0, &[], 0.0, 1.0, &cfg);
        let b = ensemble_final(&ev, factory, &u0, &[], 0.0, 1.0, &cfg);
        assert_eq!(
            a.states, b.states,
            "two runs of the same seeded ensemble differ"
        );
    }

    #[test]
    fn noise_actually_perturbs_and_decorrelates_trajectories() {
        // Sanity that the stochastic fixture is doing something: different
        // indices give different results, and they differ from the noise-free
        // trajectory.
        let ev = VmEval::new(decay());
        let cfg = IntegrateConfig::new(0.01);
        let u0 = vec![1.0; 8];
        let noisy = ensemble_final(
            &ev,
            |i: usize| Box::new(NoisyEuler::new(0.4, seed_for(1, i as u64))),
            &u0,
            &[],
            0.0,
            1.0,
            &cfg,
        );
        let clean = (-1.0_f64).exp(); // x0 e^{-1}
        let v0 = noisy.row(0)[0];
        let v1 = noisy.row(1)[0];
        assert_ne!(
            v0.to_bits(),
            v1.to_bits(),
            "distinct indices gave the same draw"
        );
        assert!((v0 - clean).abs() > 1e-9, "noise had no visible effect");
    }

    #[test]
    fn failed_trajectory_is_isolated_not_fatal() {
        let ev = VmEval::new(blowup());
        let cfg = IntegrateConfig::new(0.01);
        // x0 = 1 blows up before t = 2; x0 = -1 stays finite.
        let u0 = [1.0, -1.0];
        let ens = ensemble_final(&ev, |_| Box::new(Rk4::new()), &u0, &[], 0.0, 2.0, &cfg);

        assert!(matches!(ens.status[0], TrajStatus::Failed(_)));
        assert!(
            ens.row(0).iter().all(|x| x.is_nan()),
            "failed row must be NaN"
        );
        assert!(ens.status[1].is_ok());
        assert!(ens.row(1)[0].is_finite());
        assert_eq!(ens.n_failed(), 1);
    }

    #[test]
    fn empty_batch_is_handled() {
        let ev = VmEval::new(decay());
        let cfg = IntegrateConfig::new(0.01);
        let ens = ensemble_final(&ev, |_| Box::new(Rk4::new()), &[], &[], 0.0, 1.0, &cfg);
        assert_eq!(ens.n_ic(), 0);
        assert!(ens.states.is_empty());
    }

    #[test]
    fn direct_row_write_preserves_index_order_with_interleaved_failures() {
        // Regression for the direct-into-row-slice ensemble write (no per-traj
        // Vec + second copy): an interleaved ok/fail/ok/fail pattern must land
        // each trajectory in its OWN row in input order, with NaN confined to the
        // failed rows — even under a forced multi-thread pool where workers finish
        // out of order. blowup() diverges for x0 > 0 and decays for x0 < 0.
        let ev = VmEval::new(blowup());
        let cfg = IntegrateConfig::new(0.01);
        let n_ic = 64;
        // Even index → +1 (blows up before t=2), odd index → -1 (stays finite).
        let u0: Vec<f64> = (0..n_ic)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        let ens = pool
            .install(|| ensemble_final(&ev, |_| Box::new(Rk4::new()), &u0, &[], 0.0, 2.0, &cfg));

        assert_eq!(ens.n_ic(), n_ic);
        assert_eq!(ens.n_failed(), n_ic / 2);
        for i in 0..n_ic {
            if i % 2 == 0 {
                assert!(
                    matches!(ens.status[i], TrajStatus::Failed(_)),
                    "row {i} should have diverged"
                );
                assert!(ens.row(i)[0].is_nan(), "diverged row {i} must be NaN");
            } else {
                assert!(ens.status[i].is_ok(), "row {i} should be finite");
                assert!(
                    ens.row(i)[0].is_finite(),
                    "finite row {i} must not be NaN-smeared by a neighbour"
                );
                // -1 decaying under dx/dt = x²: x(t) = -1/(1+t), so x(2) = -1/3.
                assert!(
                    (ens.row(i)[0] - (-1.0 / 3.0)).abs() < 1e-6,
                    "row {i}: {} vs -1/3",
                    ens.row(i)[0]
                );
            }
        }
    }
}
