//! Criterion benches for the two solver-kernel work-saving optimisations.
//!
//! Both are **answer-preserving**, which is precisely why they need a speed
//! bench: no correctness test can tell whether they are still there.
//!
//! 1. **FSAL stage reuse** (`explicit::rk45::fsal_adaptive_step`, shared by
//!    `rk45`/`tsit5`/`bs3`). A genuine FSAL pair evaluates its last stage at the
//!    new point, which *is* the next step's first stage, so the kernel copies it
//!    instead of re-evaluating `f` — one RHS evaluation saved per accepted step
//!    (1 of 7 for the 5(4) pairs).
//! 2. **Frozen-Jacobian / LU reuse** (`implicit::newton::solve_substage_reuse`,
//!    used by `sdirk2`/`trbdf2`). Substages whose iteration-matrix shift `coef·h`
//!    is bit-equal share one frozen `J` and one LU factorization, so an SDIRK2
//!    step forms **two** factorizations (one at `γ·h` for the full step, one at
//!    `γ·h/2` for the step-doubling half steps) where the always-re-form path
//!    would form six.
//!
//! # What each group measures
//!
//! **FSAL — a true A/B.** The reuse is keyed on the step's start point `(u, t)`,
//! which the bench can steer from outside: the `reused` arm marches forward
//! normally, so each step's first stage matches the cached point; the
//! `recomputed` arm re-seats `(u, t)` to the same start before every step, so the
//! cached point (the *end* of the previous step) never matches and stage 0 is
//! evaluated afresh. Same kernel, same tableau, same step count — the only
//! difference is one RHS evaluation per step. The two arms are not the same
//! trajectory (steering the guard means taking different steps); the *ratio* is
//! the signal, not either absolute number.
//!
//! **LU reuse — an absolute measurement, deliberately.** There is no honest
//! outside-in A/B here: SDIRK2's saving is *intra*-step (its two substages always
//! share the shift `γ·h`, and the two step-doubling half steps share `γ·h/2`), so
//! nothing a caller can vary — including `h` — changes how often the
//! factorization is reused. This group therefore times the real kernel on a
//! stiff system large enough that the `dim × dim` Jacobian evaluation and its
//! `O(dim³)` LU dominate, and the regression signal is the number itself against
//! the committed baseline: losing the reuse takes factorizations per step from
//! two to six, which at this size is a large multiple, not a few percent.
//!
//! The **exact** statement of that contract is a unit test, not this bench:
//! `implicit::sdirk2::tests::frozen_jacobian_is_reused_across_substages_sharing_a_shift`
//! counts `eval_jac` calls and fails deterministically if the reuse is dropped
//! (as does `explicit::rk45::tests::fsal_reuse_saves_one_rhs_eval_per_continued_step`
//! for FSAL). The benches quantify what those savings are *worth*; the tests are
//! what actually blocks a regression.

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use tsdyn_ir::{Evaluator, Tape, TapeBuilder};
use tsdyn_solvers::explicit::Rk45;
use tsdyn_solvers::implicit::{BackwardEuler, Sdirk2};
use tsdyn_solvers::{Solver, SolverState};
use tsdyn_vm::Interpreter;

/// Adapts the interpreter to the `Evaluator` trait (as `tsdyn-core` does).
struct VmEval(Interpreter);

impl Evaluator for VmEval {
    fn dim(&self) -> usize {
        self.0.dim()
    }
    fn n_param(&self) -> usize {
        self.0.n_param()
    }
    fn n_scratch(&self) -> usize {
        self.0.n_scratch()
    }
    fn has_jacobian(&self) -> bool {
        self.0.has_jacobian()
    }
    fn eval(&self, u: &[f64], p: &[f64], t: f64, scratch: &mut [f64], deriv: &mut [f64]) {
        self.0.eval(u, p, t, scratch, deriv);
    }
    fn eval_jac(
        &self,
        u: &[f64],
        p: &[f64],
        t: f64,
        scratch: &mut [f64],
        deriv: &mut [f64],
        jac: &mut [f64],
    ) {
        self.0.eval_jac(u, p, t, scratch, deriv, jac);
    }
}

/// Lorenz `dx = σ(y − x); dy = x(ρ − z) − y; dz = xy − βz`, no Jacobian —
/// the explicit kernels' fixture.
fn lorenz() -> Tape {
    let mut b = TapeBuilder::new();
    let sigma = b.param(0);
    let rho = b.param(1);
    let beta = b.param(2);
    let x = b.state(0);
    let y = b.state(1);
    let z = b.state(2);
    let ymx = b.sub(y, x);
    let dx = b.mul(sigma, ymx);
    let rmz = b.sub(rho, z);
    let xrmz = b.mul(x, rmz);
    let dy = b.sub(xrmz, y);
    let xy = b.mul(x, y);
    let bz = b.mul(beta, z);
    let dz = b.sub(xy, bz);
    b.finish(&[dx, dy, dz], &[], 3, 3).unwrap()
}

/// Dimension of the stiff fixture. Big enough that the `dim × dim` Jacobian
/// evaluation and its `O(dim³)` LU are the dominant per-step cost — which is
/// exactly the work the reuse elides, and what a 2 × 2 toy would hide.
const STIFF_DIM: usize = 16;

/// The heat equation on a periodic ring, `du_i/dt = D(u_{i−1} − 2u_i + u_{i+1})`,
/// with its analytic (tridiagonal + wrap) Jacobian.
///
/// Linear, so modified Newton converges in one iteration whichever frozen `J` it
/// uses: the arms differ in *factorization* work only, never in iteration count.
/// Stiff for large `D`, which is what makes an implicit kernel the right one.
fn stiff_diffusion() -> Tape {
    let n = STIFF_DIM;
    let mut b = TapeBuilder::new();
    let d = b.param(0);
    let two = b.constant(2.0);
    let zero = b.constant(0.0);
    let neg_two_d = {
        let two_d = b.mul(two, d);
        b.neg(two_d)
    };
    let u: Vec<_> = (0..n).map(|i| b.state(i)).collect();
    let outs: Vec<_> = (0..n)
        .map(|i| {
            let two_ui = b.mul(two, u[i]);
            let lap = b.sub(u[(i + n - 1) % n], two_ui);
            let lap = b.add(lap, u[(i + 1) % n]);
            b.mul(d, lap)
        })
        .collect();
    // Row-major dim × dim: D on the two off-diagonals (with wrap), −2D on the
    // diagonal, 0 elsewhere.
    let jac: Vec<_> = (0..n)
        .flat_map(|i| {
            (0..n).map(move |j| {
                if j == i {
                    neg_two_d
                } else if j == (i + n - 1) % n || j == (i + 1) % n {
                    d
                } else {
                    zero
                }
            })
        })
        .collect();
    b.finish(&outs, &jac, n, 1).unwrap()
}

/// Steps taken per benchmark iteration — enough that the per-step difference
/// dominates the fixed setup, few enough to keep an iteration in microseconds.
const STEPS: usize = 64;

fn bench_fsal(c: &mut Criterion) {
    let ev = VmEval(Interpreter::new(lorenz()));
    let p = vec![10.0, 28.0, 8.0 / 3.0];
    let u0 = vec![1.0, 1.0, 1.0];
    let h = 1e-3;

    let mut group = c.benchmark_group("explicit/rk45_fsal");

    group.bench_function("reused", |b| {
        b.iter(|| {
            // Marching forward: each step's first stage is the previous step's
            // cached last stage, so the reuse fires on all but the first.
            let mut solver = Rk45::new();
            let mut st = SolverState::for_evaluator(&ev, u0.clone(), 0.0, p.clone());
            for _ in 0..STEPS {
                black_box(solver.step(&ev, &mut st, h));
            }
            black_box(st.u[0])
        })
    });

    group.bench_function("recomputed", |b| {
        b.iter(|| {
            // Re-seating `(u, t)` before every step makes the cached point stale,
            // so stage 0 is evaluated afresh — the pre-FSAL behaviour.
            let mut solver = Rk45::new();
            let mut st = SolverState::for_evaluator(&ev, u0.clone(), 0.0, p.clone());
            for _ in 0..STEPS {
                st.u.copy_from_slice(&u0);
                st.t = 0.0;
                black_box(solver.step(&ev, &mut st, h));
            }
            black_box(st.u[0])
        })
    });

    group.finish();
}

fn bench_sdirk_lu_reuse(c: &mut Criterion) {
    let ev = VmEval(Interpreter::new(stiff_diffusion()));
    let p = vec![1e3];
    // A smooth bump on the ring — a non-uniform profile, so the diffusion term
    // is genuinely active.
    let u0: Vec<f64> = (0..STIFF_DIM)
        .map(|i| ((i as f64) * std::f64::consts::TAU / STIFF_DIM as f64).sin())
        .collect();
    let h = 1e-4;

    let mut group = c.benchmark_group("implicit");

    // The production kernel: two frozen Jacobians + two LU factorizations per
    // step (one per distinct substage shift), whatever `h` does.
    group.bench_function("sdirk2_lu_reuse/stiff_diffusion_16", |b| {
        b.iter(|| {
            let mut solver = Sdirk2::new();
            let mut st = SolverState::for_evaluator(&ev, u0.clone(), 0.0, p.clone());
            for _ in 0..STEPS {
                black_box(solver.step(&ev, &mut st, h));
            }
            black_box(st.u[0])
        })
    });

    // Backward Euler on the same problem: the always-re-form path
    // (`solve_substage`, no cache at all), which puts a scale on what one
    // freeze + factorization costs here. Not the same method — a comparison of
    // totals would be meaningless — just the reference the absolute number above
    // is read against.
    group.bench_function("backward_euler_always_reform/stiff_diffusion_16", |b| {
        b.iter(|| {
            let mut solver = BackwardEuler::new();
            let mut st = SolverState::for_evaluator(&ev, u0.clone(), 0.0, p.clone());
            for _ in 0..STEPS {
                black_box(solver.step(&ev, &mut st, h));
            }
            black_box(st.u[0])
        })
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(50)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    targets = bench_fsal, bench_sdirk_lu_reuse
}
criterion_main!(benches);
