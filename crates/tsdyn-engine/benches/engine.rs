//! Criterion benches for the engine's two end-to-end hot paths.
//!
//! The kernel-level benches (`tsdyn-solvers`, `tsdyn-vm`) isolate individual
//! optimisations; these two measure what a user actually waits for — a full
//! `integrate` and a full map `iterate` through the same loops the FFI bridge
//! drives. They are the coarse net that catches a regression the isolated
//! benches would miss (an extra allocation per output point, a lost buffer
//! reuse, a poller that ticks too often).
//!
//! Both run on the interpreter (`tsdyn-vm`), not the JIT: the engine crate does
//! not depend on `tsdyn-jit` (that edge lives in `tsdyn-core`), and the
//! interpreter is the backend the default `backend="interp"` uses anyway.

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use tsdyn_engine::{integrate_grid, iterate_dense, IntegrateConfig};
use tsdyn_ir::{Evaluator, Tape, TapeBuilder};
use tsdyn_solvers::explicit::Rk45;
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

/// Lorenz `dx = σ(y − x); dy = x(ρ − z) − y; dz = xy − βz`.
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

/// The Hénon map `x' = 1 − a x² + y; y' = b x` — a map tape's "RHS" is the next
/// state, and the catalogue bakes `a`/`b` in as constants.
fn henon() -> Tape {
    let mut b = TapeBuilder::new();
    let x = b.state(0);
    let y = b.state(1);
    let a = b.constant(1.4);
    let bb = b.constant(0.3);
    let one = b.constant(1.0);
    let x2 = b.mul(x, x);
    let ax2 = b.mul(a, x2);
    let one_m = b.sub(one, ax2);
    let xn = b.add(one_m, y);
    let yn = b.mul(bb, x);
    b.finish(&[xn, yn], &[], 2, 0).unwrap()
}

fn bench_integrate(c: &mut Criterion) {
    let ev = VmEval(Interpreter::new(lorenz()));
    let p = vec![10.0, 28.0, 8.0 / 3.0];
    let u0 = vec![1.0, 1.0, 1.0];
    // 1000 output points over 10 time units — a short but realistic run, the
    // size a docs figure or an interactive call uses.
    let t_eval: Vec<f64> = (0..1000).map(|i| i as f64 * 0.01).collect();
    let cfg = IntegrateConfig::new(0.01);

    let mut group = c.benchmark_group("engine");
    group.bench_function("integrate_grid/lorenz_rk45_1000pts", |b| {
        b.iter(|| {
            let mut solver = Rk45::new();
            let y = integrate_grid(&ev, &mut solver, &u0, &p, &t_eval, &cfg).unwrap();
            black_box(y[0])
        })
    });
    group.finish();
}

fn bench_iterate(c: &mut Criterion) {
    let ev = VmEval(Interpreter::new(henon()));
    let u0 = vec![0.1, 0.1];

    let mut group = c.benchmark_group("engine");
    group.bench_function("iterate_dense/henon_10000_steps", |b| {
        b.iter(|| {
            let y = iterate_dense(&ev, &u0, &[], 10_000).unwrap();
            black_box(y[0])
        })
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(3));
    targets = bench_integrate, bench_iterate
}
criterion_main!(benches);
