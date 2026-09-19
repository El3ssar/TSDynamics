//! Criterion benches for Cranelift compile latency and the compiled-evaluator
//! cache.
//!
//! Compile time is the JIT's whole cost model: the generated code is faster than
//! the interpreter, so `backend="jit"` pays off only once the run is long enough
//! to amortise the compile. Two things move that break-even point, and neither
//! is visible to a correctness test:
//!
//! - **the compile itself** — dominated by Cranelift, so a cranelift bump (or a
//!   change to the `opt_level` the codegen module requests) moves it;
//! - **the cache** (`crate::cached_evaluator`) — which removes the compile from
//!   every call after the first, and which a mis-keyed refactor could silently
//!   turn into a permanent miss.
//!
//! The arms therefore measure a cold compile at two tape sizes plus a cache hit.
//! A hit is a hash + a tape equality compare + an `Arc` clone: nanoseconds, and
//! the ratio against `compile/*` is what tells you the cache is alive.

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use tsdyn_ir::{Tape, TapeBuilder};
use tsdyn_jit::{cached_evaluator, clear_cache, JitEvaluator};

/// Lorenz with its analytic Jacobian — a small tape (~20 instructions), the
/// low end of the catalogue.
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
    let neg_sigma = b.neg(sigma);
    let zero = b.constant(0.0);
    let neg_one = b.constant(-1.0);
    let neg_x = b.neg(x);
    let neg_beta = b.neg(beta);
    b.finish(
        &[dx, dy, dz],
        &[
            neg_sigma, sigma, zero, //
            rmz, neg_one, neg_x, //
            y, x, neg_beta,
        ],
        3,
        3,
    )
    .unwrap()
}

/// A method-of-lines chain `du_i/dt = D·(u_{i−1} − 2u_i + u_{i+1}) + u_i(1 − u_i)`
/// on `N` nodes with periodic wrap — the shape of the spatially-extended
/// catalogue systems (Kuramoto–Sivashinsky, Gray–Scott), whose thousands of
/// instructions are where compile latency actually hurts.
fn diffusion_chain(n: usize) -> Tape {
    let mut b = TapeBuilder::new();
    let d = b.param(0);
    let one = b.constant(1.0);
    let two = b.constant(2.0);
    let u: Vec<_> = (0..n).map(|i| b.state(i)).collect();
    let outs: Vec<_> = (0..n)
        .map(|i| {
            let left = u[(i + n - 1) % n];
            let right = u[(i + 1) % n];
            let two_ui = b.mul(two, u[i]);
            let lap = b.sub(left, two_ui);
            let lap = b.add(lap, right);
            let diff = b.mul(d, lap);
            let react = b.sub(one, u[i]);
            let react = b.mul(u[i], react);
            b.add(diff, react)
        })
        .collect();
    b.finish(&outs, &[], n, 1).unwrap()
}

fn bench_compile(c: &mut Criterion) {
    let small = lorenz();
    let large = diffusion_chain(256);

    let mut group = c.benchmark_group("jit");
    group.bench_function("compile/lorenz_with_jacobian", |b| {
        b.iter(|| black_box(JitEvaluator::new(black_box(&small)).unwrap()))
    });
    group.bench_function("compile/diffusion_chain_256", |b| {
        b.iter(|| black_box(JitEvaluator::new(black_box(&large)).unwrap()))
    });
    group.finish();
}

fn bench_cache_hit(c: &mut Criterion) {
    let large = diffusion_chain(256);
    clear_cache();
    // Prime the cache so every measured iteration is a hit.
    let _ = cached_evaluator(&large).unwrap();

    let mut group = c.benchmark_group("jit");
    group.bench_function("cache_hit/diffusion_chain_256", |b| {
        b.iter(|| black_box(cached_evaluator(black_box(&large)).unwrap()))
    });
    group.finish();
    clear_cache();
}

criterion_group! {
    name = benches;
    // A large-tape compile is ~10 ms, so keep the sample count low: this group
    // is the slowest in the suite and still has to fit in a CI job.
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(3));
    targets = bench_compile, bench_cache_hit
}
criterion_main!(benches);
