//! Criterion benches for the interpreter's **dead-register elimination**.
//!
//! `Interpreter::eval` runs the tape behind a precomputed liveness mask
//! (`rhs_live`) so a Jacobian-bearing tape never computes the registers that
//! only the Jacobian outputs read. That optimisation is correctness-tested
//! (`eval` is bit-for-bit equal with or without the mask, by construction) and
//! was therefore speed-**untested**: deleting the mask would leave the whole
//! suite green while roughly tripling every stiff-solver RHS evaluation.
//!
//! # What the three arms mean
//!
//! For the *same* right-hand side, lowered two ways:
//!
//! - `eval/plain_tape` — `eval` on the tape without Jacobian outputs. The
//!   floor: there is nothing dead to skip.
//! - `eval/jac_bearing_tape` — `eval` on the tape *with* Jacobian outputs. With
//!   DCE intact this should sit **on top of the floor**; without it, it climbs
//!   toward the third arm.
//! - `eval_jac/jac_bearing_tape` — the full pass (mask deliberately off, since
//!   the Jacobian needs every register). The ceiling, and the scale that makes
//!   the gap between the first two readable.
//!
//! So the regression signal is not an absolute number (which drifts with the
//! machine) but the **ratio** `eval/jac_bearing_tape ÷ eval/plain_tape`: ~1 when
//! DCE works, ~`eval_jac ÷ eval` when it has been lost.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use tsdyn_ir::{Tape, TapeBuilder};
use tsdyn_vm::Interpreter;

/// The Lorenz right-hand side, optionally carrying its analytic Jacobian.
///
/// `dx = σ(y − x); dy = x(ρ − z) − y; dz = xy − βz`, with `σ = p0, ρ = p1,
/// β = p2` — the same fixture the crate's unit tests use, so the bench measures
/// the tape the tests validate. The Jacobian registers (`neg_sigma`, `zero`,
/// `neg_one`, `neg_x`, `neg_beta`) are exactly the "Jacobian-only"
/// subexpressions DCE exists to skip.
fn lorenz(with_jacobian: bool) -> Tape {
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

    if !with_jacobian {
        return b.finish(&[dx, dy, dz], &[], 3, 3).unwrap();
    }
    let neg_sigma = b.neg(sigma);
    let zero = b.constant(0.0);
    let neg_one = b.constant(-1.0);
    let neg_x = b.neg(x);
    let neg_beta = b.neg(beta);
    b.finish(
        &[dx, dy, dz],
        &[
            neg_sigma, sigma, zero, // row 0
            rmz, neg_one, neg_x, // row 1
            y, x, neg_beta, // row 2
        ],
        3,
        3,
    )
    .unwrap()
}

fn bench_dead_register_elimination(c: &mut Criterion) {
    let plain = Interpreter::new(lorenz(false));
    let jac = Interpreter::new(lorenz(true));
    let u = [1.0, 1.0, 1.0];
    let p = [10.0, 28.0, 8.0 / 3.0];

    let mut group = c.benchmark_group("interp");

    let mut scratch = vec![0.0; plain.n_scratch()];
    let mut deriv = vec![0.0; plain.dim()];
    group.bench_function("eval/plain_tape", |b| {
        b.iter(|| {
            plain.eval(black_box(&u), black_box(&p), 0.0, &mut scratch, &mut deriv);
            black_box(deriv[0])
        })
    });

    let mut scratch_j = vec![0.0; jac.n_scratch()];
    let mut deriv_j = vec![0.0; jac.dim()];
    group.bench_function("eval/jac_bearing_tape", |b| {
        b.iter(|| {
            jac.eval(
                black_box(&u),
                black_box(&p),
                0.0,
                &mut scratch_j,
                &mut deriv_j,
            );
            black_box(deriv_j[0])
        })
    });

    let mut jac_out = vec![0.0; jac.dim() * jac.dim()];
    group.bench_function("eval_jac/jac_bearing_tape", |b| {
        b.iter(|| {
            jac.eval_jac(
                black_box(&u),
                black_box(&p),
                0.0,
                &mut scratch_j,
                &mut deriv_j,
                &mut jac_out,
            );
            black_box(jac_out[0])
        })
    });

    group.finish();
}

criterion_group! {
    name = benches;
    // Small sample counts and short measurement windows: these are nanosecond
    // kernels, so the statistics converge fast, and the suite has to be cheap
    // enough to run on every push (see `.github/workflows/perf-engine.yml`).
    config = Criterion::default()
        .sample_size(50)
        .warm_up_time(std::time::Duration::from_millis(500))
        .measurement_time(std::time::Duration::from_secs(2));
    targets = bench_dead_register_elimination
}
criterion_main!(benches);
