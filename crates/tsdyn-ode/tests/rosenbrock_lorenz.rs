//! Regression: Lorenz bytecode fixture integrates under Rosenbrock23 without NaNs.

use tsdyn_core::ir::CompiledOde;
use tsdyn_ode::{integrate_ode, Method};

#[test]
fn lorenz_rosenbrock_short_fixture() {
    let bytecode = include_bytes!("fixtures/lorenz_ode.bin");
    let ode = CompiledOde::from_bytes(bytecode.as_slice()).expect("decode fixture");
    let params = [10.0_f64, 28.0_f64, 8.0_f64 / 3.0_f64];
    let y0 = [1.0_f64, 1.0_f64, 1.0_f64];
    let (t, y_flat) = integrate_ode(
        &ode,
        &params,
        0.0,
        0.01,
        &y0,
        0.005,
        Method::Rosenbrock23,
        1e-6,
        1e-9,
    )
    .expect("integrate");
    assert!(t.len() >= 2);
    assert!(y_flat.iter().all(|x| x.is_finite()));
}
