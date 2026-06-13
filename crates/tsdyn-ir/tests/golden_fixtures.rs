//! Golden fixtures: the Rust half of the Python↔Rust contract.
//!
//! Each `fixtures/<system>.txt` holds the integer tape arrays the v2 Python
//! emitter produced for a real system, plus the symbolic RHS and Jacobian
//! sampled at fixed checkpoints (the ground truth, computed engine-independently
//! by the system's own `_rhs_numeric` / `jacobian`).  This test rebuilds each
//! tape via [`Tape::from_arrays`], evaluates it with the reference evaluator, and
//! asserts it reproduces the symbolic checkpoints to ~1e-12 — i.e. the migrated
//! IR still matches the symbolic right-hand side to machine precision.
//!
//! The fixtures are generated and drift-guarded by `tests/gen_fixtures.py` /
//! `tests/test_ir_contract.py`; regenerate with that script after touching the
//! emitter or the chosen systems.
#![cfg(feature = "reference")]

use std::path::{Path, PathBuf};

use tsdyn_ir::{reference, Tape};

struct Checkpoint {
    u: Vec<f64>,
    t: f64,
    deriv: Vec<f64>,
    jac: Vec<f64>,
}

struct Fixture {
    system: String,
    n_state: usize,
    n_param: usize,
    dim: usize,
    ops: Vec<i32>,
    a: Vec<i32>,
    b: Vec<i32>,
    imm: Vec<f64>,
    outputs: Vec<i32>,
    jac_outputs: Vec<i32>,
    params: Vec<f64>,
    checkpoints: Vec<Checkpoint>,
}

fn ints(rest: &str) -> Vec<i32> {
    rest.split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect()
}

fn floats(rest: &str) -> Vec<f64> {
    rest.split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect()
}

fn parse(text: &str) -> Fixture {
    let mut f = Fixture {
        system: String::new(),
        n_state: 0,
        n_param: 0,
        dim: 0,
        ops: vec![],
        a: vec![],
        b: vec![],
        imm: vec![],
        outputs: vec![],
        jac_outputs: vec![],
        params: vec![],
        checkpoints: vec![],
    };
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (key, rest) = match line.split_once(char::is_whitespace) {
            Some((k, r)) => (k, r.trim()),
            None => (line, ""), // key with no values (e.g. empty `params`)
        };
        match key {
            "system" => f.system = rest.to_string(),
            "n_state" => f.n_state = rest.parse().unwrap(),
            "n_param" => f.n_param = rest.parse().unwrap(),
            "dim" => f.dim = rest.parse().unwrap(),
            "ops" => f.ops = ints(rest),
            "a" => f.a = ints(rest),
            "b" => f.b = ints(rest),
            "imm" => f.imm = floats(rest),
            "outputs" => f.outputs = ints(rest),
            "jac_outputs" => f.jac_outputs = ints(rest),
            "params" => f.params = floats(rest),
            "cp_u" => f.checkpoints.push(Checkpoint {
                u: floats(rest),
                t: 0.0,
                deriv: vec![],
                jac: vec![],
            }),
            "cp_t" => f.checkpoints.last_mut().unwrap().t = rest.parse().unwrap(),
            "cp_deriv" => f.checkpoints.last_mut().unwrap().deriv = floats(rest),
            "cp_jac" => f.checkpoints.last_mut().unwrap().jac = floats(rest),
            other => panic!("unknown fixture key {other:?}"),
        }
    }
    f
}

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// Mixed-tolerance agreement: the tape and the symbolic engine evaluate the same
/// expression by different routes, so they agree to a few ULP; 1e-12 is slack.
#[track_caller]
fn assert_close(label: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let tol = 1e-12 * w.abs().max(1.0);
        assert!(
            (g - w).abs() <= tol,
            "{label}[{i}]: tape gave {g}, symbolic RHS gave {w} (|Δ| = {:e})",
            (g - w).abs()
        );
    }
}

fn check_fixture(path: &Path) {
    let text = std::fs::read_to_string(path).unwrap();
    let fx = parse(&text);

    let tape = Tape::from_arrays(
        &fx.ops,
        &fx.a,
        &fx.b,
        &fx.imm,
        &fx.outputs,
        &fx.jac_outputs,
        fx.n_state,
        fx.n_param,
    )
    .unwrap_or_else(|e| panic!("{}: tape rejected: {e}", fx.system));

    assert_eq!(tape.dim(), fx.dim, "{}: dim", fx.system);
    assert_eq!(tape.n_param(), fx.params.len(), "{}: n_param", fx.system);
    assert!(
        tape.has_jacobian(),
        "{}: fixture should carry a Jacobian",
        fx.system
    );
    assert!(
        !fx.checkpoints.is_empty(),
        "{}: fixture has no checkpoints",
        fx.system
    );

    for (c, cp) in fx.checkpoints.iter().enumerate() {
        let (deriv, jac) = reference::eval_jac_alloc(&tape, &cp.u, &fx.params, cp.t);
        assert_close(&format!("{} cp{c} deriv", fx.system), &deriv, &cp.deriv);
        assert_close(&format!("{} cp{c} jac", fx.system), &jac, &cp.jac);
    }
}

#[test]
fn all_golden_fixtures_match_the_symbolic_rhs() {
    let dir = fixtures_dir();
    let mut checked = 0;
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().map(|x| x == "txt").unwrap_or(false))
        .collect();
    entries.sort();
    for path in &entries {
        check_fixture(path);
        checked += 1;
    }
    // Guard against an empty/relocated fixtures directory silently passing.
    assert!(
        checked >= 6,
        "expected >= 6 golden fixtures, found {checked}"
    );
}
