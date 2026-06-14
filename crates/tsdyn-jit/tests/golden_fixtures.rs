//! The JIT, run over real v2-emitted tapes — the catalogue half of stream E2's
//! acceptance.
//!
//! Stream F1 froze golden fixtures under `tsdyn-ir/tests/fixtures/`: each holds
//! the integer tape arrays the v2 Python emitter produced for a real system, plus
//! the right-hand side and Jacobian sampled at fixed checkpoints by the system's
//! own symbolic engine (the engine-independent ground truth). This test rebuilds
//! each tape, JIT-compiles it, and asserts the native code (a) reproduces the
//! symbolic checkpoints to ~1e-12 — i.e. matches the intended RHS on real systems
//! — and (b) agrees with the canonical `reference` evaluator **bit-for-bit**.
//!
//! The fixtures are read from the sibling F1 crate (a frozen, read-only contract
//! artifact); this stream does not own or edit them.

use std::path::{Path, PathBuf};

use tsdyn_ir::{reference, Tape};
use tsdyn_jit::JitEvaluator;

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
            None => (line, ""),
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

/// The F1 fixtures live in the sibling `tsdyn-ir` crate; resolve them from this
/// crate's manifest dir (absolute, so the path holds regardless of CWD).
fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../tsdyn-ir/tests/fixtures")
}

#[track_caller]
fn assert_close(label: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let tol = 1e-12 * w.abs().max(1.0);
        assert!(
            (g - w).abs() <= tol,
            "{label}[{i}]: JIT gave {g}, symbolic ground truth {w} (|Δ| = {:e})",
            (g - w).abs()
        );
    }
}

#[track_caller]
fn assert_bits_eq(label: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{label}[{i}]: JIT gave {g:?}, reference gave {w:?}"
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

    let jit = JitEvaluator::new(&tape).unwrap_or_else(|e| panic!("{}: JIT failed: {e}", fx.system));
    for (c, cp) in fx.checkpoints.iter().enumerate() {
        let (deriv, jac) = jit.eval_jac_alloc(&cp.u, &fx.params, cp.t);

        // (a) reproduces the symbolic ground truth on a real emitted tape …
        assert_close(&format!("{} cp{c} deriv", fx.system), &deriv, &cp.deriv);
        assert_close(&format!("{} cp{c} jac", fx.system), &jac, &cp.jac);

        // (b) … and is bit-identical to the canonical reference evaluator.
        let (rd, rj) = reference::eval_jac_alloc(&tape, &cp.u, &fx.params, cp.t);
        assert_bits_eq(
            &format!("{} cp{c} deriv vs reference", fx.system),
            &deriv,
            &rd,
        );
        assert_bits_eq(&format!("{} cp{c} jac vs reference", fx.system), &jac, &rj);

        // The RHS-only entry point must agree with the combined one.
        let deriv_only = jit.eval_alloc(&cp.u, &fx.params, cp.t);
        assert_bits_eq(
            &format!("{} cp{c} eval vs eval_jac deriv", fx.system),
            &deriv_only,
            &deriv,
        );
    }
}

#[test]
fn jit_reproduces_all_golden_fixtures() {
    let dir = fixtures_dir();
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read F1 fixtures at {}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().map(|x| x == "txt").unwrap_or(false))
        .collect();
    entries.sort();

    let mut checked = 0;
    for path in &entries {
        check_fixture(path);
        checked += 1;
    }
    assert!(
        checked >= 6,
        "expected >= 6 golden fixtures, found {checked}"
    );
}
