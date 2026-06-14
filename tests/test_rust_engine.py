"""End-to-end tests for the ``tsdynamics._rust`` engine binding (stream E7).

These exercise the compiled extension through the public Python seam
(:mod:`tsdynamics.engine.run`) and via a few direct ``_rust`` calls for the
error paths. They are skipped wholesale when the extension is not built (the
default ``ci.yml`` Python job runs without it); the dedicated
``engine-bindings.yml`` job builds ``_rust`` and runs them for real.

The correctness bar:

- ``eval_rhs`` / ``eval_jac`` reproduce the pure-Python reference evaluator
  **bit-for-bit** (the interpreter mirrors the reference instruction-for-
  instruction), so they are the tolerance-tight, divergence-free signal.
- ``integrate`` / ``ensemble`` match the reference (SciPy ``solve_ivp`` over the
  same tape) on short, non-chaotic windows; maps match exactly.
"""

import numpy as np
import pytest

# The whole module is meaningless without the compiled engine.
_rust = pytest.importorskip("tsdynamics._rust")

import tsdynamics as ts  # noqa: E402
from tsdynamics.engine import run  # noqa: E402
from tsdynamics.engine.compile import lower_ode  # noqa: E402

# A small, representative spread: a 3-D chaotic flow, a stiffer-ish 3-D flow,
# and two maps. Kept short so integration stays in the non-chaotic regime where
# two correct integrators must agree tightly.
ODE_SYSTEMS = ["Lorenz", "Rossler"]
MAP_SYSTEMS = ["Henon", "Logistic"]

# Deterministic, in-basin initial conditions for the map tests. Hénon has no
# `default_ic`, so a random `U[0, 1)^2` IC (the `resolve_ic(None)` fallback)
# lands outside its bounded basin often enough to make the map test flaky —
# diverging to a non-finite state before 200 iterations. Pin a known on-attractor
# IC per map so the engine-vs-reference comparison is reproducible.
_MAP_IC = {"Henon": [0.1, 0.1], "Logistic": [0.3]}


def _sys(name):
    return getattr(ts, name)()


# ---------------------------------------------------------------------------
# Registry / dead-strip smoke test (the hand-off the F2 registry asked E7 for)
# ---------------------------------------------------------------------------


def test_solver_registry_survived_into_the_cdylib():
    names = _rust.solvers()
    assert names, "solver registry is empty — link-time registrations were dead-stripped"
    # The four explicit kernels (E3) and the two implicit ones (E4) must all be
    # reachable by name from the wheel.
    for expected in ("rk4", "rk45", "tsit5", "dop853", "rosenbrock", "trbdf2"):
        assert expected in names, f"{expected} missing from {names}"


def test_version_string():
    assert isinstance(_rust._version(), str)


# ---------------------------------------------------------------------------
# Pointwise evaluation == the reference, to machine precision
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ODE_SYSTEMS)
def test_eval_rhs_matches_reference_bit_for_bit(name):
    system = _sys(name)
    rng = np.random.default_rng(0)
    for _ in range(5):
        u = rng.standard_normal(system.dim)
        ref = run.eval_rhs(system, u, backend="reference")
        eng = run.eval_rhs(system, u, backend="interp")
        assert eng.shape == (system.dim,)
        np.testing.assert_array_equal(eng, ref)


@pytest.mark.parametrize("name", ODE_SYSTEMS)
def test_eval_jac_matches_reference_bit_for_bit(name):
    system = _sys(name)
    rng = np.random.default_rng(1)
    for _ in range(5):
        u = rng.standard_normal(system.dim)
        d_ref, j_ref = run.eval_jac(system, u, backend="reference")
        d_eng, j_eng = run.eval_jac(system, u, backend="interp")
        assert j_eng.shape == (system.dim, system.dim)
        np.testing.assert_array_equal(d_eng, d_ref)
        np.testing.assert_array_equal(j_eng, j_ref)


# ---------------------------------------------------------------------------
# Integration == the reference (short window) + shape/IC invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ODE_SYSTEMS)
def test_integrate_dense_matches_reference_short_window(name):
    system = _sys(name)
    ic = system.resolve_ic(None)
    kw = dict(final_time=3.0, dt=0.05, ic=ic, method="RK45", rtol=1e-10, atol=1e-12)
    eng = run.integrate(system, backend="interp", **kw)
    ref = run.integrate(system, backend="reference", **kw)
    assert eng.y.shape == ref.y.shape
    # First sampled row is the initial condition.
    np.testing.assert_allclose(eng.y[0], np.asarray(ic, dtype=float))
    np.testing.assert_allclose(eng.y, ref.y, atol=1e-6, rtol=1e-6)
    assert eng.meta["engine"] == "rust"


@pytest.mark.parametrize("name", ODE_SYSTEMS)
def test_ensemble_matches_reference(name):
    system = _sys(name)
    rng = np.random.default_rng(2)
    ics = system.resolve_ic(None) + 0.1 * rng.standard_normal((6, system.dim))
    kw = dict(final_time=2.0, method="RK45", rtol=1e-10, atol=1e-12)
    eng = run.ensemble(system, ics, backend="interp", **kw)
    ref = run.ensemble(system, ics, backend="reference", **kw)
    assert eng.shape == (6, system.dim)
    np.testing.assert_allclose(eng, ref, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("name", MAP_SYSTEMS)
def test_map_iteration_matches_reference_exactly(name):
    system = _sys(name)
    ic = _MAP_IC[name]
    eng = run.integrate(system, final_time=200, backend="interp", ic=ic)
    ref = run.integrate(system, final_time=200, backend="reference", ic=ic)
    assert eng.y.shape == ref.y.shape
    np.testing.assert_array_equal(eng.y, ref.y)


def test_compiled_map_diverges_loudly():
    # The compiled (interp) map loop must raise on a non-finite iterate rather
    # than return inf/NaN rows — the same diverge-loudly contract the reference
    # loop enforces. Logistic from x0 = 2 (outside [0, 1]) escapes to -inf.
    with pytest.raises(RuntimeError, match="diverged"):
        ts.Logistic().iterate(steps=60, ic=[2.0], backend="interp")


# ---------------------------------------------------------------------------
# Error paths — the binding maps each EngineError to the right exception type
# ---------------------------------------------------------------------------


def _lorenz_arrays(with_jacobian=False):
    tape = lower_ode(ts.Lorenz(), with_jacobian=with_jacobian)
    return tape.to_arrays()


def test_unknown_method_raises_value_error():
    arrays = _lorenz_arrays()
    ic = np.array([1.0, 1.0, 1.0])
    p = np.array([10.0, 28.0, 8.0 / 3.0])
    t_eval = np.array([0.0, 0.1])
    with pytest.raises(ValueError, match="unknown method"):
        _rust.integrate_dense(*arrays, ic, p, t_eval, "no-such-method", 1e-6, 1e-9, False)


def test_jit_backend_raises_not_implemented():
    arrays = _lorenz_arrays()
    ic = np.array([1.0, 1.0, 1.0])
    p = np.array([10.0, 28.0, 8.0 / 3.0])
    t_eval = np.array([0.0, 0.1])
    with pytest.raises(NotImplementedError):
        _rust.integrate_dense(*arrays, ic, p, t_eval, "rk45", 1e-6, 1e-9, True)


def test_malformed_tape_raises_value_error():
    # ops contains an unknown opcode (99); from_arrays must reject it.
    ops = np.array([1, 2, 99], dtype=np.int32)
    a = np.array([0, 0, 0], dtype=np.int32)
    b = np.array([0, 0, 1], dtype=np.int32)
    imm = np.zeros(3, dtype=np.float64)
    outputs = np.array([2], dtype=np.int32)
    jac = np.empty(0, dtype=np.int32)
    u = np.array([1.0])
    p = np.array([1.0])
    with pytest.raises(ValueError):
        _rust.eval_rhs(ops, a, b, imm, outputs, jac, 1, 1, u, p, 0.0)


def test_divergence_raises_runtime_error():
    # dx/dt = x^2 blows up in finite time from x0 = 1; integrating past the
    # singularity must raise loudly rather than return plausible numbers.
    # Tape: state u0 -> Mul(u0, u0) -> output reg 1.
    ops = np.array([1, 12], dtype=np.int32)  # State, Mul
    a = np.array([0, 0], dtype=np.int32)
    b = np.array([0, 0], dtype=np.int32)
    imm = np.zeros(2, dtype=np.float64)
    outputs = np.array([1], dtype=np.int32)
    jac = np.empty(0, dtype=np.int32)
    ic = np.array([1.0])
    p = np.empty(0)
    t_eval = np.array([0.0, 0.5, 1.5, 2.0])
    with pytest.raises(RuntimeError, match="diverged"):
        _rust.integrate_dense(
            ops, a, b, imm, outputs, jac, 1, 0, ic, p, t_eval, "rk45", 1e-8, 1e-10, False
        )
