"""Coverage for problem builders and run entry points.

:mod:`tsdynamics.engine.problem` (per-family Problem bundles) and
:mod:`tsdynamics.engine.run` (backend selection + integrate/ensemble dispatch).
Runs without the compiled engine: the ``reference`` backend exercises the full
path in pure Python, and the engine seam is checked through a fake module so the
``interp`` / ``jit`` dispatch is verified without ``tsdynamics._rust``.
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.engine import run
from tsdynamics.engine.problem import (
    DDEProblem,
    MapProblem,
    ODEProblem,
    SDEProblem,
    build_problem,
    ode_problem,
)
from tsdynamics.engine.run import EngineNotAvailableError, resolve_backend
from tsdynamics.families import Trajectory


# A diagonal-Itô SDE stand-in (the E-SDE base class is still a stub).
class _OU(ts.ContinuousSystem):
    params = {"theta": 1.5, "mu": 0.2, "sigma": 0.3}
    dim = 1

    @staticmethod
    def _equations(y, t, theta, mu, sigma):
        return [theta * (mu - y(0))]

    @staticmethod
    def _drift(y, t, theta, mu, sigma):
        return [theta * (mu - y(0))]

    @staticmethod
    def _diffusion(y, t, theta, mu, sigma):
        return [sigma + 0.0 * y(0)]


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------


def test_resolve_backend_canonicalises() -> None:
    assert resolve_backend("auto") == "interp"
    assert resolve_backend("interp") == "interp"
    assert resolve_backend("JIT") == "jit"
    assert resolve_backend("Reference") == "reference"


def test_resolve_backend_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        resolve_backend("cranelift-please")


# ---------------------------------------------------------------------------
# Problem builders + family dispatch
# ---------------------------------------------------------------------------


def test_build_problem_dispatches_by_family() -> None:
    assert isinstance(build_problem(ts.Lorenz()), ODEProblem)
    assert isinstance(build_problem(ts.Henon()), MapProblem)
    assert isinstance(build_problem(ts.MackeyGlass()), DDEProblem)
    assert isinstance(build_problem(_OU()), SDEProblem)


def test_ode_problem_carries_tape_ic_and_params() -> None:
    lor = ts.Lorenz(ic=[1.0, 2.0, 3.0])
    prob = ode_problem(lor, ic=[1.0, 2.0, 3.0], with_jacobian=True)
    assert prob.dim == 3
    assert prob.tape.has_jacobian
    np.testing.assert_array_equal(prob.ic, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(prob.params_vec(), [10.0, 28.0, 8.0 / 3.0])


def test_params_vec_reads_live_for_sweeps() -> None:
    """A control-parameter change is reflected by ``params_vec`` with no recompile."""
    lor = ts.Lorenz()
    prob = ode_problem(lor)
    tape_before = prob.tape
    lor.rho = 40.0
    assert prob.params_vec()[1] == 40.0  # rho is the 2nd control param
    assert prob.tape is tape_before  # same compiled tape — only the vector changed


def test_map_problem_has_no_params_vector() -> None:
    prob = build_problem(ts.Henon())
    assert prob.params_vec().size == 0  # map params folded into the tape


def test_dde_problem_exposes_delays() -> None:
    prob = build_problem(ts.MackeyGlass())
    assert prob.max_delay == pytest.approx(float(ts.MackeyGlass().tau))
    assert prob.delays == [pytest.approx(float(ts.MackeyGlass().tau))]


def test_build_problem_rejects_unknown_family() -> None:
    class Foo:
        dim = 1
        params: dict = {}

    with pytest.raises(TypeError, match="unrecognised family"):
        build_problem(Foo())


# ---------------------------------------------------------------------------
# Reference backend — runnable today (pure Python)
# ---------------------------------------------------------------------------


def test_reference_ode_integrate_returns_trajectory() -> None:
    lor = ts.Lorenz(ic=[1.0, 1.0, 1.0])
    traj = run.integrate(
        lor, final_time=2.0, dt=0.05, ic=[1.0, 1.0, 1.0], backend="reference", method="DOP853"
    )
    assert isinstance(traj, Trajectory)
    assert traj.y.shape == (traj.t.size, 3)
    assert traj.meta["engine"] == "rust" and traj.meta["family"] == "ode"


def test_reference_ode_integrates_the_lowered_tape() -> None:
    """Integrating the lowered tape matches integrating the symbolic RHS directly.

    Both use the same SciPy solver at the same tolerance, so the only difference
    is RHS provenance (lowered tape vs ``_rhs_numeric``).  Agreement over the
    whole window confirms the tape integrates correctly — no compilation needed
    (this stays in the fast tier).
    """
    from scipy.integrate import solve_ivp

    lor = ts.Lorenz(ic=[0.5, 0.5, 0.5])
    tape_traj = run.integrate(
        lor,
        final_time=2.0,
        dt=0.05,
        ic=[0.5, 0.5, 0.5],
        backend="reference",
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
    )
    f = lor._rhs_numeric()
    sol = solve_ivp(
        lambda t, u: f(u, t),
        (0.0, 2.0),
        np.array([0.5, 0.5, 0.5]),
        t_eval=tape_traj.t,
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
    )
    assert np.allclose(tape_traj.y, sol.y.T, rtol=1e-6, atol=1e-8)


def test_reference_map_iterate_matches_step_exactly() -> None:
    """The first reference map step equals ``_step`` to machine precision."""
    from tsdynamics.families.discrete import _unwrap_static

    h = ts.Henon(ic=[0.1, 0.1])
    traj = run.integrate(h, final_time=5, ic=[0.1, 0.1], backend="reference")
    step = _unwrap_static(type(h)._step)
    expected = np.asarray(step(np.array([0.1, 0.1]), *h.params.as_tuple()), dtype=float)
    np.testing.assert_allclose(traj.y[0], expected, rtol=1e-12, atol=1e-14)
    assert traj.y.shape == (5, 2)


def test_reference_map_diverges_loudly() -> None:
    """A diverging map raises rather than returning silent inf/NaN rows.

    The reference map iterator enforces the "diverge loudly" contract: a
    non-finite iterate is reported (naming the 0-based iterate index), never
    handed back as plausible data — matching the ODE finiteness guard in
    ``_run_continuous`` and the contract the compiled map loop will enforce
    (stream E-MAP).
    """
    import re

    from tsdynamics.engine.compile import eval_tape
    from tsdynamics.engine.problem import map_problem

    # Independently locate the 0-based iterate where the lowered tape first goes
    # non-finite, so the index assertion is pinned to the actual blow-up rather
    # than to the Logistic default parameter (and would catch an off-by-one in
    # the reported index).
    prob = map_problem(ts.Logistic(ic=[2.0]))
    x = np.asarray(prob.ic, dtype=float).ravel()
    expected_idx = None
    for i in range(60):
        x = eval_tape(prob.tape, x)
        if not np.isfinite(x).all():
            expected_idx = i
            break
    assert expected_idx is not None, "Logistic ic=[2.0] should diverge within 60 steps"

    # Logistic with an initial condition outside [0, 1] escapes to -inf.
    with pytest.raises(RuntimeError, match=r"diverged.*iteration \d+") as exc:
        run.integrate(ts.Logistic(), final_time=60, ic=[2.0], backend="reference")
    msg = str(exc.value)
    assert "Logistic" in msg
    reported = int(re.search(r"iteration (\d+)", msg).group(1))
    assert reported == expected_idx, f"reported iteration {reported}, expected {expected_idx}"


def test_reference_map_finite_orbit_iterates_without_raising() -> None:
    """A bounded orbit returns a full finite trajectory — the per-iterate guard
    must not fire on a healthy run (a ``not``-inversion or over-eager check would
    break every normal map iteration)."""
    traj = run.integrate(ts.Logistic(), final_time=60, ic=[0.2], backend="reference")
    assert traj.y.shape == (60, 1)
    assert np.all(np.isfinite(traj.y))


def test_map_time_axis_starts_at_n0() -> None:
    """A warm-restart map carries its starting iteration index on the time axis."""
    from tsdynamics.engine.problem import map_problem

    prob = map_problem(ts.Henon(ic=[0.1, 0.1]), n0=100)
    traj = run.integrate(prob, final_time=5, backend="reference")
    np.testing.assert_array_equal(traj.t, np.arange(100, 105))


def test_reference_rejects_dde_and_sde() -> None:
    with pytest.raises(NotImplementedError, match="reference"):
        run.integrate(ts.MackeyGlass(), final_time=1.0, backend="reference")
    with pytest.raises(NotImplementedError, match="reference"):
        run.integrate(_OU(), final_time=1.0, backend="reference")


def test_eval_rhs_reference_matches_symbolic(rng) -> None:
    lor = ts.Lorenz()
    f = lor._rhs_numeric()
    for _ in range(20):
        u = rng.standard_normal(3)
        got = run.eval_rhs(lor, u, 0.0, backend="reference")
        np.testing.assert_allclose(got, f(u, 0.0), rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Engine backend — availability + dispatch plumbing (mock the FFI surface)
# ---------------------------------------------------------------------------


def test_engine_backend_raises_when_extension_absent() -> None:
    """``interp``/``jit`` raise a clear error until ``tsdynamics._rust`` is built."""
    try:
        import tsdynamics._rust  # noqa: F401
    except ImportError:
        pass
    else:  # E7 built the engine extension — the "absent" path no longer applies.
        pytest.skip("the engine extension (tsdynamics._rust) is built")
    with pytest.raises(EngineNotAvailableError, match="tsdynamics._rust"):
        run.integrate(ts.Lorenz(), final_time=1.0, backend="interp")


class _FakeEngine:
    """A stand-in for ``tsdynamics._rust`` recording the calls it receives."""

    def __init__(self) -> None:
        self.integrate_calls: list[tuple] = []
        self.ensemble_calls: list[tuple] = []

    def integrate_dense(self, *args):
        self.integrate_calls.append(args)
        # Payload: (ops, a, b, imm, outputs, jac, n_state, n_param, ic, params,
        #           t_eval, method, rtol, atol, jit).
        outputs, t_eval = args[4], args[10]
        return np.zeros((len(t_eval), outputs.size))

    def integrate_ensemble_final(self, *args):
        self.ensemble_calls.append(args)
        # Payload: (ops, a, b, imm, outputs, jac, n_state, n_param, ics, params,
        #           t0, t1, first_step, method, rtol, atol, jit).
        ics = args[8]
        return np.zeros_like(ics)


def test_engine_integrate_dispatch_payload(monkeypatch) -> None:
    """``integrate(backend='interp')`` hands the engine the right tape + runtime data."""
    fake = _FakeEngine()
    monkeypatch.setattr(run, "_engine", lambda: fake)
    lor = ts.Lorenz(ic=[1.0, 1.0, 1.0])
    traj = run.integrate(
        lor, final_time=2.0, dt=0.1, ic=[1.0, 1.0, 1.0], backend="jit", method="RK45"
    )
    assert len(fake.integrate_calls) == 1
    args = fake.integrate_calls[0]
    ops, a, b, imm, outputs, jac, n_state, n_param, ic, params, t_eval, method, rtol, atol, jit = (
        args
    )
    assert (n_state, n_param) == (3, 3)
    np.testing.assert_array_equal(ic, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(params, [10.0, 28.0, 8.0 / 3.0])
    assert method == "RK45" and jit is True
    assert outputs.size == 3
    assert traj.y.shape == (t_eval.size, 3)


def test_engine_ensemble_dispatch_payload(monkeypatch) -> None:
    fake = _FakeEngine()
    monkeypatch.setattr(run, "_engine", lambda: fake)
    lor = ts.Lorenz()
    ics = np.random.default_rng(0).standard_normal((5, 3))
    out = run.ensemble(lor, ics, final_time=3.0, dt=0.05, backend="interp")
    assert out.shape == (5, 3)
    assert len(fake.ensemble_calls) == 1
    a = fake.ensemble_calls[0]
    np.testing.assert_array_equal(a[8], ics)  # the ics batch
    assert a[12] == pytest.approx(0.05)  # the fixed-step cadence (the user's dt)
    assert a[-1] is False  # jit flag (interp)


def test_ensemble_validates_ic_shape() -> None:
    with pytest.raises(ValueError, match="ics must be"):
        run.ensemble(ts.Lorenz(), np.zeros((4, 2)), final_time=1.0, backend="reference")


def test_reference_ensemble_runs_in_python() -> None:
    """The reference ensemble loops the pure-Python integrator (no engine needed)."""
    lor = ts.Lorenz()
    ics = np.array([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]])
    out = run.ensemble(lor, ics, final_time=1.0, backend="reference", method="DOP853")
    assert out.shape == (2, 3)
    assert np.all(np.isfinite(out))
