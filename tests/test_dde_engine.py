"""Engine DDE integration — the Rust method-of-steps path (stream E-DDE).

These exercise ``DelaySystem.integrate(backend="interp")``, which lowers a delay
system to a tape over ``dim + n_slots`` inputs (the delay slots) and integrates it
on the Rust engine (history ring buffer + cubic-Hermite dense interpolation,
reusing the explicit solver kernels).

The whole module is skipped when the compiled extension is not built (the default
``ci.yml`` Python job), like the other ``_rust`` tests; the ``engine-bindings.yml``
job builds it and runs these for real.

The correctness bars:

- **Absolute**: a linear DDE with a closed-form method-of-steps solution
  (``y'(t) = −y(t−1)``, constant past 1 ⇒ ``y(1) = 0``, ``y(2) = −0.5``) — no
  chaotic-divergence caveat.
- **Parity**: the Rust engine matches the v2 JiTCDDE backend within tolerance on
  Mackey–Glass and every built-in delay system (compared on an early window,
  before any chaotic amplification).
"""

from __future__ import annotations

import numpy as np
import pytest
from _sampling import DDE_HISTORIES

# The whole module is meaningless without the compiled engine.
_rust = pytest.importorskip("tsdynamics._rust")

import tsdynamics as ts  # noqa: E402
from tsdynamics import registry  # noqa: E402


def _dde_names() -> list[str]:
    return [e.name for e in registry.all_systems(family="dde")]


class _LinearDDE(ts.DelaySystem):
    """``y'(t) = −y(t − 1)``; constant past 1 ⇒ ``y(1) = 0``, ``y(2) = −0.5``.

    The textbook method-of-steps test: the solution is an exact piecewise
    polynomial, so it pins the integrator's accuracy independently of JiTCDDE.
    """

    params = {"tau": 1.0}
    dim = 1

    @staticmethod
    def _equations(y, t, *, tau):
        return [-y(0, t - tau)]


# ---------------------------------------------------------------------------
# Absolute correctness — the closed-form linear DDE
# ---------------------------------------------------------------------------


def test_linear_dde_matches_method_of_steps_closed_form():
    traj = _LinearDDE().integrate(
        backend="interp", final_time=2.0, dt=0.1, ic=[1.0], rtol=1e-9, atol=1e-11
    )
    assert traj.y[0, 0] == 1.0  # first row is the IC
    i1 = int(np.argmin(np.abs(traj.t - 1.0)))
    i2 = int(np.argmin(np.abs(traj.t - 2.0)))
    assert abs(traj.y[i1, 0] - 0.0) < 1e-6, f"y(1) = {traj.y[i1, 0]}"
    assert abs(traj.y[i2, 0] + 0.5) < 1e-6, f"y(2) = {traj.y[i2, 0]}"


@pytest.mark.parametrize("method", ["rk45", "tsit5", "dop853", "rk4"])
def test_linear_dde_every_explicit_method(method):
    """Each explicit kernel drives the method of steps to the closed form."""
    dt = 0.01 if method == "rk4" else 0.1  # the fixed-step kernel needs a small step
    traj = _LinearDDE().integrate(
        backend="interp", method=method, final_time=2.0, dt=dt, ic=[1.0], rtol=1e-9, atol=1e-11
    )
    i2 = int(np.argmin(np.abs(traj.t - 2.0)))
    assert abs(traj.y[i2, 0] + 0.5) < 1e-4, f"{method}: y(2) = {traj.y[i2, 0]}"


def test_callable_past_is_interpolated():
    """``y'(t) = −y(t−1)`` with past ``y(s) = 1 + s`` ⇒ ``y(t) = 1 − t²/2`` on [0,1]."""
    traj = _LinearDDE().integrate(
        backend="interp",
        final_time=1.0,
        dt=0.1,
        history=lambda s: [1.0 + s],
        rtol=1e-9,
        atol=1e-11,
    )
    for k, t in enumerate(traj.t):
        want = 1.0 - 0.5 * t * t
        assert abs(traj.y[k, 0] - want) < 1e-5, f"t = {t}: {traj.y[k, 0]} vs {want}"


# ---------------------------------------------------------------------------
# Parity with the v2 JiTCDDE backend (slow — JiTCDDE compiles C)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_mackey_glass_matches_jitcdde():
    """The acceptance system: Rust vs JiTCDDE, constant and callable past.

    Mackey–Glass at τ = 17 is only weakly chaotic (λ ≈ 0.009), so a tight
    tolerance keeps the two integrators in lock-step over a long window.
    """
    mg = ts.MackeyGlass()
    common = dict(final_time=120.0, dt=0.5, rtol=1e-8, atol=1e-8)
    # Constant past.
    tj = mg.integrate(backend="jitcdde", ic=[1.2], **common)
    te = mg.integrate(backend="interp", ic=[1.2], **common)
    assert np.abs(tj.y - te.y).max() < 1e-4
    # Callable past.
    hist = lambda s: [1.0 + 0.1 * np.sin(0.2 * s)]  # noqa: E731
    tjh = mg.integrate(backend="jitcdde", history=hist, **common)
    teh = mg.integrate(backend="interp", history=hist, **common)
    assert np.abs(tjh.y - teh.y).max() < 1e-4


@pytest.mark.slow
@pytest.mark.parametrize("name", _dde_names())
def test_engine_matches_jitcdde_early(name):
    """Every built-in DDE: Rust matches JiTCDDE on the pre-chaotic early window."""
    sys = getattr(ts, name)()
    history = DDE_HISTORIES[name]
    common = dict(final_time=20.0, dt=0.25, history=history, rtol=1e-8, atol=1e-8)
    tj = sys.integrate(backend="jitcdde", **common)
    te = sys.integrate(backend="interp", **common)
    assert te.y.shape == tj.y.shape
    assert np.all(np.isfinite(te.y)), f"{name}: engine produced non-finite states"
    mask = tj.t <= 8.0
    diff = np.abs(tj.y[mask] - te.y[mask]).max()
    assert diff < 5e-3, f"{name}: engine vs JiTCDDE early diff {diff}"


# ---------------------------------------------------------------------------
# Provenance and error paths
# ---------------------------------------------------------------------------


def test_engine_trajectory_provenance():
    te = _LinearDDE().integrate(backend="interp", method="tsit5", final_time=1.0, dt=0.5, ic=[1.0])
    assert te.meta["engine"] == "rust"
    assert te.meta["backend"] == "interp"
    assert te.meta["method"] == "tsit5"
    assert te.meta["family"] == "dde"


def test_reference_backend_is_unsupported_for_dde():
    with pytest.raises(NotImplementedError):
        _LinearDDE().integrate(backend="reference", final_time=1.0, dt=0.1, ic=[1.0])


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown method"):
        _LinearDDE().integrate(backend="interp", method="no-such", final_time=1.0, dt=0.1, ic=[1.0])


def test_implicit_method_is_rejected_for_dde():
    # The method of steps drives explicit kernels only (no delayed Jacobian).
    with pytest.raises(NotImplementedError, match="explicit"):
        _LinearDDE().integrate(
            backend="interp", method="rosenbrock", final_time=1.0, dt=0.1, ic=[1.0]
        )


def test_divergence_raises():
    """A super-exponentially growing DDE must surface as an error, not NaNs."""

    class _BlowUp(ts.DelaySystem):
        params = {"tau": 1.0}
        dim = 1

        @staticmethod
        def _equations(y, t, *, tau):
            from symengine import exp

            return [exp(y(0, t - tau))]

    with pytest.raises(RuntimeError):
        _BlowUp().integrate(backend="interp", final_time=100.0, dt=0.25, ic=[1.0])


def test_default_backend_is_jitcdde():
    """The default backend stays v2 (no engine routing) until the migration gate."""
    # A constant past on the engine vs the default must agree, but the default
    # must *not* carry the rust engine provenance.
    traj = _LinearDDE().integrate(final_time=1.0, dt=0.5, ic=[1.0])
    assert traj.meta.get("engine") != "rust"
