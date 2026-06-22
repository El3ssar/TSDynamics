"""Dynamic, registry-driven acceptance sweep over *every* engine ODE solver.

This is the end-to-end counterpart to the Rust unit/cross-validation tests
(``crates/tsdyn-solvers/tests/``): it drives each registered solver through the
**full Python → engine → kernel** stack (``backend="interp"``) and checks the
answer against a closed-form reference, so a solver that is registered but
mis-wired (bad tolerance threading, missing Jacobian build, alias not resolving)
fails here.

It is **dynamic**: the parametrisation reads
:func:`tsdynamics.solvers.available_for`, so a newly added solver joins these
sweeps automatically — no edit to this file. Two cheap reference problems keep CI
fast:

* a damped linear oscillator with the analytic solution
  ``e^{-0.1 t}·(cos t, −sin t)`` — *every* ODE solver must integrate it; and
* a stiff linear system (eigenvalues ``−1`` and ``−1000``, ratio 1000) with a
  matrix-exponential closed form — every **implicit** solver must integrate it
  from a step far above the explicit stability limit (the L-/A-stability proof).

The deep per-method numerics (convergence order, embedded-pair order, chaotic
cross-validation) live in the Rust tests; here we prove reachability + correctness
through the shipped engine for the whole family at once.
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics import solvers

pytest.importorskip("tsdynamics._rust")


# ── reference systems (TEST-LOCAL — not catalogue systems) ───────────────────────


class _DampedOscillator(ts.ContinuousSystem):
    """``x' = -0.1 x + y``, ``y' = -x - 0.1 y``.

    From ``u0 = [1, 0]`` the closed-form solution is
    ``x(t) = e^{-0.1 t} cos t``, ``y(t) = -e^{-0.1 t} sin t`` — a gentle,
    non-stiff, bounded problem every solver (down to order 1) can integrate.
    """

    dim = 2
    params: dict = {}
    ic = [1.0, 0.0]

    @staticmethod
    def _equations(y, t):
        return (-0.1 * y(0) + y(1), -y(0) - 0.1 * y(1))

    @staticmethod
    def exact(t: float) -> np.ndarray:
        return np.exp(-0.1 * t) * np.array([np.cos(t), -np.sin(t)])


class _StiffLinear(ts.ContinuousSystem):
    """``u' = A u`` with eigenvalues ``-1`` and ``-1000`` (stiffness ratio 1000).

    ``A = [[-500.5, 499.5], [499.5, -500.5]]``; from ``u0 = [1, 0]`` the exact
    solution is ``0.5·[e^{-t}+e^{-1000 t}, e^{-t}-e^{-1000 t}]``.
    """

    dim = 2
    params: dict = {}
    ic = [1.0, 0.0]

    @staticmethod
    def _equations(y, t):
        return (
            -500.5 * y(0) + 499.5 * y(1),
            499.5 * y(0) - 500.5 * y(1),
        )

    @staticmethod
    def exact(t: float) -> np.ndarray:
        fast = np.exp(-1000.0 * t)
        slow = np.exp(-t)
        return np.array([0.5 * (slow + fast), 0.5 * (slow - fast)])


_ODE_SOLVERS = solvers.available_for("ode")
_IMPLICIT_SOLVERS = [s for s in _ODE_SOLVERS if solvers.get(s).caps.kind == "implicit"]
_ADAPTIVE_SOLVERS = [s for s in _ODE_SOLVERS if solvers.get(s).caps.adaptive]


def test_registry_has_a_substantial_ode_family():
    # A guard that the sweeps below are actually exercising the family, not an
    # empty list (e.g. if the registry failed to populate).
    assert len(_ODE_SOLVERS) >= 20, _ODE_SOLVERS
    assert len(_IMPLICIT_SOLVERS) >= 5, _IMPLICIT_SOLVERS


@pytest.mark.parametrize("method", _ODE_SOLVERS)
def test_every_ode_solver_integrates_the_nonstiff_reference(method):
    """Every registered ODE solver reproduces the analytic damped oscillator.

    Adaptive kernels are held to ``rtol/atol`` (ignored by the fixed-step ones);
    the fixed step ``dt`` is small enough that even forward Euler (order 1) lands
    well inside the ballpark tolerance — the point is to prove the solver runs
    end-to-end and is correct, not to measure its order (the Rust tests do that).
    """
    sys = _DampedOscillator()
    final_time = 1.0
    traj = sys.integrate(
        final_time=final_time,
        dt=1e-3,
        ic=[1.0, 0.0],
        method=method,
        rtol=1e-9,
        atol=1e-11,
        backend="interp",
    )
    got = np.asarray(traj.y[-1], dtype=float)
    exact = _DampedOscillator.exact(final_time)
    err = float(np.max(np.abs(got - exact)))
    assert np.all(np.isfinite(got)), f"{method}: non-finite final state {got}"
    assert err < 2e-2, f"{method}: error {err:.3e} vs analytic {exact} (got {got})"


@pytest.mark.parametrize("method", _IMPLICIT_SOLVERS)
def test_every_implicit_solver_handles_a_stiff_system(method):
    """Every implicit solver integrates the stiff linear system from a step far
    above the explicit stability limit (``dt = 0.05`` vs ``2/1000``), matching the
    closed-form solution — the L-/A-stability + Jacobian-wiring proof through the
    engine (``run.integrate`` auto-builds the Jacobian tape for implicit methods).
    """
    sys = _StiffLinear()
    final_time = 1.0
    traj = sys.integrate(
        final_time=final_time,
        dt=0.05,
        ic=[1.0, 0.0],
        method=method,
        rtol=1e-7,
        atol=1e-10,
        backend="interp",
    )
    got = np.asarray(traj.y[-1], dtype=float)
    exact = _StiffLinear.exact(final_time)
    err = float(np.max(np.abs(got - exact)))
    assert np.all(np.isfinite(got)), f"{method}: non-finite final state {got}"
    assert err < 5e-3, f"{method}: stiff error {err:.3e} vs analytic {exact} (got {got})"


def _osc_error(method: str, rtol: float, atol: float, final_time: float) -> float:
    sys = _DampedOscillator()
    traj = sys.integrate(
        final_time=final_time,
        dt=0.5,
        ic=[1.0, 0.0],
        method=method,
        rtol=rtol,
        atol=atol,
        backend="interp",
    )
    got = np.asarray(traj.y[-1], dtype=float)
    return float(np.max(np.abs(got - _DampedOscillator.exact(final_time))))


@pytest.mark.parametrize("method", _ADAPTIVE_SOLVERS)
def test_adaptive_solver_honors_the_requested_tolerance(method):
    """A tight ``rtol`` must reach the kernel — the guard for the FFI tolerance
    table (``bridge.rs::build_solver``). Tightening ``rtol`` from ``1e-2`` to
    ``1e-11`` must improve the error by a wide margin; a solver whose tolerance is
    silently dropped would use its default at both settings and produce the *same*
    error (ratio ≈ 1). The roundoff escape (``< 1e-11``) covers the very-highest-
    order kernels (e.g. ``dop853``) that are already near machine precision at any
    tolerance on this smooth problem — their threading is additionally guaranteed
    by the same ``build_solver`` code path the lower-order kernels exercise here and
    by the registry-vs-source parity test.
    """
    e_loose = _osc_error(method, 1e-2, 1e-4, 5.0)
    e_tight = _osc_error(method, 1e-11, 1e-13, 5.0)
    assert e_tight <= e_loose + 1e-15, f"{method}: tightening rtol worsened error"
    assert e_tight < 0.1 * e_loose or e_tight < 1e-11, (
        f"{method}: error unchanged by tolerance (loose={e_loose:.2e}, "
        f"tight={e_tight:.2e}) — tolerance not threaded?"
    )


@pytest.mark.parametrize("method", _ODE_SOLVERS)
def test_interp_and_jit_backends_agree(method):
    """Each solver is bit-for-bit identical between the interpreter and JIT
    backends — both drive the same kernel over the same `Evaluator` contract, so a
    solver that diverged between them would signal a backend-specific bug.
    """
    sys = _DampedOscillator()
    kw = dict(final_time=1.0, dt=2e-3, ic=[1.0, 0.0], method=method, rtol=1e-9, atol=1e-11)
    interp = np.asarray(sys.integrate(backend="interp", **kw).y[-1], dtype=float)
    jit = np.asarray(sys.integrate(backend="jit", **kw).y[-1], dtype=float)
    np.testing.assert_array_equal(interp, jit, err_msg=f"{method}: interp != jit")
