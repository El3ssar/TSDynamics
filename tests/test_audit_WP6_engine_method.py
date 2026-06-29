"""Regression tests for WP6 — ``method="auto"`` reach + min/max Jacobian lowering.

Covers three audit findings, each of which crashed (or silently mis-lowered) on
the pre-fix logic:

- **P2-1** — ``ContinuousSystem.reinit(method="auto")`` resolved the stepper kernel
  with ``solvers.resolve("auto")``, which raised ``unknown solver method 'auto'``.
  Auto-stiffness must reach the stepping protocol identically to
  ``integrate``/``ensemble``.
- **P3-1** — ``run(events=[...], method="auto")`` reached
  ``integrate_events`` → ``solvers.resolve("auto")`` and crashed, so the advertised
  ``"auto"`` value failed the moment ``events=`` was supplied.
- **P1-1** — a system whose RHS uses ``Min``/``Max`` lowered its RHS fine but raised
  ``TapeCompileError`` when lowered ``with_jacobian=True`` (the stiff-ODE and
  map-Lyapunov kernel paths), instead of producing the a.e.-correct Jacobian.
"""

from __future__ import annotations

import numpy as np
import pytest
import symengine

import tsdynamics as ts
from tsdynamics.engine.compile import eval_tape_jac, lower_ode

pytest.importorskip("tsdynamics._rust")


# ---------------------------------------------------------------------------
# P2-1 — method="auto" through the stepping protocol (reinit/step)
# ---------------------------------------------------------------------------


def test_reinit_method_auto_resolves_like_integrate() -> None:
    """``reinit(method="auto")`` resolves the kernel instead of crashing (P2-1).

    Pre-fix ``reinit`` called ``solvers.resolve("auto")`` directly and raised
    ``ValueError("unknown solver method 'auto'")``; ``integrate(method="auto")``
    already worked.  After the fix the stepping path probes auto-stiffness and
    resolves to the same explicit kernel ``integrate`` records.
    """
    lor = ts.Lorenz()
    # Did not raise (the pre-fix crash) and selected a concrete kernel.
    lor.reinit([1.0, 1.0, 1.0], method="auto")
    assert lor._step_method_canonical == "rk45"

    # The resolved kernel matches the integrate/ensemble contract for the same IC.
    traj = ts.Lorenz().integrate(final_time=1.0, dt=0.01, ic=[1.0, 1.0, 1.0], method="auto")
    assert traj.meta["method"] == lor._step_method_canonical

    # And a step actually advances (the protocol is reachable end-to-end).
    u = lor.step(0.01)
    assert u.shape == (3,)
    assert np.all(np.isfinite(u))


# ---------------------------------------------------------------------------
# P3-1 — method="auto" through the events= seam
# ---------------------------------------------------------------------------


def test_run_events_method_auto_resolves_like_integrate() -> None:
    """``run(events=…, method="auto")`` honours auto identically to integrate (P3-1).

    Pre-fix this crashed in ``integrate_events`` (``solvers.resolve("auto")`` →
    ``ValueError``).  After the fix the events path resolves through the shared
    contract, detects crossings, and records the canonical kernel name.
    """
    sol = ts.Lorenz().run(final_time=20.0, dt=0.01, method="auto", events=[("z", 27.0, "up")])
    # Canonical name in provenance (not the raw "auto" alias), matching integrate.
    assert sol.meta["method"] == "rk45"
    # The event seam actually fired (Lorenz crosses z=27 upward many times).
    assert sol.meta["t_events"][0].size > 0


def test_events_path_still_rejects_unknown_method() -> None:
    """The auto-aware events resolution keeps rejecting v2-only / unknown names.

    Routing through the shared contract must not weaken the guard: a rejected
    name (``"LSODA"``) still raises a ``ValueError`` subclass.
    """
    with pytest.raises(ValueError):
        ts.Lorenz().run(final_time=5.0, dt=0.01, method="LSODA", events=[("z", 27.0, "up")])


# ---------------------------------------------------------------------------
# P1-1 — min/max Jacobian lowering (a.e.)
# ---------------------------------------------------------------------------


class _MaxMinSys(ts.ContinuousSystem):
    """Tiny system exercising ``Max``/``Min`` in the RHS.

    ``f₀ = -k·max(y₀, y₁)``  → ``∂f₀/∂y₀ = -k`` where ``y₀ ≥ y₁`` else ``0``;
    ``∂f₀/∂y₁ = -k`` where ``y₁ > y₀`` else ``0``.
    ``f₁ = min(y₀, 0)``      → ``∂f₁/∂y₀ = 1`` where ``y₀ ≤ 0`` else ``0``.
    """

    params = {"k": 2.0}
    dim = 2

    @staticmethod
    def _equations(y, t, k):  # type: ignore[no-untyped-def]
        return [-k * symengine.Max(y(0), y(1)), symengine.Min(y(0), symengine.Integer(0))]


def test_minmax_jacobian_resolved_a_e() -> None:
    """A ``Min``/``Max`` system lowers its Jacobian a.e. instead of raising (P1-1).

    Pre-fix ``lower_ode(..., with_jacobian=True)`` raised ``TapeCompileError`` on the
    unevaluated ``Derivative(max(...), ...)`` node (only abs/sign/floor/ceil were
    resolved).  After the fix the Jacobian follows whichever argument is the active
    extremum.
    """
    tape = lower_ode(_MaxMinSys(), with_jacobian=True)
    p = np.array([2.0])

    # y0 the max, y0 < 0: ∂f0/∂y0=-2, ∂f0/∂y1=0; ∂f1/∂y0=1 (y0<=0).
    _, j = eval_tape_jac(tape, [-1.0, -5.0], p, 0.0)
    assert j == pytest.approx(np.array([[-2.0, 0.0], [1.0, 0.0]]))

    # y1 the max, y0 > 0: ∂f0/∂y0=0, ∂f0/∂y1=-2; ∂f1/∂y0=0 (y0>0).
    _, j2 = eval_tape_jac(tape, [1.0, 3.0], p, 0.0)
    assert j2 == pytest.approx(np.array([[0.0, -2.0], [0.0, 0.0]]))


def test_minmax_jacobian_interp_equals_jit_bit_for_bit() -> None:
    """A stiff (bdf, Jacobian-needing) ``Min``/``Max`` integration is interp==jit.

    The Jacobian-bearing tape now lowers for min/max; the parity contract
    (``interp`` == ``jit`` bit-for-bit on the same lowered tape) must hold for the
    new derivative path too.
    """

    class _StiffMax(ts.ContinuousSystem):
        params = {"k": 1.0}
        dim = 2
        _default_method = "bdf"

        @staticmethod
        def _equations(y, t, k):  # type: ignore[no-untyped-def]
            return [
                -k * symengine.Max(y(0), y(1)) + 0.1,
                -y(1) + symengine.Min(y(0), symengine.Rational(1, 2)),
            ]

    s = _StiffMax()
    ti = s.integrate(final_time=5.0, dt=0.05, ic=[0.3, 0.7], backend="interp", method="bdf")
    tj = s.integrate(final_time=5.0, dt=0.05, ic=[0.3, 0.7], backend="jit", method="bdf")
    assert np.array_equal(ti.y, tj.y)
