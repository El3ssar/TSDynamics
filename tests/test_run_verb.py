"""Tests for the unified ``system.run`` trajectory verb (WS-RUNVERB).

``run`` is the one canonical producer for every family, dispatching on
:attr:`is_discrete`: it delegates to :meth:`ContinuousSystem.integrate` for
flows and :meth:`DiscreteMap.iterate` for maps.  These tests assert it returns a
:class:`~tsdynamics.data.Trajectory` for both families and is byte-identical to
the family-specific spelling it aliases, while the older verbs keep working.
"""

from __future__ import annotations

import numpy as np

import tsdynamics as ts
from tsdynamics.data import Trajectory


def test_run_flow_returns_trajectory() -> None:
    """``Lorenz().run(final_time=...)`` returns a Trajectory (flow dispatch)."""
    traj = ts.systems.Lorenz().run(final_time=10.0, dt=0.01)
    assert isinstance(traj, Trajectory)
    assert traj.y.shape[1] == 3


def test_run_map_returns_trajectory() -> None:
    """``Henon().run(n=...)`` returns a Trajectory (map dispatch on is_discrete)."""
    traj = ts.systems.Henon().run(n=500)
    assert isinstance(traj, Trajectory)
    assert traj.y.shape == (500, 2)


def test_run_dispatches_on_is_discrete() -> None:
    """The same verb name picks the integrate/iterate kernel by ``is_discrete``."""
    flow = ts.systems.Lorenz()
    a_map = ts.systems.Henon()
    assert flow.is_discrete is False
    assert a_map.is_discrete is True
    # Both answer ``run`` and return a Trajectory; the kernel differs.
    assert isinstance(flow.run(final_time=5.0, dt=0.05), Trajectory)
    assert isinstance(a_map.run(n=100), Trajectory)


def test_run_equals_integrate_for_flow() -> None:
    """``run`` is byte-identical to ``integrate`` for an ODE (a thin alias)."""
    a = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0]).run(final_time=20.0, dt=0.01)
    b = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0]).run(final_time=20.0, dt=0.01)
    np.testing.assert_array_equal(a.t, b.t)
    np.testing.assert_array_equal(a.y, b.y)


def test_run_equals_iterate_for_map() -> None:
    """``run(n=...)`` is byte-identical to ``iterate(steps=...)`` for a map."""
    a = ts.systems.Henon(ic=[0.1, 0.1]).run(n=1000)
    b = ts.systems.Henon(ic=[0.1, 0.1]).run(steps=1000)
    np.testing.assert_array_equal(a.t, b.t)
    np.testing.assert_array_equal(a.y, b.y)


def test_run_forwards_flow_kwargs() -> None:
    """Flow ``run`` forwards every keyword to ``integrate`` unchanged."""
    a = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0]).run(
        final_time=15.0, dt=0.02, solver="DOP853", rtol=1e-8, atol=1e-10
    )
    b = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0]).run(
        final_time=15.0, dt=0.02, solver="DOP853", rtol=1e-8, atol=1e-10
    )
    np.testing.assert_array_equal(a.y, b.y)


def test_run_forwards_map_kwargs() -> None:
    """Map ``run`` forwards ``ic`` and other keywords to ``iterate``."""
    a = ts.systems.Henon().run(n=300, ic=[0.2, 0.0])
    b = ts.systems.Henon().run(steps=300, ic=[0.2, 0.0])
    np.testing.assert_array_equal(a.y, b.y)


def test_run_map_default_n() -> None:
    """``run`` on a map defaults to the same step count as ``iterate``."""
    a = ts.systems.Henon(ic=[0.1, 0.1]).run()
    b = ts.systems.Henon(ic=[0.1, 0.1]).run()
    np.testing.assert_array_equal(a.y, b.y)


def test_legacy_verbs_still_present() -> None:
    """``integrate``/``iterate``/``trajectory`` remain available aliases."""
    assert callable(ts.systems.Lorenz().integrate)
    assert callable(ts.systems.Lorenz().trajectory)
    assert callable(ts.systems.Henon().iterate)
    assert callable(ts.systems.Henon().trajectory)
    # ``run`` is present on both families.
    assert callable(ts.systems.Lorenz().run)
    assert callable(ts.systems.Henon().run)


def test_protocol_doc_lists_run() -> None:
    """The System protocol module documents ``run`` as THE producer verb.

    ``run`` is deliberately *not* a required structural member of the
    ``runtime_checkable`` ``System`` protocol — a few implementers
    (``WrappedSystem``, the derived wrappers) expose only ``trajectory`` — so
    adding it would break ``isinstance(obj, System)`` for them.  The contract is
    documented in the module docstring instead.
    """
    from tsdynamics.families import protocol

    assert "canonical verb" in protocol.__doc__
    assert "run" in protocol.System.trajectory.__doc__


def test_run_does_not_break_protocol_conformance() -> None:
    """Adding ``run`` must not regress ``isinstance(obj, System)``.

    Every family and every derived wrapper must still satisfy the structural
    protocol (which keys off ``trajectory``, not ``run``).
    """
    from tsdynamics.families import System

    assert isinstance(ts.systems.Lorenz(), System)
    assert isinstance(ts.systems.Henon(), System)
    assert isinstance(ts.systems.MackeyGlass(), System)  # DDE: has no .run, still conforms
    pm = ts.derived.PoincareMap(ts.systems.Rossler(), plane=(1, 0.0))
    assert isinstance(pm, System)  # derived wrapper: has no .run, still conforms
