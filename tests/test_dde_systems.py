"""
Tests for DDE systems (``DelaySystem`` subclasses).

Registry-driven; per-system non-equilibrium histories live in
``tests/_sampling.py`` (a guard test asserts completeness).  DDE compilation
via JiTCDDE is slow, so integration tests are marked ``slow``.
"""

from __future__ import annotations

import numpy as np
import pytest
from _sampling import DDE_HISTORIES

# ---------------------------------------------------------------------------
# Instantiation — fast
# ---------------------------------------------------------------------------


def test_dde_instantiation(dde_entry) -> None:
    sys = dde_entry.cls()
    assert sys.dim is not None and sys.dim > 0
    assert sys.ic is None


def test_dde_params_as_attributes(dde_entry) -> None:
    sys = dde_entry.cls()
    for key in sys.params:
        assert hasattr(sys, key)


def test_dde_delay_params_resolve(dde_entry) -> None:
    """``_delays()`` must return positive floats matching ``_delay_params``."""
    sys = dde_entry.cls()
    delays = sys._delays()
    assert len(delays) == len(sys._delay_params)
    assert all(d > 0.0 for d in delays)


def test_dde_zero_delay_raises() -> None:
    """A zero or negative delay parameter must raise rather than silently break JiTCDDE."""
    import tsdynamics as ts

    mg = ts.MackeyGlass(params={"tau": 0.0})
    with pytest.raises(ValueError, match="must be strictly positive"):
        mg._delays()


# ---------------------------------------------------------------------------
# Integration — slow (JiTCDDE C compile)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_dde_integration_shape_and_finiteness(dde_entry) -> None:
    sys = dde_entry.cls()
    history = DDE_HISTORIES[dde_entry.name]
    traj = sys.integrate(final_time=5.0, dt=0.1, history=history)
    assert traj.t.ndim == 1
    assert traj.y.ndim == 2
    assert traj.y.shape[0] == traj.t.shape[0]
    assert traj.y.shape[1] == sys.dim
    assert np.all(np.isfinite(traj.y))


@pytest.mark.slow
def test_dde_time_starts_at_zero() -> None:
    import tsdynamics as ts

    traj = ts.MackeyGlass().integrate(final_time=3.0, dt=0.1, history=DDE_HISTORIES["MackeyGlass"])
    assert traj.t[0] == pytest.approx(0.0)


@pytest.mark.slow
def test_dde_constant_history_accepted() -> None:
    import tsdynamics as ts

    traj = ts.MackeyGlass().integrate(final_time=3.0, dt=0.2, history=lambda s: [1.5])
    assert traj.y.shape[1] == 1
    assert np.all(np.isfinite(traj.y))


@pytest.mark.slow
def test_dde_ic_used_when_no_history() -> None:
    """With no history, ``constant_past(ic)`` is used and the IC drives integration."""
    import tsdynamics as ts

    mg = ts.MackeyGlass()
    traj = mg.integrate(final_time=3.0, dt=0.5, ic=[1.5])
    assert np.all(np.isfinite(traj.y))
    np.testing.assert_array_almost_equal(mg.ic, [1.5])
