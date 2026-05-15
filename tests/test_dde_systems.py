"""
Tests for DDE systems (``DelaySystem`` subclasses).

DDE compilation via JiTCDDE is slow, so integration tests are marked ``slow``.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

_DDE_MOD = "tsdynamics.systems.continuous.delayed_systems"


# (class_name, history_fn, dim) — history must be non-equilibrium to drive dynamics.
def _mg_history(s: float) -> list[float]:
    return [1.0 + 0.1 * np.sin(0.2 * s)]


def _ikeda_history(s: float) -> list[float]:
    return [0.1 + 0.05 * np.cos(0.3 * s)]


def _sprott_history(s: float) -> list[float]:
    return [0.5 + 0.1 * np.sin(0.2 * s)]


def _scroll_history(s: float) -> list[float]:
    return [0.3 + 0.05 * np.cos(0.1 * s)]


def _piece_history(s: float) -> list[float]:
    return [0.4 + 0.05 * np.sin(0.15 * s)]


_ALL_DDE: list[tuple[str, callable, int]] = [
    ("MackeyGlass", _mg_history, 1),
    ("IkedaDelay", _ikeda_history, 1),
    ("SprottDelay", _sprott_history, 1),
    ("ScrollDelay", _scroll_history, 1),
    ("PiecewiseCircuit", _piece_history, 1),
]
_IDS = [name for name, _, _ in _ALL_DDE]


# ---------------------------------------------------------------------------
# Instantiation — fast
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("class_name,_history,expected_dim", _ALL_DDE, ids=_IDS)
def test_dde_instantiation(class_name: str, _history, expected_dim: int) -> None:
    mod = importlib.import_module(_DDE_MOD)
    cls = getattr(mod, class_name)
    sys = cls()
    assert sys.dim == expected_dim
    assert sys.ic is None


@pytest.mark.parametrize("class_name,_history,_dim", _ALL_DDE, ids=_IDS)
def test_dde_params_as_attributes(class_name: str, _history, _dim: int) -> None:
    mod = importlib.import_module(_DDE_MOD)
    cls = getattr(mod, class_name)
    sys = cls()
    for key in sys.params:
        assert hasattr(sys, key)


@pytest.mark.parametrize("class_name,_history,_dim", _ALL_DDE, ids=_IDS)
def test_dde_delay_params_resolve(class_name: str, _history, _dim: int) -> None:
    """``_delays()`` must return positive floats matching ``_delay_params``."""
    mod = importlib.import_module(_DDE_MOD)
    cls = getattr(mod, class_name)
    sys = cls()
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
@pytest.mark.parametrize("class_name,history,expected_dim", _ALL_DDE, ids=_IDS)
def test_dde_integration_shape_and_finiteness(class_name: str, history, expected_dim: int) -> None:
    mod = importlib.import_module(_DDE_MOD)
    cls = getattr(mod, class_name)
    sys = cls()
    traj = sys.integrate(final_time=5.0, dt=0.1, history=history)
    assert traj.t.ndim == 1
    assert traj.y.ndim == 2
    assert traj.y.shape[0] == traj.t.shape[0]
    assert traj.y.shape[1] == expected_dim
    assert np.all(np.isfinite(traj.y))


@pytest.mark.slow
def test_dde_time_starts_at_zero() -> None:
    import tsdynamics as ts

    traj = ts.MackeyGlass().integrate(final_time=3.0, dt=0.1, history=_mg_history)
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
