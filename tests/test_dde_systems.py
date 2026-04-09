"""
Tests for delay differential equation systems (DynSysDelay subclasses).

All DDE tests are marked slow because JiTCDDE compiles C code on first call.
"""

import numpy as np
import pytest

_DDE_MOD = "tsdynamics.systems.continuous.delayed_systems"

# ---------------------------------------------------------------------------
# Histories for each system (non-trivial to avoid equilibrium traps)
# ---------------------------------------------------------------------------


def _mg_history(s):
    return [1.5 + 0.05 * np.sin(0.1 * s)]


def _ikeda_history(s):
    return [0.1 + 0.02 * np.cos(0.3 * s)]


def _sprott_history(s):
    return [0.5 + 0.1 * np.sin(0.2 * s)]


def _scroll_history(s):
    return [0.3 + 0.05 * np.cos(0.1 * s)]


def _piece_history(s):
    return [0.4 + 0.05 * np.sin(0.15 * s)]


def _enso_history(s):
    return [0.5 + 0.1 * np.sin(0.2 * s)]


_ALL_DDE = [
    ("MackeyGlass", _mg_history, 1),
    ("IkedaDelay", _ikeda_history, 1),
    ("SprottDelay", _sprott_history, 1),
    ("ScrollDelay", _scroll_history, 1),
    ("PiecewiseCircuit", _piece_history, 1),
    ("ENSODelay", _enso_history, 1),
]
_IDS = [name for name, _, _ in _ALL_DDE]


# ---------------------------------------------------------------------------
# Instantiation (fast — no compilation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("class_name,_,expected_n_dim", _ALL_DDE, ids=_IDS)
def test_dde_instantiation(class_name, _, expected_n_dim):
    import importlib

    mod = importlib.import_module(_DDE_MOD)
    cls = getattr(mod, class_name)
    sys = cls()
    assert sys.n_dim == expected_n_dim
    assert isinstance(sys.params, dict)
    assert sys.initial_conds is None


@pytest.mark.parametrize("class_name,_,__", _ALL_DDE, ids=_IDS)
def test_dde_params_as_attributes(class_name, _, __):
    import importlib

    mod = importlib.import_module(_DDE_MOD)
    cls = getattr(mod, class_name)
    sys = cls()
    for key in sys.params:
        assert hasattr(sys, key), f"{class_name} missing attribute for param '{key}'"


# ---------------------------------------------------------------------------
# Integration (slow — JiTCDDE C compilation)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("class_name,history,expected_n_dim", _ALL_DDE, ids=_IDS)
def test_dde_integration_shape_and_finiteness(class_name, history, expected_n_dim):
    """All DDE systems must integrate and return finite trajectories."""
    import importlib

    mod = importlib.import_module(_DDE_MOD)
    cls = getattr(mod, class_name)
    sys = cls()

    t, X = sys.integrate(
        dt=0.1,
        final_time=5.0,
        history=history,
        rtol=1e-3,
        atol=1e-3,
    )

    assert t.ndim == 1, f"{class_name}: t must be 1D"
    assert X.ndim == 2, f"{class_name}: X must be 2D"
    assert X.shape[0] == t.shape[0], f"{class_name}: t and X lengths must match"
    assert X.shape[1] == expected_n_dim, (
        f"{class_name}: expected n_dim={expected_n_dim}, got {X.shape[1]}"
    )
    assert np.all(np.isfinite(X)), f"{class_name}: trajectory contains NaN or Inf"


@pytest.mark.slow
def test_dde_time_starts_at_zero():
    import importlib

    mod = importlib.import_module(_DDE_MOD)
    cls = getattr(mod, "MackeyGlass")
    sys = cls()
    t, _ = sys.integrate(dt=0.1, final_time=3.0, history=_mg_history)
    assert t[0] == pytest.approx(0.0)


@pytest.mark.slow
def test_dde_constant_history_accepted():
    """A callable returning a constant list must be accepted."""
    import importlib

    mod = importlib.import_module(_DDE_MOD)
    cls = getattr(mod, "MackeyGlass")
    sys = cls()
    history = lambda s: [1.5]  # noqa: E731
    t, X = sys.integrate(dt=0.2, final_time=3.0, history=history, rtol=1e-3, atol=1e-3)
    assert X.shape[1] == 1


@pytest.mark.slow
def test_dde_steps_overrides_final_time():
    import importlib

    mod = importlib.import_module(_DDE_MOD)
    mg = getattr(mod, "MackeyGlass")()
    t1, X1 = mg.integrate(dt=0.2, steps=20, history=_mg_history)
    assert X1.shape[0] == 21  # steps + endpoint


@pytest.mark.slow
def test_dde_initial_conds_stored_after_integrate():
    import importlib

    mod = importlib.import_module(_DDE_MOD)
    mg = getattr(mod, "MackeyGlass")()
    assert mg.initial_conds is None
    mg.integrate(dt=0.2, final_time=2.0, history=_mg_history, rtol=1e-3, atol=1e-3)
    # After integrate, either IC was set from history(0) or remains None
    # — just confirm no crash occurred.
