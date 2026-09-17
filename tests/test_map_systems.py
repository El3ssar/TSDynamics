"""
Tests for discrete map systems (``DiscreteMap`` subclasses).

All sweeps are registry-driven: a new map is covered automatically.
Iteration runs on the engine; the longer iterate/Lyapunov sweeps are marked
``slow`` on runtime grounds.

Scope note: as in ``test_ode_systems``, the iteration sweep here checks shape
and finiteness only — the pre-v6 Baker, whose every orbit collapsed to ``(0, 0)``
within ~53 iterations, passed it.  The dynamical content (bounded,
non-degenerate, recurrent, claim-consistent) is asserted per system by
``tests/test_catalogue_dynamics.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from _sampling import MAP_LYAPUNOV_EXCLUDE

import tsdynamics as ts

# ---------------------------------------------------------------------------
# Instantiation (fast)
# ---------------------------------------------------------------------------


def test_map_instantiation(map_entry) -> None:
    m = map_entry.cls()
    assert m.dim is not None and m.dim > 0


def test_map_params_as_attributes(map_entry) -> None:
    m = map_entry.cls()
    for key in m.params:
        assert hasattr(m, key)


def test_tinkerbell_uses_default_ic() -> None:
    """Tinkerbell sets ``default_ic`` because random ICs always escape the basin."""
    import tsdynamics as ts

    tb = ts.systems.Tinkerbell()
    assert tb.ic is None
    assert tb.info.default_ic is not None
    traj = tb.run(steps=100)
    np.testing.assert_array_almost_equal(tb.ic, ts.systems.Tinkerbell._default_ic)
    assert np.all(np.isfinite(traj.y))


# ---------------------------------------------------------------------------
# Iteration
# ---------------------------------------------------------------------------

_STEPS = 200


@pytest.mark.slow
def test_map_iterate_shape_and_finiteness(map_entry) -> None:
    m = map_entry.cls()
    traj = m.run(steps=_STEPS, max_retries=15)
    assert traj.t.shape == (_STEPS,)
    assert traj.y.shape == (_STEPS, m.dim)
    np.testing.assert_array_equal(traj.t, np.arange(_STEPS))
    assert np.all(np.isfinite(traj.y))


@pytest.mark.slow
def test_map_explicit_ic_is_used_but_never_latched() -> None:
    """An explicit ``run(ic=)`` starts the orbit and leaves the map alone (v6).

    See ``test_ode_systems.py::test_ode_explicit_ic_is_used_but_never_latched``
    — the same rule, in the map's own horizon word.
    """
    import tsdynamics as ts

    h = ts.systems.Henon()
    ic = np.array([0.2, 0.3])
    traj = h.run(steps=50, ic=ic)
    # A map's grid starts at the FIRST ITERATE (a flow's starts at the IC), so
    # the evidence the IC was used is f(ic), not ic.
    a, b = 1.4, 0.3
    np.testing.assert_array_almost_equal(traj.y[0], [1 - a * ic[0] ** 2 + ic[1], b * ic[0]])
    assert h.ic is None


# ---------------------------------------------------------------------------
# Lyapunov spectrum — shape/finiteness sweep
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_map_lyapunov_shape(map_entry) -> None:
    if map_entry.name in MAP_LYAPUNOV_EXCLUDE:
        pytest.skip(MAP_LYAPUNOV_EXCLUDE[map_entry.name])
    m = map_entry.cls()
    exps = ts.analysis.lyapunov_spectrum(m, n=300, k=m.dim)
    assert exps.shape == (m.dim,)
    assert np.all(np.isfinite(exps))


@pytest.mark.slow
def test_map_lyapunov_partial_spectrum(map_entry) -> None:
    if map_entry.name in MAP_LYAPUNOV_EXCLUDE:
        pytest.skip(MAP_LYAPUNOV_EXCLUDE[map_entry.name])
    m = map_entry.cls()
    exps = ts.analysis.lyapunov_spectrum(m, n=300, k=1)
    assert exps.shape == (1,)
    assert np.isfinite(exps[0])
