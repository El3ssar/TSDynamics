"""
Tests for discrete map systems (``DiscreteMap`` subclasses).

All sweeps are registry-driven: a new map is covered automatically.
Iteration is fast (Numba JIT); the first call per class triggers a one-shot
compile, so iterate/Lyapunov sweeps are marked ``slow``.
"""

from __future__ import annotations

import numpy as np
import pytest
from _sampling import MAP_LYAPUNOV_EXCLUDE

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

    tb = ts.Tinkerbell()
    assert tb.ic is None
    assert tb.default_ic is not None
    traj = tb.iterate(steps=100)
    np.testing.assert_array_almost_equal(tb.ic, ts.Tinkerbell.default_ic)
    assert np.all(np.isfinite(traj.y))


# ---------------------------------------------------------------------------
# Iteration
# ---------------------------------------------------------------------------

_STEPS = 200


@pytest.mark.slow
def test_map_iterate_shape_and_finiteness(map_entry) -> None:
    m = map_entry.cls()
    traj = m.iterate(steps=_STEPS, max_retries=15)
    assert traj.t.shape == (_STEPS,)
    assert traj.y.shape == (_STEPS, m.dim)
    np.testing.assert_array_equal(traj.t, np.arange(_STEPS))
    assert np.all(np.isfinite(traj.y))


@pytest.mark.slow
def test_map_custom_ic_stored() -> None:
    import tsdynamics as ts

    h = ts.Henon()
    ic = np.array([0.2, 0.3])
    h.iterate(steps=50, ic=ic)
    np.testing.assert_array_almost_equal(h.ic, ic)


# ---------------------------------------------------------------------------
# Lyapunov spectrum — shape/finiteness sweep
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_map_lyapunov_shape(map_entry) -> None:
    if map_entry.name in MAP_LYAPUNOV_EXCLUDE:
        pytest.skip(MAP_LYAPUNOV_EXCLUDE[map_entry.name])
    m = map_entry.cls()
    exps = m.lyapunov_spectrum(steps=300, n_exp=m.dim)
    assert exps.shape == (m.dim,)
    assert np.all(np.isfinite(exps))


@pytest.mark.slow
def test_map_lyapunov_partial_spectrum(map_entry) -> None:
    if map_entry.name in MAP_LYAPUNOV_EXCLUDE:
        pytest.skip(MAP_LYAPUNOV_EXCLUDE[map_entry.name])
    m = map_entry.cls()
    exps = m.lyapunov_spectrum(steps=300, n_exp=1)
    assert exps.shape == (1,)
    assert np.isfinite(exps[0])
