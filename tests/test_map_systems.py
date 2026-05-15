"""
Tests for discrete map systems (``DiscreteMap`` subclasses).

Iteration is fast (Numba JIT); the first call per class triggers a one-shot
compile.  Lyapunov-spectrum tests are marked ``slow`` because they re-evaluate
the Jacobian many times in Python.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

_CHAOTIC = "tsdynamics.systems.discrete.chaotic_maps"
_EXOTIC = "tsdynamics.systems.discrete.exotic_maps"
_GEOMETRIC = "tsdynamics.systems.discrete.geometric_maps"
_POLY = "tsdynamics.systems.discrete.polynomial_maps"
_POPN = "tsdynamics.systems.discrete.population_maps"

ALL_MAPS: list[tuple[str, str, int]] = [
    # Chaotic
    (_CHAOTIC, "Henon", 2),
    (_CHAOTIC, "Ulam", 1),
    (_CHAOTIC, "Ikeda", 2),
    (_CHAOTIC, "Tinkerbell", 2),
    (_CHAOTIC, "Gingerbreadman", 2),
    (_CHAOTIC, "Zaslavskii", 2),
    (_CHAOTIC, "Chirikov", 2),
    (_CHAOTIC, "FoldedTowel", 3),
    (_CHAOTIC, "GeneralizedHenon", 3),
    # Exotic
    (_EXOTIC, "Bogdanov", 2),
    (_EXOTIC, "Svensson", 2),
    (_EXOTIC, "Bedhead", 2),
    (_EXOTIC, "ZeraouliaSprott", 2),
    (_EXOTIC, "GumowskiMira", 2),
    (_EXOTIC, "Hopalong", 2),
    (_EXOTIC, "Pickover", 2),
    # Geometric
    (_GEOMETRIC, "Tent", 1),
    (_GEOMETRIC, "Baker", 2),
    (_GEOMETRIC, "Circle", 1),
    (_GEOMETRIC, "Chebyshev", 1),
    # Polynomial
    (_POLY, "Gauss", 1),
    (_POLY, "DeJong", 2),
    (_POLY, "KaplanYorke", 2),
    # Population
    (_POPN, "Logistic", 1),
    (_POPN, "Ricker", 1),
    (_POPN, "MaynardSmith", 2),
]
_IDS = [name for _, name, _ in ALL_MAPS]


# ---------------------------------------------------------------------------
# Instantiation (fast)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_path,class_name,expected_dim", ALL_MAPS, ids=_IDS)
def test_map_instantiation(module_path: str, class_name: str, expected_dim: int) -> None:
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    m = cls()
    assert m.dim == expected_dim


@pytest.mark.parametrize("module_path,class_name,_d", ALL_MAPS, ids=_IDS)
def test_map_params_as_attributes(module_path: str, class_name: str, _d: int) -> None:
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    m = cls()
    for key in m.params:
        assert hasattr(m, key)


def test_tinkerbell_uses_default_ic() -> None:
    """Tinkerbell sets ``default_ic`` because random ICs always escape the basin."""
    import tsdynamics as ts

    tb = ts.Tinkerbell()
    # Before any iterate() call, self.ic is None — class-level default_ic exists.
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
@pytest.mark.parametrize("module_path,class_name,expected_dim", ALL_MAPS, ids=_IDS)
def test_map_iterate_shape_and_finiteness(
    module_path: str, class_name: str, expected_dim: int
) -> None:
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    m = cls()
    traj = m.iterate(steps=_STEPS, max_retries=15)
    assert traj.t.shape == (_STEPS,)
    assert traj.y.shape == (_STEPS, expected_dim)
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
# Lyapunov spectrum
# ---------------------------------------------------------------------------

# Maps whose Jacobian implementation is known to be regular over the iterate.
# (Bogdanov is excluded — known singular Jacobian at origin.)
_LYAP_MAPS: list[tuple[str, str, int]] = [
    (_CHAOTIC, "Henon", 2),
    (_CHAOTIC, "Ikeda", 2),
    (_CHAOTIC, "Tinkerbell", 2),
    (_CHAOTIC, "Zaslavskii", 2),
    (_CHAOTIC, "Chirikov", 2),
    (_CHAOTIC, "FoldedTowel", 3),
    (_CHAOTIC, "GeneralizedHenon", 3),
    (_EXOTIC, "Svensson", 2),
    (_EXOTIC, "Bedhead", 2),
    (_EXOTIC, "ZeraouliaSprott", 2),
    (_EXOTIC, "GumowskiMira", 2),
    (_EXOTIC, "Hopalong", 2),
    (_EXOTIC, "Pickover", 2),
    (_GEOMETRIC, "Tent", 1),
    (_GEOMETRIC, "Baker", 2),
    (_GEOMETRIC, "Circle", 1),
    (_GEOMETRIC, "Chebyshev", 1),
    (_POLY, "Gauss", 1),
    (_POLY, "DeJong", 2),
    (_POLY, "KaplanYorke", 2),
    (_POPN, "Logistic", 1),
    (_POPN, "Ricker", 1),
    (_POPN, "MaynardSmith", 2),
]
_LYAP_IDS = [name for _, name, _ in _LYAP_MAPS]


@pytest.mark.slow
@pytest.mark.parametrize("module_path,class_name,expected_dim", _LYAP_MAPS, ids=_LYAP_IDS)
def test_map_lyapunov_shape(module_path: str, class_name: str, expected_dim: int) -> None:
    """Full-spectrum LE has correct shape and finite values."""
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    m = cls()
    exps = m.lyapunov_spectrum(steps=300, n_exp=expected_dim)
    assert exps.shape == (expected_dim,)
    assert np.all(np.isfinite(exps))


@pytest.mark.slow
@pytest.mark.parametrize("module_path,class_name,_d", _LYAP_MAPS, ids=_LYAP_IDS)
def test_map_lyapunov_partial_spectrum(module_path: str, class_name: str, _d: int) -> None:
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    m = cls()
    exps = m.lyapunov_spectrum(steps=300, n_exp=1)
    assert exps.shape == (1,)
    assert np.isfinite(exps[0])
