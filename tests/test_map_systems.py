"""
Tests for discrete map systems (DynMap subclasses).

Maps are tested with short iteration counts (100 steps) to keep CI fast.
First call triggers numba JIT compilation, which is much faster than JiTCODE.
"""

import importlib

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Complete list of all map systems: (module_path, class_name, expected_n_dim)
# ---------------------------------------------------------------------------

_CHAOTIC = "tsdynamics.systems.discrete.chaotic_maps"
_EXOTIC = "tsdynamics.systems.discrete.exotic_maps"
_GEOMETRIC = "tsdynamics.systems.discrete.geometric_maps"
_POLY = "tsdynamics.systems.discrete.polynomial_maps"
_POPN = "tsdynamics.systems.discrete.population_maps"

ALL_MAPS = [
    # ── Chaotic maps ─────────────────────────────────────────────────────
    (_CHAOTIC, "Henon", 2),
    (_CHAOTIC, "Ulam", 1),
    (_CHAOTIC, "Ikeda", 2),
    (_CHAOTIC, "Tinkerbell", 2),
    (_CHAOTIC, "Gingerbreadman", 2),
    (_CHAOTIC, "Zaslavskii", 2),
    (_CHAOTIC, "Chirikov", 2),
    (_CHAOTIC, "FoldedTowel", 3),
    (_CHAOTIC, "GeneralizedHenon", 3),
    # ── Exotic maps ──────────────────────────────────────────────────────
    (_EXOTIC, "Bogdanov", 2),
    (_EXOTIC, "Svensson", 2),
    (_EXOTIC, "Bedhead", 2),
    (_EXOTIC, "ZeraouliaSprott", 2),
    (_EXOTIC, "GumowskiMira", 2),
    (_EXOTIC, "Hopalong", 2),
    (_EXOTIC, "Pickover", 2),
    # ── Geometric maps ───────────────────────────────────────────────────
    (_GEOMETRIC, "Tent", 1),
    (_GEOMETRIC, "Baker", 2),
    (_GEOMETRIC, "Circle", 1),
    (_GEOMETRIC, "Chebyshev", 1),
    # ── Polynomial maps ──────────────────────────────────────────────────
    (_POLY, "Gauss", 1),
    (_POLY, "DeJong", 2),
    (_POLY, "KaplanYorke", 2),
    # ── Population maps ──────────────────────────────────────────────────
    (_POPN, "Logistic", 1),
    (_POPN, "Ricker", 1),
    (_POPN, "MaynardSmith", 2),
]

_IDS = [name for _, name, _ in ALL_MAPS]


# ---------------------------------------------------------------------------
# Instantiation tests (fast)
# ---------------------------------------------------------------------------

# Maps that have a fixed class-level IC because random [0,1)^n ICs always escape the basin.
_MAPS_WITH_DEFAULT_IC = {"Tinkerbell"}


@pytest.mark.parametrize("module_path,class_name,expected_n_dim", ALL_MAPS, ids=_IDS)
def test_map_instantiation(module_path, class_name, expected_n_dim):
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    m = cls()
    assert m.n_dim == expected_n_dim
    assert isinstance(m.params, dict)
    if class_name not in _MAPS_WITH_DEFAULT_IC:
        assert m.initial_conds is None


@pytest.mark.parametrize("module_path,class_name,_", ALL_MAPS, ids=_IDS)
def test_map_params_as_attributes(module_path, class_name, _):
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    m = cls()
    for key in m.params:
        assert hasattr(m, key), f"{class_name} missing attribute for param '{key}'"


# ---------------------------------------------------------------------------
# Iteration tests
# ---------------------------------------------------------------------------

_STEPS = 100


@pytest.mark.slow
@pytest.mark.parametrize("module_path,class_name,expected_n_dim", ALL_MAPS, ids=_IDS)
def test_map_iterate_shape(module_path, class_name, expected_n_dim):
    """iterate() must return (steps,) time and (steps, n_dim) trajectory."""
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    m = cls()
    ic = None  # maps now carry correct default ICs at class level

    t_idx, traj = m.iterate(initial_conds=ic, steps=_STEPS)

    assert t_idx.shape == (_STEPS,), f"{class_name}: time index shape {t_idx.shape} != ({_STEPS},)"
    assert traj.shape == (_STEPS, expected_n_dim), (
        f"{class_name}: trajectory shape {traj.shape} != ({_STEPS}, {expected_n_dim})"
    )


@pytest.mark.slow
@pytest.mark.parametrize("module_path,class_name,expected_n_dim", ALL_MAPS, ids=_IDS)
def test_map_iterate_finite(module_path, class_name, expected_n_dim):
    """Trajectory must contain only finite values (known-good ICs used where needed)."""
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    m = cls()
    ic = None  # maps now carry correct default ICs at class level

    t_idx, traj = m.iterate(initial_conds=ic, steps=_STEPS, max_retries=15)
    assert np.all(np.isfinite(traj)), f"{class_name}: trajectory contains non-finite values"


@pytest.mark.slow
@pytest.mark.parametrize("module_path,class_name,expected_n_dim", ALL_MAPS, ids=_IDS)
def test_map_iterate_time_index_is_arange(module_path, class_name, expected_n_dim):
    """Time index must be np.arange(steps)."""
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    m = cls()
    ic = None  # maps now carry correct default ICs at class level

    t_idx, _ = m.iterate(initial_conds=ic, steps=_STEPS)
    np.testing.assert_array_equal(t_idx, np.arange(_STEPS))


# ---------------------------------------------------------------------------
# Custom initial conditions
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_map_custom_initial_conds_1d():
    from tsdynamics.systems.discrete.population_maps import Logistic

    m = Logistic()
    ic = np.array([0.5])
    _, traj = m.iterate(initial_conds=ic, steps=50)
    assert np.all(np.isfinite(traj))


@pytest.mark.slow
def test_map_custom_initial_conds_2d():
    from tsdynamics.systems.discrete.chaotic_maps import Henon

    m = Henon()
    ic = np.array([0.1, 0.1])
    _, traj = m.iterate(initial_conds=ic, steps=100)
    assert traj.shape == (100, 2)
    assert np.all(np.isfinite(traj))


@pytest.mark.slow
def test_map_initial_conds_stored_after_iterate():
    from tsdynamics.systems.discrete.chaotic_maps import Henon

    m = Henon()
    assert m.initial_conds is None
    ic = np.array([0.2, 0.3])
    m.iterate(initial_conds=ic, steps=50)
    np.testing.assert_array_almost_equal(m.initial_conds, ic)


# ---------------------------------------------------------------------------
# Jacobian shape check
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_map_jacobian_shape():
    from tsdynamics.systems.discrete.chaotic_maps import Henon

    m = Henon()
    J = m.jac(np.array([0.1, 0.1]))
    assert J.shape == (2, 2)


@pytest.mark.slow
def test_map_jacobian_shape_3d():
    from tsdynamics.systems.discrete.chaotic_maps import FoldedTowel

    m = FoldedTowel()
    J = m.jac(np.array([0.1, 0.1, 0.1]))
    assert J.shape == (3, 3)


# ---------------------------------------------------------------------------
# Lyapunov spectrum — all maps that have a working Jacobian
# ---------------------------------------------------------------------------

# Maps whose _jac is known to produce valid 2D output for lyapunov_spectrum.
# Excluded: Bogdanov (singular at origin), Gingerbreadman (trivial/no _jac).
_LYAP_MAPS = [
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
@pytest.mark.parametrize("module_path,class_name,expected_n_dim", _LYAP_MAPS, ids=_LYAP_IDS)
def test_map_lyapunov_shape(module_path, class_name, expected_n_dim):
    """lyapunov_spectrum(steps=200) returns shape (n_dim,) with finite values."""
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    m = cls()

    exps = m.lyapunov_spectrum(steps=200, num_exponents=expected_n_dim)

    assert exps.shape == (expected_n_dim,), (
        f"{class_name}: lyapunov_spectrum shape {exps.shape} != ({expected_n_dim},)"
    )
    assert np.all(np.isfinite(exps)), f"{class_name}: lyapunov exponents not all finite: {exps}"


@pytest.mark.slow
@pytest.mark.parametrize("module_path,class_name,expected_n_dim", _LYAP_MAPS, ids=_LYAP_IDS)
def test_map_lyapunov_partial_spectrum(module_path, class_name, expected_n_dim):
    """Requesting num_exponents=1 always returns a 1-element array."""
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    m = cls()

    exps = m.lyapunov_spectrum(steps=200, num_exponents=1)
    assert exps.shape == (1,)
    assert np.isfinite(exps[0]), f"{class_name}: LE is not finite: {exps[0]}"
