"""Smoke tests — fast checks that the package is importable and top-level API works."""

import importlib

import pytest


def test_package_importable():
    import tsdynamics  # noqa: F401


def test_submodules_importable():
    for module in ["tsdynamics.base", "tsdynamics.utils", "tsdynamics.systems", "tsdynamics.viz"]:
        importlib.import_module(module)


def test_version_attribute_exists():
    import tsdynamics._version as v

    assert hasattr(v, "__version__")
    assert isinstance(v.__version__, str)
    assert len(v.__version__) > 0


def test_systems_subpackages_importable():
    for module in [
        "tsdynamics.systems.continuous.chaotic_attractors",
        "tsdynamics.systems.continuous.chem_bio_systems",
        "tsdynamics.systems.continuous.climate_geophysics",
        "tsdynamics.systems.continuous.coupled_systems",
        "tsdynamics.systems.continuous.delayed_systems",
        "tsdynamics.systems.continuous.exotic_systems",
        "tsdynamics.systems.continuous.neural_cognitive",
        "tsdynamics.systems.continuous.oscillatory_systems",
        "tsdynamics.systems.continuous.physical_systems",
        "tsdynamics.systems.continuous.population_dynamics",
        "tsdynamics.systems.discrete.chaotic_maps",
        "tsdynamics.systems.discrete.exotic_maps",
        "tsdynamics.systems.discrete.geometric_maps",
        "tsdynamics.systems.discrete.polynomial_maps",
        "tsdynamics.systems.discrete.population_maps",
    ]:
        importlib.import_module(module)


def test_viz_submodules_importable():
    for module in [
        "tsdynamics.viz.base",
        "tsdynamics.viz.plotters",
        "tsdynamics.viz.animators",
        "tsdynamics.viz.transforms",
    ]:
        importlib.import_module(module)


def test_utils_importable():
    from tsdynamics.utils import (  # noqa: F401
        estimate_curvature_timestep,
        estimate_dt_from_sagitta,
        estimate_dt_from_spectrum,
        staticjit,
    )


@pytest.mark.slow
def test_lorenz_integrates():
    """Lorenz integration: shapes are correct, output is finite."""
    import numpy as np

    from tsdynamics.systems.continuous.chaotic_attractors import Lorenz

    lor = Lorenz()
    t, X = lor.integrate(dt=0.05, final_time=5.0)

    assert t.ndim == 1
    assert X.ndim == 2
    assert X.shape == (t.shape[0], 3)
    assert np.all(np.isfinite(X))


@pytest.mark.slow
def test_henon_iterates():
    """Henon map iteration: shapes are correct."""
    import numpy as np

    from tsdynamics.systems.discrete.chaotic_maps import Henon

    h = Henon()
    t_idx, traj = h.iterate(steps=200)

    assert traj.shape == (200, 2)
    assert t_idx.shape == (200,)
    assert np.all(np.isfinite(traj))
