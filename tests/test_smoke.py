"""Smoke tests — fast checks that the package is importable and top-level API works."""

from __future__ import annotations

import importlib

import numpy as np
import pytest


def test_package_importable() -> None:
    import tsdynamics  # noqa: F401


def test_submodules_importable() -> None:
    for module in ["tsdynamics.families", "tsdynamics.utils", "tsdynamics.systems"]:
        importlib.import_module(module)


def test_version_attribute_exists() -> None:
    import tsdynamics

    assert isinstance(tsdynamics.__version__, str)
    assert len(tsdynamics.__version__) > 0


def test_systems_subpackages_importable() -> None:
    for module in [
        "tsdynamics.systems.continuous.chaotic_attractors",
        "tsdynamics.systems.continuous.chem_bio_systems",
        "tsdynamics.systems.continuous.climate_geophysics",
        "tsdynamics.systems.continuous.coupled_systems",
        "tsdynamics.systems.continuous.delayed_systems",
        "tsdynamics.systems.continuous.exotic_systems",
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


def test_top_level_reexports() -> None:
    """Built-in systems are accessible directly from the top-level namespace."""
    import tsdynamics as ts

    for name in (
        "Lorenz",
        "Rossler",
        "Lorenz96",
        "KuramotoSivashinsky",
        "MackeyGlass",
        "IkedaDelay",
        "Henon",
        "Logistic",
    ):
        assert hasattr(ts, name), f"{name} missing from tsdynamics top-level"


def test_utils_public_surface() -> None:
    from tsdynamics.utils import (  # noqa: F401
        SagittaDt,
        estimate_dt_from_sagitta,
        staticjit,
    )


def test_internals_not_in_top_level_all() -> None:
    """``staticjit`` and ``ParamSet`` are reachable but not advertised at top level."""
    import tsdynamics as ts

    assert "staticjit" not in ts.__all__
    assert "ParamSet" not in ts.__all__
    assert "SystemBase" not in ts.__all__


@pytest.mark.slow
def test_lorenz_integrates() -> None:
    import tsdynamics as ts

    traj = ts.Lorenz().integrate(final_time=5.0, dt=0.05)
    assert traj.y.shape == (traj.t.shape[0], 3)
    assert np.all(np.isfinite(traj.y))


@pytest.mark.slow
def test_henon_iterates() -> None:
    import tsdynamics as ts

    traj = ts.Henon().iterate(steps=200)
    assert traj.y.shape == (200, 2)
    assert traj.t.shape == (200,)
    assert np.all(np.isfinite(traj.y))
