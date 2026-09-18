"""Smoke tests — fast checks that the package is importable and top-level API works."""

from __future__ import annotations

import importlib

import numpy as np
import pytest


def test_package_importable() -> None:
    import tsdynamics  # noqa: F401


def test_submodules_importable() -> None:
    for module in ["tsdynamics.families", "tsdynamics._utils", "tsdynamics.systems"]:
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
    """Built-in systems stay accessible from the top-level namespace (lazily)."""
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
        assert hasattr(ts.systems, name), f"{name} missing from ts.systems"


def test_systems_is_the_canonical_model_path() -> None:
    """``tsdynamics.systems.<Name>`` is the canonical flat path; ``tsd.<Name>`` is its alias."""
    import tsdynamics as ts

    assert ts.systems.Lorenz is ts.systems.Lorenz
    assert ts.systems.Henon is ts.systems.Henon
    # The flat catalogue covers every registered builtin.
    from tsdynamics import registry

    flat = set(ts.systems.__all__)
    for entry in registry.all_systems():
        assert entry.name in flat, f"{entry.name} not flat-exported at tsdynamics.systems"


def test_models_do_not_clutter_top_level_namespace() -> None:
    """Models are hidden from ``dir()`` / ``__all__`` so the submodules stay findable."""
    import tsdynamics as ts

    assert "Lorenz" not in ts.__all__
    assert "Lorenz" not in dir(ts)
    # ...but the four navigable submodules and the base classes ARE on the surface.
    # (``data`` / ``derived`` / ``registry`` were demoted from tab completion by the
    # v6 namespace curation — they stay importable; see tests/test_namespace_curation.py.)
    for name in ("analysis", "systems", "viz", "ContinuousSystem"):
        assert name in dir(ts)
    for demoted in ("data", "derived", "registry", "errors"):
        assert demoted not in dir(ts)
        assert hasattr(ts, demoted)
    # An unknown attribute still raises a clean AttributeError (not a model miss).
    with pytest.raises(AttributeError):
        _ = ts.DefinitelyNotASystem


def test_utils_public_surface() -> None:
    import tsdynamics._utils as u
    from tsdynamics._utils import make_output_grid  # noqa: F401

    # The sagitta tooling moved to ``tsdynamics.analysis.sampling`` (and ``SagittaDt``
    # is hidden).  ``utils`` is the leaf package holding the values BOTH the family
    # layer and the engine layer must agree on: the output grid
    # (``make_output_grid``) and, since v6, the solver-tolerance defaults
    # (``utils/tolerances.py`` — hoisted out of sixteen duplicated literals).  Both
    # are documented contract, so both are on the surface.
    assert set(u.__all__) == {
        "make_output_grid",
        "DEFAULT_RTOL",
        "DEFAULT_ATOL",
        "DDE_RTOL",
        "DDE_ATOL",
        "DDE_LYAPUNOV_RTOL",
        "DDE_LYAPUNOV_ATOL",
        "BASIN_RTOL",
        "BASIN_ATOL",
    }
    from tsdynamics.analysis.sampling import estimate_dt_from_sagitta  # noqa: F401


def test_internals_not_in_top_level_all() -> None:
    """``ParamSet`` and ``SystemBase`` are reachable but not advertised at top level."""
    import tsdynamics as ts

    assert "ParamSet" not in ts.__all__
    assert "SystemBase" not in ts.__all__


@pytest.mark.slow
def test_lorenz_integrates() -> None:
    import tsdynamics as ts

    traj = ts.systems.Lorenz().run(final_time=5.0, dt=0.05)
    assert traj.y.shape == (traj.t.shape[0], 3)
    assert np.all(np.isfinite(traj.y))


@pytest.mark.slow
def test_henon_iterates() -> None:
    import tsdynamics as ts

    traj = ts.systems.Henon().run(steps=200)
    assert traj.y.shape == (200, 2)
    assert traj.t.shape == (200,)
    assert np.all(np.isfinite(traj.y))
