"""Shared pytest fixtures and registry-driven parametrization."""

from __future__ import annotations

import numpy as np
import pytest

from tsdynamics import registry

# ---------------------------------------------------------------------------
# Registry-driven parametrization
#
# Any test that takes one of these fixture names is automatically run once
# per registered built-in system of the matching family.  Adding a new
# system to the library therefore adds it to the bulk suite with zero
# test-file edits.
# ---------------------------------------------------------------------------

_FAMILY_FIXTURES = {
    "ode_entry": "ode",
    "dde_entry": "dde",
    "map_entry": "map",
    "sde_entry": "sde",  # no built-in SDE systems yet → empty sweep, ready for them
    "system_entry": None,  # every family
}


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    for fixture, family in _FAMILY_FIXTURES.items():
        if fixture in metafunc.fixturenames:
            entries = registry.all_systems(family=family)
            metafunc.parametrize(fixture, entries, ids=[e.name for e in entries])


@pytest.fixture
def rng() -> np.random.Generator:
    """A reproducible NumPy ``Generator`` seeded at 42."""
    return np.random.default_rng(42)
