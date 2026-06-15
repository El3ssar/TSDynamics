"""Shared pytest fixtures and registry-driven parametrization."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, settings

import tsdynamics.transforms as _transforms  # noqa: F401  (populates registry.transforms)
from tsdynamics import registry

# ---------------------------------------------------------------------------
# Hypothesis configuration (stream I-QA property-test harness)
#
# The suite runs under ``filterwarnings = ["error"]`` and a tier split, so the
# property tests need a profile that (a) drops the wall-clock ``deadline`` —
# numeric routines vary in timing and a deadline turns that into flaky
# ``DeadlineExceeded`` errors — and (b) suppresses health checks that are
# expected here (estimators are deliberately a little slow; a few semantic
# tests share a function-scoped fixture).  ``max_examples`` is kept modest so
# the fast tier stays fast; heavy point-cloud tests override it locally.
# ---------------------------------------------------------------------------

settings.register_profile(
    "tsdynamics",
    max_examples=50,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
        HealthCheck.function_scoped_fixture,
    ],
)
settings.load_profile("tsdynamics")

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

# Registry-driven parametrization over the *generic* D4 registries (the new
# analysis/transform plugin surface).  A test taking ``analysis_entry`` runs
# once per registered analysis; ``transform_entry`` once per transform.  Adding
# an analysis/transform therefore sweeps it into the meta-QA with zero edits —
# the analyses/transforms analogue of the per-system sweep above.
_REGISTRY_FIXTURES = {
    "analysis_entry": registry.analyses,
    "transform_entry": registry.transforms,
}


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    for fixture, family in _FAMILY_FIXTURES.items():
        if fixture in metafunc.fixturenames:
            entries = registry.all_systems(family=family)
            metafunc.parametrize(fixture, entries, ids=[e.name for e in entries])

    for fixture, reg in _REGISTRY_FIXTURES.items():
        if fixture in metafunc.fixturenames:
            entries = sorted(reg.all(), key=lambda e: e.name)
            metafunc.parametrize(fixture, entries, ids=[e.name for e in entries])


@pytest.fixture
def rng() -> np.random.Generator:
    """A reproducible NumPy ``Generator`` seeded at 42."""
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Shared synthetic-signal fixtures
#
# Deterministic, reproducible signals with *known qualitative character*
# (periodic / chaotic-deterministic / linear-stochastic / white) that the
# known-value and cross-quantifier tests reuse.  Compile-free (analytic or
# cheap map orbits) so they stay in the fast tier.
# ---------------------------------------------------------------------------


@pytest.fixture
def periodic_signal() -> np.ndarray:
    """A clean periodic signal (sum of two commensurate sinusoids)."""
    from _strategies import sinusoid

    return sinusoid(2048, freq=0.02) + 0.5 * sinusoid(2048, freq=0.04, phase=0.7)


@pytest.fixture
def chaotic_signal() -> np.ndarray:
    """A deterministic-chaotic series (Hénon ``x``-coordinate)."""
    from _strategies import henon_series

    return henon_series(2048)


@pytest.fixture
def noise_signal() -> np.ndarray:
    """Seeded Gaussian white noise (the maximally irregular reference)."""
    from _strategies import white_noise

    return white_noise(2048, seed=12345)


@pytest.fixture
def ar1_signal() -> np.ndarray:
    """A seeded AR(1) series (the linear-stochastic null)."""
    from _strategies import ar1

    return ar1(2048, phi=0.7, seed=2024)
