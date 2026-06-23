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


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the change-scoped selection flags (stream CI-CHANGED).

    ``--changed`` narrows the run to the tests a diff can affect (see
    ``tests/_changed_select.py``); ``--changed-since`` overrides the diff base
    (default ``origin/main`` / ``$TSD_CHANGED_BASE``).  Without ``--changed``
    nothing changes — the full suite collects as before.
    """
    import os

    from _changed_select import DEFAULT_BASE

    group = parser.getgroup("change-scoped selection")
    group.addoption(
        "--changed",
        action="store_true",
        default=False,
        help="Run only the tests a diff vs the base ref can affect "
        "(falls back to the full suite on foundational changes).",
    )
    group.addoption(
        "--changed-since",
        action="store",
        default=os.environ.get("TSD_CHANGED_BASE", DEFAULT_BASE),
        metavar="REF",
        help="Diff base for --changed (default: $TSD_CHANGED_BASE or origin/main).",
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    for fixture, family in _FAMILY_FIXTURES.items():
        if fixture in metafunc.fixturenames:
            entries = registry.all_systems(family=family)
            metafunc.parametrize(fixture, entries, ids=[e.name for e in entries])

    for fixture, reg in _REGISTRY_FIXTURES.items():
        if fixture in metafunc.fixturenames:
            entries = sorted(reg.all(), key=lambda e: e.name)
            metafunc.parametrize(fixture, entries, ids=[e.name for e in entries])


# ---------------------------------------------------------------------------
# Engine-marker auto-tagging (stream I-XVAL)
#
# Any test module that imports the compiled ``tsdynamics._rust`` extension is
# tagged with the ``engine`` marker, so the engine CI job selects them all with
# ``-m engine`` instead of a hand-maintained file list.  The tag follows the
# import (see ``_engine_marker``), so a new engine test file is covered with zero
# CI edits.  ``tests/test_engine_coverage.py`` asserts this invariant holds.
# ---------------------------------------------------------------------------

_engine_module_cache: dict[str, bool] = {}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    from _engine_marker import is_engine_test_file

    for item in items:
        path = str(item.path)
        flag = _engine_module_cache.get(path)
        if flag is None:
            flag = is_engine_test_file(path)
            _engine_module_cache[path] = flag
        if flag:
            item.add_marker(pytest.mark.engine)

    # Change-scoped selection runs LAST so it sees the final (marked) item set.
    # The deselection happens here (on every collecting process, incl. xdist
    # workers); the human-readable report is emitted by
    # pytest_report_collectionfinish below, which is the controller-side hook that
    # reliably reaches the terminal under `-n auto` (worker stdout is swallowed).
    if config.getoption("changed", default=False):
        from _changed_select import keep_item

        plan = _changed_plan(config)
        if plan.full:
            return
        kept = [item for item in items if keep_item(item, plan)]
        kept_set = set(kept)
        deselected = [item for item in items if item not in kept_set]
        if deselected:
            config.hook.pytest_deselected(items=deselected)
            items[:] = kept


_changed_plan_cache: object | None = None


def _changed_plan(config: pytest.Config) -> object:
    """The selection plan for this run (computed once per process)."""
    global _changed_plan_cache
    if _changed_plan_cache is None:
        from _changed_select import compute_plan

        _changed_plan_cache = compute_plan(config.getoption("changed_since"))
    return _changed_plan_cache


def pytest_report_collectionfinish(config: pytest.Config) -> list[str]:
    """Report what change-scoped selection kept and why (controller-side).

    Runs after collection on the controller (and in serial), so the
    ``[changed-select]`` summary survives ``pytest -n auto``.
    """
    if not config.getoption("changed", default=False):
        return []
    plan = _changed_plan(config)
    base = config.getoption("changed_since")
    if plan.full:
        return [
            f"[changed-select] full run — {plan.reason} "
            f"({len(plan.changed)} changed file(s) vs {base})"
        ]
    lines = [f"[changed-select] scoped to diff vs {base} ({len(plan.changed)} changed file(s))"]
    if plan.systems:
        lines.append(
            f"[changed-select] system sweeps limited to: {', '.join(sorted(plan.systems))}"
        )
    if plan.selected_files:
        lines.append(f"[changed-select] test files: {', '.join(sorted(plan.selected_files))}")
    return lines


def _normalize_changed_exitstatus(
    config: pytest.Config, exitstatus: int | pytest.ExitCode
) -> int | pytest.ExitCode:
    """Treat an empty ``--changed`` shard as success.

    In change-scoped CI a narrow diff can legitimately leave a marker-restricted
    shard (for example ``-m slow`` or ``-m engine``) with nothing to run. Pytest
    reports that as ``NO_TESTS_COLLECTED`` / exit code 5, but for a scoped shard
    that means "not applicable", not "failed".
    """
    if (
        config.getoption("changed", default=False)
        and exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED
    ):
        return pytest.ExitCode.OK
    return exitstatus


def pytest_sessionfinish(session: pytest.Session, exitstatus: int | pytest.ExitCode) -> None:
    """Normalize the exit code for change-scoped empty shards."""
    session.exitstatus = _normalize_changed_exitstatus(session.config, exitstatus)


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
