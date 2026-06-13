"""
Compile-cache isolation: redefined or same-named classes must never reuse
another definition's compiled dynamics, and steppers must not leak temp dirs.

Regression tests for the v2.0.0 pre-release review findings.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.utils import staticjit

# ---------------------------------------------------------------------------
# Numba iterate cache (fast)
# ---------------------------------------------------------------------------


def test_redefined_map_does_not_reuse_old_compiled_loop() -> None:
    """Two same-named map classes with different _step must run their own code."""

    class _CacheProbe(ts.DiscreteMap):
        params = {"r": 2.0}
        dim = 1

        @staticjit
        def _step(X, r):
            return (X[0] / r,)  # halving map

        @staticjit
        def _jacobian(X, r):
            return ((1.0 / r,),)

    first = _CacheProbe().iterate(steps=20, ic=[1.0]).y[0, 0]
    assert first == pytest.approx(0.5)

    class _CacheProbe(ts.DiscreteMap):  # noqa: F811 — deliberate redefinition
        params = {"r": 2.0}
        dim = 1

        @staticjit
        def _step(X, r):
            return (X[0] * r,)  # doubling map — same name, same params hash

        @staticjit
        def _jacobian(X, r):
            return ((r,),)

    second = _CacheProbe().iterate(steps=20, ic=[1.0]).y[0, 0]
    assert second == pytest.approx(2.0), (
        "redefined map class reused the old class's compiled iterate loop"
    )


def test_module_path_distinguishes_equations() -> None:
    """Same-named ODE classes with different equations get different cache paths."""

    class _OdeProbe(ts.ContinuousSystem):
        params = {"a": 1.0}
        dim = 1

        @staticmethod
        def _equations(y, t, *, a):
            return (a * y(0),)

    path_one = str(_OdeProbe()._module_path())

    class _OdeProbe(ts.ContinuousSystem):  # noqa: F811 — deliberate redefinition
        params = {"a": 1.0}
        dim = 1

        @staticmethod
        def _equations(y, t, *, a):
            return (-a * y(0),)  # sign flipped

    path_two = str(_OdeProbe()._module_path())
    assert path_one != path_two, "cache path ignores the equations content"


def test_user_class_shadowing_builtin_gets_own_cache_path() -> None:
    class Lorenz(ts.ContinuousSystem):  # shadows the builtin name
        params = {"k": 1.0}
        dim = 3

        @staticmethod
        def _equations(y, t, *, k):
            return (-k * y(0), -k * y(1), -k * y(2))

    assert str(Lorenz()._module_path()) != str(ts.Lorenz()._module_path())


def test_dde_cache_key_distinguishes_equations() -> None:
    class _DdeProbe(ts.DelaySystem):
        params = {"tau": 1.0}
        dim = 1

        @staticmethod
        def _equations(y, t, *, tau):
            return [-y(0, t - tau)]

    key_one = _DdeProbe()._cache_key()

    class _DdeProbe(ts.DelaySystem):  # noqa: F811 — deliberate redefinition
        params = {"tau": 1.0}
        dim = 1

        @staticmethod
        def _equations(y, t, *, tau):
            return [+y(0, t - tau)]  # sign flipped

    assert _DdeProbe()._cache_key() != key_one


# ---------------------------------------------------------------------------
# ODE redefinition end-to-end + stepper temp-dir hygiene (slow — compiles)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_redefined_ode_integrates_its_own_dynamics() -> None:
    class _GrowthProbe(ts.ContinuousSystem):
        params = {"a": 1.0}
        dim = 1

        @staticmethod
        def _equations(y, t, *, a):
            return (a * y(0),)  # exponential growth

    up = _GrowthProbe().integrate(final_time=1.0, dt=0.5, ic=[1.0]).y[-1, 0]
    assert up == pytest.approx(np.e, rel=1e-4)

    class _GrowthProbe(ts.ContinuousSystem):  # noqa: F811 — deliberate redefinition
        params = {"a": 1.0}
        dim = 1

        @staticmethod
        def _equations(y, t, *, a):
            return (-a * y(0),)  # exponential decay

    down = _GrowthProbe().integrate(final_time=1.0, dt=0.5, ic=[1.0]).y[-1, 0]
    assert down == pytest.approx(1.0 / np.e, rel=1e-4), (
        "redefined ODE class reused the old class's compiled module"
    )


@pytest.mark.slow
def test_reinit_does_not_accumulate_stepper_tempdirs() -> None:
    import tempfile

    pattern = os.path.join(tempfile.gettempdir(), f"tsdyn_stepper_{os.getpid()}_*")
    lor = ts.Lorenz()
    lor.reinit([1.0, 1.0, 1.0])
    baseline = len(glob.glob(pattern))
    for _ in range(5):
        lor.reinit([1.0, 1.0, 1.0])
    after = len(glob.glob(pattern))
    assert after <= baseline + 1, (
        f"reinit leaks temp dirs: {baseline} -> {after} (must reclaim the previous copy)"
    )
    # Instance death must reclaim the last one too.
    last_dir = lor._stepper_tmpdir
    del lor
    import gc

    gc.collect()
    assert not os.path.exists(last_dir), "stepper temp dir survived instance destruction"
