"""Regression test for WP9: ``zero_one_test`` must honor an explicit ``ic``.

The 0--1 test (Gottwald & Melbourne 2004, 2009) characterises a *specific*
orbit, so an explicit ``ic`` must select that orbit.  A discrete *view*
(``PoincareMap`` / ``StroboscopicMap``) has ``is_discrete=True`` and a
``trajectory`` accepting ``ic`` via ``**kwargs`` but no ``iterate`` method.
The pre-fix observable resolver gated ``ic`` on ``hasattr(system, "iterate")``,
so a caller passing ``ic`` to such a view had it silently dropped and got the
result for the wrapper's *default* orbit instead.
"""

from __future__ import annotations

import numpy as np

import tsdynamics as ts
from tsdynamics.analysis.chaos import _common as _c
from tsdynamics.analysis.chaos.zero_one import _observable


def _section_observable(pmap: ts.PoincareMap) -> np.ndarray:
    """Read the y-component observable from a fresh section of ``pmap``."""
    return np.asarray(_c._as_observable(pmap.trajectory(250, transient=0), 1), dtype=float)


def test_zero_one_observable_selects_the_requested_orbit() -> None:
    """An explicit ``ic`` must yield the observable of *that* orbit, not the default.

    A ``PoincareMap`` is a discrete view (``is_discrete=True``, has ``trajectory``
    but no ``iterate``).  Pre-fix the ``ic`` was gated on
    ``hasattr(system, "iterate")`` and dropped, so the observable matched the
    wrapper's *default* orbit (``ic=[1, 1, 0]``) rather than the requested one —
    this assertion (``got`` equals the reference orbit, and differs from the
    default orbit) would have failed.
    """
    ic_x = [0.5, -1.2, 0.4]

    pmap = ts.PoincareMap(ts.Rossler(ic=[1.0, 1.0, 0.0]), plane=(0, 0.0), dt=0.05)
    assert pmap.is_discrete
    assert not hasattr(pmap, "iterate")
    assert hasattr(pmap, "trajectory")

    got = _observable(pmap, 1, final_time=None, n=250, dt=None, transient=None, ic=ic_x)

    # The observable of the orbit literally started at ``ic_x`` (the reference).
    ref = _section_observable(ts.PoincareMap(ts.Rossler(ic=ic_x), plane=(0, 0.0), dt=0.05))
    # The observable of the wrapper's *default* orbit (what the pre-fix gate returned).
    default = _section_observable(
        ts.PoincareMap(ts.Rossler(ic=[1.0, 1.0, 0.0]), plane=(0, 0.0), dt=0.05)
    )

    assert got.size == ref.size > 0
    assert np.allclose(got, ref)  # honors ic
    assert not np.allclose(got, default)  # not the silently-substituted default orbit
