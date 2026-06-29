r"""
Result-identity gate for the Rust basin-march kernel (stream ``perf/basin-march``).

The whole per-initial-condition recurrence FSM (stepping + cell-binning + the
shared-label early-out) runs in one sequential Rust kernel
(:func:`tsdynamics.engine.run.basin_march`, driven by
:func:`tsdynamics.analysis.basins.attractors.classify_seeds`) on a supported run
— an ODE flow or a map whose ``_step`` lowers, on the ``interp`` / ``jit``
backend.  Because every cell check advances the *same* engine stepper the Python
``_AttractorMapper`` drives (a fresh-solver dense integrate over ``[t, t+dt]`` for
a flow, one lowered-IR iteration for a map), the labels are **bit-identical** to
the pure-Python loop, which stays the fallback and the oracle.

This module pins that identity: for each validation system, the public
:func:`basins_of_attraction` (which auto-selects the Rust path) must yield a
**byte-identical** basin *label image* to the same call forced onto the Python
fallback, and the Rust path must be deterministic across repeats.

The :class:`AttractorSet` identity is byte-identical too **for flows** (the
Python fallback and the Rust kernel both advance the *same* engine stepper, so
every located point matches to the bit).  For a **map** the two paths differ in
*one* respect: the Rust kernel advances the lowered-IR ``_step`` while the
Python fallback advances the pure-Python ``_step``.  These agree bit-for-bit on
a pure ``+,-,*`` step but the IR's ``x**2`` / ``x**3`` can differ by one ULP, so
along a *chaotic* map orbit the **located attractor point cloud** drifts (the
same attractor, a few cells apart) — exactly the IR-vs-NumPy caveat WS-MAPITER
documents for ``orbit_diagram`` over maps.  The basin *labels* (a robust
recurrence classification) are unaffected and stay byte-identical.  So the
AttractorSet is asserted byte-identical only where the numerics genuinely agree
(flows + non-chaotic maps); a chaotic map asserts the same attractor *count* and
representative instead.

The validation systems mirror ``tests/test_basins.py`` (TEST-LOCAL, not
catalogue): the two-well Duffing (½-basins), the magnetic pendulum (fractal
basins), and the Hénon escape basin; plus a cubic map (two fixed points) as the
simple multi-attractor map case.  The Newton ``z**3`` map is *not* parametrized
here because its complex-arithmetic ``_step`` does not lower — the kernel
transparently falls back to Python (covered by its own fallback test).
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.analysis import basins as bas
from tsdynamics.analysis.basins import attractors as _attractors
from tsdynamics.data import Box, Grid

pytest.importorskip("tsdynamics._rust")


# ===========================================================================
# Validation systems (TEST-LOCAL; not in the built-in catalogue)
# ===========================================================================


class CubicMap(ts.DiscreteMap):
    """``x -> a*x - x**3`` — two stable fixed points at ``±sqrt(a-1)`` for 1<a<2.

    A pure ``+,-,*`` step, so the lowered IR reproduces the pure-Python ``_step``
    bit-for-bit — the simple multi-attractor *map* case.
    """

    params = {"a": 1.5}
    dim = 1

    @staticmethod
    def _step(X, a):
        return (a * X[0] - X[0] ** 3,)

    @staticmethod
    def _jacobian(X, a):
        return ((a - 3.0 * X[0] ** 2,),)


class DuffingTwoWell(ts.ContinuousSystem):
    """Damped two-well Duffing ``x'' = x - x**3 - delta x'`` — wells at ``x=±1``."""

    params = {"delta": 0.3}
    dim = 2
    variables = ("x", "y")

    @staticmethod
    def _equations(Y, t, *, delta):
        x, y = Y(0), Y(1)
        return (y, x - x**3 - delta * y)


_MAGNETS = ((1.0, 0.0), (-0.5, 0.8660254037844386), (-0.5, -0.8660254037844386))


class MagneticPendulum(ts.ContinuousSystem):
    """Magnetic pendulum over three symmetric magnets — fractal basins."""

    params = {"damping": 0.2, "h": 0.3, "omega2": 0.5}
    dim = 4
    variables = ("x", "y", "vx", "vy")

    @staticmethod
    def _equations(Y, t, *, damping, h, omega2):
        x, y, vx, vy = Y(0), Y(1), Y(2), Y(3)
        ax = -omega2 * x - damping * vx
        ay = -omega2 * y - damping * vy
        for mx, my in _MAGNETS:
            den = ((x - mx) ** 2 + (y - my) ** 2 + h**2) ** 1.5
            ax = ax - (x - mx) / den
            ay = ay - (y - my) / den
        return (vx, vy, ax, ay)


# ===========================================================================
# Per-system basin runs (kept small so the gate stays in the slow tier budget)
# ===========================================================================


def _duffing_run():
    return (
        DuffingTwoWell(),
        Grid([-2.0, -2.0], [2.0, 2.0], (40, 40)),
        dict(dt=0.4, max_steps=400, consecutive_recurrences=20, attractor_locate_steps=10),
    )


def _cubic_run():
    # A 1-D map basin image (a line of cells); two fixed points at ±sqrt(0.5).
    return (
        CubicMap(),
        Grid([-2.0], [2.0], (400,)),
        dict(max_steps=200, consecutive_recurrences=8, attractor_locate_steps=5),
    )


def _henon_run():
    return (
        ts.Henon(),
        Grid([-2.0, -2.0], [2.0, 2.0], (40, 40)),
        dict(max_steps=2000),
    )


def _magnetic_pendulum_run():
    # A 2-D position slice of the 4-D flow, recurrences on the full-dim box.
    sys = MagneticPendulum()
    slice_grid = Grid([-1.3, -1.3, 0.0, 0.0], [1.3, 1.3, 0.0, 0.0], (24, 24, 1, 1))
    rbox = Box([-1.6, -1.6, -3.0, -3.0], [1.6, 1.6, 3.0, 3.0])
    return (
        sys,
        slice_grid,
        dict(
            recurrence=rbox,
            recurrence_resolution=(48, 48, 36, 36),
            dt=0.5,
            max_steps=700,
            consecutive_recurrences=20,
            attractor_locate_steps=12,
        ),
    )


#: ``name -> (factory, attractorset_exact)``.  ``attractorset_exact`` is whether the
#: located :class:`AttractorSet` is byte-identical between the Rust and Python paths.
#: It is **True for flows**: both paths advance the *same* engine stepper, so every
#: located point matches to the bit.  It is **False for maps**: the Rust kernel
#: advances the lowered-IR ``_step`` while the Python fallback advances pure-Python
#: ``_step``, and the IR's ``x**2`` / ``x**3`` can differ by one ULP — so the
#: *located point cloud* (hence the centroid, and for a chaotic map the located cell
#: set) drifts even though the basin *labels* stay byte-identical.  A False run
#: asserts the same attractor ids and representatives instead (same attractor).
_RUNS = {
    "cubic_map": (_cubic_run, False),
    "duffing_two_well": (_duffing_run, True),
    "henon_escape": (_henon_run, False),
    "magnetic_pendulum": (_magnetic_pendulum_run, True),
}


# ===========================================================================
# Helpers
# ===========================================================================


def _force_python(monkeypatch) -> None:
    """Force ``classify_seeds`` onto the pure-Python ``map_ic`` fallback (the oracle).

    Patches the support predicate the seam consults, so the *exact same*
    :func:`basins_of_attraction` code path runs the per-seed Python loop instead of
    the Rust kernel — the cleanest oracle (no re-implementation of the
    merge/harvest/painting that follows).
    """
    monkeypatch.setattr(_attractors, "_march_supported", lambda system, backend: False)


def _assert_attractor_sets_identical(a, b) -> None:
    """Same ids, same per-id cell counts, and byte-identical centres (the flow case)."""
    assert a.ids == b.ids, f"attractor ids differ: {a.ids} vs {b.ids}"
    assert a.diverged == b.diverged
    assert a.seeds == b.seeds
    for k in a.ids:
        assert a[k].cells == b[k].cells, f"id {k} cell count differs"
        # The point clouds are pooled from byte-identical engine states, so the
        # centroids match to the bit (not just to a tolerance).
        np.testing.assert_array_equal(a[k].center, b[k].center)


def _assert_attractor_sets_same(a, b) -> None:
    """Same attractor count / diverged / seeds (the map case).

    The located point cloud follows the IR (Rust) vs pure-Python (oracle) ``_step``,
    which differ by ULPs, so neither the centroid nor (for a *chaotic* map) the
    located cell count is byte-identical — they are the *same attractor* a few cells
    apart.  The byte-identical claim is carried by the basin **label image** (the
    caller asserts that separately); here assert only the structural invariants the
    ULP drift preserves: the same number of attractors and the same diverged share.
    """
    assert len(a) == len(b), f"attractor count differs: {len(a)} vs {len(b)}"
    assert a.diverged == b.diverged
    assert a.seeds == b.seeds


# ===========================================================================
# Identity gate
# ===========================================================================


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(_RUNS))
def test_rust_kernel_basin_image_bit_identical_to_python(name, monkeypatch):
    """The Rust-march basin *label image* equals the forced-Python oracle, byte for byte.

    Plus the AttractorSet: byte-identical for a flow (same engine stepper on both
    paths), same-attractor for a map (the IR/pure-Python ``_step`` ULP difference
    moves the located point cloud, never the labels).
    """
    make, attractorset_exact = _RUNS[name]

    # Rust path (the public function auto-selects it on interp).
    system, grid, kw = make()
    rust = bas.basins_of_attraction(system, grid, **kw)

    # Python oracle: identical call, support predicate forced off.
    _force_python(monkeypatch)
    system2, grid2, kw2 = make()
    py = bas.basins_of_attraction(system2, grid2, **kw2)

    # The whole label image is byte-identical (not just same fractions) for EVERY
    # validation system — the headline deliverable.
    np.testing.assert_array_equal(rust.labels, py.labels)
    assert rust.labels.dtype == py.labels.dtype
    assert rust.shape == py.shape
    if attractorset_exact:
        _assert_attractor_sets_identical(rust.attractors, py.attractors)
    else:
        _assert_attractor_sets_same(rust.attractors, py.attractors)


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(_RUNS))
def test_rust_kernel_basins_deterministic(name):
    """A repeated Rust-march run returns a byte-identical label image."""
    make, _ = _RUNS[name]
    system, grid, kw = make()
    first = bas.basins_of_attraction(system, grid, **kw)
    system2, grid2, kw2 = make()
    second = bas.basins_of_attraction(system2, grid2, **kw2)
    np.testing.assert_array_equal(first.labels, second.labels)


@pytest.mark.slow
def test_find_attractors_rust_identical_to_python(monkeypatch):
    """``find_attractors`` (Rust path) matches the forced-Python oracle attractor set."""
    rust = bas.find_attractors(
        DuffingTwoWell(),
        Box([-2.0, -2.0], [2.0, 2.0]),
        resolution=40,
        n_seeds=120,
        seed=0,
        dt=0.4,
        max_steps=400,
        consecutive_recurrences=20,
        attractor_locate_steps=10,
    )
    _force_python(monkeypatch)
    py = bas.find_attractors(
        DuffingTwoWell(),
        Box([-2.0, -2.0], [2.0, 2.0]),
        resolution=40,
        n_seeds=120,
        seed=0,
        dt=0.4,
        max_steps=400,
        consecutive_recurrences=20,
        attractor_locate_steps=10,
    )
    _assert_attractor_sets_identical(rust, py)


@pytest.mark.slow
def test_map_path_unlowerable_step_falls_back(monkeypatch):
    """A map whose ``_step`` does not lower transparently uses the Python loop.

    The complex-arithmetic Newton ``z**3`` map cannot lower to the engine IR, so
    ``classify_seeds`` must fall back rather than raise — the result still matches
    the forced-Python oracle.
    """

    class NewtonMap(ts.DiscreteMap):
        params: dict = {}
        dim = 2
        variables = ("re", "im")
        _jacobian_fd_check = False

        @staticmethod
        def _step(X):
            z = complex(X[0], X[1])
            z2 = z * z
            z3 = z2 * z
            z = (2.0 * z3 + 1.0) / (3.0 * z2)
            return (z.real, z.imag)

        @staticmethod
        def _jacobian(X):
            return ((0.0, 0.0), (0.0, 0.0))

    grid = Grid([-1.0, -1.0], [1.0, 1.0], (40, 40))
    kw = dict(consecutive_recurrences=8, attractor_locate_steps=5, max_steps=200)
    auto = bas.basins_of_attraction(NewtonMap(), grid, **kw)  # falls back internally
    _force_python(monkeypatch)
    forced = bas.basins_of_attraction(NewtonMap(), grid, **kw)
    np.testing.assert_array_equal(auto.labels, forced.labels)
    assert auto.n_attractors == 3  # the three cube roots of unity


def test_attractor_set_centroid_is_harvest_order_invariant():
    """A merged attractor's centroid must not depend on ``_att_points`` order.

    The Rust kernel harvests ``_att_points`` from a randomised hash map, so an
    unsorted pool made a merged ≥2-cloud centroid mean wobble ~1 ULP run-to-run
    and diverge from the (id-ordered) Python path.  With deliberately
    order-sensitive clouds the raw dict-order mean differs (0.333 vs 0.0); pooling
    by ascending id makes both orders agree byte-for-byte.  Regression for the
    adversarial-review finding on PR #490.
    """
    from tsdynamics.analysis.basins._common import _CellGrid
    from tsdynamics.analysis.basins.attractors import _AttractorMapper

    class _Dummy:
        is_discrete = False

    a = [np.array([1e16, 0.0])]  # id 3
    b = [np.array([-1e16, 0.0]), np.array([1.0, 0.0])]  # id 7 — order-sensitive
    merge = {3: 1, 7: 1}  # both merge into canonical id 1

    def center(order: list[int]) -> np.ndarray:
        m = _AttractorMapper(_Dummy(), _CellGrid([-2, -2], [2, 2], (2, 2)))
        m._att_points = {k: {3: a, 7: b}[k] for k in order}
        m._att_cells = {0: 3, 1: 7}
        return m.attractor_set(diverged=0, seeds=1, merge=merge)[1].center

    np.testing.assert_array_equal(center([3, 7]), center([7, 3]))
