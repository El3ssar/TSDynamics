r"""
Tests for the attractors & basins toolkit (stream **A-BASIN**, the parity moat).

Two layers:

* **Fast tier** — the basin quantifiers (entropy, uncertainty exponent, Wada,
  resilience, tipping) run on hand-built label grids with analytically known
  answers, plus the structural contract (result objects, self-registration,
  re-exports, input validation).  No integration, so these stay in the fast tier.
* **Slow tier** — literature-validated end-to-end runs on multistable systems
  defined locally here (they are not in the built-in catalogue): the **Newton
  ``z**3 = 1`` map** (three Wada basins, each ``1/3`` by C3 symmetry), the
  **two-well Duffing oscillator** (two basins, ``1/2`` each by reflection
  symmetry), the **magnetic pendulum** (three magnets, fractal basins) and the
  **Hénon** escape basin.  Basin-fraction error stays well under 1% on the Newton
  and Duffing systems.
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics import registry
from tsdynamics.analysis import basins as bas
from tsdynamics.analysis.basins.attractors import Attractor, AttractorSet
from tsdynamics.analysis.basins.basins import BasinsResult
from tsdynamics.analysis.basins.continuation import ContinuationResult
from tsdynamics.data import Ball, Box, Grid
from tsdynamics.utils import staticjit

LN2 = float(np.log(2.0))


# ===========================================================================
# Local validation systems (not in the built-in catalogue)
# ===========================================================================


class NewtonMap(ts.DiscreteMap):
    """Newton's method for ``z**3 - 1 = 0`` — three roots, Wada basins."""

    params: dict = {}
    dim = 2
    variables = ("re", "im")
    _jacobian_fd_check = False

    @staticjit
    def _step(X):
        # explicit complex multiplication (not z**3) so numba avoids transcendental
        # cpow — keeps the chaotic Julia-set boundary as reproducible as the
        # architecture allows.
        z = complex(X[0], X[1])
        z2 = z * z
        z3 = z2 * z
        z = (2.0 * z3 + 1.0) / (3.0 * z2)
        return (z.real, z.imag)

    @staticjit
    def _jacobian(X):
        return ((0.0, 0.0), (0.0, 0.0))


class CubicMap(ts.DiscreteMap):
    """``x -> a*x - x**3`` — two stable fixed points at ``±sqrt(a-1)`` for 1<a<2."""

    params = {"a": 1.5}
    dim = 1

    @staticjit
    def _step(X, a):
        return (a * X[0] - X[0] ** 3,)

    @staticjit
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
# Fast tier — structural contract
# ===========================================================================

_PUBLIC_FUNCS = [
    "find_attractors",
    "basins_of_attraction",
    "basin_fractions",
    "continuation",
    "tipping_points",
    "basin_entropy",
    "uncertainty_exponent",
    "wada_property",
    "resilience",
]
_PUBLIC_CLASSES = [
    "Attractor",
    "AttractorSet",
    "BasinsResult",
    "BasinFractions",
    "BasinEntropy",
    "UncertaintyExponent",
    "WadaResult",
    "ContinuationResult",
]


@pytest.mark.parametrize("name", _PUBLIC_FUNCS + _PUBLIC_CLASSES)
def test_public_api_reexported(name):
    assert getattr(ts, name) is getattr(bas, name)
    assert name in ts.analysis.__all__
    assert name in ts.__all__


@pytest.mark.parametrize("name", _PUBLIC_FUNCS)
def test_functions_self_register(name):
    assert name in registry.analyses
    assert registry.analyses.get(name) is getattr(bas, name)
    assert registry.analyses.entry(name).metadata["family"] == "basins"


# ===========================================================================
# Fast tier — basin entropy (Daza et al., 2016)
# ===========================================================================


def test_basin_entropy_monochromatic_is_zero():
    labels = np.ones((40, 40), dtype=int)
    be = bas.basin_entropy(labels, box_size=5)
    assert be.sb == pytest.approx(0.0)
    assert np.isnan(be.sbb)  # no boundary box
    assert be.fractal_boundary is False


def test_basin_entropy_half_split_box_is_ln2():
    # one 2x2 box, half colour 1 half colour 2 -> Gibbs entropy ln 2 (natural log).
    labels = np.array([[1, 1], [2, 2]])
    be = bas.basin_entropy(labels, box_size=2)
    assert be.sb == pytest.approx(LN2, abs=1e-9)
    assert be.sbb == pytest.approx(LN2, abs=1e-9)


def test_basin_entropy_smooth_boundary_not_fractal():
    # vertical boundary off the box edges -> boundary boxes hold only 2 colours,
    # so Sbb <= ln 2 and the fractal criterion is not met.
    labels = np.ones((40, 40), dtype=int)
    labels[17:, :] = 2
    be = bas.basin_entropy(labels, box_size=5)
    assert be.n_boundary_boxes > 0
    assert be.sbb < LN2
    assert be.fractal_boundary is False


def test_basin_entropy_interleaved_is_fractal():
    rng = np.random.default_rng(0)
    labels = rng.integers(1, 4, size=(60, 60))  # three colours, maximally mixed
    be = bas.basin_entropy(labels, box_size=5)
    assert be.sbb > LN2
    assert be.fractal_boundary is True


def test_basin_entropy_rejects_bad_box_size():
    with pytest.raises(ValueError):
        bas.basin_entropy(np.ones((4, 4), dtype=int), box_size=0)


# ===========================================================================
# Fast tier — uncertainty exponent (Grebogi et al., 1983)
# ===========================================================================


def test_uncertainty_exponent_smooth_boundary_is_one():
    # a single straight boundary in 2-D: D0 = 1, so alpha = D - D0 = 1.
    labels = np.ones((200, 200), dtype=int)
    labels[:, 100:] = 2
    ue = bas.uncertainty_exponent(labels, radii=(1, 2, 3, 4, 5, 6))
    assert ue.alpha == pytest.approx(1.0, abs=0.05)
    assert ue.boundary_dimension == pytest.approx(1.0, abs=0.05)
    assert ue.state_dimension == 2


def test_uncertainty_exponent_needs_two_radii():
    with pytest.raises(ValueError):
        bas.uncertainty_exponent(np.ones((10, 10), dtype=int), radii=(1,))


# ===========================================================================
# Fast tier — Wada (Daza et al., 2015)
# ===========================================================================


def test_wada_two_basins_is_not_wada():
    labels = np.ones((40, 40), dtype=int)
    labels[:, 20:] = 2
    wr = bas.wada_property(labels)
    assert wr.is_wada is False
    assert wr.n_basins == 2


def test_wada_three_sectors_is_not_wada():
    # three angular sectors: adjacent sectors share two-colour boundaries; all
    # three meet only at the centre, so the boundary is *not* Wada.
    n = 120
    xs = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(xs, xs, indexing="ij")
    ang = np.arctan2(yy, xx)
    labels = (np.floor((ang + np.pi) / (2 * np.pi / 3)).astype(int) % 3) + 1
    wr = bas.wada_property(labels, radii=(1, 2, 3))
    assert wr.n_basins == 3
    assert wr.is_wada is False


# ===========================================================================
# Fast tier — resilience (Halekotte & Feudel, 2020)
# ===========================================================================


def test_resilience_distance_to_boundary():
    # left half basin 1, right half basin 2; boundary at x = 0.
    grid = Grid([-1.0, -1.0], [1.0, 1.0], (101, 101))
    xs = np.linspace(-1, 1, 101)
    labels = np.where(xs[:, None] < 0.0, 1, 2) * np.ones((101, 101), dtype=int)
    att1 = Attractor(1, np.array([[-0.5, 0.0]]), cells=1)
    att2 = Attractor(2, np.array([[0.5, 0.0]]), cells=1)
    aset = AttractorSet({1: att1, 2: att2}, diverged=0, seeds=101 * 101)
    res = BasinsResult(labels=labels, grid=grid, attractors=aset)
    # attractor 1 sits at x=-0.5, the boundary is at x=0 -> resilience ~ 0.5.
    assert bas.resilience(res, 1) == pytest.approx(0.5, abs=0.03)
    assert bas.resilience(res, 2) == pytest.approx(0.5, abs=0.03)


def test_resilience_missing_attractor_raises():
    grid = Grid([0.0, 0.0], [1.0, 1.0], (10, 10))
    aset = AttractorSet({}, diverged=0, seeds=100)
    res = BasinsResult(labels=np.ones((10, 10), dtype=int), grid=grid, attractors=aset)
    with pytest.raises(ValueError):
        bas.resilience(res, 5)


# ===========================================================================
# Fast tier — tipping points on a hand-built continuation
# ===========================================================================


def _toy_continuation() -> ContinuationResult:
    values = np.array([0.0, 1.0, 2.0, 3.0])
    fractions = {
        1: np.array([0.5, 0.5, 0.0, 0.0]),  # disappears at value 2.0
        2: np.array([0.5, 0.5, 1.0, 1.0]),  # grows
        3: np.array([0.0, 0.0, 0.0, 0.4]),  # appears at value 3.0
    }
    return ContinuationResult(
        param="mu",
        values=values,
        fractions=fractions,
        attractors=[{} for _ in values],
        diverged=np.zeros(4),
    )


def test_tipping_points_detects_appear_and_disappear():
    events = bas.tipping_points(_toy_continuation())
    kinds = {(e["attractor"], e["kind"]) for e in events}
    assert (1, "disappear") in kinds
    assert (3, "appear") in kinds
    # attractor 2 only grows; no crossing of zero.
    assert all(e["attractor"] != 2 for e in events)
    disappear = next(e for e in events if e["attractor"] == 1)
    assert disappear["value"] == 2.0


def test_continuation_result_tipping_method_matches_function():
    cont = _toy_continuation()
    assert cont.tipping_points() == bas.tipping_points(cont)
    assert cont.ids == [1, 2, 3]


# ===========================================================================
# Fast tier — coercion + attractor matching
# ===========================================================================


def test_as_label_array_rejects_non_integer():
    from tsdynamics.analysis.basins._common import _as_label_array

    with pytest.raises(ValueError):
        _as_label_array(np.array([0.3, 1.7]))


def test_attractor_set_match_picks_nearest():
    a1 = Attractor(1, np.array([[0.0, 0.0]]), cells=1)
    a2 = Attractor(2, np.array([[5.0, 5.0]]), cells=1)
    aset = AttractorSet({1: a1, 2: a2}, diverged=0, seeds=10)
    assert aset.match([0.1, -0.1]) == 1
    assert aset.match([4.9, 5.2]) == 2


def test_find_attractors_rejects_unsupported_system():
    mg = ts.MackeyGlass()  # a DDE
    with pytest.raises(TypeError):
        bas.find_attractors(mg, Box([0.0], [2.0]))


# ===========================================================================
# Slow tier — literature-validated end-to-end runs
# ===========================================================================


@pytest.mark.slow
def test_newton_map_thirds_basin_fractions():
    """Newton ``z**3 = 1``: three Wada basins, each 1/3 by C3 symmetry."""
    nm = NewtonMap()
    bf = bas.basin_fractions(
        nm,
        Ball([0.0, 0.0], 1.3),
        n=30000,
        resolution=300,
        seed=0,
        consecutive_recurrences=8,
        attractor_locate_steps=5,
        max_steps=200,
    )
    assert len(bf.fractions) == 3
    assert bf.diverged < 0.01
    for frac in bf.fractions.values():
        # basin-fraction error < 0.01 (one percentage point); the realised errors
        # at this seed/N are ~0.005, comfortably inside the < 1 % acceptance bar.
        assert frac == pytest.approx(1.0 / 3.0, abs=0.01)
    # the three attractors sit on the cube roots of unity.
    centers = np.array(sorted(bf.attractors.centers.tolist()))
    expected = np.array([[-0.5, -0.8660254], [-0.5, 0.8660254], [1.0, 0.0]])
    assert np.allclose(centers, expected, atol=0.05)


@pytest.mark.slow
def test_newton_basin_image_is_wada_and_fractal():
    nm = NewtonMap()
    res = bas.basins_of_attraction(
        nm,
        Grid([-1.0, -1.0], [1.0, 1.0], (120, 120)),
        consecutive_recurrences=8,
        attractor_locate_steps=5,
        max_steps=200,
    )
    assert res.n_attractors == 3
    wada = bas.wada_property(res, radii=(1, 2, 3, 4, 5, 6))
    assert wada.n_basins == 3
    # Newton z^3 basins are Wada: as the neighbourhood grows the large majority of
    # boundary cells come to see all three roots, where a non-Wada boundary (e.g.
    # the 3-sector control above) stays near zero.  The exact fraction at the
    # fractal Julia-set boundary is floating-point sensitive and differs across CPU
    # architectures, so assert the robust climb to a high value, not a tight bound.
    assert wada.fractions[-1] > wada.fractions[0]  # climbs with radius
    assert wada.fractions[-1] > 0.5  # majority see all 3, vs the ~0.1 of a non-Wada boundary
    be = bas.basin_entropy(res, box_size=5)
    assert be.fractal_boundary is True  # Sbb > ln 2 (box-level mixing, robust)
    ue = bas.uncertainty_exponent(res)
    assert 0.0 < ue.boundary_dimension < 2.0  # a fractal boundary in the plane


@pytest.mark.slow
def test_duffing_two_well_half_basins():
    """Damped two-well Duffing: two basins, 1/2 each by reflection symmetry."""
    d = DuffingTwoWell()
    res = bas.basins_of_attraction(
        d,
        Grid([-2.0, -2.0], [2.0, 2.0], (60, 60)),
        dt=0.4,
        max_steps=400,
        consecutive_recurrences=20,
        attractor_locate_steps=10,
    )
    assert res.n_attractors == 2
    assert res.diverged_fraction == pytest.approx(0.0, abs=0.01)
    for frac in res.fractions.values():
        # basin-fraction error < 0.01 (one percentage point); reflection symmetry
        # makes the realised error ~0.001, well inside the < 1 % bar.
        assert frac == pytest.approx(0.5, abs=0.01)
    centers = np.array(sorted(res.attractors.centers.tolist()))
    assert np.allclose(centers[:, 0], [-1.0, 1.0], atol=0.05)


@pytest.mark.slow
def test_duffing_resilience_to_boundary():
    d = DuffingTwoWell()
    res = bas.basins_of_attraction(
        d, Grid([-2.0, -2.0], [2.0, 2.0], (60, 60)), dt=0.4, max_steps=400
    )
    # the saddle separating the wells sits at x=0; an attractor near x=±1 is ~1
    # away from the boundary.
    att_id = res.attractors.ids[0]
    assert bas.resilience(res, att_id) > 0.3


@pytest.mark.slow
def test_henon_escape_basin():
    """Hénon: one bounded strange attractor and an escape basin."""
    h = ts.Henon()
    res = bas.basins_of_attraction(h, Grid([-2.0, -2.0], [2.0, 2.0], (80, 80)), max_steps=3000)
    assert res.n_attractors >= 1
    # a sizeable fraction escapes the square and a sizeable fraction is captured.
    assert 0.2 < res.diverged_fraction < 0.8


@pytest.mark.slow
def test_magnetic_pendulum_three_attractors():
    """Magnetic pendulum: three magnet attractors with fractal basins."""
    mp = MagneticPendulum()
    ats = bas.find_attractors(
        mp,
        Box([-1.3, -1.3, -2.5, -2.5], [1.3, 1.3, 2.5, 2.5]),
        resolution=(60, 60, 40, 40),
        n_seeds=400,
        seed=0,
        dt=0.5,
        max_steps=800,
        consecutive_recurrences=25,
        attractor_locate_steps=15,
    )
    # the three magnets are recovered (a tiny saddle-passage artifact may appear).
    assert len(ats) >= 3
    centers = ats.centers[:, :2]
    for mag in _MAGNETS:
        d = np.linalg.norm(centers - np.array(mag), axis=1).min()
        assert d < 0.1, f"no attractor near magnet {mag}"


@pytest.mark.slow
def test_magnetic_pendulum_fractal_basin_image():
    mp = MagneticPendulum()
    slice_grid = Grid([-1.3, -1.3, 0.0, 0.0], [1.3, 1.3, 0.0, 0.0], (35, 35, 1, 1))
    rbox = Box([-1.6, -1.6, -3.0, -3.0], [1.6, 1.6, 3.0, 3.0])
    res = bas.basins_of_attraction(
        mp,
        slice_grid,
        recurrence=rbox,
        recurrence_resolution=(64, 64, 48, 48),
        dt=0.5,
        max_steps=1000,
        consecutive_recurrences=20,
        attractor_locate_steps=12,
    )
    assert res.n_attractors >= 3
    # the three magnet basins dominate the slice.
    top3 = sum(sorted((v for k, v in res.fractions.items() if k >= 1), reverse=True)[:3])
    assert top3 > 0.95
    be = bas.basin_entropy(res, box_size=4)
    assert be.fractal_boundary is True  # Sbb > ln 2


@pytest.mark.slow
def test_continuation_tracks_two_basins():
    """Continuation of the cubic map keeps two consistent attractor ids."""
    cont = bas.continuation(
        CubicMap(),
        "a",
        [1.3, 1.4, 1.5, 1.6, 1.7],
        Box([-2.0], [2.0]),
        n=1500,
        resolution=300,
        seed=0,
        min_fraction=0.05,  # drop tiny spurious sets near the unstable origin
        consecutive_recurrences=12,
        attractor_locate_steps=6,
        max_steps=400,
    )
    # two attractors (±sqrt(a-1)) persist across the whole sweep with stable ids.
    assert len(cont.ids) == 2
    for gid in cont.ids:
        assert np.all(np.isfinite(cont.fractions[gid]))  # present at every value
        assert np.all(cont.fractions[gid] > 0.2)
    # no attractor disappears over this range.
    assert bas.tipping_points(cont) == []
