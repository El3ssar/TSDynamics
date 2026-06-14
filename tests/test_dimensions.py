r"""
Tests for the fractal-dimension toolkit (stream **A-DIM**).

The headline acceptance is literature-validated: the Grassberger--Procaccia
correlation dimension of the Lorenz attractor is :math:`D_2 \approx 2.05`
(Grassberger & Procaccia 1983).  The Lorenz attractor is generated here with
``scipy`` (independent of the v2 compile backend / the Rust engine streams), so
these tests stay in the fast tier and exercise only the estimators.

The analytic point sets — uniform line / square / cube and the middle-thirds
Cantor set (:math:`D_0 = \log 2/\log 3`) — pin the estimators to exact known
dimensions where finite-sample bias is smallest.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import tsdynamics as ts
from tsdynamics import registry
from tsdynamics.analysis import dimensions as dim
from tsdynamics.analysis.dimensions._scaling import fit_scaling_region, local_slopes

# ── data generators ─────────────────────────────────────────────────────────────


def _lorenz_points(n=8000, dt=0.02, transient=1500):
    """Points on the Lorenz attractor (classic sigma=10, rho=28, beta=8/3)."""

    def rhs(_t, u):
        x, y, z = u
        return [10.0 * (y - x), x * (28.0 - z) - y, x * y - (8.0 / 3.0) * z]

    t_end = (n + transient) * dt
    t_eval = np.arange(0.0, t_end, dt)
    sol = solve_ivp(
        rhs, (0.0, t_end), [1.0, 1.0, 1.0], t_eval=t_eval, method="DOP853", rtol=1e-9, atol=1e-9
    )
    return sol.y.T[transient : transient + n]


def _cantor_points(n=20000, depth=14, seed=3):
    """Middle-thirds Cantor set; D_0 = log2/log3 ≈ 0.6309."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    scale = 1.0
    for _ in range(depth):
        scale /= 3.0
        x += (2.0 * scale) * rng.integers(0, 2, size=n)
    return x


@pytest.fixture(scope="module")
def lorenz():
    return _lorenz_points()


@pytest.fixture(scope="module")
def uniform_sets():
    rng = np.random.default_rng(0)
    return {
        "line": np.c_[rng.uniform(0, 1, 4000), np.zeros(4000)],
        "square": rng.uniform(0, 1, (5000, 2)),
        "cube": rng.uniform(0, 1, (8000, 3)),
    }


# ── literature acceptance: Lorenz D_2 ≈ 2.05 ────────────────────────────────────


def test_lorenz_correlation_dimension(lorenz):
    d = dim.correlation_dimension(lorenz, theiler_window=50)
    assert d.kind == "correlation" and d.q == 2.0
    assert abs(float(d) - 2.05) < 0.12, f"Lorenz D2 = {float(d):.3f}, expected ~2.05"


def test_lorenz_fixed_mass_dimension(lorenz):
    d = dim.fixed_mass_dimension(lorenz, theiler_window=50)
    assert 1.9 < float(d) < 2.2, f"Lorenz fixed-mass D = {float(d):.3f}"


def test_lorenz_generalized_d2(lorenz):
    d = dim.generalized_dimension(lorenz, q=2.0)
    assert 1.85 < float(d) < 2.15, f"Lorenz box D2 = {float(d):.3f}"


# ── analytic uniform sets: integer dimensions ───────────────────────────────────


@pytest.mark.parametrize("name,expected", [("line", 1.0), ("square", 2.0)])
def test_correlation_dimension_uniform(uniform_sets, name, expected):
    d = dim.correlation_dimension(uniform_sets[name])
    assert abs(float(d) - expected) < 0.12, f"{name}: D2 = {float(d):.3f}, expected {expected}"


@pytest.mark.parametrize("name,expected", [("line", 1.0), ("square", 2.0)])
def test_box_counting_uniform(uniform_sets, name, expected):
    d = dim.box_counting_dimension(uniform_sets[name])
    assert abs(float(d) - expected) < 0.12, f"{name}: D0 = {float(d):.3f}, expected {expected}"


@pytest.mark.parametrize("name,expected", [("line", 1.0), ("square", 2.0)])
def test_fixed_mass_uniform(uniform_sets, name, expected):
    d = dim.fixed_mass_dimension(uniform_sets[name])
    assert abs(float(d) - expected) < 0.15, f"{name}: D = {float(d):.3f}, expected {expected}"


def test_cantor_box_counting():
    expected = np.log(2) / np.log(3)  # 0.6309
    d = dim.box_counting_dimension(_cantor_points(), n_scales=22)
    assert abs(float(d) - expected) < 0.05, f"Cantor D0 = {float(d):.4f}, expected {expected:.4f}"


# ── generalized spectrum ────────────────────────────────────────────────────────


def test_dimension_spectrum_monofractal(uniform_sets):
    spec = dim.dimension_spectrum(uniform_sets["square"], qs=[0, 1, 2, 3, 4])
    dims = [float(spec[q]) for q in (0.0, 1.0, 2.0, 3.0, 4.0)]
    # A uniform (monofractal) set has a near-flat D_q ≈ 2 spectrum.
    assert all(1.8 < v < 2.1 for v in dims), dims
    # D_q is theoretically non-increasing in q (allow small finite-sample slack).
    assert all(dims[i + 1] <= dims[i] + 0.1 for i in range(len(dims) - 1)), dims


def test_dimension_spectrum_reuses_occupancy(uniform_sets):
    # The spectrum and a single generalized_dimension must agree for one q.
    spec = dim.dimension_spectrum(uniform_sets["square"], qs=[2.0])
    one = dim.generalized_dimension(uniform_sets["square"], q=2.0)
    assert float(spec[2.0]) == pytest.approx(float(one))


# ── correlation sum properties ──────────────────────────────────────────────────


def test_correlation_sum_is_a_cdf(uniform_sets):
    radii, c = dim.correlation_sum(uniform_sets["square"])
    assert radii.shape == c.shape
    assert np.all(c >= 0.0) and np.all(c <= 1.0)
    # C(r) is non-decreasing in r.
    order = np.argsort(radii)
    assert np.all(np.diff(c[order]) >= -1e-12)


def test_theiler_window_reduces_pair_count(uniform_sets):
    big_r = np.array([1e9])  # encloses every pair
    _, c0 = dim.correlation_sum(uniform_sets["square"], radii=big_r, theiler_window=0)
    _, cw = dim.correlation_sum(uniform_sets["square"], radii=big_r, theiler_window=20)
    # Both normalise to 1 at r→∞ (all *valid* pairs), confirming the normalisation
    # accounts for the excluded near-diagonal pairs.
    assert c0[0] == pytest.approx(1.0)
    assert cw[0] == pytest.approx(1.0)


def test_chebyshev_metric_runs(uniform_sets):
    d = dim.correlation_dimension(uniform_sets["square"], metric="chebyshev")
    assert abs(float(d) - 2.0) < 0.15


# ── input handling ──────────────────────────────────────────────────────────────


def test_accepts_trajectory_array_and_series_equivalently():
    rng = np.random.default_rng(7)
    pts = rng.uniform(0, 1, (3000, 2))
    traj = ts.Trajectory(np.arange(3000), pts, system=None)
    d_arr = float(dim.correlation_dimension(pts))
    d_traj = float(dim.correlation_dimension(traj))
    assert d_arr == pytest.approx(d_traj)


def test_one_dimensional_series_is_a_column(uniform_sets):
    series = uniform_sets["line"][:, 0]  # 1-D
    d = dim.correlation_dimension(series)
    assert abs(float(d) - 1.0) < 0.12


# ── DimensionResult API ─────────────────────────────────────────────────────────


def test_dimension_result_api(uniform_sets):
    d = dim.correlation_dimension(uniform_sets["square"])
    assert isinstance(float(d), float)
    assert float(d) == d.dimension
    assert d.local_slopes.shape == d.x.shape
    lo, hi = d.scaling_window
    assert lo < hi
    assert "correlation" in repr(d)
    a, b = d.fit_slice
    assert 0 <= a <= b < d.x.size


# ── registry self-registration ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,fn",
    [
        ("correlation_dimension", dim.correlation_dimension),
        ("generalized_dimension", dim.generalized_dimension),
        ("box_counting_dimension", dim.box_counting_dimension),
        ("information_dimension", dim.information_dimension),
        ("fixed_mass_dimension", dim.fixed_mass_dimension),
    ],
)
def test_registered_in_analyses(name, fn):
    assert name in registry.analyses
    assert registry.analyses.get(name) is fn
    assert registry.analyses.entry(name).metadata["needs"] == "trajectory"


def test_public_api_identity():
    assert ts.correlation_dimension is dim.correlation_dimension
    assert ts.fixed_mass_dimension is dim.fixed_mass_dimension
    assert ts.DimensionResult is dim.DimensionResult
    for name in ("correlation_dimension", "fixed_mass_dimension", "DimensionResult"):
        assert name in ts.__all__


# ── scaling-region fit ──────────────────────────────────────────────────────────


def test_fit_scaling_region_recovers_slope():
    x = np.linspace(0.0, 1.0, 30)
    rng = np.random.default_rng(0)
    y = 2.0 * x + 0.5 + rng.normal(0, 1e-3, x.size)
    fit = fit_scaling_region(x, y)
    assert fit.slope == pytest.approx(2.0, abs=0.02)
    assert fit.npts >= 5


def test_fit_scaling_region_picks_linear_middle():
    # A single straight region (slope 2) for |x| <= 1.5, with the curve bending
    # away at both extremes — the shape every real dimension curve has (noise at
    # small scales, saturation at large).  The fitter must lock onto the middle
    # and reject the curved ends.
    x = np.linspace(-3.0, 3.0, 121)
    bend = np.where(np.abs(x) > 1.5, 0.6 * np.sign(x) * (np.abs(x) - 1.5) ** 2, 0.0)
    y = 2.0 * x + bend
    fit = fit_scaling_region(x, y, min_window=6)
    assert fit.slope == pytest.approx(2.0, abs=0.05)
    assert x[fit.lo] >= -1.6 and x[fit.hi] <= 1.6  # window sits inside the straight middle
    assert fit.npts >= 20


def test_fit_scaling_region_too_few_points():
    with pytest.raises(ValueError, match="min_window"):
        fit_scaling_region(np.arange(3.0), np.arange(3.0), min_window=5)


def test_local_slopes_length():
    x = np.linspace(0, 1, 10)
    assert local_slopes(x, 2 * x).shape == x.shape
    assert local_slopes(x, 2 * x) == pytest.approx(2.0)


# ── error handling ──────────────────────────────────────────────────────────────


def test_unknown_metric_raises():
    with pytest.raises(ValueError, match="metric"):
        dim.correlation_dimension(np.random.default_rng(0).uniform(0, 1, (100, 2)), metric="cosine")


def test_sub_metric_exponent_raises():
    # p < 1 is not a metric; reject it consistently rather than letting one
    # estimator silently compute a quasi-norm and another crash inside scipy.
    pts = np.random.default_rng(0).uniform(0, 1, (200, 2))
    with pytest.raises(ValueError, match="Minkowski exponent must be >= 1"):
        dim.correlation_dimension(pts, metric=0.5)
    with pytest.raises(ValueError, match="Minkowski exponent must be >= 1"):
        dim.fixed_mass_dimension(pts, metric=0.5)


def test_negative_theiler_raises(uniform_sets):
    with pytest.raises(ValueError, match="theiler_window"):
        dim.correlation_sum(uniform_sets["square"], theiler_window=-1)


def test_too_few_points_raises():
    with pytest.raises(ValueError, match="at least two points"):
        dim.correlation_dimension(np.array([[0.0, 0.0]]))


def test_non_finite_raises():
    bad = np.array([[0.0, 0.0], [np.nan, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="non-finite"):
        dim.correlation_dimension(bad)
