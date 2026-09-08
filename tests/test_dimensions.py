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

import warnings

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
    d = dim.correlation_dimension(lorenz, theiler=50)
    assert d.kind == "correlation" and d.q == 2.0
    assert abs(float(d) - 2.05) < 0.12, f"Lorenz D2 = {float(d):.3f}, expected ~2.05"


def test_lorenz_fixed_mass_dimension(lorenz):
    d = dim.fixed_mass_dimension(lorenz, theiler=50)
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
    _, c0 = dim.correlation_sum(uniform_sets["square"], radii=big_r, theiler=0)
    _, cw = dim.correlation_sum(uniform_sets["square"], radii=big_r, theiler=20)
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
    # v4 (WS-NAMESPACE): the curated top-level ``__all__`` carries only headline
    # names; demoted analysis names stay reachable as flat re-exports.
    for name in ("correlation_dimension", "fixed_mass_dimension", "DimensionResult"):
        assert hasattr(ts, name)


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


def test_fit_scaling_region_min_window_2_uses_more_than_two_points():
    """A noisy straight line with ``min_window=2`` must not collapse to a 2-point fit.

    A two-point window fits any pair exactly (residual sigma = 0), so if such a
    window were allowed to set the residual threshold it would drop to 0 and only
    exactly-collinear/two-point windows would survive on a noisy curve.  The fix
    sets the threshold from windows of >= 3 points; here the whole noisy line is
    linear, so the fitter should keep a long window and recover the slope — never
    a degenerate 2-point window.  Fails on the pre-fix code.
    """
    x = np.linspace(0.0, 1.0, 40)
    rng = np.random.default_rng(0)
    y = 2.0 * x + 0.5 + rng.normal(0.0, 5e-2, x.size)
    fit = fit_scaling_region(x, y, min_window=2)
    # Pre-fix this collapsed to a 2-point window (sigma=0 forced the threshold to
    # 0); post-fix the threshold comes from >=3-point windows, so a wider window is
    # kept.  npts > 2 is the precise pre/post-fix discriminator (the estimator
    # deliberately selects the cleanest sub-window, so its slope on noisy data is
    # not expected to equal the global slope — use the default min_window for that).
    assert fit.npts > 2, f"collapsed to a {fit.npts}-point window"


def test_fit_scaling_region_min_window_2_degenerate_two_points():
    """With exactly two points and ``min_window=2`` the only window is returned."""
    fit = fit_scaling_region(np.array([0.0, 1.0]), np.array([0.5, 2.5]), min_window=2)
    assert fit.npts == 2
    assert fit.slope == pytest.approx(2.0)


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
    with pytest.raises(ValueError, match="theiler"):
        dim.correlation_sum(uniform_sets["square"], theiler=-1)


def test_too_few_points_raises():
    with pytest.raises(ValueError, match="at least two points"):
        dim.correlation_dimension(np.array([[0.0, 0.0]]))


def test_non_finite_raises():
    bad = np.array([[0.0, 0.0], [np.nan, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="non-finite"):
        dim.correlation_dimension(bad)


# ══ v6 remediation ══════════════════════════════════════════════════════════════
#
# Three defects were fixed together, because they interact through the same
# scaling-region fit:
#
#   1. box counting returned D_0 ~ 1.78 on the Lorenz attractor and produced an
#      *increasing* D_q spectrum, which is impossible for any measure;
#   2. ``fit_scaling_region`` latched onto local dips, so a denser scale grid
#      made the estimate monotonically worse while the reported stderr shrank;
#   3. the Theiler window defaulted to 0, silently inflating every dimension read
#      off a densely sampled flow.
#
# Everything below is validated against a value this library does not compute:
# an exact analytic dimension, or a published one.


def _sierpinski_points(n=20000, seed=1):
    """Chaos game on a triangle; D_q = log3/log2 = 1.58496 for every q."""
    rng = np.random.default_rng(seed)
    v = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
    p = np.zeros(2)
    out = np.empty((n, 2))
    for i in range(n):
        p = 0.5 * (p + v[rng.integers(0, 3)])
        out[i] = p
    return out


def _henon_points(n=20000, transient=1000, a=1.4, b=0.3):
    """The Henon attractor; D_0 ~ 1.26, D_2 = 1.220 +- 0.005 (Grassberger 1983)."""
    x, y = 0.1, 0.1
    out = np.empty((n + transient, 2))
    for i in range(n + transient):
        x, y = 1.0 - a * x * x + y, b * x
        out[i] = (x, y)
    return out[transient:]


# ── 1. box counting against analytic / published dimensions ─────────────────────


@pytest.mark.parametrize(
    "name,points_fn,expected,atol",
    [
        # Exact, analytic self-similar dimensions.
        ("cantor", lambda: _cantor_points(20000, depth=14), np.log(2) / np.log(3), 0.05),
        ("sierpinski", _sierpinski_points, np.log(3) / np.log(2), 0.05),
        # Published attractor dimension.
        ("henon", _henon_points, 1.26, 0.06),
    ],
)
def test_box_counting_matches_known_dimension(name, points_fn, expected, atol):
    """D_0 of sets whose dimension is known exactly or from the literature."""
    d = dim.box_counting_dimension(points_fn())
    assert abs(float(d) - expected) < atol, f"{name}: D0 = {float(d):.4f}, expected {expected:.4f}"


@pytest.mark.parametrize("d_topo,atol", [(1, 0.05), (2, 0.05), (3, 0.25)])
def test_box_counting_recovers_a_uniform_cube(d_topo, atol):
    """A uniform d-cube has D_0 = d exactly.

    The tolerances are not uniform because the achievable accuracy is not: the
    informative band is ``[min_resolution, (N/min_occupancy) ** (1/d)]`` in linear
    resolution, so it narrows geometrically with dimension.  At N = 30 000 that is
    3.0 decades for a line, 1.5 for a square and **0.44** for a cube — box
    counting a 3-D set is at the edge of feasibility at any sample size a test can
    afford, and 2.79 is what it honestly delivers (see the companion convergence
    test).  Pre-fix the 1- and 2-cubes were fine and the 3-cube came back at 2.76
    for a quite different reason: the small-box cut admitted scales down to ~1.2
    points per box, deep inside the saturation plateau.
    """
    rng = np.random.default_rng(11)
    pts = rng.uniform(0.0, 1.0, (30000, d_topo))
    d = dim.box_counting_dimension(pts)
    assert abs(float(d) - d_topo) < atol, f"{d_topo}-cube: D0 = {float(d):.4f}"


def test_box_counting_of_a_uniform_cube_converges_with_sample_size():
    """D_0 of a 3-cube must approach 3 as N grows — the estimator is consistent.

    A single-N check cannot separate "biased low" from "not enough data", so the
    *trend* is what is asserted.  Convergence is slow by nature (the usable band
    grows only as ``log N / d``), so the claim is monotone improvement plus a
    measured floor, not a rate: 0.28 -> 0.21 -> 0.16 over 5k -> 30k -> 200k.

    An independent from-scratch box count over the *whole* band of the same points
    reports ~2.95 rather than ~2.84, but that number is not the better one: at the
    coarse end of a 0.44-decade band the per-axis bin count is a small integer, so
    N(eps) jumps by factors like ``(7/6)**3``, and including that jump inflates the
    slope by luck.  Excluding it is what ``min_resolution`` is for.
    """
    rng = np.random.default_rng(4)
    errs = []
    for n in (5000, 30000, 200000):
        d = dim.generalized_dimension(rng.uniform(0.0, 1.0, (n, 3)), q=0.0)
        errs.append(abs(float(d) - 3.0))
    assert errs[0] > errs[1] > errs[2], f"3-cube D0 errors {errs} did not improve with N"
    assert errs[-1] < 0.2, f"3-cube D0 error at 200k points = {errs[-1]:.4f}"


# ── 2. the non-increasing-D_q guard ─────────────────────────────────────────────


def test_lorenz_box_counting_is_reported_as_unresolved(lorenz):
    r"""Box counting cannot resolve D_0 of Lorenz from 8000 points — and says so.

    Measured independently (a from-scratch box count over the whole informative
    range of these very points): the local slope of :math:`\log N(\epsilon)` never
    exceeds ~1.85 and plateaus near 1.75, while the q = 2 ordinate plateaus near
    2.0.  So the computed spectrum *rises* with q — impossible for any measure,
    since D_q is non-increasing in q — which means the D_0 end has not converged.
    The estimate must therefore be reported as a failure, not returned as 1.75.
    """
    from tsdynamics.errors import ConvergenceError

    with pytest.raises(ConvergenceError, match="increases with q"):
        dim.box_counting_dimension(lorenz)


def test_unresolved_spectrum_can_be_downgraded_to_a_warning(lorenz):
    """``strict=False`` returns the (documented-unreliable) number with a warning."""
    from tsdynamics.analysis.dimensions.generalized import NonMonotoneSpectrumWarning

    with pytest.warns(NonMonotoneSpectrumWarning, match="increases with q"):
        d = dim.box_counting_dimension(lorenz, strict=False)
    assert float(d) < 1.9  # the under-resolved value the warning is about


def test_monotone_spectra_pass_the_guard():
    """A resolved monofractal spectrum is returned untouched (no false positive).

    The guard must not fire on ordinary valid input: the Sierpinski gasket's true
    spectrum is flat, so finite-sample scatter is the only thing that could trip
    it.
    """
    spec = dim.dimension_spectrum(_sierpinski_points(), qs=[0, 1, 2, 3, 4, 5])
    dims = [float(spec[q]) for q in sorted(spec)]
    # No exception was raised: that is the assertion this test exists for.
    # The values drift slightly downward with q (1.577 -> 1.491 here).  That is
    # finite-sample bias, not multifractality, and it is not this estimator's:
    # an independent from-scratch box count of the same points drifts the same
    # way (1.587 at q=0 to 1.547 at q=5), because high orders are carried by the
    # few densest boxes.  A downward drift is legal for D_q, so the guard is
    # silent; only a *rise* would be impossible.
    assert all(abs(v - np.log(3) / np.log(2)) < 0.10 for v in dims), dims
    assert dims[0] == pytest.approx(np.log(3) / np.log(2), abs=0.03)


# ── 3. scaling-region fit: convergence in the grid density ──────────────────────


def test_correlation_dimension_converges_as_the_radius_grid_densifies():
    """Densifying the scale grid must not move the estimate (audit: it did).

    The log-log curve of a self-similar set is lacunar, so any window-selection
    rule scored purely on residual prefers a short window inside one ripple — and
    the finer the grid, the shorter that window gets.  Pre-fix, sweeping the
    Henon radius grid from 12 to 256 radii moved D_2 by 0.094 (and the Lorenz D_2
    by 0.64, to 1.42) while the reported stderr fell eightfold: the reported
    uncertainty was *anti-correlated* with the true error.

    Post-fix the window is admitted by abscissa **span**, a property of the curve
    rather than of its sampling, so the estimate is grid-independent.
    """
    pts = _henon_points()
    densities = (12, 24, 48, 96, 256)
    results = [dim.correlation_dimension(pts, n_radii=nr, theiler=1) for nr in densities]
    est = [float(r) for r in results]
    spread = max(est) - min(est)
    assert spread < 0.02, f"D2 varies by {spread:.4f} across grid densities {densities}: {est}"
    # ... and it converges to the published value, not merely to itself.
    assert all(abs(e - 1.220) < 0.03 for e in est), est
    # The stderr shrinks with more fitted points (as a fit standard error must)
    # and never claims precision the estimate does not have.
    stderrs = [r.stderr for r in results]
    assert stderrs[-1] < stderrs[0]
    assert all(s > 0.0 for s in stderrs)


def test_lorenz_correlation_dimension_is_grid_independent(lorenz):
    """The same sweep on the flow, where the pre-fix failure was catastrophic.

    Pre-fix: 2.062 at 48 radii, 1.419 at 96 and beyond, with stderr 1.6e-4.
    """
    est = [float(dim.correlation_dimension(lorenz, n_radii=nr, theiler=50)) for nr in (24, 96, 256)]
    assert max(est) - min(est) < 0.05, est
    assert all(abs(e - 2.05) < 0.12 for e in est), est


def test_scaling_window_may_drop_the_ends_of_a_short_curve():
    """The span floor must never clamp to the whole curve.

    The floor is ``min(min_span_frac * total, one decade)``.  Taking the *larger*
    of the two (as the first cut of this fix did) forces the entire range on any
    curve shorter than a decade — which is most real ones — so the fit is dragged
    onto the very ends the selection exists to reject.  Here the curve is 0.8
    decades long with a bent tail; the fit must exclude it.
    """
    x = np.linspace(0.0, 0.8 * np.log(10.0), 40)
    y = 2.0 * x + np.where(x > 1.2, 4.0 * (x - 1.2) ** 2, 0.0)
    fit = fit_scaling_region(x, y)
    assert fit.hi < x.size - 1, "the fit kept the bent tail (span floor clamped to the range)"
    assert fit.slope == pytest.approx(2.0, abs=0.05)


# ── 4. the automatic Theiler window ─────────────────────────────────────────────


def test_auto_theiler_is_one_for_a_decorrelated_point_set():
    """A random cloud and a map orbit need no Theiler correction."""
    from tsdynamics.analysis.dimensions._common import _auto_theiler

    rng = np.random.default_rng(3)
    assert _auto_theiler(rng.uniform(0.0, 1.0, (5000, 2))) == 1
    assert _auto_theiler(_henon_points()) == 1


def test_auto_theiler_finds_a_window_for_a_densely_sampled_flow(lorenz):
    """A flow sampled at dt = 0.02 does need one, and it is found automatically."""
    from tsdynamics.analysis.dimensions._common import _auto_theiler

    w = _auto_theiler(lorenz)
    assert w > 1
    # dt = 0.02, so this is a fraction of a Lorenz orbit — the right order.
    assert w < 100


def test_auto_theiler_declines_to_correct_a_drifting_orbit():
    r"""A non-stationary orbit gets ``w = 1``, and quietly.

    The catalogue's ``Chirikov`` standard map does not wrap its angle, so an
    orbit drifts: :math:`d(k)` climbs for every inspectable lag and no separation
    lag exists.  Reading the window off the profile's own top instead would be
    destructive — measured on this very orbit (an invariant curve, true
    :math:`D_2 = 1`), ``w = 50`` gives 1.64, ``w = 150`` gives 2.47 and
    ``w = 300`` gives 3.48 — so no correction is applied.  It must also not warn:
    nothing is wrong with the input, and there is no action for the user to take.
    """
    from tsdynamics.analysis.dimensions._common import _auto_theiler

    orbit = np.asarray(
        ts.systems.Chirikov().with_params(k=0.05).iterate(steps=3000, ic=[0.1, 0.3]).y
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here fails the test
        assert _auto_theiler(orbit) == 1
        d2 = float(dim.correlation_dimension(orbit))
    assert abs(d2 - 1.0) < 0.15, f"drifting invariant curve D2 = {d2:.3f}, expected ~1"
    # ... and the destructive alternative is destructive, which is why it is not used.
    assert float(dim.correlation_dimension(orbit, theiler=150)) > 2.0


def test_auto_theiler_improves_an_oversampled_flow():
    """On an oversampled Lorenz the automatic window beats the pre-v6 default of 0."""
    lor = _lorenz_points(n=8000, dt=0.002, transient=5000)
    d_auto = float(dim.correlation_dimension(lor))
    d_raw = float(dim.correlation_dimension(lor, theiler=0))
    assert abs(d_auto - 2.05) < abs(d_raw - 2.05), f"auto {d_auto:.3f} vs raw {d_raw:.3f}"


def test_resolved_theiler_window_is_recorded(lorenz):
    """The estimate records the window it chose — ``auto`` must not be silent."""
    res = dim.correlation_dimension(lorenz)
    assert res.meta["theiler"] > 1
    assert dim.correlation_dimension(lorenz, theiler=7).meta["theiler"] == 7


def test_theiler_rejects_an_unknown_string(uniform_sets):
    with pytest.raises(ValueError, match="theiler"):
        dim.correlation_dimension(uniform_sets["square"], theiler="sometimes")


# ── 5. the q -> 1 limit and the negative-q gap ──────────────────────────────────


def test_information_dimension_uses_the_entropy_limit_form():
    r"""At q = 1 the Renyi formula is 0/0; the entropy form is its exact limit.

    Checked by continuity against the library's own ordinate at ``q = 1 +- 1e-6``
    (where the closed form is still well conditioned): the limit value must sit
    between them, which the naive ``log(sum p^q)/(q-1)`` at exactly q = 1 cannot
    do — it is ``nan``.
    """
    from tsdynamics.analysis.dimensions.generalized import _partition_ordinate

    counts = np.array([120.0, 45.0, 30.0, 9.0, 3.0, 1.0])
    n = int(counts.sum())
    below = _partition_ordinate(counts, n, 1.0 - 1e-6)
    at = _partition_ordinate(counts, n, 1.0)
    above = _partition_ordinate(counts, n, 1.0 + 1e-6)
    assert np.isfinite(at)
    assert min(below, above) <= at <= max(below, above)
    assert at == pytest.approx(float(np.sum((counts / n) * np.log(counts / n))), rel=1e-12)


def test_negative_q_is_rejected_and_the_gap_is_named():
    """No exported function computes D_q for q < 0; the error says so."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(0.0, 1.0, (2000, 2))
    with pytest.raises(ValueError, match="no estimator for q < 0"):
        dim.generalized_dimension(pts, q=-1.0)
    # ... and the estimator the user might reach for genuinely takes no q.
    import inspect

    assert "q" not in inspect.signature(dim.fixed_mass_dimension).parameters


# ── 6. System-vs-data front-door guard ──────────────────────────────────────────


def test_dimensions_reject_a_system_with_a_named_error():
    """A System handed to a data-first estimator names itself and the fix."""
    from tsdynamics.errors import InvalidInputError

    for call in (
        lambda: dim.correlation_dimension(ts.Lorenz()),
        lambda: dim.box_counting_dimension(ts.Lorenz()),
        lambda: dim.generalized_dimension(ts.Lorenz()),
        lambda: dim.dimension_spectrum(ts.Lorenz()),
        lambda: dim.fixed_mass_dimension(ts.systems.Henon()),
    ):
        with pytest.raises(InvalidInputError, match="expects measured data, not a System"):
            call()
