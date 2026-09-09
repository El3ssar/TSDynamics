r"""The planar-flow estimators, checked against truth rather than against themselves.

Every quantity in :mod:`tsdynamics.analysis.planar` has an independent reference,
and this module uses it — a plot of a field is exactly the kind of artifact that
looks plausible while being wrong, so "it ran and produced numbers" is not the
bar anywhere here.

What each check is against:

- **nullclines** — closed form.  Lotka-Volterra's are two straight lines at
  :math:`y = \alpha/\beta` and :math:`x = \gamma/\delta`; Van der Pol's
  :math:`\dot x = 0` set is exactly :math:`y = 0` and its :math:`\dot y = 0` set
  satisfies :math:`\mu(1-x^2)y = x`.
- **the vector field** — an *integrated trajectory*: the field sampled on a
  lattice must be parallel to :math:`\dot u` along an orbit the engine produced
  by a completely separate path.
- **streamlines** — the same tangency test, plus the definition (unit-speed).
- **trace-determinant** — the hand-computed Jacobian invariants of a
  Lotka-Volterra saddle and centre.
- **FTLE** — a linear saddle :math:`(\dot x, \dot y) = (ax, -ay)` has
  :math:`\sigma \equiv a` **everywhere**, forward and backward.  And on a
  bistable flow the ridge must sit on the basin boundary, checked against an
  independent "integrate and see which well it lands in" oracle.
- **escape / transient time** — radial flows, where the first-passage time is
  :math:`\ln(R/r_0)` and :math:`\ln(r_0/\varepsilon)` in closed form.
- **invariant density** — the logistic map at :math:`r = 4`, whose natural
  measure is :math:`1/(\pi\sqrt{x(1-x)})`.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsdynamics.analysis import planar
from tsdynamics.errors import InvalidInputError, InvalidParameterError
from tsdynamics.families import ContinuousSystem
from tsdynamics.systems import LotkaVolterra, VanDerPol

# ---------------------------------------------------------------------------
# Test-local systems: linear flows whose fields are known in closed form.
# They live here, not in the catalogue, exactly like the basin validation
# systems in tests/test_basins.py.
# ---------------------------------------------------------------------------


class _LinearSaddle(ContinuousSystem):
    """``(x', y') = (a x, -a y)``: FTLE is identically ``a``, forward and backward."""

    params = {"a": 0.7}
    dim = 2
    variables = ("x", "y")
    default_ic = [0.1, 0.1]

    @staticmethod
    def _equations(Y, t, *, a):  # noqa: N803, D102
        return a * Y(0), -a * Y(1)


class _Outflow(ContinuousSystem):
    """``u' = k u``: the radius grows as ``r0 e^{kt}``, so escape times are exact."""

    params = {"k": 1.0}
    dim = 2
    variables = ("x", "y")
    default_ic = [0.1, 0.0]

    @staticmethod
    def _equations(Y, t, *, k):  # noqa: N803, D102
        return k * Y(0), k * Y(1)


class _Inflow(ContinuousSystem):
    """``u' = -k u``: the speed is the radius, so settling times are exact."""

    params = {"k": 1.0}
    dim = 2
    variables = ("x", "y")
    default_ic = [1.0, 0.0]

    @staticmethod
    def _equations(Y, t, *, k):  # noqa: N803, D102
        return -k * Y(0), -k * Y(1)


class _TwoWell(ContinuousSystem):
    """Damped unforced Duffing: stable foci at ``(+-1, 0)``, a saddle at the origin.

    Its basin boundary is the stable manifold of that saddle, which is what a
    forward FTLE ridge is supposed to find.
    """

    params = {"d": 0.35}
    dim = 2
    variables = ("x", "y")
    default_ic = [0.5, 0.0]

    @staticmethod
    def _equations(Y, t, *, d):  # noqa: N803, D102
        x, y = Y(0), Y(1)
        return y, x - x**3 - d * y


class _Spatial(ContinuousSystem):
    """A 3-D flow, for the slicing tests."""

    params = {"a": 1.0}
    dim = 3
    variables = ("x", "y", "z")
    default_ic = [0.1, 0.1, 0.1]

    @staticmethod
    def _equations(Y, t, *, a):  # noqa: N803, D102
        return a * Y(1), -a * Y(0), -Y(2)


# ---------------------------------------------------------------------------
# The slice: plane / base state / window
# ---------------------------------------------------------------------------


def test_resolve_plane_accepts_names_and_indices():
    system = _Spatial()
    assert planar.resolve_plane(system, ("x", "z")) == (0, 2, ("x", "z"))
    assert planar.resolve_plane(system, (0, 2)) == (0, 2, ("x", "z"))
    assert planar.resolve_plane(system, (-1, 0)) == (2, 0, ("z", "x"))


@pytest.mark.parametrize(
    ("plane", "match"),
    [
        (("x",), "exactly 2"),
        (("x", "q"), "not one of"),
        ((0, 7), "out of range"),
        (("y", "y"), "same coordinate twice"),
    ],
)
def test_resolve_plane_rejects_a_malformed_slice(plane, match):
    with pytest.raises(InvalidParameterError, match=match):
        planar.resolve_plane(_Spatial(), plane)


def test_the_window_is_recorded_however_it_was_chosen():
    """An auto-chosen region is the one thing a model plot can be quietly wrong about."""
    explicit_x, explicit_y, meta = planar.window_for(
        LotkaVolterra(), xlim=(0.0, 9.0), ylim=(0.0, 6.0)
    )
    assert explicit_x == (0.0, 9.0)
    assert explicit_y == (0.0, 6.0)
    assert meta["window_source"] == "explicit"
    assert meta["xlim"] == explicit_x

    auto_x, auto_y, auto_meta = planar.window_for(LotkaVolterra())
    assert auto_x[0] < 4.0 < auto_x[1]  # contains the coexistence equilibrium
    assert auto_y[0] < 2.75 < auto_y[1]
    assert "pilot orbit" in auto_meta["window_source"]


def test_a_base_state_of_the_wrong_size_is_refused():
    with pytest.raises(InvalidParameterError, match="needs 3 entries"):
        planar.flow_field(_Spatial(), at=[0.0, 0.0], xlim=(-1, 1), ylim=(-1, 1), grid=3)


def test_the_model_estimators_refuse_a_trajectory():
    """The greppable source rule, enforced: these evaluate the RHS somewhere new."""
    traj = LotkaVolterra().integrate(final_time=1.0, dt=0.1)
    for call in (planar.flow_field, planar.nullclines, planar.streamlines, planar.ftle_field):
        with pytest.raises(InvalidInputError, match="continuous system"):
            call(traj, xlim=(0.0, 1.0), ylim=(0.0, 1.0))


# ---------------------------------------------------------------------------
# nullclines vs closed form
# ---------------------------------------------------------------------------


def test_lotka_volterra_nullclines_are_the_analytic_lines():
    r"""``x' = 0`` on ``y = alpha/beta``; ``y' = 0`` on ``x = gamma/delta``."""
    system = LotkaVolterra()
    p = system.params
    curves = planar.nullclines(system, xlim=(0.2, 9.0), ylim=(0.2, 6.0), grid=201)

    y_values = np.concatenate([c[:, 1] for c in curves[0].curves])
    x_values = np.concatenate([c[:, 0] for c in curves[1].curves])
    assert np.abs(y_values - p["alpha"] / p["beta"]).max() < 1e-12
    assert np.abs(x_values - p["gamma"] / p["delta"]).max() < 1e-12
    assert [nc.label for nc in curves] == ["x", "y"]


def test_van_der_pol_nullclines_satisfy_their_defining_equations():
    system = VanDerPol()
    mu = system.params["mu"]
    curves = planar.nullclines(system, xlim=(-3.0, 3.0), ylim=(-4.0, 4.0), grid=401)

    # x' = y, so the x-nullcline is exactly the line y = 0.
    y_values = np.concatenate([c[:, 1] for c in curves[0].curves])
    assert np.abs(y_values).max() < 1e-12

    # y' = mu (1 - x^2) y - x, resolved only as well as the marching-squares lattice.
    residual = np.concatenate(
        [np.abs(mu * (1 - c[:, 0] ** 2) * c[:, 1] - c[:, 0]) for c in curves[1].curves]
    )
    assert residual.max() < 1e-3
    assert len(curves[1].curves) >= 2, "the y-nullcline has branches either side of x = +-1"


def test_the_nullcline_crossings_are_the_equilibria():
    """The check worth making every time, and it is a check on *both* computations.

    The nullclines come from marching squares over a lattice and the equilibria
    from multi-start Newton — entirely separate code — so agreement is evidence,
    not tautology.
    """
    from tsdynamics.analysis import fixed_points

    system = LotkaVolterra()
    curves = planar.nullclines(system, xlim=(0.2, 9.0), ylim=(0.2, 6.0), grid=401)
    equilibria = [
        fp.x for fp in fixed_points(system, seed=0) if 0.2 < fp.x[0] < 9.0 and 0.2 < fp.x[1] < 6.0
    ]
    assert equilibria, "the coexistence equilibrium is inside the window"
    for point in equilibria:
        for nullcline in curves:
            pts = np.vstack(nullcline.curves)
            gap = np.hypot(pts[:, 0] - point[0], pts[:, 1] - point[1]).min()
            assert gap < 5e-2, f"equilibrium {point} is not on the {nullcline.label}-nullcline"


def test_nullclines_of_a_named_off_plane_component_are_available():
    curves = planar.nullclines(
        _Spatial(),
        plane=("x", "z"),
        xlim=(-1.0, 1.0),
        ylim=(-1.0, 1.0),
        grid=41,
        components=("y",),
    )
    assert [nc.label for nc in curves] == ["y"]


# ---------------------------------------------------------------------------
# the field is the field: tangency to an integrated orbit
# ---------------------------------------------------------------------------


def _cosine_with_field(system, traj, field, skip=200):
    """Cosine of the angle between ``d(orbit)/dt`` and the sampled field along an orbit."""
    from scipy.interpolate import RegularGridInterpolator

    pts = traj.y[skip:]
    times = traj.t[skip:]
    gu = RegularGridInterpolator((field.ys, field.xs), field.u)
    gv = RegularGridInterpolator((field.ys, field.xs), field.v)
    query = np.column_stack([pts[:, 1], pts[:, 0]])
    velocity = np.gradient(pts, times, axis=0)
    fu, fv = gu(query), gv(query)
    return (velocity[:, 0] * fu + velocity[:, 1] * fv) / (
        np.hypot(velocity[:, 0], velocity[:, 1]) * np.hypot(fu, fv)
    )


def test_the_sampled_field_is_tangent_to_an_integrated_trajectory():
    """The one check that says the arrows mean what the picture claims."""
    system = VanDerPol()
    traj = system.integrate(final_time=20.0, dt=0.005, ic=[0.5, 0.5])
    field = planar.flow_field(system, xlim=(-3.0, 3.0), ylim=(-4.0, 4.0), grid=201)
    cosine = _cosine_with_field(system, traj, field)
    assert cosine.min() > 0.999, f"worst angle {np.degrees(np.arccos(cosine.min())):.3f} deg"


def test_the_field_keeps_its_true_magnitudes():
    """``flow_field`` never normalizes — normalization is the *drawing*'s decision."""
    system = VanDerPol()
    field = planar.flow_field(system, xlim=(-2.0, 2.0), ylim=(-2.0, 2.0), grid=9)
    assert np.allclose(field.speed, np.hypot(field.u, field.v))
    assert field.speed.max() > 1.5, "a real field's magnitudes vary; these were flattened"


def test_a_slice_freezes_the_off_plane_coordinates_where_it_says_it_does():
    system = _Spatial()
    field = planar.flow_field(
        system, plane=("x", "z"), at=[0.0, 2.0, 0.0], xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), grid=5
    )
    # x' = a y with y frozen at 2 -> the whole u channel is 2 everywhere.
    assert np.allclose(field.u, 2.0)
    assert field.meta["at"] == [0.0, 2.0, 0.0]


# ---------------------------------------------------------------------------
# streamlines
# ---------------------------------------------------------------------------


def test_streamlines_are_unit_speed_integral_curves_inside_the_window():
    system = VanDerPol()
    xlim, ylim = (-3.0, 3.0), (-4.0, 4.0)
    curves = planar.streamlines(system, xlim=xlim, ylim=ylim, seeds=4, steps=60)
    assert len(curves) == 16

    rhs = system._rhs_numeric()
    angles = []
    for curve in curves:
        assert curve.shape[1] == 2
        # inside the window, up to the one step that leaves it
        assert (curve[:, 0] >= xlim[0] - 1e-9).all() and (curve[:, 0] <= xlim[1] + 1e-9).all()
        step = np.diff(curve, axis=0)
        keep = np.hypot(step[:, 0], step[:, 1]) > 1e-12
        mid = 0.5 * (curve[:-1] + curve[1:])[keep]
        field = np.array([rhs(point, 0.0) for point in mid])
        cosine = (step[keep, 0] * field[:, 0] + step[keep, 1] * field[:, 1]) / (
            np.hypot(step[keep, 0], step[keep, 1]) * np.hypot(field[:, 0], field[:, 1])
        )
        angles.append(np.degrees(np.arccos(np.clip(np.abs(cosine), 0.0, 1.0))))
    off = np.concatenate(angles)
    # The chord of a step is compared with the field at its midpoint, so the
    # residual is the step's own truncation error; the tail is the tightest turn
    # the arc-length step still resolves.
    assert np.percentile(off, 99) < 1.5, f"p99 angle {np.percentile(off, 99):.2f} deg"
    assert off.max() < 5.0, f"worst angle {off.max():.2f} deg"


def test_streamline_arc_length_is_exactly_uniform():
    """Arc-length parametrization is the definition, and it is what makes the picture read.

    The unit field satisfies ``|u'| = 1`` identically, so re-projecting the
    Runge-Kutta increment onto unit length is a projection onto the constraint,
    not a fudge — and the sample spacing comes out uniform to floating point.
    """
    curves = planar.streamlines(
        VanDerPol(), xlim=(-3.0, 3.0), ylim=(-4.0, 4.0), seeds=2, steps=50, both_ways=False
    )
    for curve in curves:
        lengths = np.hypot(*np.diff(curve, axis=0).T)
        interior = lengths[:-1]  # the last step may be clipped by the window
        if interior.size:
            assert np.ptp(interior) < 1e-12 * max(float(interior.mean()), 1e-12) + 1e-12


def test_a_streamline_stops_rather_than_striding_through_an_equilibrium():
    """A fixed arc-length step through a fixed point would draw a line that is not there.

    On ``(ax, -ay)`` the ``x = 0`` axis is invariant and flows into the origin,
    so a streamline seeded on it must *stop* at the origin.  Without the turn
    guard the march would sail through and continue out the other side, drawing
    a straight segment across an equilibrium — a picture of something that never
    happens.
    """
    curves = planar.streamlines(
        _LinearSaddle(), xlim=(-0.5, 0.5), ylim=(-1.0, 1.0), seeds=(1, 2), steps=200
    )
    assert curves
    for curve in curves:
        assert np.allclose(curve[:, 0], 0.0, atol=1e-9), "the x = 0 axis is invariant"
        signs = np.sign(curve[:, 1])
        assert len(set(signs[signs != 0])) == 1, "the streamline crossed the equilibrium"
        assert np.abs(curve[:, 1]).min() < 0.05, "and it did reach it"


# ---------------------------------------------------------------------------
# trace-determinant
# ---------------------------------------------------------------------------


def test_classify_linear_reads_the_plane_the_way_the_textbook_does():
    assert planar.classify_linear(0.0, -1.0) == "saddle"
    assert planar.classify_linear(-3.0, 1.0) == "stable node"
    assert planar.classify_linear(3.0, 1.0) == "unstable node"
    assert planar.classify_linear(-1.0, 4.0) == "stable focus"
    assert planar.classify_linear(1.0, 4.0) == "unstable focus"
    assert planar.classify_linear(0.0, 4.0) == "centre"
    assert planar.classify_linear(1.0, 0.0) == "degenerate"


def test_lotka_volterra_lands_on_the_trace_determinant_plane_where_it_should():
    r"""Hand-computed: the origin is a saddle ``(0.7, -0.44)``, coexistence a centre."""
    system = LotkaVolterra()
    p = system.params
    result = planar.trace_determinant(system, points=[[0.0, 0.0], [4.0, 2.75]])

    assert result.trace[0] == pytest.approx(p["alpha"] - p["gamma"])
    assert result.determinant[0] == pytest.approx(-p["alpha"] * p["gamma"])
    assert result.classes[0] == "saddle"

    assert result.trace[1] == pytest.approx(0.0, abs=1e-9)
    assert result.determinant[1] == pytest.approx(p["beta"] * p["delta"] * 4.0 * 2.75)
    assert result.classes[1] == "centre"

    taus = result.parabola[:, 0]
    assert np.allclose(result.parabola[:, 1], taus**2 / 4.0)
    assert result.trace_range[0] < 0.0 < result.trace_range[1], "the tau = 0 axis must be in view"


def test_the_trace_determinant_plane_uses_the_named_sub_block():
    """On a 3-D flow, ``plane=`` selects which 2x2 linearization is classified."""
    result = planar.trace_determinant(_Spatial(), plane=("x", "y"), points=[[0.0, 0.0, 0.0]])
    assert result.trace[0] == pytest.approx(0.0)
    assert result.determinant[0] == pytest.approx(1.0)  # rotation block
    assert result.classes[0] == "centre"


# ---------------------------------------------------------------------------
# FTLE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backward", [False, True])
def test_ftle_of_a_linear_saddle_is_its_exponent_everywhere(backward):
    r"""``(ax, -ay)`` has flow map ``diag(e^{aT}, e^{-aT})``, so ``sigma == a`` exactly."""
    system = _LinearSaddle()
    field = planar.ftle_field(
        system, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), grid=21, time=2.0, backward=backward
    )
    assert np.isfinite(field.values).all()
    assert np.abs(field.values - system.params["a"]).max() < 1e-8
    assert field.meta["backward"] is backward
    assert field.meta["ftle_time"] == 2.0


def test_a_zero_ftle_horizon_is_refused():
    with pytest.raises(InvalidParameterError, match="non-zero"):
        planar.ftle_field(_LinearSaddle(), xlim=(-1, 1), ylim=(-1, 1), grid=5, time=0.0)


def test_the_ftle_ridge_sits_on_the_basin_boundary_of_a_bistable_flow():
    """The physical claim of the plot: forward FTLE ridges are repelling LCS.

    The oracle is deliberately *not* the basin machinery — it is a direct
    "integrate every cell and see which well it ends in" ensemble, so nothing in
    the comparison shares code with the field being checked.
    """
    from tsdynamics.engine import run as engine_run

    system = _TwoWell()
    n = 61
    axis = np.linspace(-2.0, 2.0, n)
    gx, gy = np.meshgrid(axis, axis)
    final = engine_run.ensemble(system, np.column_stack([gx.ravel(), gy.ravel()]), final_time=60.0)
    well = np.sign(final[:, 0]).reshape(n, n)
    sigma = planar.ftle_field(system, xlim=(-2.0, 2.0), ylim=(-2.0, 2.0), grid=n, time=6.0).values

    rng = np.random.default_rng(0)
    offsets, null = [], []
    for row in range(3, n - 3):
        crossings = np.array([c for c in range(3, n - 4) if well[row, c] * well[row, c + 1] < 0])
        if crossings.size == 0:
            continue
        peak = int(np.nanargmax(sigma[row, 3 : n - 3])) + 3
        offsets.append(np.abs(crossings - peak).min())
        null.append(np.abs(crossings - rng.integers(3, n - 3)).min())

    offsets, null = np.asarray(offsets, float), np.asarray(null, float)
    assert offsets.size > 40
    assert np.median(offsets) <= 2.0, "the strongest FTLE cell in a row is not on the boundary"
    assert np.median(offsets) < 0.4 * np.median(null), "no better than a random column"
    assert (offsets <= 2).mean() > 0.5


# ---------------------------------------------------------------------------
# escape / transient time
# ---------------------------------------------------------------------------


def test_escape_time_of_a_radial_outflow_is_the_analytic_log():
    r"""``r(t) = r_0 e^{kt}``, so leaving radius ``R`` takes exactly ``ln(R / r_0)``."""
    step = 6.0 / 60
    field = planar.escape_time_field(
        _Outflow(),
        xlim=(-1.0, 1.0),
        ylim=(-1.0, 1.0),
        grid=21,
        final_time=6.0,
        chunks=60,
        escape=4.0,
    )
    gx, gy = np.meshgrid(field.xs, field.ys)
    radius = np.hypot(gx, gy)
    truth = np.log(4.0 / np.where(radius > 0, radius, np.nan))
    residual = field.values - truth
    finite = np.isfinite(residual)
    # A sampled first-passage time is the first chunk *at or after* the true one.
    assert residual[finite].min() >= -1e-9
    assert residual[finite].max() <= step + 1e-9
    assert field.meta["time_resolution"] == pytest.approx(step)
    assert "distance from the window centre" in field.meta["escape"]


def test_escape_time_defaults_to_leaving_the_drawn_window():
    field = planar.escape_time_field(
        _Outflow(), xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), grid=11, final_time=4.0, chunks=8
    )
    assert field.meta["escape"] == "left the drawn window (in-plane)"
    assert np.isfinite(field.values).any()


def test_a_point_that_never_escapes_is_nan_not_the_horizon():
    field = planar.escape_time_field(
        _Inflow(), xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), grid=11, final_time=4.0, chunks=8
    )
    assert np.isnan(field.values).all(), "an inflow never leaves; that is NaN, not final_time"


def test_transient_time_of_a_radial_inflow_is_the_analytic_log():
    r"""``|f| = r = r_0 e^{-kt}``, so slowing below ``eps`` takes ``ln(r_0 / eps)``."""
    step = 8.0 / 80
    tol = 1e-2
    field = planar.transient_time_field(
        _Inflow(),
        xlim=(-1.0, 1.0),
        ylim=(-1.0, 1.0),
        grid=21,
        final_time=8.0,
        chunks=80,
        tol=tol,
    )
    gx, gy = np.meshgrid(field.xs, field.ys)
    radius = np.hypot(gx, gy)
    truth = np.log(np.where(radius > tol, radius / tol, np.nan))
    residual = field.values - truth
    finite = np.isfinite(residual)
    assert residual[finite].min() >= -1e-9
    assert residual[finite].max() <= step + 1e-9
    assert field.meta["settled"] == f"speed |f(u)| < {tol:g}"


def test_the_default_settling_threshold_is_read_off_the_field_and_recorded():
    """An absolute speed threshold means nothing without the flow's own scale.

    A fixed ``1e-3`` makes the field almost entirely "did not settle" on a system
    whose speeds are of order ``1e-2``; the default is a percent of the median
    lattice speed, and — like the window — it is recorded rather than applied
    silently.
    """
    from tsdynamics.systems import Brusselator

    system = Brusselator(b=1.5)  # below the Hopf threshold: a stable focus
    field = planar.transient_time_field(
        system, xlim=(0.2, 2.6), ylim=(0.4, 3.4), grid=31, final_time=20.0, chunks=40
    )
    assert "1% of the median lattice speed" in field.meta["settled"]
    assert np.isfinite(field.values).mean() > 0.9, "the default left the field almost empty"


def test_the_batched_speed_is_the_per_row_speed_exactly():
    """The vectorization is a speed-up, not an approximation."""
    system = VanDerPol()
    batched = planar._batched_speed(system)
    per_row = system._rhs_numeric()
    states = np.random.default_rng(1).uniform(-2.0, 2.0, size=(200, 2))
    reference = np.array([float(np.linalg.norm(per_row(row, 0.0))) for row in states])
    assert np.abs(batched(states) - reference).max() == 0.0


def test_an_explicit_arrival_test_overrides_the_speed_default_and_is_recorded():
    field = planar.transient_time_field(
        _Inflow(),
        xlim=(-1.0, 1.0),
        ylim=(-1.0, 1.0),
        grid=11,
        final_time=4.0,
        chunks=8,
        settled=lambda states: np.hypot(states[:, 0], states[:, 1]) < 0.05,
    )
    assert field.meta["settled"] == "caller-supplied arrival test"
    assert np.isfinite(field.values).any()


# ---------------------------------------------------------------------------
# the natural measure
# ---------------------------------------------------------------------------


def test_the_logistic_invariant_density_matches_the_arcsine_law():
    r"""``rho(x) = 1 / (pi sqrt(x (1-x)))`` at ``r = 4`` — an exact reference."""
    from tsdynamics.systems import Logistic

    orbit = Logistic(r=4.0).iterate(steps=200_000, ic=[0.4])
    centres, density, edges = planar.invariant_density(orbit.y[:, 0], bins=200)
    truth = 1.0 / (np.pi * np.sqrt(np.clip(centres * (1.0 - centres), 1e-30, None)))
    inside = (centres > 0.05) & (centres < 0.95)
    relative = np.abs(density[inside] - truth[inside]) / truth[inside]

    assert np.median(relative) < 0.05
    assert relative.max() < 0.25
    assert edges.size == centres.size + 1
    assert np.trapezoid(density, centres) == pytest.approx(1.0, abs=0.05)


def test_the_two_dimensional_measure_normalizes_and_is_image_shaped():
    rng = np.random.default_rng(3)
    x = rng.normal(size=20_000)
    y = rng.normal(size=20_000)
    xs, ys, density = planar.invariant_density_2d(x, y, bins=40)
    assert density.shape == (ys.size, xs.size)
    area = float(np.diff(xs).mean() * np.diff(ys).mean())
    assert density.sum() * area == pytest.approx(1.0, abs=0.02)


def test_the_density_estimators_refuse_empty_or_mismatched_input():
    with pytest.raises(InvalidInputError, match="finite sample"):
        planar.invariant_density([np.nan, np.inf])
    with pytest.raises(InvalidInputError, match="same length"):
        planar.invariant_density_2d([1.0, 2.0], [1.0])


def test_the_batched_sliced_field_is_the_per_point_field_exactly():
    """The lockstep march is a speed-up, not a different computation.

    Every seed marches together through one batched right-hand-side call per
    Runge-Kutta stage — 102,400 evaluations at the defaults, which is the whole
    cost of the plot when made one Python call at a time.  A kernel that declines
    a batch falls back to the per-point callable, so the two must agree to the
    last bit or the drawn curves would depend on which path a system took.
    """
    rng = np.random.default_rng(4)
    for system, plane, base in (
        (VanDerPol(), (0, 1), np.zeros(2)),
        (LotkaVolterra(), (0, 1), np.zeros(2)),
        (_Spatial(), (0, 2), np.array([0.0, 2.0, 0.0])),
    ):
        i, j, _ = planar.resolve_plane(system, plane)
        batched = planar._batched_sliced_rhs(system, i, j, base)
        per_point = planar._sliced_rhs(system, i, j, base)
        points = rng.uniform(0.1, 2.0, size=(64, 2))
        reference = np.array([per_point(float(p[0]), float(p[1])) for p in points])
        assert np.abs(batched(points) - reference).max() == 0.0, type(system).__name__
