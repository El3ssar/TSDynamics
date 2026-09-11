"""A-ORBIT: return maps, bifurcation quantification, and registry wiring.

The orbit-diagram and Poincaré-section *paths* are exercised in
``test_analysis.py``; this module covers the A-ORBIT additions — the
first-return map, ``OrbitDiagram`` period/bifurcation quantifiers, and the
``registry.analyses`` self-registration.
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics import registry
from tsdynamics.analysis import ReturnMap, return_map


def _functionality(cur: np.ndarray, suc: np.ndarray) -> float:
    """How single-valued ``suc = F(cur)`` is: median successor jump between
    current-adjacent points, as a fraction of the successor range.  Near 0 for
    a tight 1-D map; ~0.3+ for a 2-D cloud."""
    order = np.argsort(cur)
    s = suc[order]
    rng = s.max() - s.min()
    return float(np.median(np.abs(np.diff(s))) / rng)


# ---------------------------------------------------------------------------
# return_map — extremum mode on synthetic series (fast, no integration)
# ---------------------------------------------------------------------------


class TestReturnMapSeries:
    def test_known_maxima(self) -> None:
        s = np.array([0, 1, 0, 2, 0, 3, 0, 2.5, 0], dtype=float)
        rm = return_map(s, kind="max")
        np.testing.assert_allclose(rm.values, [1.0, 2.0, 3.0, 2.5])
        np.testing.assert_allclose(rm.current, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(rm.successor, [2.0, 3.0, 2.5])
        assert isinstance(rm, ReturnMap)
        assert len(rm) == 3

    def test_minima(self) -> None:
        s = np.array([0, -1, 0, -2, 0, -3, 0], dtype=float)
        rm = return_map(s, kind="min")
        np.testing.assert_allclose(rm.values, [-1.0, -2.0, -3.0])

    def test_parabolic_refinement_sharpens_peak(self) -> None:
        # Sample a cosine peak off-grid: the true maximum (1.0) lies between
        # samples, so the parabolic-refined value beats the raw sample max.
        t = np.linspace(-0.37, 2 * np.pi - 0.37, 64)  # peak of cos at t=0 is off-grid
        s = np.cos(t)
        rm = return_map(s, kind="max")
        assert rm.values.size == 1
        assert s.max() < rm.values[0] <= 1.0 + 1e-9

    def test_flat_and_iter(self) -> None:
        s = np.array([0, 1, 0, 2, 0, 3, 0], dtype=float)
        rm = return_map(s, kind="max")
        cur, suc = rm.flat()
        assert cur.shape == suc.shape == (2,)
        pairs = list(rm)
        assert pairs[0] == (1.0, 2.0)

    def test_constant_amplitude_collapses_to_diagonal(self) -> None:
        # A pure sine: every maximum is equal → the return map is one point on
        # the diagonal (current == successor).
        t = np.linspace(0, 60, 6000)
        rm = return_map(np.sin(2 * np.pi * t), kind="max")
        assert rm.values.size > 5
        np.testing.assert_allclose(rm.current, rm.successor, atol=1e-6)
        np.testing.assert_allclose(rm.values, rm.values[0], atol=1e-6)

    def test_too_short_series_is_empty(self) -> None:
        rm = return_map(np.array([1.0, 2.0]), kind="max")
        assert rm.values.size == 0
        assert len(rm) == 0


# ---------------------------------------------------------------------------
# return_map — input validation
# ---------------------------------------------------------------------------


class TestReturnMapValidation:
    def test_bad_kind(self) -> None:
        with pytest.raises(ValueError, match="kind must be"):
            return_map(np.zeros(10), kind="bogus")

    def test_2d_raw_series_rejected(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            return_map(np.zeros((10, 2)), kind="max")

    def test_poincare_needs_plane(self) -> None:
        with pytest.raises(ValueError, match="plane"):
            return_map(ts.systems.Rossler(), kind="poincare")

    def test_poincare_rejects_raw_series(self) -> None:
        with pytest.raises(TypeError, match="System or Trajectory"):
            return_map(np.zeros(10), kind="poincare", plane=(0, 0.0))

    def test_discrete_map_rejected_for_extrema(self) -> None:
        with pytest.raises(TypeError, match="continuous flow"):
            return_map(ts.systems.Henon(), kind="max")

    def test_unknown_named_observable(self) -> None:
        traj = ts.systems.Lorenz().run(final_time=1.0, dt=0.1, ic=[1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="unknown component"):
            return_map(traj, "nope", kind="max")


# ---------------------------------------------------------------------------
# return_map — Poincaré mode on trajectory data (fast, synthetic)
# ---------------------------------------------------------------------------


class TestReturnMapPoincareData:
    @staticmethod
    def _circle_traj() -> ts.data.Trajectory:
        # a clean limit cycle: (sin, cos) crosses the x=0 plane once per period
        t = np.linspace(0.0, 10.0 * np.pi, 4000)
        y = np.column_stack([np.sin(t), np.cos(t)])
        return ts.data.Trajectory(t, y, None)

    def test_crossings_from_data(self) -> None:
        rm = return_map(self._circle_traj(), 1, kind="poincare", plane=(0, 0.0), direction=1)
        assert rm.kind == "poincare"
        assert rm.values.size > 2
        # y = cos at the up-crossings of sin is ≈ +1 each period → a fixed point
        np.testing.assert_allclose(rm.values, 1.0, atol=1e-3)

    def test_transient_drops_leading_crossings(self) -> None:
        traj = self._circle_traj()
        full = return_map(traj, 1, kind="poincare", plane=(0, 0.0), direction=1)
        skipped = return_map(
            traj, 1, kind="poincare", plane=(0, 0.0), direction=1, skip_crossings=2
        )
        assert skipped.values.size == full.values.size - 2
        np.testing.assert_array_equal(skipped.values, full.values[2:])


# ---------------------------------------------------------------------------
# OrbitDiagram.periods / bifurcation_points (fast — logistic map)
# ---------------------------------------------------------------------------


class TestOrbitDiagramQuantifiers:
    def test_period_doubling_sequence(self) -> None:
        od = ts.analysis.orbit_diagram(
            ts.systems.Logistic(),
            "r",
            [2.8, 3.2, 3.5, 3.56],
            points_per_value=120,
            transient=2000,
            carry_state=False,
            ic=[0.5],
        )
        p = od.periods()
        assert p[0] == 1  # fixed point
        assert p[1] == 2  # 2-cycle
        assert p[2] == 4  # 4-cycle
        assert p[3] == 8  # 8-cycle

    def test_chaotic_band_is_aperiodic(self) -> None:
        od = ts.analysis.orbit_diagram(
            ts.systems.Logistic(), "r", [3.9], points_per_value=200, transient=500, ic=[0.5]
        )
        assert od.periods()[0] == 0  # too many branches → reported aperiodic

    def test_empty_value_is_minus_one(self) -> None:
        # r > 4 escapes [0, 1]: the sweep records an empty set (diverges).
        with pytest.warns(RuntimeWarning, match="diverged"):
            od = ts.analysis.orbit_diagram(
                ts.systems.Logistic(), "r", [4.5], points_per_value=50, transient=50, ic=[0.5]
            )
        assert od.periods()[0] == -1

    def test_bifurcation_points_match_literature(self) -> None:
        # Logistic period-doubling onsets: r1 = 3, r2 = 1 + sqrt(6) ≈ 3.449.
        od = ts.analysis.orbit_diagram(
            ts.systems.Logistic(),
            "r",
            np.linspace(2.9, 3.6, 400),
            points_per_value=64,
            transient=2000,
            ic=[0.5],
        )
        bp = od.bifurcation_points()
        assert np.min(np.abs(bp - 3.0)) < 0.03
        assert np.min(np.abs(bp - (1.0 + np.sqrt(6.0)))) < 0.02

    def test_to_plot_spec_default_is_clean(self) -> None:
        # The default plot is the textbook bifurcation picture: just the scatter,
        # no period/bifurcation text overlay (which piles up illegibly in the
        # chaotic cascade).  It also must not walk periods() at all when clean.
        od = ts.analysis.orbit_diagram(
            ts.systems.Logistic(),
            "r",
            np.linspace(2.8, 4.0, 60),
            points_per_value=48,
            transient=400,
            ic=[0.5],
        )
        calls = {"n": 0}
        real_periods = ts.analysis.OrbitDiagram.periods

        def counting_periods(self, **kw):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            return real_periods(self, **kw)

        try:
            ts.analysis.OrbitDiagram.periods = counting_periods  # type: ignore[method-assign]
            spec = od.to_plot_spec()
        finally:
            ts.analysis.OrbitDiagram.periods = real_periods  # type: ignore[method-assign]
        assert calls["n"] == 0  # clean plot never computes the period sweep
        assert not spec.annotations  # no vline smear

    def test_to_plot_spec_annotate_computes_periods_once(self) -> None:
        # Regression: the annotated path previously recomputed periods() once
        # directly and again inside bifurcation_points() (and a third walk).  With
        # annotate=True it must compute the period sweep exactly once per call.
        od = ts.analysis.orbit_diagram(
            ts.systems.Logistic(),
            "r",
            np.linspace(2.8, 3.6, 60),
            points_per_value=48,
            transient=400,
            ic=[0.5],
        )
        calls = {"n": 0}
        real_periods = ts.analysis.OrbitDiagram.periods

        def counting_periods(self, **kw):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            return real_periods(self, **kw)

        try:
            ts.analysis.OrbitDiagram.periods = counting_periods  # type: ignore[method-assign]
            spec = od.to_plot_spec(annotate=True)
        finally:
            ts.analysis.OrbitDiagram.periods = real_periods  # type: ignore[method-assign]
        assert calls["n"] == 1
        # And the labelled onset annotations are produced when opted in.
        assert any(a.kind == "vline" for a in spec.annotations)

    def test_bifurcation_points_from_precomputed_periods_match(self) -> None:
        # The factored helper must agree with the public bifurcation_points().
        od = ts.analysis.orbit_diagram(
            ts.systems.Logistic(),
            "r",
            np.linspace(2.9, 3.6, 80),
            points_per_value=48,
            transient=400,
            ic=[0.5],
        )
        p = od.periods()
        np.testing.assert_array_equal(
            od._bifurcation_points_from_periods(p), od.bifurcation_points()
        )


# ---------------------------------------------------------------------------
# registry self-registration
# ---------------------------------------------------------------------------


def test_orbit_analyses_self_register() -> None:
    names = registry.analyses.names()
    for n in ("orbit_diagram", "poincare_section", "return_map"):
        assert n in names
        assert registry.analyses.get(n) is getattr(ts.analysis, n)


# ---------------------------------------------------------------------------
# Slow: flows — the literature-validated return maps
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_lorenz_z_maxima_cusp_map() -> None:
    """Lorenz (1963): successive maxima of z form a near-1-D cusp map."""
    rm = ts.analysis.return_map(
        ts.systems.Lorenz(ic=[1.0, 1.0, 1.0]),
        "z",
        kind="max",
        final_time=400.0,
        dt=0.01,
        transient=40.0,
    )
    cur, suc = rm.flat()
    assert len(rm) > 100
    # the classic z-maxima live in a tight band around the cusp
    assert rm.values.min() > 28.0
    assert rm.values.max() < 50.0
    # and the map is effectively single-valued (a 1-D function)
    assert _functionality(cur, suc) < 0.05


@pytest.mark.slow
def test_rossler_poincare_return_map_is_1d() -> None:
    """y at successive x=0 crossings of Rössler is a tight 1-D return map."""
    rm = ts.analysis.return_map(
        ts.systems.Rossler(ic=[1.0, 1.0, 0.0]),
        "y",
        kind="poincare",
        plane=(0, 0.0),
        n=400,
        skip_crossings=20,
        dt=0.03,
    )
    assert rm.kind == "poincare"
    assert len(rm) > 100
    assert _functionality(*rm.flat()) < 0.05


@pytest.mark.slow
def test_periods_on_flow_bifurcation_diagram() -> None:
    """`periods()` reads the Rössler period-doubling route on a Poincaré section.

    A periodic-orbit branch recorded from a flow differs only by integration
    noise, so this exercises the scale-relative negligible-spread guard that
    keeps `_count_branches` honest for flows (period-1 must not shatter).
    """
    found = {}
    for c in (2.6, 3.5, 5.7):
        pmap = ts.derived.PoincareMap(
            ts.systems.Rossler(ic=[1.0, 1.0, 0.0]), plane=(0, 0.0), dt=0.03
        )
        od = ts.analysis.orbit_diagram(
            pmap, "c", [c], points_per_value=80, transient=100, component=1, ic=[3.0, 3.0, 0.0]
        )
        found[c] = int(od.periods()[0])
    assert found[2.6] == 1  # period-1 limit cycle
    assert found[3.5] == 2  # period-2
    assert found[5.7] == 0  # chaotic band → aperiodic


@pytest.mark.slow
def test_system_and_trajectory_paths_agree() -> None:
    """The same integration, read as a System or a Trajectory, gives the same map."""
    ic = [1.0, 1.0, 1.0]
    transient = 30.0
    rm_sys = ts.analysis.return_map(
        ts.systems.Lorenz(ic=ic), "z", kind="max", final_time=200.0, dt=0.01, transient=transient
    )
    traj = ts.systems.Lorenz(ic=ic).run(final_time=200.0, dt=0.01, ic=ic)
    rm_traj = ts.analysis.return_map(traj.after(transient), "z", kind="max")
    np.testing.assert_allclose(rm_sys.values, rm_traj.values)


# ---------------------------------------------------------------------------
# The one-liner: a bifurcation diagram OF A FLOW
#
# ``ts.analysis.orbit_diagram(model, "rho", values)`` used to refuse with a
# TypeError that named ``orbit_diagram`` (a function the caller had not typed)
# and told them to go and read about PoincareMap / StroboscopicMap.  A
# bifurcation diagram of a flow is the single most canonical use of the
# function, so it now works: the flow is reduced to its successive-maxima (peak)
# map, and the choice is recorded rather than made silently.
# ---------------------------------------------------------------------------


class TestBifurcationDiagramOfAFlow:
    """The headline call must return a diagram, not a lecture."""

    def test_a_raw_flow_is_accepted(self) -> None:
        od = ts.analysis.orbit_diagram(
            ts.systems.Lorenz(ic=[1.0, 1.0, 1.0]),
            "rho",
            np.linspace(0.0, 50.0, 12),
            points_per_value=30,
            transient=40,
        )
        assert len(od) == 12
        x, y = od.flat()
        assert x.size == y.size > 0
        assert np.all(np.isfinite(y))

    def test_the_chosen_section_is_recorded_not_silent(self) -> None:
        """A section always has to be chosen; choosing silently is its own trap."""
        od = ts.analysis.orbit_diagram(
            ts.systems.Lorenz(ic=[1.0, 1.0, 1.0]),
            "rho",
            [28.0],
            points_per_value=20,
            transient=30,
            component="z",
        )
        assert od.meta["section"] == "successive maxima of z"
        assert od.meta["section_auto"] is True
        # ... and the figure itself says so, so the picture is reproducible.
        spec = od.to_plot_spec()
        assert "successive maxima of z" in spec.title

    def test_the_fixed_point_branch_is_recorded_not_dropped(self) -> None:
        """Below the Hopf value the flow settles: that equilibrium IS the branch.

        Lorenz's non-trivial equilibria sit at ``x = ±sqrt(beta (rho - 1))``; a
        converged column must record that point rather than an empty set.
        """
        lor = ts.systems.Lorenz(ic=[1.0, 1.0, 1.0])
        od = ts.analysis.orbit_diagram(lor, "rho", [10.0], points_per_value=20, transient=30)
        (points,) = od.points
        assert points.shape[0] >= 1
        expected = np.sqrt(lor.beta * (10.0 - 1.0))
        assert np.allclose(np.abs(points[:, 0]), expected, atol=1e-6)

    def test_section_override_uses_a_poincare_map(self) -> None:
        od = ts.analysis.orbit_diagram(
            ts.systems.Rossler(ic=[1.0, 1.0, 0.0]),
            "c",
            [4.0],
            points_per_value=20,
            transient=25,
            section=("y", 0.0, "up"),
        )
        assert "Poincaré section" in od.meta["section"]
        assert od.meta["section_auto"] is False
        assert od.points[0].shape[0] == 20

    def test_a_map_is_unaffected(self) -> None:
        od = ts.analysis.orbit_diagram(
            ts.systems.Logistic(), "r", [3.2, 3.9], points_per_value=40, transient=200
        )
        assert od.meta["section"] == "map iterates"
        assert od.meta["section_auto"] is False
        assert int(od.periods()[0]) == 2  # the period-2 window

    def test_orbit_diagram_is_the_one_spelling(self) -> None:
        """The ``bifurcation_diagram`` alias was deleted in v6 (one concept, one name).

        It was the same object under a second name, so every shared error message
        named a function half its callers had never typed.  Guessing it must now
        teach the survivor rather than fail bare.
        """
        assert registry.analyses.get("orbit_diagram") is ts.analysis.orbit_diagram
        assert "bifurcation_diagram" not in registry.analyses.names()
        for namespace, prefix in ((ts, "tsdynamics"), (ts.analysis, "tsdynamics.analysis")):
            # v6: an exact hit in a redirect table is an ImportError (MovedInV6),
            # because ``from X import Y`` discards an AttributeError's message.
            with pytest.raises((AttributeError, ImportError)) as excinfo:
                _ = namespace.bifurcation_diagram
            message = str(excinfo.value)
            assert "orbit_diagram" in message, f"{prefix} must name the survivor"
            assert "renamed in v6" in message


class TestBifurcationDiagramRefusals:
    """When it does refuse, the message must contain the line to type.

    There is one spelling now — ``orbit_diagram`` — so a message names the call
    the user made.  While the ``bifurcation_diagram`` alias existed, one
    implementation carried two spellings and a message could only ever name one
    of them, contradicting the ``TypeError`` Python raised a line earlier.
    """

    def test_a_stochastic_system_is_refused_with_a_runnable_line(self) -> None:
        from tsdynamics.errors import InvalidInputError

        with pytest.raises(InvalidInputError) as excinfo:
            ts.analysis.orbit_diagram(ts.systems.OrnsteinUhlenbeck(), "theta", [1.0])
        message = str(excinfo.value)
        assert "ts.analysis.orbit_diagram(ts.systems.Lorenz(), 'rho'" in message

    def test_a_non_system_is_refused_with_a_runnable_line(self) -> None:
        from tsdynamics.errors import InvalidInputError

        with pytest.raises(InvalidInputError) as excinfo:
            ts.analysis.orbit_diagram([1.0, 2.0], "r", [1.0])
        assert "ts.analysis.orbit_diagram(ts.systems.Lorenz(), 'rho'" in str(excinfo.value)

    def test_section_on_an_already_discrete_view_names_the_swept_parameter(self) -> None:
        from tsdynamics.errors import InvalidParameterError

        with pytest.raises(InvalidParameterError) as excinfo:
            ts.analysis.orbit_diagram(ts.systems.Logistic(), "r", [3.5], section=("x", 0.0))
        assert "ts.analysis.orbit_diagram(system, 'r', values)" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The flow path must never hand back a blank diagram
#
# ``transient`` / ``n`` count PEAKS on the flow path, and the defaults (500/200)
# are sized for a map's cheap iterates.  A slow oscillator — or a DDE, whose
# peaks are a delay apart — makes far fewer peaks than that inside ``max_time``,
# and dropping the first ``transient`` of them then left an EMPTY column: a
# blank figure whose only clue was one RuntimeWarning.
# ---------------------------------------------------------------------------


class TestFlowColumnsAreNeverSilentlyEmpty:
    def test_a_short_run_keeps_its_last_peaks_and_says_so(self) -> None:
        """Fewer peaks than ``transient`` records the most asymptotic ones available."""
        from tsdynamics.analysis.orbits.orbit_diagram import _short_column

        rec = np.arange(30.0).reshape(30, 1)
        with pytest.warns(RuntimeWarning) as record:
            column = _short_column(rec, transient=500, n=200, idx=[0], max_time=1e4, dim=3)

        assert column.shape == (30, 1)
        assert np.array_equal(column, rec)  # the last min(n, found) peaks
        message = str(record[0].message)
        assert "transient=500" in message
        assert "NOT fully discarded" in message
        assert "ts.analysis.orbit_diagram(system, param, values, max_time=100000)" in message

    def test_a_partial_column_still_discards_the_transient(self) -> None:
        from tsdynamics.analysis.orbits.orbit_diagram import _short_column

        rec = np.arange(60.0).reshape(60, 1)
        with pytest.warns(RuntimeWarning, match="only 10 of 200 peaks"):
            column = _short_column(rec, transient=50, n=200, idx=[0], max_time=1e4, dim=3)
        assert column.shape == (10, 1)
        assert column[0, 0] == 50.0

    def test_no_peaks_at_all_points_at_the_other_two_views(self) -> None:
        """A monotone component is the peak map's one real failure: say what to type."""
        from tsdynamics.analysis.orbits.orbit_diagram import _short_column

        with pytest.warns(RuntimeWarning) as record:
            column = _short_column(np.empty((0, 1)), transient=5, n=5, idx=[0], max_time=1e4, dim=3)
        assert column.shape == (0, 1)
        message = str(record[0].message)
        assert "section=('z', 27.0, 'up')" in message
        assert "component=1" in message

    def test_a_scalar_flow_is_not_told_to_type_a_component_it_does_not_have(self) -> None:
        """A 1-D DDE has no second component; suggesting ``component=1`` would be a lie."""
        from tsdynamics.analysis.orbits.orbit_diagram import _short_column

        with pytest.warns(RuntimeWarning) as record:
            _short_column(np.empty((0, 1)), transient=5, n=5, idx=[0], max_time=1e4, dim=1)
        message = str(record[0].message)
        assert "section=" in message
        assert "component=" not in message

    def test_a_slow_flow_sweep_returns_points_rather_than_a_blank_picture(self) -> None:
        """End to end: a flow whose peaks are expensive still yields a drawable diagram."""
        with pytest.warns(RuntimeWarning):
            od = ts.analysis.orbit_diagram(
                ts.systems.Rossler(ic=[1.0, 1.0, 0.0]),
                "c",
                [4.0, 4.5],
                points_per_value=40,
                transient=200,
                max_time=60.0,
            )
        x, y = od.flat()
        assert x.size == y.size > 0
        assert np.all(np.isfinite(y))


def test_repr_reports_the_range_of_a_ragged_diagram() -> None:
    """Flow columns are ragged (an equilibrium records one point, a chaotic band ``n``).

    Quoting the first column's size described a 39 000-point Lorenz diagram as
    "1 points/value".
    """
    from tsdynamics.analysis.orbits.orbit_diagram import OrbitDiagram

    ragged = OrbitDiagram(
        param="rho",
        values=np.array([1.0, 2.0]),
        points=[np.zeros((1, 1)), np.zeros((200, 1))],
        components=(0,),
    )
    # v6 repr: the answer first, then the supporting lines (CONTRACT §4.3).
    head = repr(ragged).splitlines()[0]
    assert head == "OrbitDiagram  rho ∈ [1, 2] · 2 values × 1–200 points"

    even = OrbitDiagram(
        param="r",
        values=np.array([1.0, 2.0]),
        points=[np.zeros((40, 1)), np.zeros((40, 1))],
        components=(0,),
    )
    assert repr(even).splitlines()[0] == "OrbitDiagram  r ∈ [1, 2] · 2 values × 40 points"
