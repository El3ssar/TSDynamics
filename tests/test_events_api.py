"""The general ``system.run(events=[...])`` API (stream WS-EVENTSAPI).

Exercises the scipy-shaped event surface on :class:`ContinuousSystem`: event-spec
coercion, crossing direction, multiple events, terminal (arbitrary) stopping, the
compiled-engine path cross-checked against the dependency-light SciPy fallback,
and the analytic crossings of a linear oscillator.  ``PoincareMap.as_events`` is
checked to reproduce a section through the same seam.
"""

from __future__ import annotations

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.engine.run import Event, EventSolution, integrate_events
from tsdynamics.errors import InvalidInputError, InvalidParameterError

# The engine path is compiled; reference-path assertions run without it, but the
# whole module needs the extension to exercise both halves.
pytest.importorskip("tsdynamics._rust")

LORENZ_IC = [1.0, 1.0, 1.0]


class _EventsHarmonic(ts.ContinuousSystem):
    """Linear oscillator ``x'' = -ω² x`` — an analytic crossing oracle.

    With ``ic=(1, 0)`` and ``ω=1``: ``x(t) = cos t``, ``v(t) = -sin t``, so the
    section ``x = 0`` is crossed *downward* (``v < 0``) at ``t = π/2 + 2πk`` and
    *upward* (``v > 0``) at ``t = 3π/2 + 2πk``.  Non-chaotic, so two independent
    integrators agree over the whole window (no sensitive divergence).
    """

    dim = 2
    params = {"omega": 1.0}
    variables = ("x", "v")
    default_ic = (1.0, 0.0)

    @staticmethod
    def _equations(y, t, omega):  # noqa: D102
        return [y(1), -(omega**2) * y(0)]


# ---------------------------------------------------------------------------
# Event-spec coercion
# ---------------------------------------------------------------------------


class TestEventSpec:
    def test_plane_tuple_with_direction_word(self):
        e = Event(("z", 27.0, "up"))
        assert e.direction == +1
        assert e.terminal is False
        assert e.condition == ("z", 27.0)

    def test_plane_tuple_two_element_default_direction(self):
        e = Event(("z", 27.0))
        assert e.direction == 0

    def test_callable_carries_scipy_attributes(self):
        def g(y, t):
            return y(2) - 27.0

        g.terminal = True
        g.direction = -1
        e = Event.coerce(g)
        assert e.terminal is True
        assert e.direction == -1
        assert e.name == "g"

    def test_event_passthrough(self):
        e = Event(("x", 0.0))
        assert Event.coerce(e) is e

    def test_signed_direction_normalised(self):
        assert Event(("x", 0.0), direction=5).direction == +1
        assert Event(("x", 0.0), direction=-3).direction == -1

    def test_bad_condition_type_raises(self):
        with pytest.raises(InvalidInputError):
            Event(42)

    def test_bad_direction_word_raises(self):
        with pytest.raises(InvalidParameterError):
            Event(("z", 27.0, "sideways"))

    def test_bad_plane_length_raises(self):
        with pytest.raises(InvalidParameterError):
            Event(("z", 27.0, "up", "extra"))


# ---------------------------------------------------------------------------
# Analytic crossings — the linear oscillator
# ---------------------------------------------------------------------------


class TestAnalyticCrossings:
    def test_down_crossings_match_analytic(self):
        sys = _EventsHarmonic()
        sol = sys.run(
            final_time=20.0, dt=0.01, ic=(1.0, 0.0), method="rk4", events=[("x", 0.0, "down")]
        )
        t_cross = sol.meta["t_events"][0]
        expected = np.array([np.pi / 2 + 2 * np.pi * k for k in range(3)])
        assert t_cross.size >= expected.size
        assert np.allclose(t_cross[: expected.size], expected, atol=1e-3)
        # Crossing states sit on the section (x ≈ 0) with v < 0 (downward).
        y_cross = sol.meta["y_events"][0]
        assert np.allclose(y_cross[:, 0], 0.0, atol=1e-3)
        assert np.all(y_cross[: expected.size, 1] < 0.0)

    def test_up_crossings_match_analytic(self):
        sys = _EventsHarmonic()
        sol = sys.run(
            final_time=20.0, dt=0.01, ic=(1.0, 0.0), method="rk4", events=[("x", 0.0, "up")]
        )
        t_cross = sol.meta["t_events"][0]
        expected = np.array([3 * np.pi / 2 + 2 * np.pi * k for k in range(3)])
        assert np.allclose(t_cross[: expected.size], expected, atol=1e-3)

    def test_both_equals_up_plus_down(self):
        sys = _EventsHarmonic()
        ic = (1.0, 0.0)
        kw = dict(final_time=20.0, dt=0.01, ic=ic, method="rk4")
        up = sys.run(events=[("x", 0.0, "up")], **kw).meta["t_events"][0]
        down = sys.run(events=[("x", 0.0, "down")], **kw).meta["t_events"][0]
        both = sys.run(events=[("x", 0.0, "both")], **kw).meta["t_events"][0]
        assert both.size == up.size + down.size
        # `both` is the sorted merge of `up` and `down`.
        assert np.allclose(both, np.sort(np.concatenate([up, down])), atol=1e-9)


# ---------------------------------------------------------------------------
# run(events=) behaviour
# ---------------------------------------------------------------------------


class TestRunEvents:
    def test_returns_trajectory_with_event_meta(self):
        lor = ts.Lorenz()
        sol = lor.run(final_time=30.0, dt=0.01, ic=LORENZ_IC, events=[("z", 27.0, "up")])
        assert isinstance(sol, ts.Trajectory)
        assert sol.y.shape[1] == 3
        assert len(sol.meta["t_events"]) == 1
        assert sol.meta["n_events"] == 1
        assert sol.meta["terminated"] is False
        ev = sol.meta["t_events"][0]
        yv = sol.meta["y_events"][0]
        assert ev.size == yv.shape[0] > 0
        # Every recorded crossing sits on the z = 27 plane.
        assert np.allclose(yv[:, 2], 27.0, atol=1e-6)

    def test_events_none_is_plain_run(self):
        lor = ts.Lorenz()
        a = lor.run(final_time=10.0, dt=0.01, ic=LORENZ_IC)
        b = lor.run(final_time=10.0, dt=0.01, ic=LORENZ_IC, events=None)
        assert np.array_equal(a.y, b.y)
        assert "t_events" not in a.meta

    def test_multiple_events_collected_independently(self):
        lor = ts.Lorenz()
        sol = lor.run(
            final_time=30.0,
            dt=0.01,
            ic=LORENZ_IC,
            events=[("z", 27.0, "up"), ("z", 27.0, "down")],
        )
        up, down = sol.meta["t_events"]
        assert up.size > 0 and down.size > 0
        # up/down on the same plane interleave: counts differ by at most one.
        assert abs(up.size - down.size) <= 1

    def test_terminal_event_truncates(self):
        lor = ts.Lorenz()

        def escape(y, t):
            return y(0) ** 2 + y(1) ** 2 + y(2) ** 2 - 40.0**2

        escape.terminal = True
        full = lor.run(final_time=200.0, dt=0.01, ic=LORENZ_IC)
        sol = lor.run(final_time=200.0, dt=0.01, ic=LORENZ_IC, events=[escape])
        assert sol.meta["terminated"] is True
        # The trajectory stopped well before the full horizon.
        assert sol.t[-1] < full.t[-1] - 1.0
        # Exactly one terminal firing, and the state there is on the ball.
        fire = sol.meta["y_events"][0]
        assert fire.shape[0] == 1
        assert np.isclose(np.linalg.norm(fire[0]), 40.0, atol=1e-3)

    def test_terminal_with_nonterminal_companion(self):
        lor = ts.Lorenz()

        # A time-dependent terminal event (stop at t = 20): exercises an event
        # condition that depends on `t`, not just the state.
        def stop_at(y, t):
            return t - 20.0

        stop_at.terminal = True
        sol = lor.run(
            final_time=200.0,
            dt=0.01,
            ic=LORENZ_IC,
            events=[Event(("z", 27.0, "up")), stop_at],
        )
        t_stop = sol.meta["t_events"][1][0]
        z_cross = sol.meta["t_events"][0]
        assert np.isclose(t_stop, 20.0, atol=1e-6)
        assert sol.meta["terminated"] is True
        assert np.isclose(sol.t[-1], 20.0, atol=0.011)
        # Companion crossings happen and are all before the terminal stop.
        assert z_cross.size > 0
        assert np.all(z_cross <= t_stop + 1e-9)


# ---------------------------------------------------------------------------
# Engine vs the SciPy fallback (independent implementations)
# ---------------------------------------------------------------------------


class TestEngineVsReference:
    def test_oscillator_crossings_agree(self):
        sys = _EventsHarmonic()
        kw = dict(final_time=18.0, dt=0.02, ic=(1.0, 0.0), method="rk45", rtol=1e-10, atol=1e-12)
        eng = sys.run(backend="interp", events=[("x", 0.0, "up")], **kw)
        ref = sys.run(backend="reference", events=[("x", 0.0, "up")], **kw)
        te, tr = eng.meta["t_events"][0], ref.meta["t_events"][0]
        assert te.size == tr.size > 0
        # Non-chaotic flow: two independent integrators + root finders agree.
        assert np.allclose(te, tr, atol=1e-6)

    def test_lorenz_early_crossings_agree(self):
        lor = ts.Lorenz()
        kw = dict(final_time=12.0, dt=0.005, ic=LORENZ_IC, method="rk45", rtol=1e-9, atol=1e-11)
        eng = lor.run(backend="interp", events=[("z", 27.0, "up")], **kw)
        ref = lor.run(backend="reference", events=[("z", 27.0, "up")], **kw)
        te, tr = eng.meta["t_events"][0], ref.meta["t_events"][0]
        n = min(te.size, tr.size, 3)
        assert n >= 1
        # Compare only the first few crossings (Lorenz is chaotic: independent
        # integrators diverge later, but early crossings track the same orbit).
        assert np.allclose(te[:n], tr[:n], atol=1e-3)


# ---------------------------------------------------------------------------
# PoincareMap is one consumer of the events seam
# ---------------------------------------------------------------------------


class TestPoincareConsumer:
    def test_as_events_reproduces_section(self):
        ros = ts.Rossler()
        ic = [3.0, 3.0, 0.5]
        pmap = ts.PoincareMap(ros, plane=("y", 0.0, "up"), dt=0.01)
        pmap.reinit(ic)
        section = pmap.trajectory(15)

        sol = ros.run(
            final_time=500.0,
            dt=0.01,
            ic=ic,
            method="rk4",
            events=pmap.as_events(),
        )
        crossings = sol.meta["y_events"][0]
        assert crossings.shape[0] >= section.y.shape[0]
        # Same fixed-step march from the same IC → the same crossings.
        assert np.allclose(crossings[: section.y.shape[0]], section.y, atol=1e-6)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


class TestEventGuards:
    def test_map_problem_rejected(self):
        from tsdynamics.engine.problem import build_problem

        prob = build_problem(ts.Henon())
        with pytest.raises(InvalidInputError):
            integrate_events(prob, [("x", 0.0)], final_time=10.0)

    def test_empty_events_rejected(self):
        from tsdynamics.engine.problem import build_problem

        prob = build_problem(ts.Lorenz(), ic=LORENZ_IC)
        with pytest.raises(InvalidParameterError):
            integrate_events(prob, [], final_time=10.0)

    def test_event_solution_shape(self):
        sys = _EventsHarmonic()
        sol = sys._run_events(final_time=10.0, dt=0.02, events=[("x", 0.0, "down")], ic=(1.0, 0.0))
        assert isinstance(sol, ts.Trajectory)
        # Round-trip the transport object directly too.
        from tsdynamics.engine.problem import ode_problem

        out = integrate_events(
            ode_problem(sys, ic=np.array([1.0, 0.0])), [("x", 0.0)], final_time=10.0, dt=0.02
        )
        assert isinstance(out, EventSolution)
        assert out.y.shape[1] == 2
