"""WS-POINCARE-API — the named viz-ready ``poincare_section`` entry point.

Covers the two deliverables of stream WS-POINCARE-API (issue #209):

1. the friendly ``plane`` spelling — a component **name** (``"y"``) instead of an
   opaque index, with an optional direction word (``"up"`` / ``"down"`` /
   ``"both"``) as a third tuple element — accepted by ``poincare_section`` and
   ``PoincareMap`` alike, and answer-identical to the old ``(index, value)`` form;
2. the :class:`~tsdynamics.derived.PoincareSection` result type — a thin
   :class:`~tsdynamics.data.Trajectory` subclass carrying ``POINCARE_SECTION`` plot
   intent plus the ``.summary()`` / ``.to_dict()`` / ``.plot`` result surface.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import tsdynamics as ts
from tsdynamics.analysis._result import VisualizationNotInstalled
from tsdynamics.derived import PoincareMap, PoincareSection
from tsdynamics.errors import ConvergenceError, InvalidParameterError
from tsdynamics.families import Trajectory
from tsdynamics.viz.spec import PlotKind


def _rossler():
    """A deterministic Rössler (fixed IC) so two sections are bit-comparable."""
    return ts.Rossler(ic=[1.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# Friendly plane spelling — names resolve, and the section is identical
# ---------------------------------------------------------------------------


def test_named_axis_matches_index_form() -> None:
    """``plane=("y", 0.0)`` resolves to the same section as ``plane=(1, 0.0)``."""
    named = ts.poincare_section(_rossler(), plane=("y", 0.0), crossings=40, dt=0.05)
    index = ts.poincare_section(_rossler(), plane=(1, 0.0), crossings=40, dt=0.05)
    assert np.array_equal(named.y, index.y)
    assert np.array_equal(named.t, index.t)
    assert named.meta["plane"] == (1, 0.0)


def test_direction_word_in_plane_sets_and_overrides_direction() -> None:
    """A third ``plane`` element is the crossing direction and beats ``direction=``."""
    down = ts.poincare_section(_rossler(), plane=("x", 0.0, "down"), crossings=30, dt=0.05)
    explicit = ts.poincare_section(_rossler(), plane=(0, 0.0), direction=-1, crossings=30, dt=0.05)
    assert down.meta["direction"] == -1
    assert np.array_equal(down.y, explicit.y)
    # the in-plane direction word wins over a conflicting direction= argument
    forced = ts.poincare_section(
        _rossler(), plane=("x", 0.0, "down"), direction=+1, crossings=30, dt=0.05
    )
    assert forced.meta["direction"] == -1


def test_poincare_map_accepts_named_plane_and_direction_word() -> None:
    """``PoincareMap`` resolves a named plane to the index form on ``.plane``."""
    pm = PoincareMap(_rossler(), plane=("y", 0.0, "down"))
    assert pm.plane == (1, 0.0)
    assert pm.direction == -1
    assert isinstance(pm.run(20), PoincareSection)


def test_general_normal_plane_passes_through() -> None:
    """An arbitrary normal vector still works (and is not name-resolved)."""
    sec = ts.poincare_section(_rossler(), plane=([0.0, 1.0, 0.0], 0.0), crossings=20, dt=0.05)
    # normal is along +y, so the recorded crossings sit on y ≈ 0
    assert np.max(np.abs(sec.y[:, 1])) < 1e-6


# ---------------------------------------------------------------------------
# The PoincareSection result type (intent + result surface)
# ---------------------------------------------------------------------------


def test_returns_poincare_section_that_is_a_trajectory() -> None:
    sec = ts.poincare_section(_rossler(), plane=("y", 0.0), crossings=30, dt=0.05)
    assert isinstance(sec, PoincareSection)
    assert isinstance(sec, Trajectory)
    # trajectory affordances survive (named components, shapes)
    assert sec.y.shape == (30, 3)
    assert np.array_equal(sec["x"], sec.y[:, 0])


def test_section_carries_poincare_intent_and_spec() -> None:
    sec = ts.poincare_section(_rossler(), plane=("y", 0.0), crossings=40, dt=0.05)
    assert sec.meta["plot_kind"] == "poincare_section"
    spec = sec.__plot_spec__()
    assert spec.kind == PlotKind.POINCARE_SECTION
    assert spec.ndim == 2
    assert spec.aspect == "equal"
    assert spec.layers[0].kind == PlotKind.SCATTER


def test_section_summary_and_repr() -> None:
    sec = ts.poincare_section(_rossler(), plane=("y", 0.0, "up"), crossings=25, dt=0.05)
    summary = sec.summary()
    assert "PoincareSection" in summary
    assert "crossings = 25" in summary
    assert "up" in summary
    assert repr(sec) == "PoincareSection(crossings=25, dim=3)"


def test_section_to_dict_is_json_serializable() -> None:
    sec = ts.poincare_section(_rossler(), plane=("y", 0.0), crossings=20, dt=0.05)
    d = sec.to_dict()
    assert set(d) >= {"t", "y", "n_crossings", "plane", "direction", "meta"}
    assert d["n_crossings"] == 20
    assert d["plane"] == [1, 0.0]
    # the whole thing must round-trip through stdlib json (no numpy leaks)
    assert isinstance(json.loads(json.dumps(d))["y"], list)


def test_section_has_populated_provenance_meta() -> None:
    sec = ts.poincare_section(_rossler(), plane=("y", 0.0), crossings=15, dt=0.05)
    assert isinstance(sec.meta, dict) and sec.meta
    assert sec.meta.get("system") == "Rossler"


def test_section_plot_seam_raises_without_a_backend(monkeypatch) -> None:
    # The matplotlib backend auto-registers on render as of stream VIZ-MPL-CORE;
    # force an empty registry to keep testing the genuine no-backend path.
    from tsdynamics import registry
    from tsdynamics.viz import render as render_mod

    saved = registry.renderers.all()
    registry.renderers.clear()
    monkeypatch.setattr(render_mod, "register_builtin_renderers", lambda *a, **k: [])
    try:
        sec = ts.poincare_section(_rossler(), plane=("y", 0.0), crossings=15, dt=0.05)
        with pytest.raises(VisualizationNotInstalled):
            sec.plot().render()
    finally:
        registry.renderers.clear()
        for entry in saved:
            registry.renderers.register(entry.name, entry.obj, replace=True)


# ---------------------------------------------------------------------------
# Data path (a measured Trajectory) — same friendly spelling, same result type
# ---------------------------------------------------------------------------


def test_data_path_returns_named_section() -> None:
    traj = _rossler().run(final_time=80.0, dt=0.02)
    sec = ts.poincare_section(traj, plane=("z", 0.0, "up"))
    assert isinstance(sec, PoincareSection)
    assert sec.meta["plot_kind"] == "poincare_section"
    assert sec.meta["plane"] == (2, 0.0)
    assert sec.meta["direction"] == 1
    if sec.n_steps:  # interpolated crossings land on the plane (dt-limited)
        assert np.max(np.abs(sec.y[:, 2])) < 1e-3


# ---------------------------------------------------------------------------
# Backward compatibility — the old spellings still work
# ---------------------------------------------------------------------------


def test_index_form_and_skip_crossings_still_work() -> None:
    sec = ts.poincare_section(_rossler(), plane=(0, 0.0), crossings=20, skip_crossings=5, dt=0.05)
    assert isinstance(sec, PoincareSection)
    assert sec.y.shape == (20, 3)
    assert np.max(np.abs(sec.y[:, 0])) < 1e-6


# ---------------------------------------------------------------------------
# Error reporting (all value-naming InvalidParameterError, ⊂ ValueError)
# ---------------------------------------------------------------------------


def test_unknown_component_name_raises() -> None:
    with pytest.raises(InvalidParameterError, match="not a declared component"):
        ts.poincare_section(_rossler(), plane=("w", 0.0), crossings=5)


def test_unknown_direction_word_raises() -> None:
    with pytest.raises(InvalidParameterError, match="up.*down.*both"):
        ts.poincare_section(_rossler(), plane=("y", 0.0, "sideways"), crossings=5)


def test_malformed_plane_raises() -> None:
    with pytest.raises(InvalidParameterError, match="axis, offset"):
        ts.poincare_section(_rossler(), plane=(1,), crossings=5)


def test_plane_errors_are_value_errors() -> None:
    """``except ValueError`` still catches the new section errors."""
    assert issubclass(InvalidParameterError, ValueError)
    with pytest.raises(ValueError):
        PoincareMap(_rossler(), plane=("nope", 0.0))


def test_parse_plane_out_of_range_is_invalid_parameter() -> None:
    """An out-of-range index raises the typed ``InvalidParameterError``."""
    with pytest.raises(InvalidParameterError, match="out of range"):
        PoincareMap(_rossler(), plane=(7, 0.0))


def test_parse_plane_zero_normal_is_invalid_parameter() -> None:
    """A zero normal vector raises the typed ``InvalidParameterError``."""
    with pytest.raises(InvalidParameterError, match="non-zero"):
        PoincareMap(_rossler(), plane=(np.zeros(3), 0.0))


# ---------------------------------------------------------------------------
# ConvergenceError contract — a plane that misses the attractor (CLAUDE.md)
# ---------------------------------------------------------------------------


def test_section_missing_attractor_raises_convergence_error() -> None:
    """A plane the bounded Rössler attractor never reaches raises ConvergenceError.

    CLAUDE.md promises no-crossing-within-``max_time`` raises a
    :class:`~tsdynamics.errors.ConvergenceError` (a ``RuntimeError``), not a bare
    ``RuntimeError``.  ``x = 1e6`` is far outside the attractor, so no crossing is
    ever found within the short ``max_time`` — on both the engine fast path
    (default backend) and the pure-Python loop (``backend="reference"``).
    """
    pm = PoincareMap(_rossler(), plane=("x", 1e6), dt=0.05, max_time=50.0)
    pm.reinit([1.0, 1.0, 0.0])
    with pytest.raises(ConvergenceError):
        pm.run(5)

    pm_ref = PoincareMap(_rossler(), plane=("x", 1e6), dt=0.05, max_time=50.0)
    pm_ref.reinit([1.0, 1.0, 0.0])
    with pytest.raises(ConvergenceError):
        pm_ref.run(5, backend="reference")


def test_section_missing_attractor_is_runtime_error() -> None:
    """``except RuntimeError`` still catches the no-crossing failure (additive)."""
    pm = PoincareMap(_rossler(), plane=("x", 1e6), dt=0.05, max_time=50.0)
    pm.reinit([1.0, 1.0, 0.0])
    with pytest.raises(RuntimeError):
        pm.run(5, backend="reference")


# ---------------------------------------------------------------------------
# No plane named — one is CHOSEN and RECORDED (stream v6 API-FOOTGUNS)
# ---------------------------------------------------------------------------
#
# ``ts.PoincareMap(ts.systems.Lorenz())`` and ``ts.poincare_section(Lorenz())``
# used to answer with Python's raw binder error — *missing 1 required positional
# argument: 'plane'* — which names the parameter and says nothing about what a
# plane is, what shape it takes, or which one works for this system.  For one of
# the six headline names on the curated top level that is not an acceptable first
# contact, so a section is now chosen the way ``bifurcation_diagram`` chooses a
# flow's discrete view: from the data, and never silently.


def test_poincare_map_without_a_plane_chooses_one() -> None:
    """The headline call runs, and the section it picked is a real section."""
    pmap = PoincareMap(_rossler())
    assert pmap.plane_auto is True
    axis, offset = pmap.plane
    assert axis in range(3)
    assert np.isfinite(offset)
    section = pmap.run(40)
    assert section.y.shape == (40, 3)
    # the crossings really do lie on the plane it chose
    assert np.allclose(section.y[:, axis], offset, atol=1e-6)


def test_poincare_section_without_a_plane_chooses_one() -> None:
    """``ts.poincare_section(system)`` — the headline name — works with no plane."""
    section = ts.poincare_section(_rossler(), crossings=30)
    assert isinstance(section, PoincareSection)
    assert section.n_steps == 30
    assert section.meta["plane_auto"] is True


def test_auto_plane_offset_is_crossed_by_construction() -> None:
    """The median offset is what makes the auto section safe: the orbit straddles it.

    A naive default (``x = 0``) can miss an attractor entirely and then raise
    ``ConvergenceError`` — the very failure mode a beginner cannot diagnose.  Half
    the probe orbit lies on each side of the median, so a crossing must exist.
    """
    from tsdynamics.derived.poincare import auto_plane

    system = _rossler()
    axis, offset = auto_plane(system)
    y = system.copy().run(final_time=100.0, dt=0.05, transient=50.0).y[:, axis]
    assert y.min() < offset < y.max()


def test_auto_choice_is_recorded_everywhere_it_is_read() -> None:
    """meta, the attribute and the summary all say the section was chosen."""
    section = ts.poincare_section(_rossler(), crossings=20)
    assert section.meta["plane_auto"] is True
    assert "chosen automatically" in section.summary()
    named = ts.poincare_section(_rossler(), plane=("y", 0.0), crossings=20)
    assert named.meta["plane_auto"] is False
    assert "chosen automatically" not in named.summary()


def test_named_plane_is_unaffected_by_the_auto_path() -> None:
    """Passing a plane still bypasses the probe entirely (byte-identical section)."""
    a = ts.poincare_section(_rossler(), plane=("y", 0.0, "up"), crossings=25)
    b = ts.poincare_section(_rossler(), plane=(1, 0.0), crossings=25)
    assert np.array_equal(a.y, b.y)
    assert a.meta["plane_auto"] is False


def test_auto_plane_from_data_uses_the_samples() -> None:
    """The data overload chooses from the trajectory itself — no probe run."""
    traj = _rossler().run(final_time=120.0, dt=0.01, transient=40.0)
    section = ts.poincare_section(traj)
    assert section.meta["plane_auto"] is True
    axis, offset = section.meta["plane"]
    assert traj.y[:, axis].min() < offset < traj.y[:, axis].max()


def test_auto_plane_refuses_a_fixed_point_with_a_usable_message() -> None:
    """A flat orbit has no section, and the refusal names planes to pass instead."""
    from tsdynamics.derived.poincare import auto_plane

    flat = Trajectory(
        t=np.linspace(0.0, 1.0, 50),
        y=np.tile([1.0, 2.0, 3.0], (50, 1)),
        system=_rossler(),
        meta={},
    )
    with pytest.raises(InvalidParameterError) as excinfo:
        auto_plane(flat)
    message = str(excinfo.value)
    assert "fixed point" in message
    assert "plane=" in message


def test_rebuilt_map_keeps_the_same_section_and_its_provenance() -> None:
    """``with_params`` re-parametrizes; it does not re-choose (or re-label) the section."""
    pmap = PoincareMap(_rossler())
    rebuilt = pmap.with_params(c=5.8)
    assert rebuilt.plane == pmap.plane
    assert rebuilt.plane_auto is True


# ---------------------------------------------------------------------------
# A section has to be RE-crossed (adversarial follow-up to the auto-plane rule)
# ---------------------------------------------------------------------------
#
# Choosing purely by spread picks exactly the wrong component on a *driven*
# system.  A carried drive phase (Duffing's ``z' = omega``) advances
# monotonically, so over any window it has the widest interquartile range of any
# coordinate — and crosses its own median once, ever.  The rule's guarantee
# ("half the orbit lies on each side of the median") is a statement about being
# *reached*, not about being *returned to*, and a Poincaré map is a return map.
#
# Measured before this was fixed: 30 of 137 catalogue flows chose such a
# coordinate, and the resulting ``PoincareMap`` marched to ``max_time = 1e4``
# before raising ``ConvergenceError`` — a slow, unexplained failure landing on
# the beginner the auto plane exists for.


def test_auto_plane_refuses_a_monotone_drive_phase() -> None:
    """Duffing's widest-spread coordinate is its drive phase; the section is not.

    ``z`` sweeps 70 units of phase per probe window (the widest IQR in the
    system) and never comes back, so the choice must fall to a component that
    oscillates — and the map must then really find its crossings.
    """
    from tsdynamics.derived.poincare import auto_plane

    duffing = ts.systems.Duffing()
    axis, offset = auto_plane(duffing)
    assert axis != 2, "picked the monotone drive phase z (widest spread, never recrossed)"
    section = PoincareMap(duffing).run(5)
    assert section.y.shape == (5, 3)
    assert np.allclose(section.y[:, axis], offset, atol=1e-6)


def test_auto_plane_prefers_spread_among_the_components_it_accepts() -> None:
    """The spread rule still decides — it is applied *after* the recrossing filter.

    Rössler's ``x`` and ``y`` both oscillate; the wider one is chosen, which is
    the classical section.  (A test that only checked "some component" would
    pass on the narrowest one and lose the reason the rule exists.)
    """
    from tsdynamics.derived.poincare import _MIN_RECROSSINGS, _recrossings, auto_plane

    system = _rossler()
    axis, offset = auto_plane(system)
    orbit = system.copy().run(final_time=100.0, dt=0.01, transient=50.0, solver="rk4").y
    spread = np.percentile(orbit, 75.0, axis=0) - np.percentile(orbit, 25.0, axis=0)
    accepted = [
        i
        for i in range(orbit.shape[1])
        if _recrossings(orbit[:, i], float(np.median(orbit[:, i]))) >= _MIN_RECROSSINGS
    ]
    assert axis in accepted
    assert spread[axis] == pytest.approx(max(spread[i] for i in accepted), rel=0.2)
    assert _recrossings(orbit[:, axis], offset) >= _MIN_RECROSSINGS


def test_recrossings_counts_returns_not_visits() -> None:
    """The predicate itself: a ramp crosses once, an oscillation many times."""
    from tsdynamics.derived.poincare import _recrossings

    ramp = np.linspace(-1.0, 1.0, 500)
    assert _recrossings(ramp, 0.0) == 1
    wave = np.sin(np.linspace(0.1, 20.0 * np.pi + 0.1, 5000))  # ten full periods
    assert _recrossings(wave, 0.0) == 10
    # samples sitting exactly on the plane are not two crossings
    touch = np.array([-1.0, 0.0, -1.0, 0.0, -1.0])
    assert _recrossings(touch, 0.0) == 0


def test_auto_plane_says_so_when_nothing_recrosses() -> None:
    """A purely advancing record is refused with the reason, not with a bad section."""
    from tsdynamics.derived.poincare import auto_plane

    ramp = np.linspace(0.0, 100.0, 400)
    advancing = Trajectory(
        t=np.linspace(0.0, 100.0, 400),
        y=np.column_stack([ramp, 2.0 * ramp, 0.5 * ramp]),
        system=_rossler(),
        meta={},
    )
    with pytest.raises(InvalidParameterError) as excinfo:
        auto_plane(advancing)
    message = str(excinfo.value)
    assert "recrosses" in message
    assert "drive phase" in message
    assert "plane=" in message


def test_auto_plane_gives_a_slow_oscillator_a_longer_look() -> None:
    """A relaxation oscillator has few periods in the short window — and still gets one.

    FitzHughNagumo crosses its median twice in 50 post-transient units, which is
    below the acceptance count; refusing it would be a false negative, so the
    probe window is stretched once before giving up.
    """
    from tsdynamics.derived.poincare import _MIN_RECROSSINGS, _recrossings, auto_plane

    system = ts.systems.FitzHughNagumo()
    axis, offset = auto_plane(system)
    short = system.copy().run(final_time=100.0, dt=0.01, transient=50.0, solver="rk4").y
    assert _recrossings(short[:, axis], offset) < _MIN_RECROSSINGS  # the short look cannot decide
    long = system.copy().run(final_time=1000.0, dt=0.01, transient=500.0, solver="rk4").y
    assert _recrossings(long[:, axis], float(np.median(long[:, axis]))) >= _MIN_RECROSSINGS


def test_auto_plane_refuses_a_map_by_naming_the_family() -> None:
    """``PoincareMap(Henon())`` is a category mistake, and is answered as one.

    Adversarial follow-up to the auto-plane work: the probe used to simply *run*
    on whatever it was handed, so a discrete map reported the first thing the
    probe tripped over — ``dt is not a valid Henon.run() keyword`` — under a
    heading that read "every probe run of Henon failed or diverged … if it has no
    default_ic, the random start may be leaving the attractor's basin, so
    reinit(ic) it first".  Every clause of that is wrong about a map, and it
    sends the reader after an initial condition that was never the problem.
    """
    from tsdynamics.errors import InvalidParameterError

    with pytest.raises(InvalidParameterError) as excinfo:
        PoincareMap(ts.systems.Henon())
    message = str(excinfo.value)
    assert "discrete map" in message
    assert "no Poincaré section" in message
    assert "diverged" not in message and "default_ic" not in message


def test_auto_plane_refuses_a_stochastic_system_by_naming_the_family() -> None:
    """An SDE has no transversal to cross — so it is refused, not probed."""
    from tsdynamics.errors import InvalidParameterError

    with pytest.raises(InvalidParameterError) as excinfo:
        PoincareMap(ts.systems.OrnsteinUhlenbeck())
    message = str(excinfo.value)
    assert "stochastic" in message
    assert "differentiable" in message
    assert "rk4" not in message  # not the probe's solver complaint
