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
from tsdynamics.errors import InvalidParameterError
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
    named = ts.poincare_section(_rossler(), plane=("y", 0.0), n=40, dt=0.05)
    index = ts.poincare_section(_rossler(), plane=(1, 0.0), n=40, dt=0.05)
    assert np.array_equal(named.y, index.y)
    assert np.array_equal(named.t, index.t)
    assert named.meta["plane"] == (1, 0.0)


def test_direction_word_in_plane_sets_and_overrides_direction() -> None:
    """A third ``plane`` element is the crossing direction and beats ``direction=``."""
    down = ts.poincare_section(_rossler(), plane=("x", 0.0, "down"), n=30, dt=0.05)
    explicit = ts.poincare_section(_rossler(), plane=(0, 0.0), direction=-1, n=30, dt=0.05)
    assert down.meta["direction"] == -1
    assert np.array_equal(down.y, explicit.y)
    # the in-plane direction word wins over a conflicting direction= argument
    forced = ts.poincare_section(_rossler(), plane=("x", 0.0, "down"), direction=+1, n=30, dt=0.05)
    assert forced.meta["direction"] == -1


def test_poincare_map_accepts_named_plane_and_direction_word() -> None:
    """``PoincareMap`` resolves a named plane to the index form on ``.plane``."""
    pm = PoincareMap(_rossler(), plane=("y", 0.0, "down"))
    assert pm.plane == (1, 0.0)
    assert pm.direction == -1
    assert isinstance(pm.trajectory(20), PoincareSection)


def test_general_normal_plane_passes_through() -> None:
    """An arbitrary normal vector still works (and is not name-resolved)."""
    sec = ts.poincare_section(_rossler(), plane=([0.0, 1.0, 0.0], 0.0), n=20, dt=0.05)
    # normal is along +y, so the recorded crossings sit on y ≈ 0
    assert np.max(np.abs(sec.y[:, 1])) < 1e-6


# ---------------------------------------------------------------------------
# The PoincareSection result type (intent + result surface)
# ---------------------------------------------------------------------------


def test_returns_poincare_section_that_is_a_trajectory() -> None:
    sec = ts.poincare_section(_rossler(), plane=("y", 0.0), n=30, dt=0.05)
    assert isinstance(sec, PoincareSection)
    assert isinstance(sec, Trajectory)
    # trajectory affordances survive (named components, shapes)
    assert sec.y.shape == (30, 3)
    assert np.array_equal(sec["x"], sec.y[:, 0])


def test_section_carries_poincare_intent_and_spec() -> None:
    sec = ts.poincare_section(_rossler(), plane=("y", 0.0), n=40, dt=0.05)
    assert sec.meta["plot_kind"] == "poincare_section"
    spec = sec.to_plot_spec()
    assert spec.kind == PlotKind.POINCARE_SECTION
    assert spec.ndim == 2
    assert spec.aspect == "equal"
    assert spec.layers[0].kind == PlotKind.SCATTER


def test_section_summary_and_repr() -> None:
    sec = ts.poincare_section(_rossler(), plane=("y", 0.0, "up"), n=25, dt=0.05)
    summary = sec.summary()
    assert "PoincareSection" in summary
    assert "crossings = 25" in summary
    assert "up" in summary
    assert repr(sec) == "PoincareSection(crossings=25, dim=3)"


def test_section_to_dict_is_json_serializable() -> None:
    sec = ts.poincare_section(_rossler(), plane=("y", 0.0), n=20, dt=0.05)
    d = sec.to_dict()
    assert set(d) >= {"t", "y", "n_crossings", "plane", "direction", "meta"}
    assert d["n_crossings"] == 20
    assert d["plane"] == [1, 0.0]
    # the whole thing must round-trip through stdlib json (no numpy leaks)
    assert isinstance(json.loads(json.dumps(d))["y"], list)


def test_section_has_populated_provenance_meta() -> None:
    sec = ts.poincare_section(_rossler(), plane=("y", 0.0), n=15, dt=0.05)
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
        sec = ts.poincare_section(_rossler(), plane=("y", 0.0), n=15, dt=0.05)
        with pytest.raises(VisualizationNotInstalled):
            sec.plot()
    finally:
        registry.renderers.clear()
        for entry in saved:
            registry.renderers.register(entry.name, entry.obj, replace=True)


# ---------------------------------------------------------------------------
# Data path (a measured Trajectory) — same friendly spelling, same result type
# ---------------------------------------------------------------------------


def test_data_path_returns_named_section() -> None:
    traj = _rossler().integrate(final_time=80.0, dt=0.02)
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
    sec = ts.poincare_section(_rossler(), plane=(0, 0.0), n=20, skip_crossings=5, dt=0.05)
    assert isinstance(sec, PoincareSection)
    assert sec.y.shape == (20, 3)
    assert np.max(np.abs(sec.y[:, 0])) < 1e-6


# ---------------------------------------------------------------------------
# Error reporting (all value-naming InvalidParameterError, ⊂ ValueError)
# ---------------------------------------------------------------------------


def test_unknown_component_name_raises() -> None:
    with pytest.raises(InvalidParameterError, match="not a declared component"):
        ts.poincare_section(_rossler(), plane=("w", 0.0), n=5)


def test_unknown_direction_word_raises() -> None:
    with pytest.raises(InvalidParameterError, match="up.*down.*both"):
        ts.poincare_section(_rossler(), plane=("y", 0.0, "sideways"), n=5)


def test_malformed_plane_raises() -> None:
    with pytest.raises(InvalidParameterError, match="axis, offset"):
        ts.poincare_section(_rossler(), plane=(1,), n=5)


def test_plane_errors_are_value_errors() -> None:
    """``except ValueError`` still catches the new section errors."""
    assert issubclass(InvalidParameterError, ValueError)
    with pytest.raises(ValueError):
        PoincareMap(_rossler(), plane=("nope", 0.0))
