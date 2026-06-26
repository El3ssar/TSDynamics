"""Governance gate for the frozen PlotKind/mark vocabulary (stream VIZ-VOCAB).

The :class:`~tsdynamics.viz.spec.PlotKind` enum is a **closed contract**: adding a
renderer never needs a new kind, and adding a kind is a deliberate, reviewed
change (HITL maintainer sign-off).  This module is the gate that keeps the
vocabulary frozen and the schema backward-compatible:

1. The enum partitions exactly into the *semantic kinds* a whole
   :class:`~tsdynamics.viz.spec.PlotSpec` can be and the *layer marks* a single
   :class:`~tsdynamics.viz.spec.Layer` can draw — disjoint and exhaustive.
2. The exact membership of each set is pinned here (the
   ``EXPECTED_SEMANTIC_KINDS`` / ``EXPECTED_MARKS`` frozen lists), so a kind
   cannot be added or removed without editing this gate — the place a reviewer
   looks for a vocabulary change.
3. Every spec (across every kind, the new schema fields, and the new channels)
   round-trips **byte-identical** through ``to_dict`` / ``from_dict``, and old
   serialized dicts (without the new ``Axis.categories`` / ``Colorbar.cmap``…
   keys) still load — the schema additions are additive.

Engine-free, fast tier (imports only the backend-agnostic spec IR).
"""

from __future__ import annotations

import numpy as np
import pytest

from tsdynamics.viz.spec import Annotation, Axis, Colorbar, Layer, Legend, PlotKind, PlotSpec

# ---------------------------------------------------------------------------
# The frozen vocabulary (the reviewed contract — edit deliberately, with HITL
# sign-off; the membership guards below pin the exact set).
# ---------------------------------------------------------------------------

EXPECTED_SEMANTIC_KINDS: frozenset[str] = frozenset(
    {
        "time_series",
        "phase_portrait_2d",
        "phase_portrait_3d",
        "spacetime",
        # the spatial-field kind (stream VIZ-SPATIAL-FIELD): a spatially-extended
        # system's field at one instant, reshaped to its grid — a 1-D profile line
        # or a 2-D heatmap; an animation plays it over time (the field movie).
        "spatial_field",
        "composite",
        "bifurcation",
        "orbit_diagram",
        "cobweb",
        "return_map",
        "poincare_section",
        "basins_image",
        "recurrence_plot",
        "power_spectrum",
        "spectrogram",
        "scaling_fit",
        "dimension_spectrum",
        "diagnostic_curve",
        "complexity_curve",
        "line_family",
        "ensemble_fan",
        "histogram_null",
        "lyapunov_spectrum",
        "eigenvalue_plane",
        "fixed_points_overlay",
        "vector_field",
        "phase_portrait_field",
        "continuation",
        "categorical_bar",
        "feature_bars",
        "trajectory_animation",
        "ensemble_animation",
    }
)

EXPECTED_MARKS: frozenset[str] = frozenset(
    {
        "line",
        "line3d",
        "scatter",
        "markers",
        "image",
        "quiver",
        "surface3d",
        "histogram",
        "bar",
        "area",
        "errorbar",
    }
)


# ---------------------------------------------------------------------------
# 1 + 2 — the closed vocabulary is exactly the frozen set and partitions cleanly
# ---------------------------------------------------------------------------


def test_semantic_kinds_membership_is_frozen():
    """The semantic-kind set is exactly the pinned contract (no drift)."""
    assert {k.value for k in PlotKind.semantic_kinds()} == EXPECTED_SEMANTIC_KINDS


def test_layer_marks_membership_is_frozen():
    """The layer-mark set is exactly the pinned contract (no drift)."""
    assert {k.value for k in PlotKind.layer_marks()} == EXPECTED_MARKS


def test_vocabulary_partitions_the_enum():
    """Semantic kinds and marks are disjoint and together exhaust the enum."""
    semantic = PlotKind.semantic_kinds()
    marks = PlotKind.layer_marks()
    assert semantic.isdisjoint(marks)
    assert semantic | marks == set(PlotKind)


def test_no_new_member_escapes_a_set():
    """Every enum member is classified as exactly one of semantic / mark.

    Guards against adding a member to the enum but forgetting to place it in a
    governance set (it would be drawable nowhere and slip the frozen contract).
    """
    for kind in PlotKind:
        assert PlotKind.is_semantic(kind) ^ PlotKind.is_mark(kind), kind


def test_gapfill_required_kinds_present():
    """The kinds the gap-fill batches name must all exist (forward guarantee)."""
    required = {
        "dimension_spectrum",
        "eigenvalue_plane",
        "lyapunov_spectrum",
        "vector_field",
        "phase_portrait_field",
        "spectrogram",
        "fixed_points_overlay",
        "ensemble_fan",
        "categorical_bar",
        "feature_bars",
        "complexity_curve",
        "continuation",
    }
    values = {k.value for k in PlotKind}
    assert required <= values


# ---------------------------------------------------------------------------
# 3 — every spec round-trips byte-identical; schema additions are additive
# ---------------------------------------------------------------------------


def _minimal_layer(mark: PlotKind) -> Layer:
    """A tiny valid layer for ``mark`` (just enough channel data to round-trip)."""
    x = np.linspace(0.0, 1.0, 4)
    base = {"x": x, "y": x[::-1]}
    if mark in (PlotKind.LINE3D, PlotKind.SURFACE3D):
        base["z"] = x
    if mark == PlotKind.QUIVER:
        base |= {"u": x, "v": x}
    if mark == PlotKind.IMAGE:
        base = {"x": x, "y": x, "c": np.outer(x, x)}
    return Layer(mark, base, label=mark.value)


@pytest.mark.parametrize("kind", sorted(PlotKind, key=lambda k: k.value), ids=lambda k: k.value)
def test_every_kind_spec_round_trips_byte_identical(kind: PlotKind):
    """A spec built with each kind round-trips byte-identical through to_dict."""
    mark = kind if PlotKind.is_mark(kind) else PlotKind.LINE
    ndim = 3 if kind in (PlotKind.PHASE_PORTRAIT_3D,) else 2
    spec = PlotSpec(
        kind=kind,
        layers=[_minimal_layer(mark)],
        x=Axis(label="x"),
        y=Axis(label="y"),
        z=Axis(label="z") if ndim == 3 else None,
        ndim=ndim,
    )
    once = spec.to_dict()
    twice = PlotSpec.from_dict(once).to_dict()
    assert once == twice


def test_full_featured_spec_round_trips_byte_identical():
    """A spec exercising every new schema field + channel round-trips losslessly."""
    cat = Axis(label="basin", scale="categorical", categories=["A", "B", "C"])
    layer = Layer(
        PlotKind.BAR,
        {
            "x": np.array([0.0, 1.0, 2.0]),
            "y": np.array([0.6, 0.3, 0.1]),
            "cat": np.array([0, 1, 2]),
            "err": np.array([0.05, 0.04, 0.02]),
            "size": np.array([10.0, 20.0, 30.0]),
            "lo": np.array([0.55, 0.26, 0.08]),
            "hi": np.array([0.65, 0.34, 0.12]),
            "c": np.array([1.0, 2.0, 3.0]),
        },
        label="fractions",
        style={"color": "tab:blue", "alpha": 0.8},
    )
    spec = PlotSpec(
        kind=PlotKind.CATEGORICAL_BAR,
        layers=[layer],
        x=cat,
        y=Axis(label="fraction"),
        clim=(1.0, 3.0),
        colorbar=Colorbar(label="id", cmap="tab20", norm="log", discrete=True),
        legend=Legend(show=True, location="upper right", title="basins"),
        annotations=[Annotation(kind="vline", x=1.5, text="tip")],
        meta={"system": "Duffing"},
    )
    once = spec.to_dict()
    twice = PlotSpec.from_dict(once).to_dict()
    assert once == twice
    # The new fields actually serialized (not silently dropped).
    assert once["x"]["scale"] == "categorical"
    assert once["x"]["categories"] == ["A", "B", "C"]
    assert once["colorbar"]["cmap"] == "tab20"
    assert once["colorbar"]["norm"] == "log"
    assert once["colorbar"]["discrete"] is True
    assert set(once["layers"][0]["data"]) >= {"lo", "hi", "err", "cat", "size", "c"}


def test_old_serialized_dict_without_new_keys_still_loads():
    """A pre-VIZ-VOCAB dict (no categories / cmap / norm / discrete) loads with defaults.

    The schema additions are additive: ``from_dict`` tolerates the older shape so
    a cached spec keeps deserializing.
    """
    legacy = {
        "kind": "recurrence_plot",
        "layers": [{"kind": "image", "data": {"c": [[1.0, 0.0], [0.0, 1.0]]}, "style": {}}],
        "x": {"label": "i", "scale": "linear", "limits": None, "ticks": None, "tickformat": None},
        "y": {"label": "j", "scale": "linear", "limits": None, "ticks": None, "tickformat": None},
        "z": None,
        "clim": [0.0, 1.0],
        "colorbar": {
            "label": "R",
            "location": "right",
            "ticks": None,
            "tickformat": None,
            "show": True,
        },
        "legend": None,
        "title": "R",
        "ndim": 2,
        "aspect": "equal",
        "annotations": [],
        "meta": {},
    }
    spec = PlotSpec.from_dict(legacy)
    assert spec.kind is PlotKind.RECURRENCE_PLOT
    # The new fields default cleanly when absent in the source dict.
    assert spec.x.categories is None
    assert spec.colorbar is not None
    assert spec.colorbar.cmap is None
    assert spec.colorbar.norm is None
    assert spec.colorbar.discrete is False
