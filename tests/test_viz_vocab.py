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

v6 vocabulary surgery — why seven semantic kinds left
-----------------------------------------------------
Editing this gate is the deliberate act it exists to force, so the reasoning is
recorded here rather than in a commit message.  An AST scan of every
``PlotSpec(...)`` / ``pb.spec(...)`` construction outside ``viz/render`` (plus a
value-literal cross-check) found **eight** semantic kinds that no code path in the
library ever produced.  Seven were removed; one was kept.

Removed, with the reason each was already dead:

- ``power_spectrum``, ``spectrogram``, ``histogram_null``, ``feature_bars`` —
  these name the PSD, the spectrogram, the surrogate null distribution and the
  Hjorth/feature bar chart.  Their producers left the library in the v6 scope
  surgery (commit ``5d841149`` deleted ``transforms/`` / ``entropy/`` /
  ``surrogate/``), and ``CLAUDE.md`` now explicitly forbids re-adding generic
  time-series statistics here.  A vocabulary member for an analysis the project
  has ruled out of scope is a promise the library will never keep.
- ``complexity_curve`` — same batch; nothing has produced it since.
- ``trajectory_animation``, ``ensemble_animation`` — superseded by the orthogonal
  :class:`~tsdynamics.viz.spec.Animation` modifier (PR #463).  A spec of *any*
  kind animates by carrying an ``Animation``, so animation stopped being a kind;
  these two survived only as "kept so an old serialized spec round-trips", and a
  payload naming them has never been written by any released version that also
  had the ``Animation`` modifier.

**Kept:** ``bifurcation``.  The scan flags it as unproduced by a *default* path,
but it is genuinely reachable and genuinely drawn — ``OrbitDiagram.plot.bifurcation()``
routes ``kind="bifurcation"`` through ``to_plot_spec`` into ``pb.spec``, and the
dispatcher's ``"bifurcation_diagram"`` alias resolves to it.  Verified by
rendering: it returns a real ``Figure`` of the cascade.

Also kept, deliberately, though neither has a producer:

- the ``surface3d`` **mark** — the only 3-D mark all three drawing backends
  already implement end to end (~103 lines in mpl / plotly / three.js).  The plan
  is to give it a producer, not to delete the implementation.  No producer was
  added in this pass, so it stays a hand-buildable primitive.
- the ``histogram`` **mark** — ``_plotbuilder.histogram()`` has zero callers now
  that the surrogate null distribution is gone, but a *mark* is a drawing
  primitive rather than an analysis name: matplotlib and plotly both draw it, and
  a user hand-building a ``Layer`` can use it today.  A dead mark is a spare
  primitive; a dead semantic kind is a false advertisement.  That asymmetry is the
  line this surgery drew.

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
        "scaling_fit",
        "dimension_spectrum",
        "diagnostic_curve",
        "line_family",
        "ensemble_fan",
        "lyapunov_spectrum",
        "eigenvalue_plane",
        "fixed_points_overlay",
        "vector_field",
        "phase_portrait_field",
        "continuation",
        "categorical_bar",
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
    """The kinds the gap-fill batches name must all exist (forward guarantee).

    Trimmed in v6: ``spectrogram`` / ``feature_bars`` / ``complexity_curve`` were
    dropped from this list together with the enum members, because the analyses
    that would have filled those gaps are out of scope (see the module docstring).
    """
    required = {
        "dimension_spectrum",
        "eigenvalue_plane",
        "lyapunov_spectrum",
        "vector_field",
        "phase_portrait_field",
        "fixed_points_overlay",
        "ensemble_fan",
        "categorical_bar",
        "continuation",
    }
    values = {k.value for k in PlotKind}
    assert required <= values


def test_removed_kinds_stay_removed():
    """The seven kinds the v6 surgery deleted must not creep back unreviewed.

    The other half of the freeze: the membership guards above would also pass if
    someone re-added a member *and* re-listed it, so this names the seven
    explicitly.  Re-adding one means giving it a producer and editing this test —
    exactly the review the gate exists to force.
    """
    removed = {
        "power_spectrum",
        "spectrogram",
        "histogram_null",
        "feature_bars",
        "complexity_curve",
        "trajectory_animation",
        "ensemble_animation",
    }
    values = {k.value for k in PlotKind}
    assert removed.isdisjoint(values)
    for name in ("POWER_SPECTRUM", "SPECTROGRAM", "HISTOGRAM_NULL", "FEATURE_BARS"):
        assert not hasattr(PlotKind, name)


def test_every_semantic_kind_has_a_producer_or_a_recorded_exemption():
    """No semantic kind may exist purely because this gate says it exists.

    The invariant the surgery established, kept live: every semantic kind is
    either **produced** somewhere in the library (an AST scan for
    ``PlotKind.<NAME>`` / its string value outside ``viz/render`` and outside the
    enum's own definition site) or listed in the exemption table below with a
    reason.  Without this, a kind can be added, never wired to anything, and
    survive forever on the strength of a membership assertion — which is how the
    seven removed kinds lasted as long as they did.
    """
    import ast
    import pathlib

    import tsdynamics

    root = pathlib.Path(tsdynamics.__file__).parent
    by_value = {k.value: k.name for k in PlotKind}
    produced: set[str] = set()
    for path in root.rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        if rel.startswith("viz/render/") or rel == "viz/spec.py":
            continue  # a renderer *consumes* kinds; spec.py *defines* them
        tree = ast.parse(path.read_text(), str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "PlotKind"
                and node.attr in PlotKind.__members__
            ):
                produced.add(node.attr)
            elif isinstance(node, ast.Constant) and node.value in by_value:
                produced.add(by_value[node.value])

    # Kinds with no in-library producer, each for a recorded reason.  **Empty**
    # after the v6 surgery: all 25 surviving semantic kinds are produced.  The
    # table is the documented escape hatch — a kind that legitimately has no
    # producer goes here with its reason, in review, rather than silently.
    exempt: set[str] = set()
    orphans = {k.name for k in PlotKind.semantic_kinds()} - produced - exempt
    assert not orphans, (
        f"semantic kind(s) {sorted(orphans)} are produced by nothing in the library. "
        "Give them a producer, remove them, or add a recorded exemption here."
    )


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
    # ``COMPOSITE`` is the one kind whose *structure* is constrained: since v6 a
    # composite must carry at least one panel (``PlotSpec.__post_init__``), because
    # a panel-less composite rendered as a blank figure.  Give it one so the
    # round-trip still covers the kind — panels must survive to_dict/from_dict too.
    panels = (
        [PlotSpec(kind=PlotKind.TIME_SERIES, layers=[_minimal_layer(PlotKind.LINE)])]
        if kind is PlotKind.COMPOSITE
        else []
    )
    spec = PlotSpec(
        kind=kind,
        layers=[_minimal_layer(mark)],
        x=Axis(label="x"),
        y=Axis(label="y"),
        z=Axis(label="z") if ndim == 3 else None,
        ndim=ndim,
        panels=panels,
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


# ── every semantic kind must declare what its axes MEAN ───────────────────────


def test_every_semantic_kind_declares_a_frame() -> None:
    """A semantic kind without a ``_KIND_FRAME`` row silently overlays wrongly.

    Overlay legality is decided by :class:`~tsdynamics.viz._frames.Frame`, and a
    kind with no row falls through to ``_DEFAULT_KIND_FRAME`` — ``(SCALING, 1)``.
    That default is *plausible* for a scaling fit and wrong for everything else,
    and the failure is silent: the spec composes onto axes it has nothing to do
    with instead of raising.  So the frame table has to be exhaustive over the
    semantic kinds, and a new kind must not be able to skip it.

    Layer *marks* are deliberately exempt: a mark is a drawing primitive and
    carries no claim about what the axes mean, so it legitimately takes the
    default when someone hand-builds a spec whose kind is a bare mark.
    """
    from tsdynamics.viz._frames import _KIND_FRAME
    from tsdynamics.viz.spec import PlotKind

    # Two semantic kinds resolve their frame from the spec's own contents rather
    # than from a static table, so a row would be meaningless for them.
    frame_resolved_dynamically = {"spatial_field", "composite"}

    missing = sorted(
        k.value
        for k in PlotKind.semantic_kinds()
        if k.value not in _KIND_FRAME and k.value not in frame_resolved_dynamically
    )
    assert not missing, (
        f"semantic kinds with no _KIND_FRAME row: {missing}. Add a row to "
        "viz/_frames.py::_KIND_FRAME saying what the axes of this kind MEAN, or "
        "— if the frame genuinely depends on the spec's contents — handle it in "
        "frame_of() and add it to frame_resolved_dynamically here."
    )
