"""The visualization layer's visibility contract — contract §11 (v6 round 9).

The owner's complaint was *"people overwhelmed with stuff they won't use directly
anyway"*, and the chair's answer was a rule with two halves that have to be tested
together, because each alone is a lie:

1. **the listing shrank** — ``ts.viz.transforms`` 22 → 5, ``Geometry`` 19 → 4,
   ``PlotTransform`` 22 → 11, ``Primitive`` 10 → 4, ``Frame`` 9 → 4, ``Theme``
   19 → 16, ``dir(p.title)`` 47 → 0, and 405 module-level leaks → 0;
2. **nothing stopped resolving** — every one of those names is still bound, still
   importable, still callable, still doing its job.

Half 2 is guardrail **G1** (*"simple must NOT mean that it hinders the
customizability of the library"*), and it is the half a hiding sweep gets wrong.
So this module does not merely assert the new counts: for every name it claims to
have hidden, it **reaches the name and uses it**.  If a future edit turns a
``__dir__`` curation into a real removal, the test that catches it is here.

Section 3 pins the **inherited-builtin** listings.  Subclassing ``str`` or
``dict`` is how a value stays a drop-in replacement for the plain thing it
replaces, and the tab surface pays for it in full; the chair ruled the defect
four times elsewhere, and the completion sweep found three more here, each
reached from a *listed* name: ``p.kind`` 89 → 42 and ``g.frame.space`` 60 → 13
(two ``StrEnum``\\ s donating 47 text verbs apiece) and
``ts.viz.compatibility()`` 12 → 1 (a ``dict`` donating ``keys``/``values``/… to a
record whose one verb is ``rows()``).  Measured first: **zero** call sites invoke
any of them, and every advertised operation on all three is a dunder.

Section 4 pins the **enforcement clause**: a module that declares ``__all__``
must define ``__dir__`` returning ``sorted(__all__)``.  ``__all__`` governs only
``from X import *``; it has zero effect on ``dir()``/TAB, so a module that
declares one and stops has a documented surface and an actual surface that
disagree — which is how 32 viz modules came to offer 405 names no ``__all__``
claimed.  The sweep is over the whole package, so a **new** module cannot regrow
the leak.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys

import numpy as np
import pytest

pytest.importorskip("matplotlib")

import tsdynamics as ts  # noqa: E402
import tsdynamics.viz as viz  # noqa: E402


def _public(obj: object) -> list[str]:
    """The tab surface: sorted, non-underscored names of ``dir(obj)``."""
    return sorted(n for n in dir(obj) if not n.startswith("_"))


@pytest.fixture(scope="module")
def traj():
    """A tiny Lorenz orbit — every listing below is measured on a real object."""
    return ts.systems.Lorenz().run(final_time=2.0, dt=0.05, ic=[1.0, 1.0, 1.0])


@pytest.fixture
def plot(traj):
    """One rendered-able :class:`Plot`, closed down after the test."""
    import matplotlib.pyplot as plt

    p = ts.plot(traj)
    yield p
    plt.close("all")


# ---------------------------------------------------------------------------
# 1. The namespaces
# ---------------------------------------------------------------------------

#: ``ts.viz.transforms.<TAB>`` — the four shared registry verbs plus ``allow``,
#: the fifth that lives only here.
_TRANSFORMS_SURFACE = ["allow", "find", "get", "names", "register"]

#: The seventeen names that left that listing in v6 round 9.  Every one is still
#: bound; the test below imports all seventeen by name.
_TRANSFORMS_DEMOTED = [
    "FrameSpace",
    "Geometry",
    "Part",
    "PlotTransform",
    "Presentation",
    "Primitive",
    "T",
    "compatibility",
    "draw",
    "geometry",
    "get_primitive",
    "make_frame",
    "plot",
    "plot_transform",
    "primitive_names",
    "register_primitive",
    "transforms",
]


def test_ts_viz_transforms_is_five_verbs():
    """22 -> 5.  The one viz namespace the C5 package sweep never covered."""
    from tsdynamics.viz import transforms as tr

    assert sorted(tr.__all__) == _TRANSFORMS_SURFACE
    assert dir(tr) == _TRANSFORMS_SURFACE
    for name in _TRANSFORMS_SURFACE:
        assert callable(getattr(tr, name)), name


def test_every_demoted_transforms_name_still_resolves_and_still_works():
    """G1: the seventeen are a *discovery* change.  Reach each one and use it."""
    from tsdynamics.viz import transforms as tr

    for name in _TRANSFORMS_DEMOTED:
        assert hasattr(tr, name), f"{name} stopped resolving — that is a removal, not a hide"
    # ...and the from-import spelling, which is how the in-tree code reaches them.
    mod = importlib.import_module("tsdynamics.viz.transforms")
    for name in _TRANSFORMS_DEMOTED:
        assert getattr(mod, name) is not None

    # The reason six of them are demoted rather than kept: they ARE the ts.viz
    # spelling of the same object, measured, not asserted by eye.
    for name in ("plot", "draw", "geometry", "compatibility", "make_frame", "T"):
        assert getattr(tr, name) is getattr(ts.viz, name), name
    # ...and `plot_transform` is `register` under a second name (C3).
    assert tr.plot_transform is tr.register


def test_ts_viz_promotes_the_warning_the_docs_tell_you_to_catch():
    """``VisualizationDegraded`` is typeable now (13 -> 14) — the G1 counterweight."""
    assert "VisualizationDegraded" in ts.viz.__all__
    assert "VisualizationDegraded" in dir(ts.viz)
    from tsdynamics.viz import VisualizationDegraded

    assert VisualizationDegraded is ts.viz.VisualizationDegraded
    assert issubclass(VisualizationDegraded, UserWarning)
    # It is the same class the internal address holds — one object, one warning.
    from tsdynamics.viz.render import VisualizationDegraded as FromTheInternalAddress

    assert VisualizationDegraded is FromTheInternalAddress


def test_ts_viz_spec_promotes_the_two_types_a_renderer_author_must_declare():
    """Writing a renderer is a declared plugin door; both types were unreachable."""
    spec = ts.viz.spec
    assert "RendererCapabilities" in spec.__all__
    assert "RenderResult" in spec.__all__
    caps = spec.RendererCapabilities(name="mine", supports_3d=True, writes_static=frozenset({".x"}))
    assert caps.supports_3d and ".x" in caps.writes
    result = spec.RenderResult(backend="mine", payload={"ok": True})
    assert result.backend == "mine"


@pytest.mark.parametrize("name", ["SCHEMA_VERSION", "to_dict_envelope", "from_dict_envelope"])
def test_the_demoted_envelope_names_still_resolve(name):
    """Off ``ts.viz.spec.__all__``, still reachable — a listing edit, nothing more."""
    assert name not in ts.viz.spec.__all__
    assert getattr(ts.viz.spec, name) is not None
    assert getattr(ts.viz, name) is not None


def test_ts_viz_binds_nothing_it_does_not_declare():
    """The ``_INTERNAL_NAMES`` table is the truth, or the next sweep is mis-informed.

    It used to be decorative: eleven names were bound on ``ts.viz`` and named by
    neither ``__all__`` nor the table, so ``from tsdynamics.viz import os``
    worked and nobody had decided that.
    """
    declared = set(viz.__all__) | set(viz._INTERNAL_NAMES)
    bound = {n for n in vars(viz) if not n.startswith("_")}
    assert bound <= declared, f"bound on ts.viz but declared nowhere: {sorted(bound - declared)}"


@pytest.mark.parametrize("name", ["os", "Any", "Sequence", "TYPE_CHECKING"])
def test_the_accidental_imports_stop_resolving(name):
    """``ts.viz.os`` was an import, not a decision.  Underscored now."""
    assert not hasattr(viz, name)


# ---------------------------------------------------------------------------
# 2. The records — what each one lists, and that the rest still works
# ---------------------------------------------------------------------------

#: ``dir(p)`` — 41 names (39 + the two *restored* fields).  Exact and sorted:
#: this is the listing the contract fixes, so a builder who produces a different
#: one has failed.  Nothing was hidden here; the overwhelm was never on ``Plot``.
_PLOT_SURFACE = [
    "add", "animate", "animation", "ax", "axes", "background", "camera", "clock",
    "colorize", "fig", "font", "from_dict", "gridlines", "head", "hline",
    "is_animated", "is_composite", "kind", "layers", "layout", "limits", "meta",
    "palette", "panels", "recolor", "relabel", "render", "rescale", "save",
    "show", "size", "span", "style", "text", "theme", "ticks", "title",
    "to_dict", "to_json", "trail", "vline",
]  # fmt: skip


def test_a_plot_lists_forty_one_names_including_the_two_restored_ones(plot):
    """``layout`` and ``animation`` are read by *executed* documentation lines."""
    assert _public(plot) == _PLOT_SURFACE
    assert "layout" in _PLOT_SURFACE and "animation" in _PLOT_SURFACE


def test_the_restored_fields_are_the_ones_documentation_reads(traj):
    """``grid.layout.mode`` and ``b.animation.fps`` — the two doctest-executed reads."""
    import matplotlib.pyplot as plt

    grid = ts.viz.grid(ts.plot(traj), ts.plot(traj), ts.plot(traj), cols=2)
    assert grid.layout.mode == "grid"
    movie = ts.plot(traj, animate={"fps": 24})
    assert movie.animation.fps == 24
    plt.close("all")


def test_the_escape_hatches_are_documented_not_merely_present(plot):
    """G1 names ``Plot.ax`` by name; a route with no docstring is not a route."""
    from tsdynamics.viz.spec import Plot

    for name in ("fig", "ax", "axes"):
        doc = getattr(Plot, name).__doc__ or ""
        assert doc.strip(), f"Plot.{name} has no docstring"
    # `layers` / `panels` are dataclass fields, so their documentation is the
    # class docstring — which must say what they are FOR, not merely what they are.
    klass_doc = " ".join((Plot.__doc__ or "").split())
    assert "the route to one curve" in klass_doc
    assert "the route to one panel" in klass_doc
    # ...and they still work.
    assert [lyr.label for lyr in plot.layers] == [None]
    assert plot.panels == []


def test_the_title_sheds_the_str_methods_but_is_still_a_string(traj):
    """47 inherited ``str`` methods leave ``dir(p.title)``; every one still runs."""
    import matplotlib.pyplot as plt

    p = ts.plot(traj, title="Lorenz")
    assert _public(p.title) == []
    assert p.title == "Lorenz"
    assert p.title.upper() == "LORENZ"  # the hidden methods still work
    assert f"{p.title:>10}" == "    Lorenz"
    assert "".join(sorted(p.title)) == "".join(sorted("Lorenz"))
    # ...and the read/call duality the subclass exists for is untouched.
    with pytest.raises(ts.errors.InvalidInputError, match="relabel"):
        p.title("nope")
    plt.close("all")


def test_a_geometry_lists_its_five_documented_doors(traj):
    """19 -> 5: ``g["x"]`` / ``.parts`` / ``.frame`` / ``.primitives`` / ``.meta``.

    ``parts`` is listed because the class **hands it back**: two of ``Geometry``'s
    own error messages end in *"iterate g.parts"*, and a blind user who opened
    the arrays escape hatch specifically to reach the arrays could not
    tab-complete the name those errors name.  A remedy a library prints must be
    a name the reader can then find.
    """
    g = ts.viz.geometry(traj, "phase_portrait")
    assert dir(g) == ["__getitem__", "frame", "meta", "parts", "primitives"]
    assert g["x"].shape == traj.y[:, 0].shape
    assert g.frame.space is not None
    assert g.primitives is None or isinstance(g.primitives, frozenset)
    assert isinstance(dict(g.meta), dict)
    assert len(g.parts) >= 1


def test_the_geometry_errors_name_a_door_that_is_listed(traj):
    """Every ``g.<name>`` a ``Geometry`` error tells you to use is in ``dir(g)``."""
    import re

    from tsdynamics.viz.transforms import _base

    named = set(re.findall(r"\bg\.([a-z_]+)\b", _base.Geometry.__doc__ or ""))
    for source in (_base.Geometry._stacked, _base.Geometry.__array__):
        named |= set(re.findall(r"\bg\.([a-z_]+)\b", source.__doc__ or ""))
        named |= set(re.findall(r"iterate g\.([a-z_]+)", inspect.getsource(source)))
    g = ts.viz.geometry(traj, "phase_portrait")
    assert named, "the probe found no g.<name> remedy to check"
    assert named <= set(dir(g)), sorted(named - set(dir(g)))


_GEOMETRY_HIDDEN = [
    "aspect", "axes", "axis_labels", "axis_limits", "axis_scales", "channel_names",
    "channels", "chosen_primitive", "clim", "color_label", "kind", "legend",
    "space", "title", "transform",
]  # fmt: skip


@pytest.mark.parametrize("name", _GEOMETRY_HIDDEN)
def test_every_hidden_geometry_name_still_resolves(name, traj):
    """G1 again: the lowering reads all sixteen on every plot this library draws."""
    g = ts.viz.geometry(traj, "phase_portrait")
    assert name not in dir(g)
    getattr(g, name)  # raises if the hide became a removal


def test_geometry_iteration_and_length_survive_the_curated_dir(traj):
    """``__dir__`` is not ``__getattr__``: the protocol dunders are untouched."""
    g = ts.viz.geometry(traj, "time_series", components=["x", "y"])
    assert len(g) == len(list(g)) >= 1
    assert g[0] is g.parts[0]


def test_the_renamed_geometry_field_answers_by_name(traj):
    """``primitive`` was one letter from ``primitives`` and meant the opposite."""
    g = ts.viz.geometry(traj, "phase_portrait")
    with pytest.raises(AttributeError, match="chosen_primitive"):
        _ = g.primitive
    assert g.chosen_primitive in (None, "line", "line3d", "points", "points3d")


def test_the_renamed_geometry_keyword_answers_by_name():
    """A transform author mid-migration gets a sentence, not a raw ``TypeError``."""
    from tsdynamics.viz.spec import FrameSpace, Geometry, make_frame

    frame = make_frame(FrameSpace.STATE2, ("x", "y"))
    channels = {"x": np.arange(4.0), "y": np.arange(4.0)}
    with pytest.raises(ts.errors.InvalidParameterError, match="chosen_primitive"):
        Geometry("mine", frame, channels=channels, primitive="line")
    built = Geometry("mine", frame, channels=channels, chosen_primitive="line")
    assert built.chosen_primitive == "line"


#: ``dir(ts.viz.transforms.get(name))`` — the eleven declaration names.
_TRANSFORM_RECORD_SURFACE = [
    "aliases", "available", "default_primitive", "doc", "frame", "kind", "name",
    "primitives", "requires", "source", "subjects",
]  # fmt: skip

_TRANSFORM_RECORD_HIDDEN = [
    "accepts_subject", "analysis", "compute", "describe_primitives", "example",
    "labels", "ndim", "presentation", "role", "shape_dependent",
]  # fmt: skip


def test_a_transform_record_lists_its_declaration():
    """22 -> 11: what the row SAYS, not how the library drives it."""
    record = ts.viz.transforms.get("phase_portrait")
    assert _public(record) == _TRANSFORM_RECORD_SURFACE


@pytest.mark.parametrize("name", _TRANSFORM_RECORD_HIDDEN)
def test_every_hidden_transform_record_name_still_resolves(name):
    """Ten hidden names, ten still-working attributes — swept over every record."""
    for record in map(ts.viz.transforms.get, ts.viz.transforms.names()):
        assert name not in dir(record), f"{record.name}.{name}"
        getattr(record, name)


def test_a_primitive_lists_the_four_names_it_is_asked_about():
    """10 -> 4.  ``primitives.get("line").requires`` is the advertised read."""
    line = ts.viz.primitives.get("line")
    assert _public(line) == ["doc", "marks", "name", "requires"]
    assert "x" in line.requires
    for name in ("accepts_frame", "build", "consumes", "emits_frame", "frames", "options"):
        assert name not in dir(line)
        getattr(line, name)


def test_a_frame_lists_what_it_says_not_the_overlay_algebra(traj):
    """9 -> 4.  ``compatible_with`` / ``merge`` are run for you by every ``a + b``."""
    frame = ts.viz.geometry(traj, "phase_portrait").frame
    assert _public(frame) == ["axes", "describe", "ndim", "space"]
    for name in ("compatible_with", "from_dict", "is_free", "merge", "to_dict"):
        assert name not in dir(frame)
        getattr(frame, name)
    assert frame.compatible_with(frame) is True  # ...and it still answers


def test_a_theme_lists_all_sixteen_fields_and_no_helpers():
    """19 -> 16.  ``styling.md`` reads five fields directly, so every field stays."""
    theme = ts.viz.themes.get("default")
    listed = _public(theme)
    assert len(listed) == 16
    for field in ("palette", "font_family", "title_size", "background", "name"):
        assert field in listed
    for helper in ("merged", "to_dict", "from_dict"):
        assert helper not in listed
        getattr(theme, helper)
    assert theme.merged(name="mine").name == "mine"  # ...still the derive engine


def test_the_taught_derive_route_needs_no_hidden_helper():
    """Hiding ``Theme.merged`` costs nothing because ``themes.register`` derives."""
    dark = ts.viz.themes.get("dark")
    ts.viz.themes.register("_visibility_probe", dark, title_size=31.0)
    try:
        derived = ts.viz.themes.get("_visibility_probe")
        assert derived.title_size == 31.0
        assert derived.background == ts.viz.themes.get("dark").background
    finally:
        from tsdynamics.viz.style import THEMES

        THEMES.pop("_visibility_probe", None)


def test_a_style_key_lists_the_vocabulary_not_the_validator():
    """5 -> 4.  ``validate`` is what ``normalize_style`` runs, not part of the vocabulary."""
    key = ts.viz.styles.get("lw")
    assert _public(key) == ["aliases", "doc", "honored_by", "name"]
    assert "validate" not in dir(key)
    assert key.validate(2.0) == 2.0


def test_renderer_capabilities_lists_the_declaration_not_the_dispatch():
    """The three predicates are the dispatcher's questions of the declaration."""
    caps = ts.viz.spec.RendererCapabilities(
        name="probe", writes_static=frozenset({".png"}), supports_3d=True
    )
    for predicate in ("can_render", "can_render_spec", "can_save"):
        assert predicate not in dir(caps)
    assert caps.can_save(".png") is True  # ...still the save contract
    assert caps.can_render("line") is True
    assert "writes_static" in dir(caps) and "render_kwargs" in dir(caps)


#: The IR round-trip plumbing hidden on every noun that carries it, and the noun
#: to probe it on.  ``Plot`` is deliberately absent: ``p.to_dict`` / ``p.to_json``
#: / ``ts.viz.load`` **are** the round trip a user drives.
_IR_HIDDEN = [
    ("Axis", "to_dict"),
    ("Axis", "from_dict"),
    ("Annotation", "to_dict"),
    ("Annotation", "from_mapping"),
    ("Colorbar", "coerce"),
    ("Legend", "coerce"),
    ("Layout", "grid"),
    ("Layout", "to_dict"),
    ("Animation", "head_indices"),
    ("Animation", "playback_seconds"),
    ("Animation", "DEFAULT_FRAMES"),
    ("Layer", "to_dict"),
]


@pytest.mark.parametrize(("noun", "name"), _IR_HIDDEN)
def test_the_ir_plumbing_is_hidden_and_still_public(noun, name):
    """Twelve names off twelve listings; each still resolves on an instance."""
    from tsdynamics.viz.spec import (
        Animation,
        Annotation,
        Axis,
        Colorbar,
        Layer,
        Layout,
        Legend,
        PlotKind,
    )

    instance = {
        "Axis": Axis(),
        "Annotation": Annotation(kind="vline", x=1.0),
        "Colorbar": Colorbar(),
        "Legend": Legend(),
        "Layout": Layout(),
        "Animation": Animation(),
        "Layer": Layer(kind=PlotKind.LINE),
    }[noun]
    assert name not in dir(instance)
    assert getattr(instance, name) is not None


def test_the_two_animation_sums_documentation_runs_stay_listed():
    """``frame_count`` and ``tail_samples`` are executed on three ``animation.md`` lines."""
    from tsdynamics.viz.spec import Animation

    anim = Animation(fps=24, duration=2.0)
    assert "frame_count" in dir(anim)
    assert "tail_samples" in dir(anim)
    assert anim.frame_count(1000) == 48


def test_a_layer_keeps_all_five_fields():
    """``p.layers[0].data[...]`` and ``[lyr.label for lyr in ...]`` are runnable doc lines."""
    from tsdynamics.viz.spec import Layer, PlotKind

    layer = Layer(kind=PlotKind.LINE, data={"x": np.arange(3.0)}, label="orbit")
    assert _public(layer) == ["data", "kind", "label", "style", "transform"]


def test_the_json_round_trip_the_user_drives_is_untouched(plot, tmp_path):
    """Hiding a noun's ``to_dict`` cannot break the round trip, because ``Plot`` walks it."""
    path = tmp_path / "p.json"
    plot.save(str(path))
    back = ts.viz.load(str(path))
    assert back.kind == plot.kind
    assert len(back.layers) == len(plot.layers)


# ---------------------------------------------------------------------------
# 3. Inherited noise — the three values that subclass a builtin
# ---------------------------------------------------------------------------
#
# Subclassing a builtin is how a value stays a drop-in replacement for the plain
# thing it replaces, and the tab surface pays for it in full.  The chair ruled
# this defect four times (``CountResult``'s 11 ``int`` members, ``ParamSet``'s
# mutators, ``count``/``index`` on the two records, and ``dir(p.title)``'s 47
# ``str`` methods); the sweep that followed found three more in viz, all of them
# reached from a *listed* name:
#
#   ``p.kind``              89 -> 42   (``PlotKind``, a ``StrEnum``)
#   ``g.frame.space``       60 -> 13   (``FrameSpace``, a ``StrEnum``)
#   ``ts.viz.compatibility()``  12 -> 1    (``CompatibilityMatrix``, a ``dict``)
#
# Each is hidden from ``__dir__`` only.  The tests below pin the listing *and*
# call every hidden member, because the whole risk of this kind of edit is that
# a later refactor reads "hidden" as "removable".


def test_a_plot_kind_lists_the_vocabulary_not_the_string_methods():
    """``p.kind.<TAB>`` is the closed vocabulary, not 47 ways to manipulate text."""
    from tsdynamics.viz.spec import PlotKind

    listing = _public(PlotKind.TIME_SERIES)
    members = {m.name for m in PlotKind}
    # The 36 members are the point: tabbing a kind is how you discover the set.
    assert members <= set(listing)
    # What remains beside them is the enum API and the four governance verbs.
    assert sorted(set(listing) - members) == [
        "is_mark",
        "is_semantic",
        "layer_marks",
        "name",
        "semantic_kinds",
        "value",
    ]
    assert len(listing) == len(members) + 6 == 42


def test_a_frame_space_lists_the_vocabulary_not_the_string_methods():
    """The twin ruling, on the other ``StrEnum`` a user can reach."""
    from tsdynamics.viz._frames import FrameSpace

    listing = _public(FrameSpace.STATE2)
    members = {m.name for m in FrameSpace}
    assert sorted(set(listing) - members) == ["name", "value"]
    assert len(listing) == len(members) + 2 == 13


@pytest.mark.parametrize("value_name", ["PlotKind.TIME_SERIES", "FrameSpace.STATE2"])
def test_the_str_enums_are_still_strings_in_every_way_that_matters(value_name):
    """G1: hiding ``str``'s methods must not touch what being a ``str`` buys."""
    import json

    from tsdynamics.viz._frames import FrameSpace
    from tsdynamics.viz.spec import PlotKind

    value = {"PlotKind.TIME_SERIES": PlotKind.TIME_SERIES, "FrameSpace.STATE2": FrameSpace.STATE2}[
        value_name
    ]
    # value equality, serialization and formatting — none of them read dir()
    assert value == value.value
    assert isinstance(value, str)
    assert json.dumps(value) == f'"{value.value}"'
    assert f"{value}" == value.value
    # ...and every hidden method is still there and still correct.
    for method in ("upper", "lower", "title", "capitalize", "casefold", "strip"):
        assert getattr(value, method)() == getattr(value.value, method)()
    assert value.split("_") == value.value.split("_")
    assert value.startswith(value.value[:2])
    assert value.replace("_", "-") == value.value.replace("_", "-")
    assert value.zfill(30) == value.value.zfill(30)


def test_the_compatibility_matrix_lists_its_one_verb():
    """12 -> 1: ``rows()``.  The programmable half the docstring advertises is dunders."""
    matrix = ts.viz.compatibility()
    assert _public(matrix) == ["rows"]
    # Everything the class docstring calls "programmable" still works...
    assert matrix["phase_portrait"]
    assert "psd" in matrix
    assert len(matrix) == len(matrix.rows()) > 0
    assert isinstance(matrix, dict)
    assert sorted(matrix)[0] == min(matrix)
    # ...and the repr is still the answer.
    assert "FROM DATA" in repr(matrix)


def test_every_hidden_dict_method_on_the_matrix_still_resolves():
    """G1: hidden is not removed — all eleven are callable and correct."""
    matrix = ts.viz.compatibility()
    assert sorted(matrix.keys()) == sorted(ts.viz.transforms.names())  # the 2nd spelling
    assert matrix.get("psd") == matrix["psd"]
    assert len(matrix.values()) == len(matrix.items()) == len(matrix)
    assert matrix.copy() == matrix
    for method in ("pop", "popitem", "clear", "update", "setdefault", "fromkeys"):
        assert callable(getattr(matrix, method))


def test_no_viz_value_regrows_an_inherited_builtin_listing():
    """The sweep that found these three, frozen so a fourth cannot land unnoticed.

    A new ``str``/``dict``/``list``/``int`` subclass on the viz surface is a
    perfectly good design; shipping one whose listing is mostly the builtin's is
    the thing this catches.
    """
    from tsdynamics.viz._frames import FrameSpace
    from tsdynamics.viz.spec import PlotKind

    subjects = {
        "PlotKind": PlotKind.TIME_SERIES,
        "FrameSpace": FrameSpace.STATE2,
        "CompatibilityMatrix": ts.viz.compatibility(),
        "Plot.title": ts.plot(np.linspace(0, 1, 8), title="t").title,
    }
    for label, value in subjects.items():
        base = next(b for b in (str, dict, list, int) if isinstance(value, b))
        leaked = [n for n in _public(value) if hasattr(base, n)]
        assert not leaked, f"{label} leaks {len(leaked)} inherited {base.__name__} names: {leaked}"


# ---------------------------------------------------------------------------
# 4. The enforcement clause — a module that declares __all__ defines __dir__
# ---------------------------------------------------------------------------


def _viz_modules():
    """Import every ``tsdynamics.viz`` submodule; skip the ones this env cannot."""
    found = ["tsdynamics.viz"]
    for info in pkgutil.walk_packages(viz.__path__, prefix="tsdynamics.viz."):
        try:
            importlib.import_module(info.name)
        except ImportError:  # pragma: no cover - an optional backend is absent
            continue
        found.append(info.name)
    return [sys.modules[name] for name in sorted(found)]


def test_every_viz_module_that_declares_all_also_defines_dir():
    """The §11.1 enforcement clause, swept — so a NEW module cannot regrow the leak.

    ``__all__`` governs only ``from X import *``.  Before this sweep, 32 viz
    modules declared one and none defined ``__dir__``, so ``dir()`` offered 405
    names no ``__all__`` claimed.
    """
    missing = [
        m.__name__ for m in _viz_modules() if hasattr(m, "__all__") and "__dir__" not in vars(m)
    ]
    assert missing == [], f"declare __all__ but no __dir__: {missing}"


def test_no_viz_module_leaks_a_public_name_its_all_does_not_claim():
    """The measurable form of the same clause: the leak count is zero, and stays zero."""
    leaks = {}
    for module in _viz_modules():
        names = getattr(module, "__all__", None)
        if names is None:
            continue
        extra = sorted(set(_public(module)) - set(names))
        if extra:
            leaks[module.__name__] = extra
    assert leaks == {}, f"public names outside __all__: {leaks}"


def test_a_curated_module_dir_is_exactly_its_sorted_all():
    """Not a subset, not a superset — the listing IS the declaration."""
    for module in _viz_modules():
        names = getattr(module, "__all__", None)
        if names is None:
            continue
        assert dir(module) == sorted(names), module.__name__


def test_hiding_a_module_name_did_not_unbind_it():
    """G1 over the module sweep: every formerly-leaked name is still an attribute.

    The sample is the four modules the contract named as worst offenders, probed
    through the import machinery rather than through ``dir``.
    """
    probes = {
        "tsdynamics.viz.compose": ["Any", "Layer", "Layout", "Legend"],
        "tsdynamics.viz.producers": ["Any", "Callable", "AUTOSTYLE_PIVOT"],
        "tsdynamics.viz.style": ["Any", "Callable", "LayoutEngineName"],
        "tsdynamics.viz.render.caps": ["Any", "PlotKind", "PlotSpec", "Protocol"],
    }
    for name, attrs in probes.items():
        module = importlib.import_module(name)
        for attr in attrs:
            assert hasattr(module, attr), f"{name}.{attr} stopped resolving"
            assert attr not in dir(module), f"{name}.{attr} is still listed"


# ---------------------------------------------------------------------------
# 5. `allow` — the extension point's other half (the highest-value G1 item)
# ---------------------------------------------------------------------------


def test_a_custom_primitive_is_refused_by_a_message_that_names_the_fix(traj):
    """Measured before v6 round 9: the refusal named nothing, and ``allow`` had 0 doc mentions.

    A primitive a user registers is admitted by **no** shipped row — all 39 were
    frozen before it existed — so ``allow`` is the only fix, and the error is the
    only place a user will ever be told.
    """
    import matplotlib.pyplot as plt

    name = "_visibility_stem"

    @ts.viz.primitives.register(name, requires=("x", "y"), marks=("line", "points"))
    def stem(part, **options):
        """A vertical drop to the baseline plus a marker at each point."""
        return [{"x": part["x"], "y": part["y"], "mark": "line"}]

    with pytest.raises(ts.errors.InvalidParameterError) as excinfo:
        ts.plot(traj, "time_series", primitive=name)
    message = str(excinfo.value)
    assert "ts.viz.transforms.allow" in message
    assert f"allow('time_series', {name!r})" in message

    # ...and the line it hands back RUNS, which is the whole point of a remedy.
    ts.viz.transforms.allow("time_series", name)
    built = ts.plot(traj, "time_series", primitive=name)
    assert built.layers
    plt.close("all")


def test_allow_is_documented_where_it_is_listed():
    """It is one of five names on ``ts.viz.transforms``; it had better say what it does."""
    doc = ts.viz.transforms.allow.__doc__ or ""
    assert "ts.viz.transforms.allow" in doc or "allow(" in doc
    assert "primitives.register" in doc
