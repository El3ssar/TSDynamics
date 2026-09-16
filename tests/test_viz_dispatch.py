"""Backend dispatch + capability fallback + kind aliasing (stream VIZ-DISPATCH).

Exercises :mod:`tsdynamics.viz.render` without any real plotting backend: fake
renderers (plain callables carrying a ``.capabilities`` descriptor) are
registered into ``registry.renderers`` and removed by a fixture, so the dispatch
logic — named/default selection, capability-aware fallback, the
``VisualizationDegraded`` warning, the kind-alias normalisation, and the
no-backend / unknown-name errors — is tested in isolation.

Engine-free, fast tier; imports no matplotlib/plotly (proving the dispatch seam
itself stays plot-free).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tsdynamics import registry
from tsdynamics.analysis._result import VisualizationNotInstalled
from tsdynamics.viz.render import (
    RendererCapabilities,
    VisualizationDegraded,
    normalize_kind,
    register_builtin_renderers,
    render_spec,
    select_renderer,
)
from tsdynamics.viz.spec import Layer, PlotKind, PlotSpec

# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


def _make_renderer(caps: RendererCapabilities | None):
    """A fake renderer callable that records its call and carries ``caps``."""

    def _render(spec, **kw):
        return {"backend": getattr(caps, "name", "anon"), "spec": spec, "kw": kw}

    if caps is not None:
        _render.capabilities = caps  # type: ignore[attr-defined]
    return _render


@pytest.fixture
def clean_renderers():
    """Empty the renderers registry for the test, restoring it afterwards."""
    saved = list(registry.renderers.all())
    registry.renderers.clear()
    try:
        yield registry.renderers
    finally:
        registry.renderers.clear()
        for entry in saved:
            registry.renderers.register(entry.name, entry.obj)


def _line_spec() -> PlotSpec:
    import numpy as np

    x = np.linspace(0.0, 1.0, 4)
    return PlotSpec(kind=PlotKind.TIME_SERIES, layers=[Layer(PlotKind.LINE, {"x": x, "y": x})])


def _vector_field_spec() -> PlotSpec:
    import numpy as np

    g = np.zeros((3, 3))
    return PlotSpec(
        kind=PlotKind.VECTOR_FIELD,
        layers=[Layer(PlotKind.QUIVER, {"x": g, "y": g, "u": g, "v": g})],
    )


# ---------------------------------------------------------------------------
# normalize_kind / _KIND_ALIAS
# ---------------------------------------------------------------------------


def test_normalize_kind_passes_real_kinds_through():
    assert normalize_kind(PlotKind.TIME_SERIES) is PlotKind.TIME_SERIES
    assert normalize_kind("recurrence_plot") is PlotKind.RECURRENCE_PLOT


def test_normalize_kind_resolves_accessor_aliases():
    # The result.plot.phase() / .image() / .section() accessor spellings.
    assert normalize_kind("phase") is PlotKind.PHASE_PORTRAIT_2D
    assert normalize_kind("phase3d") is PlotKind.PHASE_PORTRAIT_3D
    assert normalize_kind("image") is PlotKind.IMAGE
    assert normalize_kind("section") is PlotKind.POINCARE_SECTION
    assert normalize_kind("bifurcation_diagram") is PlotKind.BIFURCATION


def test_aliases_to_removed_kinds_are_gone():
    """The ``"spectrum"`` / ``"psd"`` / ``"histogram"`` aliases left with their kinds.

    They pointed at ``POWER_SPECTRUM`` / ``HISTOGRAM_NULL``, which the v6
    vocabulary surgery deleted (see ``tests/test_viz_vocab.py``).  An alias that
    still *resolved* would be worse than a missing one: it would name a kind
    nothing draws.
    """
    for alias in ("spectrum", "psd", "histogram_null"):
        with pytest.raises(ValueError):
            normalize_kind(alias)
    # "histogram" is no longer an alias, but it IS a real layer mark, so it must
    # resolve to the mark rather than to the deleted HISTOGRAM_NULL kind.
    assert normalize_kind("histogram") is PlotKind.HISTOGRAM


def test_every_kind_alias_targets_a_live_kind():
    """No alias may point at a kind that is not in the vocabulary."""
    from tsdynamics.viz.render import _KIND_ALIAS

    for alias, target in _KIND_ALIAS.items():
        assert target in set(PlotKind), f"alias {alias!r} targets a dead kind"


def test_normalize_kind_rejects_garbage():
    with pytest.raises(ValueError):
        normalize_kind("definitely_not_a_kind")


# ---------------------------------------------------------------------------
# register_builtin_renderers (the matplotlib backend ships as of VIZ-MPL-CORE)
# ---------------------------------------------------------------------------


def test_register_builtin_renderers_registers_matplotlib(clean_renderers):
    """The installed matplotlib backend is discovered and registered (stream VIZ-MPL-CORE).

    Registration is *lazy* in its plot import: it adds the matplotlib backend to
    the registry but the matplotlib library is imported only on the first actual
    render — registration itself stays side-effect-light — and re-running is
    idempotent.
    """
    pytest.importorskip("matplotlib")
    newly = register_builtin_renderers()
    assert "matplotlib" in newly
    assert "matplotlib" in clean_renderers
    # Re-running is idempotent: matplotlib is not registered twice.
    assert register_builtin_renderers() == []


# ---------------------------------------------------------------------------
# Selection + capability-aware fallback (>= 2 registered backends)
# ---------------------------------------------------------------------------


def test_named_backend_used_when_capable(clean_renderers):
    caps = RendererCapabilities.all_kinds("alpha")
    clean_renderers.register("alpha", _make_renderer(caps))
    name, _renderer = select_renderer(_line_spec(), "alpha")
    assert name == "alpha"


def test_named_backend_falls_back_when_incapable(clean_renderers):
    line_only = RendererCapabilities.of_kinds("plotly", [PlotKind.LINE, PlotKind.TIME_SERIES])
    universal = RendererCapabilities.all_kinds("matplotlib")
    clean_renderers.register("matplotlib", _make_renderer(universal))
    clean_renderers.register("plotly", _make_renderer(line_only))

    spec = _vector_field_spec()
    with pytest.warns(VisualizationDegraded):
        name, _renderer = select_renderer(spec, "plotly")
    assert name == "matplotlib"  # fell back to the capable backend


def test_default_selection_picks_first_capable(clean_renderers):
    line_only = RendererCapabilities.of_kinds("plotly", [PlotKind.LINE, PlotKind.TIME_SERIES])
    universal = RendererCapabilities.all_kinds("matplotlib")
    # plotly registered first, but it cannot draw a vector field → matplotlib wins.
    clean_renderers.register("plotly", _make_renderer(line_only))
    clean_renderers.register("matplotlib", _make_renderer(universal))
    name, _renderer = select_renderer(_vector_field_spec(), None)
    assert name == "matplotlib"


def test_render_spec_dispatches_and_forwards_kwargs(clean_renderers):
    clean_renderers.register("alpha", _make_renderer(RendererCapabilities.all_kinds("alpha")))
    out = render_spec(_line_spec(), "alpha", dpi=120)
    assert out["backend"] == "alpha"
    assert out["kw"] == {"dpi": 120}


def test_capability_less_callable_is_a_universal_fallback(clean_renderers):
    """A plain callable with no .capabilities draws anything (the fake-renderer idiom)."""
    clean_renderers.register("plain", _make_renderer(None))
    name, _renderer = select_renderer(_vector_field_spec(), None)
    assert name == "plain"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_render_with_no_backend_raises_not_installed(clean_renderers, monkeypatch):
    # The matplotlib backend now auto-registers on render; stub that out to keep
    # exercising the genuine "no backend installed" path faithfully.
    from tsdynamics.viz import render as render_mod

    monkeypatch.setattr(render_mod, "register_builtin_renderers", lambda *a, **k: [])
    with pytest.raises(VisualizationNotInstalled):
        render_spec(_line_spec())


def test_unknown_backend_name_lists_the_registered_backends(clean_renderers):
    """A bad ``backend=`` is a bad option value: name the choices and a line to type.

    It used to surface the registry's bare ``KeyError``, whose message named the
    mistake and nothing else — and whose ``__str__`` is ``repr(arg)``, so a
    multi-line remedy would print with literal ``\\n`` in it.
    """
    from tsdynamics.errors import InvalidParameterError

    clean_renderers.register("alpha", _make_renderer(RendererCapabilities.all_kinds("alpha")))
    with pytest.raises(InvalidParameterError) as excinfo:
        render_spec(_line_spec(), "nope")
    message = str(excinfo.value)
    assert "'nope'" in message
    assert "alpha" in message
    assert "spec.render('alpha')" in message


# ---------------------------------------------------------------------------
# Unknown render keywords (the "a typo must not be swallowed" gate)
# ---------------------------------------------------------------------------
#
# Before this gate, ``spec.render(backend=B, totally_bogus_kwarg=42)`` was
# accepted with no warning on **all four** in-tree backends: three of the four
# render cores end in a ``**_kw`` / ``**_ignored`` catch-all, and the fourth is
# reached through a ``def _render(spec, /, **kw)`` wrapper.  A misspelled option
# name was therefore indistinguishable from a correct one — the exact silent
# no-op the viz rework exists to eliminate, sitting in the dispatcher itself.


@pytest.mark.parametrize("backend", ["matplotlib", "plotly", "json", "threejs"])
def test_unknown_render_kwarg_raises_naming_the_key_and_the_backend(backend):
    """Every in-tree backend rejects an unknown render keyword, loudly."""
    pytest.importorskip("matplotlib")  # plotly falls back to mpl if absent
    if backend == "plotly":
        pytest.importorskip("plotly")
    from tsdynamics.errors import InvalidParameterError

    register_builtin_renderers()
    with pytest.raises(InvalidParameterError) as exc:
        _line_spec().render(backend, totally_bogus_kwarg=42)
    msg = str(exc.value)
    assert "totally_bogus_kwarg" in msg
    assert backend in msg
    # The message must also say what this backend *does* take, per backend.
    expected = {
        "matplotlib": "figsize",
        "plotly": "include_plotlyjs",
        "json": "indent",
        "threejs": "max_points",
    }[backend]
    assert expected in msg


@pytest.mark.parametrize(
    ("backend", "kwargs"),
    [
        ("matplotlib", {"figsize": (4.0, 3.0)}),
        ("plotly", {"include_plotlyjs": "cdn"}),
        ("json", {"indent": 2}),
        ("threejs", {"decimals": 3}),
    ],
)
def test_legitimate_backend_kwargs_still_pass_through(backend, kwargs):
    """The per-backend check must not break a genuine backend option."""
    pytest.importorskip("matplotlib")
    if backend == "plotly":
        pytest.importorskip("plotly")
    register_builtin_renderers()
    assert _line_spec().render(backend, **kwargs) is not None


def test_a_kwarg_of_the_wrong_backend_is_rejected():
    """``figsize`` is matplotlib's; asking plotly for it is an error, not a no-op."""
    pytest.importorskip("plotly")
    from tsdynamics.errors import InvalidParameterError

    register_builtin_renderers()
    with pytest.raises(InvalidParameterError, match="figsize"):
        _line_spec().render("plotly", figsize=(4.0, 3.0))


def test_out_of_tree_backend_with_var_keyword_keeps_its_pass_through(clean_renderers):
    """An undeclared plugin taking ``**kwargs`` keeps the documented free pass-through.

    The check must be *per backend*: we cannot know what a third-party renderer
    reads, so a catch-all signature with no declaration is left alone.
    """
    clean_renderers.register("alpha", _make_renderer(RendererCapabilities.all_kinds("alpha")))
    out = render_spec(_line_spec(), "alpha", whatever_it_wants=1)
    assert out["kw"] == {"whatever_it_wants": 1}


def test_out_of_tree_backend_may_opt_in_by_declaring_render_kwargs(clean_renderers):
    """A plugin that declares ``render_kwargs`` gets the same protection."""
    from tsdynamics.errors import InvalidParameterError

    caps = RendererCapabilities.all_kinds("beta", render_kwargs={"scale"})
    clean_renderers.register("beta", _make_renderer(caps))
    assert render_spec(_line_spec(), "beta", scale=2)["kw"] == {"scale": 2}
    with pytest.raises(InvalidParameterError, match="scal"):
        render_spec(_line_spec(), "beta", scal=2)


def test_out_of_tree_backend_with_explicit_signature_is_introspected(clean_renderers):
    """A plugin with no catch-all is validated from its own signature."""
    from tsdynamics.errors import InvalidParameterError

    caps = RendererCapabilities.all_kinds("gamma")

    def _render(spec, *, scale=1.0):
        return {"kw": {"scale": scale}}

    _render.capabilities = caps
    clean_renderers.register("gamma", _render)
    assert render_spec(_line_spec(), "gamma", scale=3.0)["kw"] == {"scale": 3.0}
    with pytest.raises(InvalidParameterError):
        render_spec(_line_spec(), "gamma", nope=1)


def test_declared_render_kwargs_match_each_backend_core():
    """The declaration table must match what each backend's real render fn accepts.

    ``_BUILTIN_RENDER_KWARGS`` is hand-written (it has to be: every in-tree
    renderer is registered as a ``**kw`` wrapper, so introspecting the registered
    callable recovers nothing).  This is what stops it drifting: it introspects
    each backend's *actual* entry point and fails if the declaration promises a
    keyword the backend does not take, or omits one it does.
    """
    import inspect

    pytest.importorskip("matplotlib")
    pytest.importorskip("plotly")
    from tsdynamics.viz.render.caps import _BUILTIN_RENDER_KWARGS

    register_builtin_renderers()

    def named_kwargs(fn):
        sig = inspect.signature(fn)
        return {
            name
            for name, p in sig.parameters.items()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and name not in ("spec", "warn")
        }

    from tsdynamics.viz.render.mpl import _core as mpl_core
    from tsdynamics.viz.render.plotly import _core as plotly_core

    cores = {
        "matplotlib": named_kwargs(mpl_core.render),
        "plotly": named_kwargs(plotly_core.render),
        # json / threejs register a closure that IS the entry point.
        "json": named_kwargs(registry.renderers.get("json")),
        "threejs": named_kwargs(registry.renderers.get("threejs")),
    }
    for backend, actual in cores.items():
        assert _BUILTIN_RENDER_KWARGS[backend] == actual, backend


# ---------------------------------------------------------------------------
# v6 — the ts.viz surface, the renderers registry, and the save contract
# ---------------------------------------------------------------------------

#: The contract's ``ts.viz.<TAB>``, §2.5.  Exact and sorted: a builder who
#: produces a different listing has failed.
_VIZ_TAB_SURFACE = [
    "Plot",
    "compatibility",
    "draw",
    "geometry",
    "grid",
    "load",
    "plot",
    "primitives",
    "renderers",
    "spec",
    "styles",
    "themes",
    "transforms",
]

#: The contract's ``ts.viz.spec.<TAB>``, §2.7 — the 19 IR nouns.
_SPEC_TAB_SURFACE = [
    "Animation",
    "Annotation",
    "Axis",
    "Colorbar",
    "Frame",
    "FrameSpace",
    "Geometry",
    "Layer",
    "Layout",
    "Legend",
    "Part",
    "PlotKind",
    "PlotTransform",
    "Presentation",
    "SCHEMA_VERSION",
    "T",
    "from_dict_envelope",
    "make_frame",
    "to_dict_envelope",
]


def test_ts_viz_tab_surface_is_the_contract() -> None:
    """13 names: four registries, two doors, one arranger, one type, and the IR."""
    import tsdynamics as ts

    assert sorted(ts.viz.__all__) == _VIZ_TAB_SURFACE
    assert dir(ts.viz) == _VIZ_TAB_SURFACE
    for name in _VIZ_TAB_SURFACE:
        assert getattr(ts.viz, name) is not None, name


def test_ts_viz_spec_holds_the_ir_and_every_noun_resolves() -> None:
    """The IR is one dot away — and ten of the nineteen come from sibling modules."""
    import tsdynamics as ts

    assert sorted(ts.viz.spec.__all__) == _SPEC_TAB_SURFACE
    assert dir(ts.viz.spec) == _SPEC_TAB_SURFACE
    for name in _SPEC_TAB_SURFACE:
        assert getattr(ts.viz.spec, name) is not None, name
    # Plot is deliberately NOT here: it is the one type you annotate, and it
    # lives one level up.
    assert "Plot" not in ts.viz.spec.__all__
    assert ts.viz.Plot.__name__ == "Plot"


def test_touching_the_viz_surface_imports_no_plot_library() -> None:
    """``ts.viz`` and ``ts.viz.spec`` must stay free of matplotlib and plotly."""
    import subprocess
    import sys

    code = (
        "import sys, tsdynamics as ts;"
        "_ = ts.viz.__all__; _ = ts.viz.spec.__all__; _ = ts.viz.spec.Geometry;"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('matplotlib', 'plotly')];"
        "assert not bad, bad; print('CLEAN')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "CLEAN" in out.stdout


def test_a_name_that_moved_says_where_it_went() -> None:
    """The error message *is* the migration guide."""
    import tsdynamics as ts

    with pytest.raises(AttributeError, match=r"PlotSpec is now Plot"):
        _ = ts.viz.PlotSpec
    with pytest.raises(AttributeError, match=r"transforms\.names\(\)"):
        _ = ts.viz.list_transforms
    # A guess is still an ordinary miss, so hasattr keeps working.
    assert not hasattr(ts.viz, "definitely_not_a_name")


def test_the_four_registries_share_one_shape() -> None:
    """``register`` / ``names`` / ``find`` / ``get`` — learn one, know all four."""
    import tsdynamics as ts

    for name in ("transforms", "primitives", "renderers", "themes"):
        registry_obj = getattr(ts.viz, name)
        missing = [
            verb for verb in ("names", "get") if not callable(getattr(registry_obj, verb, None))
        ]
        assert not missing, f"ts.viz.{name} is missing {missing}"
        assert registry_obj.names(), f"ts.viz.{name}.names() is empty"


def test_the_four_registries_all_answer_find() -> None:
    """``find`` is the fourth shared verb; three of four answer it today."""
    import tsdynamics as ts

    for name in ("transforms", "primitives", "renderers", "themes"):
        assert callable(getattr(getattr(ts.viz, name), "find", None)), name


def test_renderers_introspection_is_honest_before_the_first_render() -> None:
    """Measured before v6: ``names()`` answered ``[]`` until something had drawn."""
    import subprocess
    import sys

    code = (
        "import tsdynamics as ts;"
        "print(','.join(ts.viz.renderers.names()));"
        "print(','.join(ts.viz.renderers.find(writes='.svg')))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    listed, svg = out.stdout.strip().splitlines()
    assert listed.split(",")[0] == "matplotlib", listed
    assert svg == "matplotlib", svg


def test_writes_is_split_into_static_and_animated_because_savefig_disagrees() -> None:
    """Measured: ``savefig`` refuses ``.mp4``; ``FuncAnimation.save`` writes it.

    One undivided ``writes`` set made ``Plot.save`` promise four formats
    matplotlib can never produce (``.apng`` ``.m4v`` ``.mov`` ``.webm``, all
    static) while refusing ``.webp``, which it can.
    """
    import tsdynamics as ts

    caps = ts.viz.renderers.get("matplotlib")
    assert caps.can_save(".png") and not caps.can_save(".png", animated=True)
    assert caps.can_save(".mp4", animated=True) and not caps.can_save(".mp4")
    assert caps.can_save(".gif") and caps.can_save(".gif", animated=True)
    assert ".webp" in caps.writes_static, "declared and, before v6, unreachable"
    assert ".pgf" in caps.writes_static, "works, and was simply never declared"
    assert caps.writes == caps.writes_static | caps.writes_animated


@pytest.mark.parametrize(
    "ext",
    [
        ".png",
        ".pdf",
        ".svg",
        ".svgz",
        ".eps",
        ".ps",
        ".pgf",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".webp",
        ".gif",
    ],
)
def test_every_extension_matplotlib_declares_statically_can_actually_be_written(
    tmp_path, ext: str
) -> None:
    """The declaration and the writer must agree — in both directions."""
    pytest.importorskip("matplotlib")
    import warnings as _w

    import matplotlib.pyplot as plt
    import numpy as np

    t = np.linspace(0.0, 1.0, 8)
    spec = PlotSpec(kind=PlotKind.TIME_SERIES, layers=[Layer(PlotKind.LINE, {"x": t, "y": t})])
    target = tmp_path / f"figure{ext}"
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        spec.save(str(target))
    assert target.stat().st_size > 0
    plt.close("all")


def test_a_third_party_backends_declared_extension_becomes_saveable(tmp_path) -> None:
    """``save`` asks the backends and believes them — it keeps no table of its own.

    Before v6 a registered backend could be *rendered* by name but its declared
    extension could **never** be saved, because a hardcoded ``_WRITABLE_EXT`` in
    ``spec.py`` was consulted first.
    """
    import numpy as np

    written: list[str] = []

    def _render(spec, /, *, path=None, **kw):
        # A file-writing backend: it draws nothing, so with no ``path`` it has
        # nothing to hand back — which is exactly the shape ``save`` must cope
        # with when it does not recognise the writer up front.
        if path is None:
            return None
        Path(path).write_text("TIKZ")
        written.append(path)
        return path

    _render.capabilities = RendererCapabilities.all_kinds("tikz", writes=(".tikz",))
    registry.renderers.register("tikz", _render)
    try:
        t = np.linspace(0.0, 1.0, 8)
        spec = PlotSpec(kind=PlotKind.TIME_SERIES, layers=[Layer(PlotKind.LINE, {"x": t, "y": t})])
        target = tmp_path / "figure.tikz"
        assert "tikz" in ts_viz_renderers_find(".tikz")
        spec.save(str(target), backend="tikz")
        assert written and target.read_text() == "TIKZ"
    finally:
        registry.renderers.unregister("tikz")


def ts_viz_renderers_find(ext: str) -> list[str]:
    """Helper: which backends declare they can write ``ext``."""
    import tsdynamics as ts

    return ts.viz.renderers.find(writes=ext)
