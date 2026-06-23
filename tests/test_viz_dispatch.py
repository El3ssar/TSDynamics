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
    # The result.plot.phase() / .image() / .spectrum() accessor spellings.
    assert normalize_kind("phase") is PlotKind.PHASE_PORTRAIT_2D
    assert normalize_kind("phase3d") is PlotKind.PHASE_PORTRAIT_3D
    assert normalize_kind("image") is PlotKind.IMAGE
    assert normalize_kind("spectrum") is PlotKind.POWER_SPECTRUM
    assert normalize_kind("section") is PlotKind.POINCARE_SECTION


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


def test_unknown_backend_name_raises_keyerror(clean_renderers):
    clean_renderers.register("alpha", _make_renderer(RendererCapabilities.all_kinds("alpha")))
    with pytest.raises(KeyError):
        render_spec(_line_spec(), "nope")
