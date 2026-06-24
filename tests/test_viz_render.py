"""Conformance gate for the matplotlib reference renderer (stream VIZ-RENDER-GATE).

The matplotlib backend is the **conformance oracle**: it must draw *every* mark
and *every* semantic :class:`~tsdynamics.viz.spec.PlotKind` in the frozen
vocabulary on the Agg canvas without error.  This gate freezes that:

1. **Every layer mark renders** — a minimal spec carrying each
   :meth:`PlotKind.layer_marks` mark renders to a :class:`matplotlib.figure.Figure`.
2. **Every semantic kind renders** — a minimal spec of each
   :meth:`PlotKind.semantic_kinds` kind (2-D, 3-D, image, bar, …) renders.
3. **Registry conformance** — every registered analysis whose result carries a
   ``to_plot_spec`` produces a spec whose semantic kind and layer marks are real
   :class:`~tsdynamics.viz.spec.PlotKind` members that round-trip through
   ``to_dict`` / ``from_dict`` (engine-free; the synthetic builders live in
   :mod:`tests.test_viz_fake_renderer`).
4. **No-plot-import guard** — a fresh ``import tsdynamics`` pulls in **no**
   matplotlib (registration is lazy, only on first render).
5. **Negative control** — a deliberately broken renderer / spec fails, so the
   gate is not a tautology.

Engine-free.  Skips the render assertions where matplotlib is absent.
"""

from __future__ import annotations

import importlib
import io
import subprocess
import sys

import numpy as np
import pytest

from tsdynamics import registry
from tsdynamics.viz.spec import Axis, Colorbar, Layer, PlotKind, PlotSpec

pytest.importorskip("matplotlib")

from tsdynamics.viz.render import register_builtin_renderers  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _mpl_backend():
    """Ensure the matplotlib backend is registered for the whole module."""
    register_builtin_renderers()
    if "matplotlib" not in registry.renderers:  # pragma: no cover - matplotlib present here
        pytest.skip("matplotlib backend did not register")
    yield


# ---------------------------------------------------------------------------
# Minimal renderable specs for every mark / kind
# ---------------------------------------------------------------------------

_X = np.linspace(0.0, 1.0, 6)
_GRID = np.add.outer(_X, _X)


def _layer_for_mark(mark: PlotKind) -> Layer:
    """A minimal valid layer carrying the channels ``mark`` consumes."""
    if mark in (PlotKind.LINE3D,):
        return Layer(mark, {"x": _X, "y": _X[::-1], "z": _X})
    if mark == PlotKind.SURFACE3D:
        return Layer(mark, {"x": _X, "y": _X, "z": _GRID})
    if mark == PlotKind.IMAGE:
        return Layer(mark, {"x": _X, "y": _X, "c": _GRID})
    if mark == PlotKind.QUIVER:
        g = np.zeros((3, 3))
        return Layer(mark, {"x": g, "y": g, "u": g + 1.0, "v": g + 1.0})
    if mark == PlotKind.HISTOGRAM:
        return Layer(mark, {"x": _X})
    if mark == PlotKind.AREA:
        return Layer(mark, {"x": _X, "lo": _X - 0.1, "hi": _X + 0.1})
    if mark == PlotKind.ERRORBAR:
        return Layer(mark, {"x": _X, "y": _X, "err": _X * 0.1})
    if mark == PlotKind.BAR:
        return Layer(mark, {"x": _X, "y": _X})
    # LINE / SCATTER / MARKERS
    return Layer(mark, {"x": _X, "y": _X[::-1]})


_IMAGE_KINDS = frozenset(
    {
        PlotKind.IMAGE,
        PlotKind.BASINS_IMAGE,
        PlotKind.SPACETIME,
        PlotKind.SPECTROGRAM,
    }
)


def _spec_for_kind(kind: PlotKind) -> PlotSpec:
    """A minimal renderable spec for a *semantic* kind (2-D / 3-D / image)."""
    if kind in (PlotKind.PHASE_PORTRAIT_3D,):
        return PlotSpec(
            kind=kind,
            ndim=3,
            x=Axis(),
            y=Axis(),
            z=Axis(),
            layers=[_layer_for_mark(PlotKind.LINE3D)],
        )
    if kind in _IMAGE_KINDS:
        return PlotSpec(kind=kind, layers=[_layer_for_mark(PlotKind.IMAGE)], colorbar=Colorbar())
    if kind == PlotKind.RECURRENCE_PLOT:
        # Sparse recurrence plot — a scatter, never a dense image (anti-OOM).
        return PlotSpec(kind=kind, aspect="equal", layers=[_layer_for_mark(PlotKind.SCATTER)])
    if kind in (PlotKind.CATEGORICAL_BAR, PlotKind.FEATURE_BARS, PlotKind.LYAPUNOV_SPECTRUM):
        return PlotSpec(kind=kind, layers=[_layer_for_mark(PlotKind.BAR)])
    if kind == PlotKind.HISTOGRAM_NULL:
        return PlotSpec(kind=kind, layers=[_layer_for_mark(PlotKind.HISTOGRAM)])
    if kind in (PlotKind.TRAJECTORY_ANIMATION, PlotKind.ENSEMBLE_ANIMATION):
        # Animation kinds render their final frame (no animation backend ships).
        return PlotSpec(kind=kind, layers=[_layer_for_mark(PlotKind.LINE)])
    if kind == PlotKind.COMPOSITE:
        # A composite is a multi-panel figure: it tiles ``panels``, not ``layers``.
        from tsdynamics.viz.spec import Layout

        panel = PlotSpec(kind=PlotKind.TIME_SERIES, layers=[_layer_for_mark(PlotKind.LINE)])
        return PlotSpec(kind=kind, panels=[panel, panel], layout=Layout(mode="stack"))
    return PlotSpec(kind=kind, layers=[_layer_for_mark(PlotKind.LINE)])


# ---------------------------------------------------------------------------
# 1 — every layer mark renders on Agg
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mark", sorted(PlotKind.layer_marks(), key=lambda k: k.value), ids=str)
def test_every_mark_renders(mark: PlotKind):
    from matplotlib.figure import Figure

    spec = PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_3D
        if mark in (PlotKind.LINE3D, PlotKind.SURFACE3D)
        else PlotKind.TIME_SERIES,
        ndim=3 if mark in (PlotKind.LINE3D, PlotKind.SURFACE3D) else 2,
        z=Axis() if mark in (PlotKind.LINE3D, PlotKind.SURFACE3D) else None,
        layers=[_layer_for_mark(mark)],
    )
    fig = spec.render("matplotlib")
    assert isinstance(fig, Figure), mark
    assert fig.axes, f"{mark} produced no axes"


# ---------------------------------------------------------------------------
# 2 — every semantic kind renders on Agg
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(PlotKind.semantic_kinds(), key=lambda k: k.value), ids=str)
def test_every_semantic_kind_renders(kind: PlotKind):
    from matplotlib.figure import Figure

    fig = _spec_for_kind(kind).render("matplotlib")
    assert isinstance(fig, Figure), kind
    assert fig.axes, f"{kind} produced no axes"


# ---------------------------------------------------------------------------
# 3 — registry conformance: every result's spec is coercible-kind + round-trips
# ---------------------------------------------------------------------------


def _result_builders() -> dict[type, object]:
    """The synthetic result builders from the fake-renderer gate (engine-free)."""
    module = importlib.import_module("test_viz_fake_renderer")
    return dict(module._BUILDERS)  # type: ignore[attr-defined]


def test_every_result_spec_is_coercible_and_round_trips():
    """Every synthetic result either yields a valid PlotSpec that round-trips, or raises."""
    from tsdynamics.analysis._result import VisualizationNotInstalled

    builders = _result_builders()
    assert builders, "the fake-renderer gate's builders must be importable"
    for cls, build in builders.items():
        result = build()
        try:
            spec = result.to_plot_spec()
        except VisualizationNotInstalled:
            continue  # the documented "nothing to draw" path
        assert isinstance(spec, PlotSpec), cls.__name__
        assert isinstance(spec.kind, PlotKind), cls.__name__
        for layer in spec.layers:
            assert isinstance(layer.kind, PlotKind), cls.__name__
        rebuilt = PlotSpec.from_dict(spec.to_dict())
        assert rebuilt.kind == spec.kind, cls.__name__
        assert len(rebuilt.layers) == len(spec.layers), cls.__name__


def test_every_registered_analysis_result_renders_or_is_documented():
    """Every synthetic result's spec renders on matplotlib (or documents no spec)."""
    from matplotlib.figure import Figure

    from tsdynamics.analysis._result import VisualizationNotInstalled

    for cls, build in _result_builders().items():
        result = build()
        try:
            spec = result.to_plot_spec()
        except VisualizationNotInstalled:
            continue
        fig = spec.render("matplotlib")
        assert isinstance(fig, Figure), cls.__name__


# ---------------------------------------------------------------------------
# 4 — no-plot-import guard
# ---------------------------------------------------------------------------


def test_import_tsdynamics_pulls_no_matplotlib():
    """A fresh ``import tsdynamics`` must not import matplotlib (lazy registration)."""
    code = (
        "import sys\n"
        "import tsdynamics\n"
        "import tsdynamics.viz\n"
        "for banned in ('matplotlib', 'matplotlib.pyplot', 'plotly'):\n"
        "    assert banned not in sys.modules, banned\n"
        "print('OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "OK"


# ---------------------------------------------------------------------------
# 5 — negative control (the gate is not a tautology)
# ---------------------------------------------------------------------------


def test_negative_control_broken_layer_is_caught():
    """A spec carrying a non-PlotKind layer kind cannot be constructed/validated.

    Proves the kind assertions above have teeth: a stray string is rejected at
    Layer construction, so a broken spec can never slip the conformance check.
    """
    with pytest.raises(ValueError):
        Layer("definitely_not_a_mark", {"x": _X, "y": _X})


def test_negative_control_unrenderable_spec_surfaces():
    """A spec whose only layer carries no usable channels still returns a Figure or errors.

    The renderer must not silently swallow a structurally-empty spec into a
    non-figure; it returns a Figure (possibly empty axes) — never ``None``.
    """
    from matplotlib.figure import Figure

    spec = PlotSpec(kind=PlotKind.TIME_SERIES, layers=[])
    fig = spec.render("matplotlib")
    assert isinstance(fig, Figure)


def test_render_buffer_is_a_nonempty_png():
    """A rendered figure saves to a real, non-empty PNG (a render-to-image smoke)."""
    spec = _spec_for_kind(PlotKind.PHASE_PORTRAIT_2D)
    fig = spec.render("matplotlib")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    data = buf.getvalue()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    assert len(data) > 1000, "suspiciously small PNG (empty render?)"
