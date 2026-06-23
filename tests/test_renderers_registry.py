"""Meta-QA over the visualization renderers registry (stream WS-VIZREG / VIZ-MPL-CORE).

The renderers registry (:data:`tsdynamics.registry.renderers`) is the third
generic :class:`~tsdynamics.registry.Registry` container, mirroring
:data:`~tsdynamics.registry.analyses` / :data:`~tsdynamics.registry.transforms`.

As of stream VIZ-MPL-CORE the matplotlib reference renderer is the **first real
backend**, which flips visualization from *deferred* to *live*: the registry
still ships **empty at import** (no backend auto-registers when you
``import tsdynamics`` / ``import tsdynamics.viz`` — and no plot library is pulled
in), but :func:`tsdynamics.viz.render.register_builtin_renderers` now discovers
the installed matplotlib backend on the **first render**, so
``render("matplotlib")`` succeeds and returns a figure.

These tests freeze that live contract:

- the registry exists, is a :class:`~tsdynamics.registry.Registry` tagged
  ``"renderer"``, and is a *distinct* instance from the analyses/transforms ones;
- it is **empty at import** — a fresh ``import tsdynamics`` registers no backend
  and pulls in **no plot library** (core stays plotting-free; registration is
  lazy, only on first render);
- :mod:`tsdynamics.viz` wires entry-point discovery (the ``tsdynamics.renderers``
  group) exactly like analyses/transforms;
- with **no** backend registered (forced empty), resolving through
  :meth:`tsdynamics.viz.spec.PlotSpec.render` raises a *helpful*,
  message-carrying ``VisualizationNotInstalled`` (the canonical named exception,
  not a bare ``KeyError``);
- once the builtin matplotlib backend registers, ``render("matplotlib")``
  dispatches to it and returns a real :class:`matplotlib.figure.Figure` (proving
  the seam is live), and an unknown backend name then raises a naming ``KeyError``.

The "no backend installed" path is exercised by *forcing* an empty registry
(monkeypatching :func:`register_builtin_renderers` to a no-op and clearing the
registry), so it stays a faithful test of the missing-backend behaviour even
though a real backend now ships.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from tsdynamics import registry
from tsdynamics.registry import Registry
from tsdynamics.viz import RENDERERS_GROUP, PlotKind, PlotSpec, discover_plugins
from tsdynamics.viz.spec import Layer

# ---------------------------------------------------------------------------
# The registry exists, is the right kind, and is distinct
# ---------------------------------------------------------------------------


def test_renderers_registry_exists_and_is_a_registry():
    """``registry.renderers`` is a :class:`Registry` tagged ``"renderer"``."""
    assert isinstance(registry.renderers, Registry)
    assert registry.renderers.kind == "renderer"


def test_renderers_in_registry_all():
    """``renderers`` is part of the registry module's public surface."""
    assert "renderers" in registry.__all__


def test_renderers_registry_is_distinct_instance():
    """The three generic registries are distinct container instances."""
    assert registry.renderers is not registry.analyses
    assert registry.renderers is not registry.transforms
    # …and distinctly tagged.
    kinds = {registry.analyses.kind, registry.transforms.kind, registry.renderers.kind}
    assert kinds == {"analysis", "transform", "renderer"}


def test_renderers_registry_empty_at_import():
    """No backend auto-registers at import — registration is lazy (first render).

    A fresh ``import tsdynamics`` (in a clean subprocess, immune to backends a
    prior in-process render registered) leaves the renderers registry empty and
    pulls in **no** plot library: the matplotlib backend registers only on the
    first :meth:`PlotSpec.render`, not at import.
    """
    code = textwrap.dedent(
        """
        import sys
        import tsdynamics
        import tsdynamics.viz
        from tsdynamics import registry

        # Registration is lazy: nothing registers merely from importing.
        assert registry.renderers.names() == [], registry.renderers.names()
        assert len(registry.renderers) == 0
        # …and no plot library was dragged in.
        for banned in ("matplotlib", "matplotlib.pyplot", "plotly", "plotly.graph_objects"):
            assert banned not in sys.modules, banned
        print("OK")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "OK"


# ---------------------------------------------------------------------------
# Core stays plotting-free
# ---------------------------------------------------------------------------


def test_importing_viz_pulls_no_plot_library():
    """A fresh import of tsdynamics / tsdynamics.viz must not import a plot backend.

    Run in a clean subprocess so the assertion is about *import*, not about
    whatever a prior in-process test may have rendered (a render lazily registers
    and imports matplotlib — by design, but only on first render, never at
    import).
    """
    code = textwrap.dedent(
        """
        import sys
        import tsdynamics  # noqa: F401
        import tsdynamics.viz  # noqa: F401

        for banned in ("matplotlib", "matplotlib.pyplot", "plotly", "plotly.graph_objects"):
            assert banned not in sys.modules, banned
        print("OK")
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "OK"


# ---------------------------------------------------------------------------
# Entry-point discovery is wired like analyses/transforms
# ---------------------------------------------------------------------------


def test_renderers_group_name():
    """The entry-point group is the documented ``tsdynamics.renderers``."""
    assert RENDERERS_GROUP == "tsdynamics.renderers"


def test_discover_plugins_is_idempotent_and_returns_a_list():
    """``discover_plugins`` returns the newly-registered names; re-running is safe.

    With no renderer plugins installed it returns an empty list and leaves the
    registry empty — and calling it again does not error or duplicate.
    """
    first = discover_plugins()
    assert isinstance(first, list)
    second = discover_plugins()
    assert isinstance(second, list)
    # Re-running registers nothing new (names already present are skipped).
    assert second == []


# ---------------------------------------------------------------------------
# spec.render() resolves through the registry with a helpful error when empty
# ---------------------------------------------------------------------------


def _scaling_spec() -> PlotSpec:
    """A minimal, backend-agnostic spec to render in the tests below."""
    return PlotSpec(kind=PlotKind.SCALING_FIT, layers=[])


@pytest.fixture
def _forced_empty_registry(monkeypatch):
    """Force a genuinely empty renderers registry for the no-backend path.

    A real backend (matplotlib) now ships, and :meth:`PlotSpec.render` lazily
    registers it on first use.  To keep exercising the *missing-backend* contract
    faithfully, this fixture clears the registry **and** stubs
    :func:`register_builtin_renderers` to a no-op for the duration of the test, so
    ``render`` finds nothing to register and takes the empty-registry branch.  The
    registry is restored afterwards.
    """
    from tsdynamics.viz import render as render_mod

    saved = registry.renderers.all()
    registry.renderers.clear()
    monkeypatch.setattr(render_mod, "register_builtin_renderers", lambda *a, **k: [])
    try:
        yield
    finally:
        registry.renderers.clear()
        for entry in saved:
            registry.renderers.register(entry.name, entry.obj, replace=True)


def test_render_with_empty_registry_raises_helpful_error(_forced_empty_registry):
    """With NO backend registered, ``render('matplotlib')`` raises a *helpful* error.

    Not a bare ``KeyError`` / ``ImportError`` — the canonical, message-carrying
    ``VisualizationNotInstalled``.  (The registry is forced empty here; in normal
    use the matplotlib backend auto-registers and this path is not reached.)
    """
    from tsdynamics.analysis._result import VisualizationNotInstalled

    spec = _scaling_spec()
    with pytest.raises(VisualizationNotInstalled) as excinfo:
        spec.render("matplotlib")

    msg = str(excinfo.value)
    assert msg.strip(), "the no-backend error must carry a message"
    # The message must guide the user, not just surface a missing key.
    assert "backend" in msg.lower()
    # It is the canonical, named exception — NOT a bare KeyError (which would only
    # echo the missing name).  VisualizationNotInstalled subclasses ImportError so
    # the optional-dependency machinery reads like any optional dependency
    # (``except ImportError`` catches it).
    assert isinstance(excinfo.value, ImportError)
    assert not isinstance(excinfo.value, KeyError)


def test_render_default_backend_also_raises_when_empty(_forced_empty_registry):
    """``render()`` with no backend argument also raises cleanly when none is registered."""
    from tsdynamics.analysis._result import VisualizationNotInstalled

    with pytest.raises(VisualizationNotInstalled):
        _scaling_spec().render()


# ---------------------------------------------------------------------------
# Once a backend registers, the seam is live (and unknown names raise on name)
# ---------------------------------------------------------------------------


@pytest.fixture
def _temp_backend():
    """Register a trivial renderer, yield its name, then remove it.

    Proves the registry seam dispatches by name without depending on a real
    plotting backend.  The trivial renderer is removed afterwards; the builtin
    matplotlib backend that a render lazily auto-registers is left in place (it is
    part of the live registry once viz is exercised).
    """
    name = "_test_backend"
    seen: dict[str, PlotSpec] = {}

    def _render(spec: PlotSpec, **kw):
        seen["spec"] = spec
        seen["kw"] = kw
        return "rendered"

    registry.renderers.register(name, _render)
    try:
        yield name, seen
    finally:
        if name in registry.renderers:
            registry.renderers.unregister(name)


def test_render_dispatches_to_a_registered_backend(_temp_backend):
    """With a backend registered, ``render(name)`` calls it and forwards kwargs."""
    name, seen = _temp_backend
    spec = _scaling_spec()

    result = spec.render(name, dpi=120)

    assert result == "rendered"
    assert seen["spec"] is spec
    assert seen["kw"] == {"dpi": 120}


def test_render_unknown_backend_raises_naming_keyerror(_temp_backend):
    """With at least one backend registered, an unknown name is a naming KeyError.

    (When the registry is empty the no-backend ``VisualizationNotInstalled`` wins;
    once a backend exists, an unrecognized name surfaces the registry's
    name-not-found ``KeyError`` instead — the registry's own helpful lookup.)
    """
    spec = _scaling_spec()
    with pytest.raises(KeyError) as excinfo:
        spec.render("definitely_not_a_backend")
    assert "definitely_not_a_backend" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The live matplotlib reference backend (stream VIZ-MPL-CORE)
# ---------------------------------------------------------------------------


def test_register_builtin_renderers_picks_up_matplotlib():
    """``register_builtin_renderers`` discovers and registers the matplotlib backend."""
    pytest.importorskip("matplotlib")
    from tsdynamics.viz.render import register_builtin_renderers

    register_builtin_renderers()
    assert "matplotlib" in registry.renderers
    # …and it advertises all kinds + 3-D support (the conformance oracle).
    caps = getattr(registry.renderers.get("matplotlib"), "capabilities", None)
    assert caps is not None
    assert caps.kinds is None  # all-kinds shorthand
    assert caps.supports_3d is True


def test_render_matplotlib_returns_a_figure():
    """``render('matplotlib')`` now returns a real matplotlib Figure (seam is live)."""
    pytest.importorskip("matplotlib")
    import numpy as np
    from matplotlib.figure import Figure

    spec = PlotSpec(
        kind=PlotKind.TIME_SERIES,
        layers=[Layer(PlotKind.LINE, {"x": np.linspace(0, 1, 5), "y": np.zeros(5)})],
    )
    fig = spec.render("matplotlib")
    assert isinstance(fig, Figure)
    assert fig.axes  # at least one Axes was drawn
    fig.clear()


# An Agg golden/smoke pass: a minimal spec for *every* 2-D semantic PlotKind must
# render to a Figure with axes and no exception.  3-D phase-portrait kinds (whose
# only sensible layer is a 3-D mark) are excluded — they are the VIZ-MPL-3D
# follow-up — as are the deferred animation kinds.
_THREE_D_KINDS = frozenset({PlotKind.PHASE_PORTRAIT_3D})
_ANIMATION_KINDS = frozenset({PlotKind.TRAJECTORY_ANIMATION, PlotKind.ENSEMBLE_ANIMATION})
_TWO_D_SEMANTIC_KINDS = sorted(
    PlotKind.semantic_kinds() - _THREE_D_KINDS - _ANIMATION_KINDS,
    key=lambda k: k.value,
)


def _minimal_layers_for(kind: PlotKind) -> list[Layer]:
    """Build a minimal but representative layer set for a semantic 2-D kind."""
    import numpy as np

    x = np.linspace(0.0, 1.0, 8)
    y = np.sin(2 * np.pi * x)
    if kind in (
        PlotKind.IMAGE,
        PlotKind.SPACETIME,
        PlotKind.SPECTROGRAM,
        PlotKind.RECURRENCE_PLOT,
        PlotKind.BASINS_IMAGE,
    ):
        img = np.abs(np.outer(y, y)) + 0.01  # strictly positive (log-norm safe)
        return [Layer(PlotKind.IMAGE, {"z": img})]
    if kind in (PlotKind.VECTOR_FIELD, PlotKind.PHASE_PORTRAIT_FIELD):
        return [Layer(PlotKind.QUIVER, {"x": x, "y": y, "u": y, "v": x, "c": y})]
    if kind in (PlotKind.CATEGORICAL_BAR, PlotKind.FEATURE_BARS):
        return [Layer(PlotKind.BAR, {"cat": np.arange(3.0), "y": np.array([1.0, 2.0, 3.0])})]
    if kind == PlotKind.ENSEMBLE_FAN:
        return [Layer(PlotKind.AREA, {"x": x, "lo": y - 0.1, "hi": y + 0.1, "y": y})]
    if kind in (PlotKind.DIMENSION_SPECTRUM, PlotKind.SCALING_FIT):
        return [Layer(PlotKind.ERRORBAR, {"x": x, "y": y, "err": np.full_like(x, 0.05)})]
    if kind == PlotKind.HISTOGRAM_NULL:
        return [Layer(PlotKind.HISTOGRAM, {"x": y})]
    if kind in (
        PlotKind.PHASE_PORTRAIT_2D,
        PlotKind.POINCARE_SECTION,
        PlotKind.RETURN_MAP,
        PlotKind.BIFURCATION,
        PlotKind.ORBIT_DIAGRAM,
        PlotKind.EIGENVALUE_PLANE,
        PlotKind.FIXED_POINTS_OVERLAY,
    ):
        return [Layer(PlotKind.SCATTER, {"x": x, "y": y})]
    # default: a line (TIME_SERIES, COBWEB, POWER_SPECTRUM, DIAGNOSTIC_CURVE,
    # COMPLEXITY_CURVE, LINE_FAMILY, LYAPUNOV_SPECTRUM, CONTINUATION, …)
    return [Layer(PlotKind.LINE, {"x": x, "y": y})]


@pytest.mark.parametrize("kind", _TWO_D_SEMANTIC_KINDS, ids=lambda k: k.value)
def test_every_2d_kind_renders_on_agg(kind):
    """Every 2-D semantic PlotKind renders to an Agg Figure with axes, no exception."""
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    colorbar = None
    if kind in PlotSpec._COLOR_KINDS:
        from tsdynamics.viz.spec import Colorbar

        colorbar = Colorbar()
    spec = PlotSpec(kind=kind, layers=_minimal_layers_for(kind), colorbar=colorbar)
    fig = spec.render("matplotlib")
    try:
        assert isinstance(fig, Figure)
        assert fig.axes, f"{kind.value} produced no axes"
    finally:
        fig.clear()


def test_render_result_envelope_wraps_the_figure():
    """The core's ``render_result`` returns a typed RenderResult around the Figure."""
    pytest.importorskip("matplotlib")
    import numpy as np
    from matplotlib.figure import Figure

    from tsdynamics.viz.render.caps import RenderResult
    from tsdynamics.viz.render.mpl._core import render_result

    spec = PlotSpec(
        kind=PlotKind.TIME_SERIES,
        layers=[Layer(PlotKind.LINE, {"x": np.arange(4.0), "y": np.arange(4.0)})],
    )
    result = render_result(spec)
    assert isinstance(result, RenderResult)
    assert result.backend == "matplotlib"
    assert isinstance(result.figure, Figure)
    assert result.kind == PlotKind.TIME_SERIES
    result.figure.clear()


def test_3d_mark_renders_on_a_3d_axes():
    """A 3-D mark renders on an mplot3d Axes3D (drawn by VIZ-MPL-3D)."""
    pytest.importorskip("matplotlib")
    import numpy as np
    from matplotlib.figure import Figure

    from tsdynamics.viz.render import register_builtin_renderers
    from tsdynamics.viz.spec import Axis

    register_builtin_renderers()
    xyz = {"x": np.arange(3.0), "y": np.arange(3.0), "z": np.arange(3.0)}
    spec = PlotSpec(
        kind=PlotKind.PHASE_PORTRAIT_3D,
        layers=[Layer(PlotKind.LINE3D, xyz)],
        z=Axis(),
        ndim=3,
    )
    fig = spec.render("matplotlib")
    assert isinstance(fig, Figure)
    assert getattr(fig.axes[0], "name", "") == "3d"


def test_matplotlib_is_kept_last_so_an_explicit_backend_wins_by_default():
    """matplotlib stays the last-resort fallback; an explicit renderer wins default selection.

    A renderer the caller registers before the builtins (e.g. a test double or a
    plugin) must remain the default ``render()`` pick — the matplotlib oracle is
    re-seated last on each ``register_builtin_renderers`` so it never shadows it.
    """
    pytest.importorskip("matplotlib")
    from tsdynamics.viz.render import register_builtin_renderers

    # Ensure matplotlib is registered (possibly first, from earlier renders).
    register_builtin_renderers()
    name = "_priority_probe"
    captured: list[PlotSpec] = []

    def _probe(spec: PlotSpec, **_kw):
        captured.append(spec)
        return "probe"

    registry.renderers.register(name, _probe)
    try:
        out = _scaling_spec().render()  # no backend → default selection
        assert out == "probe", "the explicitly-registered renderer must win default selection"
        assert captured  # it was actually called
        # matplotlib is still present, just not the default pick.
        assert "matplotlib" in registry.renderers
        assert registry.renderers.names()[-1] == "matplotlib"
    finally:
        registry.renderers.unregister(name)
