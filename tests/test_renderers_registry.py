"""Meta-QA over the visualization renderers registry (stream WS-VIZREG).

The renderers registry (:data:`tsdynamics.registry.renderers`) is the third
generic :class:`~tsdynamics.registry.Registry` container, mirroring
:data:`~tsdynamics.registry.analyses` / :data:`~tsdynamics.registry.transforms`.
It ships **empty** — visualization is deferred (decision D6), so no backend
registers in v4.0 — and is the seam a future multi-backend plotting suite plugs
into by construction.

These tests freeze that contract:

- the registry exists, is a :class:`~tsdynamics.registry.Registry` tagged
  ``"renderer"``, and is a *distinct* instance from the analyses/transforms ones;
- it is initially empty (no backend ships);
- ``import tsdynamics`` (and ``import tsdynamics.viz``) pull in **no plot
  library** — core stays plotting-free;
- :mod:`tsdynamics.viz` wires entry-point discovery (the ``tsdynamics.renderers``
  group) exactly like analyses/transforms;
- resolving a backend through :meth:`tsdynamics.viz.spec.PlotSpec.render` raises a
  *helpful*, message-carrying ``VisualizationNotInstalled`` (the canonical named
  exception, not a bare ``KeyError``) while the registry is empty;
- once a backend registers, ``render(name)`` dispatches to it (proving the seam
  is live), and an unknown backend name then raises a naming ``KeyError``.

This file imports no plotting backend itself; the "backend" used to exercise the
live-dispatch path is a trivial local callable, registered and removed within a
fixture so no global state leaks.
"""

from __future__ import annotations

import sys

import pytest

from tsdynamics import registry
from tsdynamics.registry import Registry
from tsdynamics.viz import RENDERERS_GROUP, PlotKind, PlotSpec, discover_plugins

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


def test_renderers_registry_ships_empty():
    """No rendering backend ships in v4.0 — the registry starts empty."""
    assert len(registry.renderers) == 0
    assert registry.renderers.names() == []


# ---------------------------------------------------------------------------
# Core stays plotting-free
# ---------------------------------------------------------------------------


def test_importing_viz_pulls_no_plot_library():
    """Importing tsdynamics / tsdynamics.viz must not import a plot backend."""
    import tsdynamics  # noqa: F401  (already imported; assert the seam stayed clean)
    import tsdynamics.viz  # noqa: F401

    for banned in ("matplotlib", "matplotlib.pyplot", "plotly", "plotly.graph_objects"):
        assert banned not in sys.modules, f"{banned!r} was imported by tsdynamics"


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


def test_render_with_empty_registry_raises_helpful_error():
    """``render('matplotlib')`` raises a *helpful* error, not a bare KeyError/ImportError."""
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
    # the deferred-viz machinery reads like any optional dependency
    # (``except ImportError`` catches it).
    assert isinstance(excinfo.value, ImportError)
    assert not isinstance(excinfo.value, KeyError)


def test_render_default_backend_also_raises_when_empty():
    """``render()`` with no backend argument is the common path and raises cleanly."""
    from tsdynamics.analysis._result import VisualizationNotInstalled

    with pytest.raises(VisualizationNotInstalled):
        _scaling_spec().render()


# ---------------------------------------------------------------------------
# Once a backend registers, the seam is live (and unknown names raise on name)
# ---------------------------------------------------------------------------


@pytest.fixture
def _temp_backend():
    """Register a trivial renderer, yield its name, then remove it.

    Proves the registry seam dispatches by name without shipping (or importing)
    a real plotting backend, and restores the empty registry afterwards.
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
        registry.renderers.unregister(name)
    assert len(registry.renderers) == 0  # back to the shipped (empty) state


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
