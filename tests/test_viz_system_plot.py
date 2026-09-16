"""System families are Plottable end-to-end (stream VIZ-SYSTEM-PLOT).

``ts.systems.Lorenz().plot()`` must resolve through the visualization seam exactly like
an analysis result: a system describes itself with ``__plot_spec__()`` (a default
trajectory delegated to :meth:`tsdynamics.data.Trajectory.to_plot_spec`), and
``.plot()`` routes that spec to a backend (raising the documented
``VisualizationNotInstalled`` until one registers).  The mixin imports the viz
package only lazily, so ``import tsdynamics`` stays visualization-free.

Engine-backed (a default trajectory is integrated), so this skips cleanly where
the compiled extension is absent.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytest.importorskip("tsdynamics._rust")

import tsdynamics as ts
from tsdynamics.analysis._result import VisualizationNotInstalled
from tsdynamics.families._plottable import SystemPlottable
from tsdynamics.viz.spec import PlotKind, PlotSpec


def test_every_family_base_is_plottable():
    """The four family bases inherit the Plottable mixin."""
    for cls in (
        ts.ContinuousSystem,
        ts.DelaySystem,
        ts.DiscreteMap,
        ts.StochasticSystem,
    ):
        assert issubclass(cls, SystemPlottable)


def test_continuous_system_to_plot_spec_resolves():
    """A flow integrates a default trajectory and yields a valid spec."""
    spec = ts.systems.Lorenz().__plot_spec__()
    assert isinstance(spec, PlotSpec)
    assert isinstance(spec.kind, PlotKind)
    assert spec.layers


def test_discrete_map_to_plot_spec_resolves():
    """A map's default spec resolves too (dispatching on is_discrete)."""
    spec = ts.systems.Henon().__plot_spec__()
    assert isinstance(spec, PlotSpec)
    assert isinstance(spec.kind, PlotKind)


def test_to_plot_spec_forwards_trajectory_kwargs():
    """__plot_spec__ passes integration kwargs through to the family trajectory."""
    spec = ts.systems.Lorenz().__plot_spec__(final_time=5.0, dt=0.05)
    assert isinstance(spec, PlotSpec)


@pytest.fixture
def _no_backend(monkeypatch):
    """Force an empty renderers registry so the ``.plot`` seam raises.

    The matplotlib backend lazily auto-registers on first render as of stream
    VIZ-MPL-CORE, so the genuine no-backend path is exercised by clearing the
    registry and stubbing :func:`register_builtin_renderers` to a no-op, then
    restoring it.
    """
    from tsdynamics import registry
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


def test_render_raises_without_a_backend(_no_backend):
    """`.plot().render()` resolves end-to-end to the render seam (no backend → error).

    ``.plot()`` itself is backend-free since v6: it *builds* the
    :class:`~tsdynamics.viz.spec.PlotSpec` (the same thing ``ts.plot(system)``
    returns), and ``.render()`` is what needs a backend.
    """
    with pytest.raises(VisualizationNotInstalled):
        ts.systems.Lorenz().plot().render()


def test_system_plot_returns_a_spec_like_every_other_door(_no_backend):
    """``system.plot()`` and ``ts.plot(system)`` are the same kind of thing."""
    from tsdynamics.viz.spec import PlotSpec

    assert isinstance(ts.systems.Lorenz().plot(final_time=2.0, dt=0.05), PlotSpec)


def test_repr_mimebundle_is_noop_without_backend(_no_backend):
    """The notebook hook no-ops (returns None) until a backend registers."""
    assert ts.systems.Lorenz()._repr_mimebundle_() is None


def test_import_tsdynamics_stays_viz_free():
    """Importing tsdynamics (hence the family bases + the mixin) must not import viz."""
    code = (
        "import tsdynamics\n"
        "import sys\n"
        "assert 'tsdynamics.viz' not in sys.modules, 'viz eagerly imported by the plottable mixin'\n"
        "assert 'matplotlib' not in sys.modules, 'matplotlib eagerly imported'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
