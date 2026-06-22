"""System families are Plottable end-to-end (stream VIZ-SYSTEM-PLOT).

``ts.Lorenz().plot()`` must resolve through the visualization seam exactly like
an analysis result: a system describes itself with ``to_plot_spec()`` (a default
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
    spec = ts.Lorenz().to_plot_spec()
    assert isinstance(spec, PlotSpec)
    assert isinstance(spec.kind, PlotKind)
    assert spec.layers


def test_discrete_map_to_plot_spec_resolves():
    """A map's default spec resolves too (dispatching on is_discrete)."""
    spec = ts.Henon().to_plot_spec()
    assert isinstance(spec, PlotSpec)
    assert isinstance(spec.kind, PlotKind)


def test_to_plot_spec_forwards_trajectory_kwargs():
    """to_plot_spec passes integration kwargs through to the family trajectory."""
    spec = ts.Lorenz().to_plot_spec(final_time=5.0, dt=0.05)
    assert isinstance(spec, PlotSpec)


def test_plot_raises_without_a_backend():
    """`.plot()` resolves end-to-end to the render seam (no backend → documented error)."""
    from tsdynamics import registry

    if len(registry.renderers):  # pragma: no cover - defensive if a backend leaked in
        pytest.skip("a renderer backend is registered in this session")
    with pytest.raises(VisualizationNotInstalled):
        ts.Lorenz().plot()


def test_repr_mimebundle_is_noop_without_backend():
    """The notebook hook no-ops (returns None) until a backend registers."""
    from tsdynamics import registry

    if len(registry.renderers):  # pragma: no cover
        pytest.skip("a renderer backend is registered in this session")
    assert ts.Lorenz()._repr_mimebundle_() is None


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
