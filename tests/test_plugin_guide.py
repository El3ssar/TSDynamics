"""Execute the extension guide's package files through real entry-point discovery."""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path


def test_documented_addon_loads_models_analyses_plots_renderer_and_solver(tmp_path):
    guide = Path(__file__).resolve().parents[1] / "docs/project/plugins.md"
    page = guide.read_text()
    modules = dict(
        re.findall(r'```python title="src/oscillator_addon/(\w+)\.py"\n(.*?)```', page, re.S)
    )
    assert set(modules) == {"models", "measurements", "plots", "primitives", "renderers", "solvers"}
    project = tomllib.loads("\n".join(re.findall(r"```toml[^\n]*\n(.*?)```", page, re.S)))
    points = project["project"]["entry-points"]
    assert len(points) == 6
    package = tmp_path / "oscillator_addon"
    package.mkdir()
    (package / "__init__.py").write_text("")
    for module, source in modules.items():
        (package / f"{module}.py").write_text(source)
    info = tmp_path / "oscillator_addon-0.1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Name: oscillator-addon\nVersion: 0.1.0\n")
    (info / "entry_points.txt").write_text(
        "\n".join(
            f"[{group}]\n" + "\n".join(f"{name} = {target}" for name, target in entries.items())
            for group, entries in points.items()
        )
    )
    source = """
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import tsdynamics as ts

model = ts.systems.MyOscillator(omega=2.0, ic=[1.0, 0.0])
traj = model.run(final_time=0.1, dt=0.01, backend="interp")
assert traj.y.shape == (11, 2)
assert ts.systems.get("MyOscillator") is type(model)
assert type(model) in ts.systems.find("MyOscillator")

measure = ts.analysis.sampled_amplitude
assert measure in ts.analysis.find(traj)
assert measure in ts.analysis.find("sampled amplitude")
result = measure(traj, component="x")
assert float(result) == np.ptp(traj["x"]) / 2.0

assert "oscillator_phase" in ts.viz.transforms.names()
assert "oscillator_stems" in ts.viz.primitives.names()
assert "model_summary" in ts.viz.renderers.names()
plot = ts.plot(traj, "oscillator_phase")
assert plot.kind.value == "phase_portrait_2d"
summary = plot.render(backend="model_summary")
assert summary.payload["kind"] == "phase_portrait_2d"
assert summary.payload["layers"] == 1
ts.viz.transforms.allow("oscillator_phase", "oscillator_stems")
stems = ts.plot(traj, "oscillator_phase", primitive="oscillator_stems")
assert len(stems.layers) == 2

direct = model.run(final_time=0.1, dt=0.01, solver="rk4", backend="interp")
aliased = model.run(final_time=0.1, dt=0.01, solver="oscillator_rk4", backend="interp")
np.testing.assert_array_equal(direct.y, aliased.y)
assert aliased.meta["solver"] == "rk4"
"""
    result = subprocess.run(
        [sys.executable, "-W", "error", "-c", source, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
