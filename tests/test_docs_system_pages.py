"""The catalogue's copyable examples and uncached illustration acquisition."""

from __future__ import annotations

import html
import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tsdynamics import registry

_ROOT = Path(__file__).resolve().parents[1]


def test_landing_page_workflow_runs_and_exports_a_plot(tmp_path, monkeypatch):
    """Execute the exact HTML example that first-time readers copy."""
    source = (_ROOT / "docs/overrides/home.html").read_text()
    examples = re.findall(
        r'<pre><code class="language-python">(.*?)</code></pre>', source, re.DOTALL
    )
    assert examples, "The landing page must retain a runnable Python workflow."
    monkeypatch.chdir(tmp_path)
    namespace = {}
    for example in examples:
        exec(compile(html.unescape(example), "landing-page-workflow", "exec"), namespace)
    assert (tmp_path / "lorenz.png").read_bytes().startswith(b"\x89PNG")
    assert np.isfinite(np.asarray(namespace["spectrum"])).all()


@pytest.fixture(scope="module")
def docs_hook():
    spec = importlib.util.spec_from_file_location(
        "docs_system_page_hook", _ROOT / "hooks/docs_autogen.py"
    )
    assert spec is not None and spec.loader is not None
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    return hook


@pytest.mark.parametrize(
    "name", [entry.name for entry in registry.all_systems() if entry.is_builtin]
)
def test_every_system_quickstart_runs_from_an_empty_namespace(docs_hook, name):
    """Check the exact generated first example, including family/IC conventions.

    A dead-spelling blacklist alone missed VossDelay's escaping default history.
    This runs the actual small example and builds its lazy plot, without importing
    a drawing backend or relying on injected names.
    """
    record = docs_hook._catalog.load_catalog().by_name(name)
    block = "\n".join(docs_hook._define_block(record))
    source = re.search(r"```python\n(.*?)```", block, re.DOTALL).group(1)
    namespace = {}
    exec(compile(source, f"generated:{name}", "exec"), namespace)
    trajectory = namespace["traj"]
    assert len(trajectory) > 1
    assert np.isfinite(trajectory.y).all(), name


def test_explicit_array_initial_condition_is_not_compared_as_a_boolean(docs_hook):
    system = SimpleNamespace(dim=3)
    initial = np.array([0.1, 0.2, 0.3])
    np.testing.assert_array_equal(docs_hook._figures._resolve_ic(system, initial), initial)


@pytest.mark.parametrize("acquire", ["engine", "scipy", "viewer", "sampling_pilot"])
def test_uncached_illustration_can_retry_without_a_removed_resolver(
    docs_hook, monkeypatch, acquire
):
    """A random-IC retry must work for a model with no declared basin start."""
    attempts = []
    t = np.linspace(0.0, 2.0, 128)
    y = np.column_stack((np.sin(t), np.cos(t), t))

    class NoDeclaredStart:
        dim = 3
        _default_ic = None

        def run(self, **kwargs):
            attempts.append(np.asarray(kwargs["ic"]).copy())
            if len(attempts) == 1:
                raise RuntimeError("first initial condition escapes")
            return SimpleNamespace(t=t, y=y)

        def _rhs_numeric(self):
            return lambda state, time: -state

    entry = SimpleNamespace(
        name="NoDeclaredStart",
        family="ode",
        dim=3,
        cls=NoDeclaredStart,
        viewer={},
        projection=None,
        projection2=None,
        variables=("x", "y", "z"),
    )
    figures, viewer, pilot = docs_hook._figures, docs_hook._viewer, docs_hook._plot_dt
    monkeypatch.setattr(pilot, "choose_plot_dt", lambda *args, **kwargs: 0.01)
    monkeypatch.setattr(viewer, "_smooth", lambda points: points)
    if acquire == "scipy":
        import scipy.integrate

        def solve(fun, span, initial, **kwargs):
            attempts.append(np.asarray(initial).copy())
            return SimpleNamespace(t=t, y=y.T, status=0, success=len(attempts) > 1)

        monkeypatch.setattr(scipy.integrate, "solve_ivp", solve)
        result = figures._ode_trajectory_scipy(entry, {"final_time": 2.0})[1]
    elif acquire == "engine":
        result = figures._ode_trajectory_engine(entry, {"final_time": 2.0})[1]
    elif acquire == "viewer":
        result = viewer._ode_cloud(entry, second=False)
    else:
        result = pilot._ode_pilot(entry, 2.0, 0.01)
    assert result is not None and len(result) > 50
    assert len(attempts) == 2
    assert not np.array_equal(attempts[0], attempts[1])
    np.testing.assert_array_equal(attempts[0], np.random.default_rng(42).uniform(0.0, 1.0, 3))


def test_delay_viewer_honors_the_same_bounded_history_as_the_static_figure(docs_hook, monkeypatch):
    seen = []

    class Delay:
        dim = 1

        def run(self, **kwargs):
            seen.append(kwargs["history"](0.0))
            return SimpleNamespace(y=np.linspace(0.1, 0.2, 300).reshape(-1, 1))

        def _delays(self):
            return [0.2]

    monkeypatch.setattr(docs_hook._viewer, "_smooth", lambda points: points)
    entry = SimpleNamespace(name="VossDelay", cls=Delay)
    assert docs_hook._viewer._dde_delay_embedding(entry) is not None
    assert seen == [[0.15]]


def test_missing_illustrations_have_an_explicit_reader_visible_state(docs_hook, monkeypatch):
    record = docs_hook._catalog.load_catalog().by_name("Lorenz")
    monkeypatch.setattr(docs_hook, "WITH_FIGURES", True)
    monkeypatch.setitem(
        docs_hook._figures.RENDER_FAILURES, record.name, "No bounded orbit was found."
    )
    block = "\n".join(
        docs_hook._attractor_block(record, docs_hook._system_uri(record), False, False)
    )
    assert "Illustration unavailable" in block
    assert "No bounded orbit was found" in block
    assert "TSD_DOCS_FIGURES=1" not in block


def test_every_generated_page_has_a_readable_route_from_example_to_evidence(docs_hook, monkeypatch):
    """Cover the page template for every live model without recomputing metrics.

    Numerical evidence has its own diagnostics tests. Here its deliberately
    unavailable state checks that the page still explains the model, gives a
    runnable entry point and exposes methods with resolvable documentation links.
    """
    import posixpath

    cards = {
        key: {"status": "unavailable", "reason": "Not measured in this template test."}
        for key in ("lyapunov_spectrum", "kaplan_yorke", "divergence", "equilibria")
    }
    monkeypatch.setattr(docs_hook._properties, "compute_properties", lambda record: cards)
    monkeypatch.setattr(docs_hook, "WITH_FIGURES", False)
    for record in docs_hook._catalog.load_catalog().records:
        page = docs_hook._system_page(record, has_viewer=False, has_figure=False)
        assert page.index("## Start here") < page.index("## Dynamics at the default parameters")
        assert page.index("## Dynamics at the default parameters") < page.index("## Definition")
        assert "## Parameters" in page and "## Reference" in page
        assert page.count('class="ts-prop"') == 4
        assert "TODO" not in page
        for target in re.findall(r"\]\(([^)]+\.md)(?:#[^)]*)?\)", page):
            relative = posixpath.normpath(
                posixpath.join(posixpath.dirname(docs_hook._system_uri(record)), target)
            )
            assert (_ROOT / "docs" / relative).exists(), (record.name, target)
