"""The catalogue's copyable examples and uncached illustration acquisition."""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tsdynamics import registry

_ROOT = Path(__file__).resolve().parents[1]


def test_landing_page_workflow_runs_and_exports_a_plot(tmp_path, monkeypatch):
    """Execute the exact HTML example that first-time readers copy."""
    pytest.importorskip("matplotlib")
    source = (_ROOT / "docs/overrides/home.html").read_text()
    examples = re.findall(
        r'{%\s*filter tsd_highlight\("python"\)\s*%}(.*?){%\s*endfilter\s*%}',
        source,
        re.DOTALL,
    )
    assert examples, "The landing page must retain a runnable Python workflow."
    monkeypatch.chdir(tmp_path)
    namespace = {}
    for example in examples:
        exec(compile(textwrap.dedent(example).strip(), "landing-page-workflow", "exec"), namespace)
    assert (tmp_path / "lorenz.png").read_bytes().startswith(b"\x89PNG")
    assert np.isfinite(np.asarray(namespace["spectrum"])).all()


def test_every_quickstart_runs_without_docs_or_plotting_dependencies():
    """Keep model examples executable in core/dev environments, without extras."""
    script = textwrap.dedent(
        """
        import importlib.abc
        import importlib.util
        import re
        import sys
        from pathlib import Path

        class WithoutOptionalExtras(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                blocked = {
                    'mkdocs', 'mkdocstrings', 'material', 'mkdocs_glightbox',
                    'markdown', 'pymdownx', 'griffe', 'matplotlib', 'plotly',
                }
                if fullname.split('.')[0] in blocked:
                    raise ModuleNotFoundError('Optional extra blocked: ' + fullname, name=fullname)

        sys.meta_path.insert(0, WithoutOptionalExtras())
        import numpy as np
        from tsdynamics import registry

        spec = importlib.util.spec_from_file_location(
            'lean_docs_hook', Path('hooks/docs_autogen.py').resolve()
        )
        hook = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hook)
        catalogue = hook._catalog.load_catalog()
        names = [entry.name for entry in registry.all_systems() if entry.is_builtin]
        for name in names:
            block = '\\n'.join(hook._define_block(catalogue.by_name(name)))
            source = re.search(r'```python\\n(.*?)```', block, re.DOTALL).group(1)
            namespace = {}
            exec(compile(source, 'lean-quickstart:' + name, 'exec'), namespace)
            trajectory = namespace['traj']
            assert len(trajectory) > 1, name
            assert np.isfinite(trajectory.y).all(), name
        print(len(names))
        """
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script], cwd=_ROOT, text=True, capture_output=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert int(result.stdout.strip()) == sum(entry.is_builtin for entry in registry.all_systems())


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


@pytest.fixture
def new_catalogue_model(docs_hook, tmp_path, monkeypatch):
    """A source-backed future builtin, with no editorial entry or handwritten page."""
    import tsdynamics as ts

    name = "DocumentationProbe"
    module_name = "tsdynamics.systems.continuous._documentation_probe"
    source = tmp_path / "future_model.py"
    source.write_text(
        textwrap.dedent('''
        import tsdynamics as ts

        class DocumentationProbe(ts.ContinuousSystem):
            """A neutral rotation with one contracting direction.

            Parameters
            ----------
            rate : float
                Decay rate of the third state, per time unit.
            """
            params = {"rate": 0.01}
            variables = ("u", "v", "w")
            default_ic = [1.0, 0.0, 1.0]

            @staticmethod
            def _equations(y, t, *, rate):
                return (-y(1), y(0), -rate * y(2))
    ''')
    )
    monkeypatch.setitem(registry._BY_NAME, name, [])
    spec = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    # Equivalent to exporting the new class from its catalogue category.
    monkeypatch.setattr(ts.systems, name, module.DocumentationProbe, raising=False)
    docs_hook._catalog.load_catalog.cache_clear()
    record = docs_hook._catalog.load_catalog().by_name(name)
    assert record is not None
    monkeypatch.setenv("TSD_DOCS_PROPS", "1")
    monkeypatch.setattr(docs_hook._properties, "CACHE_DIR", tmp_path / "properties")
    monkeypatch.setattr(docs_hook._figures, "CACHE_DIR", tmp_path / "figures-cache")
    monkeypatch.setattr(docs_hook._figures, "OUT_DIR", tmp_path / "figures")
    monkeypatch.setattr(docs_hook._viewer, "CACHE_DIR", tmp_path / "viewers")
    yield record
    docs_hook._catalog.load_catalog.cache_clear()


def test_a_new_system_automatically_gets_a_complete_measured_page(
    docs_hook, new_catalogue_model, monkeypatch
):
    """Exercise registration through real equations, poster, viewer and diagnostics."""
    pytest.importorskip("matplotlib")
    record = new_catalogue_model
    assert record.name not in docs_hook._catalog._ed_section("systems")
    assert record.param_roles["rate"] == "Decay rate of the third state, per time unit."
    monkeypatch.setattr(docs_hook, "_ONLY", {record.name})
    monkeypatch.setattr(docs_hook, "WITH_FIGURES", True)
    monkeypatch.delenv("TSD_DOCS_FIGURES", raising=False)
    monkeypatch.setattr(docs_hook, "_GENERATED", {})
    monkeypatch.setattr(docs_hook, "_VIEWERS", {})
    monkeypatch.setattr(docs_hook, "_build_gallery", lambda: None)
    docs_hook.on_config(SimpleNamespace(nav=[{"Systems": []}]))
    page = docs_hook._GENERATED[docs_hook._system_uri(record)]
    assert page.startswith('---\ntitle: "DocumentationProbe"\n---\n')
    assert "Decay rate of the third state" in page
    assert "\\dot" in page and "rate" in page
    assert 'class="ts-prop"' in page and "Not computed" not in page
    assert "No primary literature reference is recorded" in page
    assert (docs_hook._figures.OUT_DIR / f"{record.name}.png").read_bytes().startswith(b"\x89PNG")
    assert docs_hook._VIEWERS[f"assets/threejs/{record.name}.html"]

    props = docs_hook._properties.compute_properties(record)
    np.testing.assert_allclose(props["lyapunov_spectrum"]["value"], [0.0, 0.0, -0.01], atol=2e-4)
    assert props["equilibria"]["status"] == "measured"
    assert props["provenance"]["model_identity"]["module"] == record.cls.__module__
    monkeypatch.setattr(
        docs_hook._properties,
        "_compute_diagnostics",
        lambda entry: pytest.fail("A cached new model must not recompute its diagnostics."),
    )
    assert docs_hook._properties.compute_properties(record) == props


def test_a_session_only_system_has_an_honest_unavailable_card(docs_hook, monkeypatch):
    """Never replace an ephemeral class with a same-named built-in calculation."""
    import tsdynamics as ts

    monkeypatch.setitem(registry._BY_NAME, "SessionOnlyModel", [])

    class SessionOnlyModel(ts.ContinuousSystem):
        variables = ("x",)

        @staticmethod
        def _equations(y, t):
            return (-y(0),)

    entry = next(
        item for item in registry.all_systems(builtin=None) if item.cls is SessionOnlyModel
    )
    record = docs_hook._catalog._merge_record(entry, {})
    result = docs_hook._properties._compute_diagnostics(record)
    assert result["lyapunov_spectrum"]["status"] == "unavailable"
    assert "current session" in result["lyapunov_spectrum"]["reason"]
    assert "value" not in result["lyapunov_spectrum"]


def test_worker_timeout_keeps_completed_roots_and_never_invents_rates(
    docs_hook, new_catalogue_model, monkeypatch
):
    import json

    partial = {
        "name": new_catalogue_model.name,
        "family": new_catalogue_model.family,
        "fingerprint": docs_hook._properties._diagnostics_module().fingerprint(new_catalogue_model),
        "equilibria": {"status": "measured", "value": 1, "settings": {"seed": 0}},
        "provenance": {"parameters": {"rate": 0.01}},
    }

    def interrupted(command, **kwargs):
        assert kwargs["timeout"] == docs_hook._properties.NEW_MODEL_TIMEOUT
        raise subprocess.TimeoutExpired(command, kwargs["timeout"], output=json.dumps(partial))

    monkeypatch.setattr(docs_hook._properties.subprocess, "run", interrupted)
    result = docs_hook._properties._compute_diagnostics(new_catalogue_model)
    assert result["equilibria"] == partial["equilibria"]
    assert result["lyapunov_spectrum"]["status"] == "unavailable"
    assert "budget" in result["lyapunov_spectrum"]["reason"]
    assert "value" not in result["kaplan_yorke"]


def test_external_source_identities_are_distinct_and_verified(
    docs_hook, new_catalogue_model, monkeypatch, tmp_path
):
    """Same-named, byte-identical class bodies in different modules cannot alias."""
    import inspect
    from dataclasses import replace

    original = new_catalogue_model
    other_file = tmp_path / "another_model.py"
    other_file.write_text(Path(inspect.getsourcefile(original.cls)).read_text())
    module_name = "tsdynamics.systems.continuous._another_documentation_probe"
    spec = importlib.util.spec_from_file_location(module_name, other_file)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    # The second identity is used only for this fingerprint/dispatch check, not
    # added as a second same-named builtin in the public catalogue.
    monkeypatch.setattr(registry, "register_class", lambda cls: None)
    spec.loader.exec_module(module)
    other = replace(original, cls=module.DocumentationProbe, module=module_name)
    diagnostics = docs_hook._properties._diagnostics_module()
    assert diagnostics.fingerprint(original) != diagnostics.fingerprint(other)
    assert docs_hook._properties.cache_key(original) != docs_hook._properties.cache_key(other)

    worker_spec = importlib.util.spec_from_file_location(
        "docs_identity_worker", _ROOT / "docs/_tooling/diagnostics_worker.py"
    )
    worker = importlib.util.module_from_spec(worker_spec)
    worker_spec.loader.exec_module(worker)
    request = {
        "module": original.cls.__module__,
        "qualname": original.cls.__qualname__,
        "source_file": str(Path(inspect.getsourcefile(original.cls)).resolve()),
        "family": original.family,
        "fingerprint": diagnostics.fingerprint(other),
    }
    with pytest.raises(ValueError, match="differs from the request"):
        worker.resolve_entry(request)


def test_a_user_model_named_like_a_builtin_is_measured_as_its_own_class(
    docs_hook, monkeypatch, tmp_path
):
    """An external Lorenz with linear decay must not get the built-in's spectrum."""
    monkeypatch.syspath_prepend(str(tmp_path))
    source = tmp_path / "external_lorenz.py"
    source.write_text(
        textwrap.dedent("""
        import tsdynamics as ts

        class Lorenz(ts.ContinuousSystem):
            variables = ("a", "b", "c")
            default_ic = [1.0, 1.0, 1.0]

            @staticmethod
            def _equations(y, t):
                return (-y(0), -2*y(1), -3*y(2))
    """)
    )
    monkeypatch.setitem(registry._BY_NAME, "Lorenz", list(registry._BY_NAME["Lorenz"]))
    spec = importlib.util.spec_from_file_location("external_lorenz", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "external_lorenz", module)
    spec.loader.exec_module(module)
    entry = next(item for item in registry.all_systems(builtin=False) if item.cls is module.Lorenz)
    record = docs_hook._catalog._merge_record(entry, {})
    monkeypatch.setenv("TSD_DOCS_PROPS", "1")
    result = docs_hook._properties._compute_diagnostics(record)
    np.testing.assert_allclose(result["lyapunov_spectrum"]["value"], [-1, -2, -3], atol=1e-5)
    assert result["provenance"]["model_identity"]["module"] == "external_lorenz"


def test_an_importable_nested_class_keeps_its_qualified_identity(docs_hook, monkeypatch, tmp_path):
    """Indented class source and dotted qualnames are valid reproducible models."""
    module_name = "nested_documentation"
    source = tmp_path / f"{module_name}.py"
    source.write_text(
        textwrap.dedent('''
        import tsdynamics as ts

        class Models:
            class NestedDocumentationModel(ts.ContinuousSystem):
                """Two independent contracting directions."""
                variables = ("u", "v")
                default_ic = [1.0, 1.0]

                @staticmethod
                def _equations(y, t):
                    return (-y(0), -2*y(1))
    ''')
    )
    monkeypatch.setitem(registry._BY_NAME, "NestedDocumentationModel", [])
    spec = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    entry = next(
        item
        for item in registry.all_systems(builtin=False)
        if item.cls is module.Models.NestedDocumentationModel
    )
    record = docs_hook._catalog._merge_record(entry, {})
    monkeypatch.setenv("TSD_DOCS_PROPS", "1")
    properties = docs_hook._properties
    assert properties.cache_key(record)
    result = properties._compute_diagnostics(record)
    np.testing.assert_allclose(result["lyapunov_spectrum"]["value"], [-1, -2], atol=1e-5)
    assert result["provenance"]["model_identity"]["class"] == "Models.NestedDocumentationModel"


def test_every_non_field_catalogue_model_automatically_has_an_animated_view(docs_hook, monkeypatch):
    """Do not silently leave scalar, noisy or uncurated larger models static."""
    monkeypatch.delenv("TSD_DOCS_FIGURES", raising=False)
    viewer = docs_hook._viewer
    for record in docs_hook._catalog.load_catalog().records:
        if not viewer._is_field(record):
            assert viewer.eligible(record), record.name


@pytest.mark.parametrize(
    "name",
    [
        "Logistic",
        "Henon",
        "FoldedTowel",
        "OrnsteinUhlenbeck",
        "Oregonator",
        "BlinkingVortex",
        "Hopfield",
        "MackeyGlass",
    ],
)
def test_generated_viewers_have_real_timelines_and_honest_coordinates(docs_hook, name):
    """Exercise actual scalar/planar/3-D iterates, time paths and projections."""
    record = docs_hook._catalog.load_catalog().by_name(name)
    payload = docs_hook._viewer._build_payload(record, second=False)
    assert payload is not None
    animation = payload["metadata"]["animation"]
    assert animation["n_samples"] > 8
    assert animation["duration"] == docs_hook._viewer._DURATION_S
    view = payload["metadata"]["documentation_view"]
    positions = np.asarray(payload["geometries"][0]["positions"]).reshape(-1, 3)
    assert np.isfinite(positions).all()
    assert len(positions) > 8
    if record.family == "map":
        assert payload["geometries"][0]["type"] == "points"
        assert view["sampling"] == "iteration"
        assert "iteration order" in view["description"]
    elif name in ("OrnsteinUhlenbeck", "Oregonator"):
        assert view["coordinates"][0] == "t"
        assert view["sampling"] == "time"
        assert np.all(np.diff(positions[:, 0]) > 0)
    elif name == "BlinkingVortex":
        assert view["coordinates"] == ["r cos θ", "r sin θ"]
    elif name == "MackeyGlass":
        assert "t − τ" in view["coordinates"][1]
    else:
        assert view["coordinates"] == list(record.variables[:3])


@pytest.mark.parametrize("name", ["Logistic", "Henon", "FoldedTowel"])
def test_map_animation_preserves_successive_iterates_without_connecting_them(docs_hook, name):
    """Point reveals must not spatially sort, interpolate or pool independent orbits."""
    record = docs_hook._catalog.load_catalog().by_name(name)
    actual = docs_hook._viewer._map_cloud(record)
    orbit = record.cls().run(steps=8_000, seed=0).y[100:]
    expected = np.column_stack((orbit[:-1, 0], orbit[1:, 0])) if record.dim == 1 else orbit[:, :3]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("kind", ["figure", "viewer"])
def test_inherited_equation_changes_invalidate_rendered_model_caches(docs_hook, monkeypatch, kind):
    """A subclass's unchanged body cannot hide a repaired parent equation."""
    import diagnostics

    record = docs_hook._catalog.load_catalog().by_name("Coullet")
    cache_key = (
        docs_hook._figures.cache_key
        if kind == "figure"
        else lambda entry: docs_hook._viewer.cache_key(entry, second=False)
    )
    original = cache_key(record)
    class_digest = diagnostics._class_digest
    monkeypatch.setattr(
        diagnostics,
        "_class_digest",
        lambda cls: b"repaired parent equation" if cls.__name__ == "Arneodo" else class_digest(cls),
    )
    assert cache_key(record) != original


@pytest.mark.parametrize("kind", ["figure", "viewer", "gallery"])
def test_native_kernel_changes_invalidate_rendered_model_caches(
    docs_hook, monkeypatch, tmp_path, kind
):
    """Change a real native source file while leaving the model and recipe alone."""
    import diagnostics

    crate = tmp_path / "crates"
    native = crate / "integration/src/lib.rs"
    native.parent.mkdir(parents=True)
    (crate / "Cargo.toml").write_text("[workspace]\n")
    (crate / "Cargo.lock").write_text("version = 4\n")
    native.write_text("fn integrate() -> f64 { 1.0 }\n")
    monkeypatch.setattr(diagnostics, "ROOT", tmp_path)
    monkeypatch.setattr(docs_hook._figures, "ROOT", tmp_path)
    record = docs_hook._catalog.load_catalog().by_name("Lorenz")
    if kind == "gallery":

        def cache_key(entry):
            return docs_hook._gallery.cache_key("subject = 1", "ts.plot(subject)")
    else:
        cache_key = (
            docs_hook._figures.cache_key
            if kind == "figure"
            else lambda entry: docs_hook._viewer.cache_key(entry, second=False)
        )
    original = cache_key(record)
    native.write_text("fn integrate() -> f64 { 2.000 }\n")
    assert cache_key(record) != original


def test_gallery_cache_does_not_recompute_figures_just_for_a_release_stamp(docs_hook, monkeypatch):
    import tsdynamics

    original = docs_hook._gallery.cache_key("subject = 1", "ts.plot(subject)")
    monkeypatch.setattr(tsdynamics, "__version__", "9999.0.0")
    assert docs_hook._gallery.cache_key("subject = 1", "ts.plot(subject)") == original


def test_render_source_fingerprints_are_portable_and_follow_live_changes(docs_hook, tmp_path):
    """Content and logical labels matter; installation location and CRLF do not."""
    checkout = tmp_path / "checkout.py"
    wheel = tmp_path / "wheel.py"
    checkout.write_bytes(b"def render():\n return 1\n")
    wheel.write_bytes(b"def render():\r\n return 1\r\n")

    def signature(path):
        return (("package/viz/render.py", str(path), path.stat().st_mtime_ns, path.stat().st_size),)

    digest = docs_hook._figures._render_sources_digest
    original = digest(signature(checkout))
    assert digest(signature(wheel)) == original
    checkout.write_bytes(b"def render():\n return 2.000\n")
    assert digest(signature(checkout)) != original


def test_render_fingerprints_ignore_docs_and_release_stamps_but_keep_plot_hooks(
    docs_hook, tmp_path
):
    """A wording/release edit is cheap; a changed __plot_spec__ is a real cache miss."""
    source = tmp_path / "model.py"
    template = '''
__version__ = "VERSION"
class View:
    """WORDS"""
    def __plot_spec__(self):
        return VALUE
'''

    def digest(version, words, value):
        source.write_text(
            template.replace("VERSION", version).replace("WORDS", words).replace("VALUE", value)
        )
        signature = (
            ("package/viz/view.py", str(source), source.stat().st_mtime_ns, source.stat().st_size),
        )
        return docs_hook._figures._render_sources_digest(signature)

    original = digest("1.0", "Old documentation", "1")
    assert digest("2.0", "New documentation", "1") == original
    assert digest("2.0", "New documentation", "2") != original


@pytest.mark.parametrize("renderer", ["figure", "gallery"])
@pytest.mark.parametrize("fail", [False, True])
def test_docs_rendering_restores_the_callers_drawing_state(
    docs_hook, tmp_path, monkeypatch, renderer, fail
):
    """A docs build must not change how the next ordinary still/movie is drawn."""
    import logging

    matplotlib = pytest.importorskip("matplotlib")
    import tsdynamics as ts

    matplotlib.get_backend()  # Resolve the lazy backend before taking the snapshot.
    previous_theme = ts.viz.get_theme()
    font_logger = logging.getLogger("matplotlib.font_manager")
    previous_level = font_logger.level

    def unavailable(*args, **kwargs):
        raise RuntimeError("deliberate illustration failure")

    with matplotlib.rc_context(
        {
            "figure.facecolor": "#fdfcfb",
            "axes.facecolor": "#f1f2f3",
            "savefig.transparent": False,
            "axes.grid": True,
            "font.size": 13,
        }
    ):
        ts.viz.set_theme("dark")
        caller_theme = ts.viz.get_theme()
        before = matplotlib.rcParams.copy()
        try:
            if renderer == "gallery":
                monkeypatch.setattr(docs_hook._gallery, "CACHE_DIR", tmp_path / "gallery")
                if fail:
                    monkeypatch.setattr(docs_hook._gallery, "_render_one", unavailable)
                build = docs_hook._gallery.render_all(only={"ensemble_fan"}, force=True)
                assert bool(build.failures) == fail
                if not fail:
                    assert len(build.cells) == 1
                    assert build.cells[0].cached_path.is_file()
                    assert 'ts.plot(batch, "ensemble_fan")' in build.cells[0].snippet
            else:
                monkeypatch.setattr(docs_hook._figures, "CACHE_DIR", tmp_path / "cache")
                monkeypatch.setattr(docs_hook._figures, "OUT_DIR", tmp_path / "out")
                monkeypatch.setitem(
                    docs_hook._figures.FIG_OVERRIDES,
                    "Lorenz",
                    {"final_time": 4.0, "dt": 0.02},
                )
                if fail:
                    monkeypatch.setattr(docs_hook._figures, "_render_ode", unavailable)
                rendered = docs_hook._figures.render(
                    docs_hook._catalog.load_catalog().by_name("Lorenz")
                )
                assert (rendered is None) == fail
                if not fail:
                    assert rendered.is_file()
            assert matplotlib.rcParams == before
            assert ts.viz.get_theme() == caller_theme
            assert font_logger.level == previous_level
        finally:
            ts.viz.set_theme(previous_theme)
            font_logger.setLevel(previous_level)
