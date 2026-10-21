"""Installed system plugins use real metadata discovery in isolated interpreters."""

from __future__ import annotations

import subprocess
import sys
import textwrap


def _distribution(site, name, entries, modules):
    info = site / f"{name.replace('-', '_')}-0.0.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Name: {name}\nVersion: 0.0.0\n")
    (info / "entry_points.txt").write_text(f"[tsdynamics.systems]\n{entries}\n")
    for module, source in modules.items():
        (site / f"{module}.py").write_text(textwrap.dedent(source))


def _flow(name):
    return f"""
        import tsdynamics as ts
        from tsdynamics import ContinuousSystem, __version__

        PLUGIN_VERSION = __version__
        assert ContinuousSystem is ts.ContinuousSystem

        class {name}(ts.ContinuousSystem):
            params = {{"rate": 1.0}}
            dim = 1
            variables = ("x",)

            @staticmethod
            def _equations(y, t, *, rate):
                return (rate * y(0),)
    """


def _run(site, source):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]);\n" + textwrap.dedent(source),
            str(site),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_installed_system_joins_catalogue_search_and_runs(tmp_path):
    _distribution(
        tmp_path,
        "published-system",
        "PublishedFlow = published_flow:PublishedFlow",
        {
            "published_flow": textwrap.dedent(_flow("PublishedFlow"))
            + "\nfrom tsdynamics import plot as PLUGIN_PLOT\n"
            + "PLUGIN_BUILTIN = ts.systems.Lorenz\n"
        },
    )
    _run(
        tmp_path,
        """
        import math
        import tsdynamics as ts
        from tsdynamics import registry

        class LocalProbe(ts.ContinuousSystem):
            params = {}
            dim = 1

            @staticmethod
            def _equations(y, t):
                return (y(0),)

        catalogue = ts.systems
        from published_flow import PublishedFlow, PLUGIN_VERSION, PLUGIN_PLOT, PLUGIN_BUILTIN

        assert PLUGIN_VERSION == ts.__version__
        assert PLUGIN_PLOT is ts.plot is ts.viz.plot
        assert PLUGIN_BUILTIN is ts.systems.Lorenz
        assert catalogue.PublishedFlow is PublishedFlow
        assert catalogue.get("PublishedFlow") is PublishedFlow
        assert "PublishedFlow" in catalogue.names()
        assert "PublishedFlow" in catalogue.__all__
        assert "PublishedFlow" in dir(catalogue)
        assert sorted(catalogue.__all__) == dir(catalogue)
        assert catalogue.find("PublishedFlow") == [PublishedFlow]
        assert PublishedFlow in catalogue.find(family="ode", dim=1)
        assert "ts.systems.PublishedFlow" in repr(catalogue.find("PublishedFlow"))
        assert "LocalProbe" not in catalogue.names()
        assert catalogue.find("LocalProbe") == []
        assert registry.get("LocalProbe", builtin=False).cls is LocalProbe
        assert registry.get("PublishedFlow", builtin=False).cls is PublishedFlow
        assert not registry.get("PublishedFlow").is_builtin
        assert PublishedFlow not in [entry.cls for entry in registry.all_systems()]
        assert catalogue.Lorenz.__module__.startswith("tsdynamics.systems.")
        traj = PublishedFlow(ic=[1.0]).run(final_time=0.1, dt=0.1, backend="reference")
        assert traj.y.shape == (2, 1)
        assert math.isclose(traj.y[-1, 0], math.exp(0.1), rel_tol=1e-6)
        """,
    )


def test_invalid_system_plugins_warn_without_shadowing_or_hiding_valid_models(tmp_path):
    _distribution(
        tmp_path,
        "mixed-systems",
        """PublishedFlow = published_flow:PublishedFlow
BrokenFlow = broken_flow:BrokenFlow
Lorenz = conflicting_flow:Lorenz
get = reserved_flow:get
Alias = published_flow:PublishedFlow
NotASystem = invalid_flow:NotASystem
Existing = existing_flow:Existing
AbstractFlow = abstract_flow:AbstractFlow""",
        {
            "published_flow": _flow("PublishedFlow"),
            "broken_flow": "raise RuntimeError('broken dependency')\n",
            "conflicting_flow": _flow("Lorenz"),
            "reserved_flow": _flow("get"),
            "invalid_flow": "NotASystem = object()\n",
            "existing_flow": "import preexisting_flow\n" + textwrap.dedent(_flow("Existing")),
            "preexisting_flow": _flow("Existing"),
            "abstract_flow": """
                import tsdynamics as ts
                class AbstractFlow(ts.ContinuousSystem):
                    pass
            """,
        },
    )
    _run(
        tmp_path,
        """
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import tsdynamics as ts
            catalogue = ts.systems
        messages = [str(warning.message) for warning in caught]
        assert any("BrokenFlow" in message and "broken dependency" in message for message in messages)
        for name in ("Lorenz", "get", "Alias", "NotASystem", "Existing", "AbstractFlow"):
            assert any(f"ignored system plugin {name!r}" in message for message in messages), messages
        assert catalogue.get("Lorenz") is catalogue.Lorenz
        assert catalogue.Lorenz.__module__.startswith("tsdynamics.systems.")
        assert callable(catalogue.get)
        assert catalogue.get("PublishedFlow").__name__ == "PublishedFlow"
        for name in ("BrokenFlow", "Alias", "NotASystem", "Existing", "AbstractFlow"):
            assert name not in catalogue.names()
            assert catalogue.find(name) == []
        """,
    )


def test_duplicate_entry_names_are_rejected_before_importing_either_target(tmp_path):
    for suffix in ("one", "two"):
        _distribution(
            tmp_path,
            f"ambiguous-{suffix}",
            f"AmbiguousFlow = ambiguous_{suffix}:AmbiguousFlow",
            {f"ambiguous_{suffix}": "raise AssertionError('ambiguous target was imported')\n"},
        )
    _distribution(
        tmp_path,
        "published-system",
        "PublishedFlow = published_flow:PublishedFlow",
        {"published_flow": _flow("PublishedFlow")},
    )
    _run(
        tmp_path,
        """
        import sys
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import tsdynamics as ts
            catalogue = ts.systems
        assert len(caught) == 1, caught
        assert "duplicate plugin name 'AmbiguousFlow'" in str(caught[0].message)
        assert "ambiguous_one" not in sys.modules and "ambiguous_two" not in sys.modules
        assert "AmbiguousFlow" not in catalogue.names()
        assert catalogue.get("PublishedFlow").__name__ == "PublishedFlow"

        from tsdynamics import plugins
        try:
            plugins.load_plugins(plugins.SYSTEMS_GROUP, strict=True, reject_duplicates=True)
        except ValueError as exc:
            assert "AmbiguousFlow" in str(exc)
        else:
            raise AssertionError("strict duplicate checking must fail")
        """,
    )
