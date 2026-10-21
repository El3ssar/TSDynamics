"""Installed analysis plugins initialize against the complete public package."""

from __future__ import annotations

import subprocess
import sys
import textwrap


def _distribution(site, entries, modules):
    info = site / "published_analyses-0.0.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Name: published-analyses\nVersion: 0.0.0\n")
    (info / "entry_points.txt").write_text(f"[tsdynamics.analyses]\n{entries}\n")
    for module, source in modules.items():
        (site / f"{module}.py").write_text(textwrap.dedent(source))


def _published_analysis():
    return """
        import tsdynamics as ts
        from tsdynamics import ContinuousSystem, __version__, plot, viz
        from tsdynamics.analysis.results import ScalarResult

        # A plugin can inspect compatibility, use the catalogue, and import
        # plotting facilities while its module initializes.
        FAMILY = ContinuousSystem
        VERSION = __version__
        LORENZ = ts.systems.Lorenz
        assert callable(plot)
        assert callable(viz.plot)

        def published_measure(data) -> ScalarResult:
            "Return the sample sum with the installed package version."
            return ScalarResult(value=float(sum(data)), meta={"version": VERSION})
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


def test_installed_analysis_imports_public_exports_and_joins_the_namespace(tmp_path):
    _distribution(
        tmp_path,
        "published_measure = published_analysis:published_measure",
        {"published_analysis": _published_analysis()},
    )
    _run(
        tmp_path,
        """
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import tsdynamics as ts
        from tsdynamics import registry
        import published_analysis as plugin

        assert not caught, [str(w.message) for w in caught]
        assert plugin.FAMILY is ts.ContinuousSystem
        assert plugin.VERSION == ts.__version__
        assert plugin.LORENZ is ts.systems.Lorenz
        assert registry.analyses.get("published_measure") is plugin.published_measure
        assert ts.analysis.published_measure is plugin.published_measure
        assert "published_measure" in ts.analysis.__all__
        assert "published_measure" in dir(ts.analysis)
        assert plugin.published_measure in ts.analysis.find("published_measure")
        result = ts.analysis.published_measure([1.0, 2.0, 3.0])
        assert float(result) == 6.0
        assert result.meta["version"] == ts.__version__
        """,
    )


def test_broken_analysis_warns_and_healthy_analysis_survives_a_direct_submodule_import(tmp_path):
    _distribution(
        tmp_path,
        """broken_measure = broken_analysis:broken_measure
published_measure = published_analysis:published_measure""",
        {
            "broken_analysis": "raise RuntimeError('broken analysis dependency')\n",
            "published_analysis": _published_analysis(),
        },
    )
    _run(
        tmp_path,
        """
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import tsdynamics.analysis as analyses
        import tsdynamics as ts
        from tsdynamics import registry

        messages = [str(w.message) for w in caught]
        assert len(messages) == 1, messages
        assert "broken_measure" in messages[0] and "broken analysis dependency" in messages[0]
        assert "broken_measure" not in registry.analyses.names()
        assert "broken_measure" not in analyses.__all__
        assert analyses is ts.analysis
        assert float(analyses.published_measure([2.0, 4.0])) == 6.0
        assert registry.analyses.get("published_measure") is analyses.published_measure
        """,
    )
