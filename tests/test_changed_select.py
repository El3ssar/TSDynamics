"""Guards for the change-scoped test selector (``tests/_changed_select.py``).

These are pure-Python and git-free: the classifier is exercised with synthetic
file sets, and the meta-QA keeps the selection tables in sync with the tree
(every analysis subpackage mapped; every referenced test file present).  The
selector's job is to *never under-select*, so the unit tests focus on the
escalation rules — the cases that must fall back to the full suite.
"""

from __future__ import annotations

import types
from pathlib import Path

import _changed_select as cs
import pytest

import conftest
import tsdynamics.analysis
from tsdynamics import registry


def _module_to_src(module: str) -> str:
    return "src/" + module.replace(".", "/") + ".py"


# ---------------------------------------------------------------------------
# Escalation: foundational / unknown changes must run the full suite.
# ---------------------------------------------------------------------------


def test_git_failure_falls_back_to_full() -> None:
    assert cs.classify(None).full


def test_foundational_changes_force_full() -> None:
    for path in [
        "src/tsdynamics/engine/run.py",
        "src/tsdynamics/families/base.py",
        "src/tsdynamics/solvers/select.py",
        "src/tsdynamics/derived/poincare.py",
        "src/tsdynamics/data/trajectory.py",
        "src/tsdynamics/utils/grids.py",
        "src/tsdynamics/registry.py",
        "src/tsdynamics/__init__.py",
        "crates/tsdyn-engine/src/dde.rs",
        "tests/conftest.py",
        "tests/_strategies.py",
        "tests/_changed_select.py",
        "pyproject.toml",
        "uv.lock",
        ".github/workflows/ci.yml",
    ]:
        assert cs.classify({path}).full, f"{path} should force a full run"


def test_unknown_source_path_escalates() -> None:
    # A source file matching no rule must escalate, never silently select nothing.
    assert cs.classify({"src/tsdynamics/brand_new_module.py"}).full


def test_unmapped_analysis_area_escalates() -> None:
    assert cs.classify({"src/tsdynamics/analysis/teleportation/core.py"}).full


def test_systems_init_change_forces_full() -> None:
    assert cs.classify({"src/tsdynamics/systems/continuous/__init__.py"}).full


def test_system_module_with_no_registered_class_escalates() -> None:
    assert cs.classify({"src/tsdynamics/systems/continuous/does_not_exist.py"}).full


# ---------------------------------------------------------------------------
# Scoped selection: the happy paths.
# ---------------------------------------------------------------------------


def test_system_module_scopes_to_its_own_systems() -> None:
    by_mod = cs._systems_by_module()
    assert by_mod, "registry produced no system modules"
    module, names = next(iter(sorted(by_mod.items())))
    plan = cs.classify({_module_to_src(module)})
    assert not plan.full
    assert plan.systems == names
    # The cheap integrity guards always ride along.
    for guard in cs._ALWAYS_GUARDS:
        assert guard in plan.selected_files


def test_analysis_area_selects_its_tests_and_crosscut() -> None:
    plan = cs.classify({"src/tsdynamics/analysis/recurrence/rqa.py"})
    assert not plan.full
    assert {"test_recurrence.py", "test_property_recurrence.py"} <= plan.selected_files
    # The cross-quantifier / analysis-pack gates span several areas → always run.
    assert set(cs._CROSSCUT_ANALYSIS_TESTS) <= plan.selected_files
    assert not plan.systems


def test_removed_series_statistics_paths_escalate() -> None:
    """The deleted generic-statistics areas are unmapped, so a stray path escalates.

    ``analysis/entropy/`` and ``analysis/surrogate/`` (and ``transforms/``) left the
    tree in the v6 scope surgery.  Nothing should quietly select a shrunken set if
    such a path ever reappears in a diff — it must fall through to a full run.
    """
    for path in (
        "src/tsdynamics/analysis/entropy/core.py",
        "src/tsdynamics/analysis/surrogate/generators.py",
        "src/tsdynamics/transforms/spectral.py",
    ):
        assert cs.classify({path}).full, path


def test_orbits_area_includes_orbit_diagram_perf() -> None:
    # Regression for the WS-MAPITER engine-routing test being dropped.
    plan = cs.classify({"src/tsdynamics/analysis/orbits/orbit_diagram.py"})
    assert not plan.full
    assert "test_orbit_diagram_perf.py" in plan.selected_files


def test_system_change_does_not_pull_crosscut_analysis() -> None:
    # A pure system-module change must NOT drag in the cross-quantifier gates
    # (they run on fixed signals, not registry systems).
    by_mod = cs._systems_by_module()
    module, _ = next(iter(sorted(by_mod.items())))
    plan = cs.classify({_module_to_src(module)})
    assert not (set(cs._CROSSCUT_ANALYSIS_TESTS) & plan.selected_files)


def test_changed_test_file_is_selected() -> None:
    plan = cs.classify({"tests/test_smoke.py"})
    assert not plan.full
    assert "test_smoke.py" in plan.selected_files


def test_a_docs_page_selects_the_gate_that_executes_it() -> None:
    """A documentation change runs the doctest gate — the known v6 gap, closed.

    ``tests/test_doctests.py`` *executes* every runnable ``python`` fence on every
    ``docs/**.md`` page under ``filterwarnings = error``.  Before this rule,
    ``docs/`` and ``*.md`` sat in the ignore table as "no bearing on the test
    suite", so ``classify(['docs/analysis/lyapunov.md'])`` selected three cheap
    registry guards and **not** the gate that runs the page — a docs-only PR could
    break every example on it and go green.
    """
    for page in ("docs/index.md", "docs/analysis/lyapunov.md", "docs/tutorials/basics.md"):
        plan = cs.classify({page})
        assert not plan.full, plan.reason
        assert "test_doctests.py" in plan.selected_files, page
        assert not plan.systems


def test_the_repo_root_files_the_doctest_gate_reads_are_not_ignored() -> None:
    """``README.md`` / ``CLAUDE.md`` / ``mkdocs.yml`` select the gate that reads them.

    ``test_doctests.py`` checks the catalogue counts written in their prose
    against the live registry and parses ``mkdocs.yml``'s ``exclude_docs`` block,
    so none of the three is ignorable.  The two tables are kept from disagreeing
    by :func:`test_no_file_is_both_docs_gated_and_ignored`.
    """
    from _doctest_select import REPO_ROOT

    for name in sorted(cs._DOCS_GATE_FILES):
        assert (REPO_ROOT / name).exists(), f"{name} is gated but does not exist"
        plan = cs.classify({name})
        assert not plan.full, plan.reason
        assert "test_doctests.py" in plan.selected_files, name


def test_docs_tooling_selects_the_tests_that_import_it() -> None:
    """``docs/_tooling/`` is code the suite imports, not prose.

    The gallery builder, the committed golden-figure corpus and
    ``editorial.json`` are all read by tests; ignoring the directory hid those
    dependencies completely.
    """
    for path in ("docs/_tooling/gallery.py", "docs/_tooling/editorial.json"):
        plan = cs.classify({path})
        assert not plan.full, plan.reason
        assert set(cs._DOCS_TOOLING_TESTS) <= plan.selected_files, path


def test_planning_and_changelog_stay_ignored() -> None:
    """Only the documentation the suite *reads* is gated; the rest still costs nothing."""
    plan = cs.classify({"planning/notes.md", "CHANGELOG.md", ".claude/settings.json"})
    assert not plan.full
    assert plan.selected_files == set(cs._ALWAYS_GUARDS)
    assert not plan.systems


def test_no_file_is_both_docs_gated_and_ignored() -> None:
    """One file, one classification — the two tables may not claim the same name.

    The doc-gate check runs *first*, so an overlap would be a silently dead
    ignore entry rather than an error.  This makes it loud.
    """
    overlap = cs._DOCS_GATE_FILES & cs._IGNORE_FILES
    assert not overlap, f"claimed by both the docs gate and the ignore table: {sorted(overlap)}"
    assert not any(p.startswith("docs") for p in cs._IGNORE_PREFIXES)


def test_docs_gate_tests_exist() -> None:
    tests_dir = Path(__file__).parent
    for name in set(cs._DOCS_GATE_TESTS) | set(cs._DOCS_TOOLING_TESTS):
        assert (tests_dir / name).exists(), f"docs lane references a missing test file: {name}"


def test_no_changes_runs_only_guards() -> None:
    plan = cs.classify(set())
    assert not plan.full
    assert plan.selected_files == set(cs._ALWAYS_GUARDS)


def test_mixed_system_and_area() -> None:
    by_mod = cs._systems_by_module()
    module, names = next(iter(sorted(by_mod.items())))
    plan = cs.classify(
        {_module_to_src(module), "src/tsdynamics/analysis/chaos/gali.py", "tests/test_derived.py"}
    )
    assert not plan.full
    assert plan.systems == names
    assert {"test_chaos.py", "test_derived.py"} <= plan.selected_files


# ---------------------------------------------------------------------------
# keep_item predicate.
# ---------------------------------------------------------------------------


def _fake_item(filename: str, entry: object | None = None) -> object:
    callspec = types.SimpleNamespace(params={"e": entry}) if entry is not None else None
    return types.SimpleNamespace(path=Path("tests") / filename, callspec=callspec)


def test_keep_item_by_selected_file() -> None:
    plan = cs.Plan(full=False, reason="t", selected_files={"test_recurrence.py"})
    assert cs.keep_item(_fake_item("test_recurrence.py"), plan)
    assert not cs.keep_item(_fake_item("test_dimensions.py"), plan)


def test_keep_item_by_system_param() -> None:
    entries = list(registry.all_systems())
    changed, other = entries[0], entries[-1]
    assert changed.name != other.name
    plan = cs.Plan(full=False, reason="t", systems={changed.name})
    # A sweep item bound to the changed system survives...
    assert cs.keep_item(_fake_item("test_jacobians.py", changed), plan)
    # ...one bound to a DIFFERENT system is dropped (the per-system scoping).
    assert not cs.keep_item(_fake_item("test_jacobians.py", other), plan)


def test_keep_item_by_system_name_string() -> None:
    # The by-name sweeps (test_rust_engine, INTEGRATION_SAMPLE, DDE names, the
    # hand-listed cases) parametrize over a NAME STRING, not a SystemEntry — they
    # must still be scoped to their system, not silently dropped.
    name = next(iter(registry.all_systems())).name
    item = types.SimpleNamespace(
        path=Path("tests/test_rust_engine.py"),
        callspec=types.SimpleNamespace(params={"name": name}),
    )
    assert cs.keep_item(item, cs.Plan(full=False, reason="t", systems={name}))
    assert not cs.keep_item(item, cs.Plan(full=False, reason="t", systems={"NotARealSystem"}))
    # A string that is not a system name is NOT treated as a system binding.
    assert cs.system_name_of(_fake_item("test_solvers.py", "bdf")) is None


def test_keep_item_handwritten_per_system_test_in_sweep_file() -> None:
    # A non-parametrized per-system test (no callspec) in a sweep file must run
    # whenever ANY system module changed — it is the bespoke regression the
    # parametrized sweep cannot reach (e.g. test_lorenz96_integrates).
    item = _fake_item("test_ode_systems.py")  # callspec=None
    assert cs.keep_item(item, cs.Plan(full=False, reason="t", systems={"Lorenz"}))
    # ...but not when no system changed (e.g. an analysis-only diff).
    assert not cs.keep_item(item, cs.Plan(full=False, reason="t", systems=set()))
    # And a no-callspec test in a NON-sweep file is still governed by file selection.
    assert not cs.keep_item(
        _fake_item("test_smoke.py"), cs.Plan(full=False, reason="t", systems={"Lorenz"})
    )


def test_changed_no_tests_collected_exits_success() -> None:
    exitstatus = pytest.ExitCode.NO_TESTS_COLLECTED
    session = types.SimpleNamespace(
        config=types.SimpleNamespace(getoption=lambda name, default=False: name == "changed"),
        exitstatus=exitstatus,
    )
    conftest.pytest_sessionfinish(session, exitstatus)
    assert exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED
    assert session.exitstatus == pytest.ExitCode.OK


def test_plain_no_tests_collected_stays_nonzero() -> None:
    exitstatus = pytest.ExitCode.NO_TESTS_COLLECTED
    session = types.SimpleNamespace(
        config=types.SimpleNamespace(getoption=lambda name, default=False: False),
        exitstatus=exitstatus,
    )
    conftest.pytest_sessionfinish(session, exitstatus)
    assert exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED
    assert session.exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED


# ---------------------------------------------------------------------------
# Meta-QA: keep the selection tables in sync with the tree.
# ---------------------------------------------------------------------------


def test_every_analysis_subpackage_is_mapped() -> None:
    """A new ``analysis/<area>/`` must be added to ``_AREA_TESTS`` (or it would
    silently escalate every touch to a full run)."""
    analysis_dir = Path(next(iter(tsdynamics.analysis.__path__)))
    areas = {
        p.name for p in analysis_dir.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))
    }
    missing = areas - set(cs._AREA_TESTS)
    assert not missing, f"unmapped analysis areas (add to _AREA_TESTS): {sorted(missing)}"


def test_every_area_test_file_is_in_some_lane() -> None:
    """Every test file carrying a lane prefix is selected by that lane's change.

    The blind spot this closes: a change anywhere under ``src/tsdynamics/viz/``
    selected exactly three test files while sixteen ``test_viz_*.py`` existed, so
    a viz change could reach ``main`` with most of its own tests deselected.  A
    hand-written tuple over a *growing* family of files is the defect, and it
    recurs in every area — so the check is generic: for each prefix in
    :data:`_changed_select.LANE_PREFIXES`, classify a real source path in that
    area and assert that **no** existing test file with that prefix is missing
    from the plan.  Adding ``tests/test_viz_newthing.py`` needs no edit here;
    adding ``tests/test_basins_newthing.py`` fails until it is mapped.
    """
    tests_dir = Path(__file__).parent
    stale: list[str] = []
    for prefix, probe in sorted(cs.LANE_PREFIXES.items()):
        assert (Path(__file__).parents[1] / probe).exists(), (
            f"LANE_PREFIXES probe {probe!r} no longer exists; point it at a real "
            f"source file in the {prefix!r} lane."
        )
        plan = cs.classify({probe})
        assert not plan.full, f"probe {probe!r} escalated to a full run: {plan.reason}"
        carrying = sorted(p.name for p in tests_dir.glob(f"{prefix}*.py"))
        assert carrying, f"no test file carries the lane prefix {prefix!r} — stale table entry."
        stale += [
            f"{name} (changing {probe} does not select it)"
            for name in carrying
            if name not in plan.selected_files
        ]
    assert not stale, "test files in no lane: " + ", ".join(stale)


def test_viz_lane_is_discovered_not_hand_listed() -> None:
    """``viz_tests()`` picks up every ``test_viz_*.py`` on disk, plus the exceptions."""
    tests_dir = Path(__file__).parent
    on_disk = {p.name for p in tests_dir.glob("test_viz_*.py")}
    assert len(on_disk) > 3, "expected the viz test family to be larger than the old hand list"
    lane = set(cs.viz_tests())
    assert on_disk <= lane
    assert set(cs._VIZ_EXTRA_TESTS) <= lane


def test_referenced_test_files_exist() -> None:
    tests_dir = Path(__file__).parent
    referenced = (
        set(cs._ALWAYS_GUARDS)
        | set(cs.viz_tests())
        | set(cs._CROSSCUT_ANALYSIS_TESTS)
        | set(cs._SYSTEM_SWEEP_FILES)
    )
    for files in cs._AREA_TESTS.values():
        referenced.update(files)
    missing = {name for name in referenced if not (tests_dir / name).exists()}
    assert not missing, f"selection references nonexistent test files: {sorted(missing)}"


def test_no_system_name_collides_with_known_non_system_param_strings() -> None:
    """The string-name match in system_name_of must not capture non-system
    parametrize ids (those would be wrongly *kept*, i.e. over-selected — safe —
    but this documents that today there is no collision)."""
    non_system_param_ids = {
        "bdf",
        "BDF",
        "gear",
        "line",
        "square",
        "recurrence_matrix",
        "rqa",
        "windowed_rqa",
    }
    names = cs._all_system_names()
    assert not (non_system_param_ids & names)


def test_benchmarks_edit_selects_the_harness_gate() -> None:
    """Editing the bench harness selects the test that imports it, not a full run.

    ``benchmarks/analysis_bench.py`` is loaded by *path* from
    ``test_perf_regression.py`` (it is not an installed package), so the selector
    must recognise ``benchmarks/`` explicitly: ignoring it would hide a real test
    dependency, and leaving it unrecognised would escalate every harness tweak to
    the whole suite.
    """
    plan = cs.classify(["benchmarks/analysis_bench.py"])
    assert not plan.full, plan.reason
    assert "test_perf_regression.py" in plan.selected_files
