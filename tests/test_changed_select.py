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
    plan = cs.classify({"src/tsdynamics/analysis/entropy/core.py"})
    assert not plan.full
    assert {"test_entropy.py", "test_property_entropy.py"} <= plan.selected_files
    # The cross-quantifier / analysis-pack gates span several areas → always run.
    assert set(cs._CROSSCUT_ANALYSIS_TESTS) <= plan.selected_files
    assert not plan.systems


def test_transforms_change_selects_transform_and_crosscut_tests() -> None:
    plan = cs.classify({"src/tsdynamics/transforms/spectral.py"})
    assert not plan.full
    assert set(cs._TRANSFORM_TESTS) <= plan.selected_files
    # test_known_quantifiers also uses a transform (spectral_entropy).
    assert set(cs._CROSSCUT_ANALYSIS_TESTS) <= plan.selected_files


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


def test_docs_and_markdown_are_ignored() -> None:
    plan = cs.classify({"docs/index.md", "README.md", "planning/notes.md"})
    assert not plan.full
    # Only the always-on guards remain — nothing escalated.
    assert plan.selected_files == set(cs._ALWAYS_GUARDS)
    assert not plan.systems


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
    plan = cs.Plan(full=False, reason="t", selected_files={"test_entropy.py"})
    assert cs.keep_item(_fake_item("test_entropy.py"), plan)
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


def test_referenced_test_files_exist() -> None:
    tests_dir = Path(__file__).parent
    referenced = (
        set(cs._ALWAYS_GUARDS)
        | set(cs._TRANSFORM_TESTS)
        | set(cs._VIZ_TESTS)
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


def test_scripts_ignore_is_safe_no_test_imports_scripts() -> None:
    """The selector ignores ``scripts/`` — assert no test module imports from it,
    so ignoring it cannot hide a real test dependency."""
    import ast

    def imports_scripts(node: ast.AST) -> bool:
        if isinstance(node, ast.Import):
            return any(a.name.split(".")[0] == "scripts" for a in node.names)
        if isinstance(node, ast.ImportFrom):
            return (node.module or "").split(".")[0] == "scripts"
        return False

    this_file = Path(__file__).name
    tests_dir = Path(__file__).parent
    offenders = [
        f.name
        for f in tests_dir.glob("test_*.py")
        if f.name != this_file  # skip self (mentions "scripts" in this assertion)
        and any(imports_scripts(n) for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))))
    ]
    assert not offenders, f"tests import from scripts/ (ignore would hide changes): {offenders}"
