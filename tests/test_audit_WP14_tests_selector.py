"""Regression: catalogue RHS-correctness gates are scoped on a system edit.

Finding X3:X3-1.  The change-scoped selector (``tests/_changed_select.py``) must
run the catalogue correctness gates — the golden lowered-tape snapshot, the
analytic literature identities, and the engine cross-validation sweep — whenever
any ``src/tsdynamics/systems/`` kernel changes.  Those gates are the project's
primary defence against operator-precedence / transcription bugs in a system
kernel body (the ``p**1/2`` class of defect).  They are NOT registry-parametrized
per system (their cases are bare tuples / plain functions), so the per-system
scoping in :func:`keep_item` cannot reach them; before the fix a kernel edit ran
none of them on a change-scoped PR, deferring the catch to merge / nightly.

These assertions fail on the pre-fix selector (which selected only the three
``_ALWAYS_GUARDS`` plus the changed module's systems) and pass after.
"""

from __future__ import annotations

from pathlib import Path

import tests._changed_select as cs

# A representative catalogue kernel module (one ODE, one map) — editing either
# must pull the catalogue gates.  Derived from the live registry so a module
# rename does not silently turn these into full-run escalations.
_MAP_MODULE = "src/tsdynamics/systems/discrete/chaotic_maps.py"


def _a_real_system_module(prefix: str) -> str:
    """A repo-relative path of a registered built-in system module under ``prefix``."""
    from tsdynamics import registry

    for entry in registry.all_systems(builtin=True):
        rel = "src/" + entry.module.replace(".", "/") + ".py"
        if rel.startswith(prefix):
            return rel
    raise AssertionError(f"no registered system module under {prefix}")


def test_catalogue_gates_constant_files_exist() -> None:
    """Every gate the selector names is a real test file (no dangling reference)."""
    tests_dir = Path(__file__).parent
    missing = {name for name in cs._CATALOGUE_GATES if not (tests_dir / name).exists()}
    assert not missing, f"_CATALOGUE_GATES references nonexistent files: {sorted(missing)}"


def test_map_module_edit_selects_catalogue_gates() -> None:
    """A discrete-map kernel edit scopes in (not escalates to full) the gates."""
    plan = cs.classify({_MAP_MODULE})
    assert not plan.full, plan.reason
    for gate in cs._CATALOGUE_GATES:
        assert gate in plan.selected_files, f"{gate} not scoped on a map-module edit"


def test_ode_module_edit_selects_catalogue_gates() -> None:
    """An ODE-flow kernel edit likewise scopes in the catalogue gates."""
    plan = cs.classify({_a_real_system_module("src/tsdynamics/systems/continuous/")})
    assert not plan.full, plan.reason
    assert set(cs._CATALOGUE_GATES) <= plan.selected_files


def test_equation_reference_gate_specifically_scoped() -> None:
    """The golden-tape snapshot (the named ``p**1/2`` defence) is among them.

    This is the exact assertion the finding calls out: editing a system module
    must select ``test_equation_reference.py``.
    """
    plan = cs.classify({_MAP_MODULE})
    assert "test_equation_reference.py" in plan.selected_files


def test_non_system_edit_does_not_pull_catalogue_gates() -> None:
    """Answer-preserving: a viz-only edit must NOT add the catalogue gates.

    The gates are tied to the *systems* branch; an unrelated scoped edit keeps
    its own selection untouched.
    """
    plan = cs.classify({"src/tsdynamics/viz/spec.py"})
    assert not plan.full, plan.reason
    assert not (set(cs._CATALOGUE_GATES) & plan.selected_files)


def test_catalogue_gates_carry_no_system_param() -> None:
    """Document why file-level selection (not per-system scoping) is required.

    The gates' kept items must survive purely by filename membership, since they
    carry no ``SystemEntry`` to scope on.  A plan that scopes a system module but
    omits these files would never keep their items.
    """
    plan = cs.classify({_MAP_MODULE})
    # Each gate is kept via selected_files regardless of plan.systems content.
    for gate in cs._CATALOGUE_GATES:
        assert gate in plan.selected_files
