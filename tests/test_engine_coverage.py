"""Meta-QA: every engine test reaches CI; no hand-list can silently drop one.

Stream **I-XVAL**.  The engine CI job used to enumerate its test files by hand
(``engine-bindings.yml``), so a module that ``importorskip``\\s ``tsdynamics._rust``
but was forgotten from the list silently *skipped* in the default matrix and
*never ran* in the engine job — reported as success.  That blind spot needed the
out-of-band patch in PR #72 (and still left ``test_bdf_stiff.py`` /
``test_cfam_seam.py`` uncovered).

This module removes the blind spot structurally and guards that it stays gone:

* the ``engine`` marker is **auto-applied** to any module that imports the
  compiled extension (``conftest`` + :mod:`_engine_marker`), so coverage follows
  the import, not a list;
* ``engine-bindings.yml`` selects those tests with ``-m engine`` — no file list —
  and builds the real ``tsdynamics._rust`` engine to run them (including the
  catalogue gate ``test_xval_catalogue.py``).

Post-M3 the engine is the sole integration backend, so ``ci.yml`` builds it too
and runs the whole suite; ``engine-bindings.yml`` stays as the focused FFI job.
These checks are pure source/workflow inspection (no compiled engine needed), so
they run anywhere — the regression is caught even before the extension is built.
"""

from __future__ import annotations

import re
from pathlib import Path

from _engine_marker import ENGINE_IMPORT, is_engine_test_file

_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parent
_WORKFLOWS = _REPO_ROOT / ".github" / "workflows"

#: Modules that must always be recognised as engine-dependent.  A regression in
#: the detector (too narrow) trips here before it can silently drop coverage.
_ANCHORS = frozenset(
    {
        "test_rust_engine.py",
        "test_dde_engine.py",
        "test_engine_wire.py",
        "test_bdf_stiff.py",
        "test_xval_catalogue.py",
    }
)

#: Modules that reference ``tsdynamics._rust`` only in prose / metadata and must
#: NOT be tagged engine (they need no compiled extension to run).  ``conftest``
#: itself isn't ``test_*`` so it's excluded from the glob; this module is here
#: because it documents the import forms in prose but imports nothing.
_NON_ENGINE = frozenset({"test_packaging.py", "test_sde.py", "test_engine_coverage.py"})


def test_ast_detector_is_backed_by_the_regex_safety_net() -> None:
    """Every module the AST detector flags also matches the regex twin.

    The production detector (:func:`_engine_marker.is_engine_test_file`) is an AST
    walk; :data:`_engine_marker.ENGINE_IMPORT` is an implementation-disjoint regex
    kept as a cross-check.  The regex over-matches import forms in prose (it has no
    docstring awareness), so it is a *superset* safety net: it may flag a module
    the AST does not (a harmless false-positive — the module still runs in both CI
    matrices), but it must never MISS one the AST flags.  A regex that fell behind
    a newly-recognised AST import form (silently shrinking engine coverage) trips
    here.
    """
    for path in sorted(_TESTS_DIR.glob("test_*.py")):
        source = path.read_text(encoding="utf-8")
        if is_engine_test_file(path):
            assert ENGINE_IMPORT.search(source), (
                f"{path.name}: AST flags it engine but the regex safety net misses it"
            )


def test_known_engine_modules_are_detected() -> None:
    """The anchor engine modules are all classified as engine-dependent."""
    detected = {p.name for p in _TESTS_DIR.glob("test_*.py") if is_engine_test_file(p)}
    missing = _ANCHORS - detected
    assert not missing, f"engine modules not detected (would be dropped from CI): {sorted(missing)}"


def test_prose_only_modules_are_not_engine() -> None:
    """Modules that merely mention the engine in text are not tagged engine."""
    for name in _NON_ENGINE:
        path = _TESTS_DIR / name
        if path.exists():
            assert not is_engine_test_file(path), f"{name} should not be an engine module"


def test_this_meta_module_is_wheel_free() -> None:
    """This coverage meta-test must run without the engine (it is pure inspection).

    It mentions ``tsdynamics._rust`` only as a string constant, so it must NOT be
    tagged ``engine`` — otherwise the wheel-free guarantee in the module docstring
    would be a coincidence rather than a property.
    """
    assert not is_engine_test_file(Path(__file__)), (
        "test_engine_coverage must stay wheel-free (not engine-tagged)"
    )


def test_conftest_autotags_the_engine_marker() -> None:
    """The collection hook that applies the ``engine`` marker is wired in ``conftest``."""
    src = (_TESTS_DIR / "conftest.py").read_text(encoding="utf-8")
    assert "pytest_collection_modifyitems" in src
    assert "is_engine_test_file" in src
    assert "pytest.mark.engine" in src


def _strip_yaml_comments(text: str) -> str:
    """Drop ``#``-comments line by line (these workflows have no ``#`` in strings)."""
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def _pytest_run_commands(text: str) -> list[str]:
    """The ``pytest`` invocation lines of a workflow (comments stripped)."""
    return [line for line in _strip_yaml_comments(text).splitlines() if "pytest" in line]


def test_engine_bindings_selects_by_marker_not_a_file_list() -> None:
    """``engine-bindings.yml`` runs ``-m engine`` and enumerates no test files.

    This is the structural fix for the PR #72 blind spot: with marker selection
    there is no hand-maintained list to forget to update.  Any ``tests/test_*.py``
    path in a pytest command would reintroduce the list, so reject it outright —
    only the pytest run lines are scanned (a prose comment naming a file is fine).
    """
    wf = (_WORKFLOWS / "engine-bindings.yml").read_text(encoding="utf-8")
    assert re.search(r"-m\s+[\"']?engine", wf), (
        "engine-bindings.yml must select tests with -m engine"
    )
    handlisted = [
        m for cmd in _pytest_run_commands(wf) for m in re.findall(r"tests/test_\w+\.py", cmd)
    ]
    assert not handlisted, f"engine-bindings.yml hand-lists test files: {sorted(set(handlisted))}"


def test_engine_bindings_builds_the_real_engine() -> None:
    """``engine-bindings.yml`` builds the real engine crate ``tsdyn-core``.

    Guards against silently reverting to the retired v2-seed ``tsdynamics-core``
    accelerator: the engine job must build the real binding crate ``tsdyn-core``
    (→ ``tsdynamics._rust``) that backs the whole post-M3 integration path.
    """
    wf = (_WORKFLOWS / "engine-bindings.yml").read_text(encoding="utf-8")
    assert "tsdyn-core" in wf, "engine-bindings.yml must build the real engine (crates/tsdyn-core)"


def test_engine_bindings_covers_the_engine_source_tree() -> None:
    """The engine job triggers on the whole package, not just engine/ + families/.

    The engine integrate / variational-Lyapunov path reaches ``utils/grids.py``,
    ``derived/`` and the solver layer too; a path filter narrower than
    ``src/tsdynamics/**`` would let a pure-source regression there bypass the
    focused engine job.
    """
    wf = (_WORKFLOWS / "engine-bindings.yml").read_text(encoding="utf-8")
    assert "src/tsdynamics/**" in wf, (
        "engine-bindings.yml must trigger on the whole package (src/tsdynamics/**), not a sub-tree"
    )
