"""Detect engine-dependent test modules (stream I-XVAL).

A test module *depends on the compiled engine* when it imports the
``tsdynamics._rust`` extension — directly or via ``pytest.importorskip``.  The
``conftest`` collection hook reads this to auto-apply the ``engine`` marker to
every test in such a module, and the engine CI job selects them with
``-m engine``.  Because the marker follows the import (not a hand-maintained
file list), a new engine test file joins the engine job with zero CI edits — the
gap that silently dropped ``test_bdf_stiff.py`` / ``test_cfam_seam.py`` from the
old hand-list and needed the out-of-band patch in PR #72.

Detection is **AST-based**, not a substring or raw-source regex: it recognises a
real ``import``/``importorskip`` of the extension and is *not* fooled by a module
that merely mentions ``tsdynamics._rust`` in a docstring or string literal (e.g.
``test_packaging.py`` asserting the maturin ``module-name``, or
``test_engine_coverage.py`` documenting the import forms).  A regex twin
(:data:`ENGINE_IMPORT`) is kept only as the *independent cross-check* the
coverage meta-test compares against — never as the source of truth.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_RUST = "tsdynamics._rust"

#: Regex twin of the AST detector, kept solely so the coverage meta-test has an
#: implementation-disjoint second opinion.  It over-matches the import forms in
#: prose (raw source has no docstring awareness), so it is a *safety net* — every
#: real import the AST flags must also match here — never the production detector.
ENGINE_IMPORT = re.compile(
    r"""importorskip\(\s*["']tsdynamics\._rust["']\s*\)"""
    r"""|import_module\(\s*["']tsdynamics\._rust["']\s*\)"""
    r"""|^\s*import\s+tsdynamics\._rust\b"""
    r"""|^\s*from\s+tsdynamics\s+import\s+_rust\b"""
    r"""|^\s*from\s+tsdynamics\._rust\s+import\b""",
    re.MULTILINE,
)


def is_engine_test_source(source: str) -> bool:
    """Whether ``source`` actually imports the compiled ``tsdynamics._rust`` extension.

    Walks the AST and recognises, as genuine code (not prose):
    ``import tsdynamics._rust`` (``... as x``), ``from tsdynamics import _rust``,
    ``from tsdynamics._rust import …``, ``pytest.importorskip("tsdynamics._rust")``,
    and ``importlib.import_module("tsdynamics._rust")``.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover - test files parse
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == _RUST for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == _RUST:
                return True
            if mod == "tsdynamics" and any(alias.name == "_rust" for alias in node.names):
                return True
        elif isinstance(node, ast.Call):
            func = node.func
            fname = (
                func.attr
                if isinstance(func, ast.Attribute)
                else func.id
                if isinstance(func, ast.Name)
                else ""
            )
            if fname in {"importorskip", "import_module"} and node.args:
                arg0 = node.args[0]
                if isinstance(arg0, ast.Constant) and arg0.value == _RUST:
                    return True
    return False


def is_engine_test_file(path: str | Path) -> bool:
    """Whether the test file at ``path`` imports ``tsdynamics._rust``."""
    try:
        return is_engine_test_source(Path(path).read_text(encoding="utf-8"))
    except OSError:
        return False
