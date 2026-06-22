"""Repo-root pytest configuration for the executable-documentation gate.

The bulk of the suite's fixtures and registry-driven parametrization live in
``tests/conftest.py`` (the ``testpaths`` root).  This *repo-root* conftest exists
for the doctest gate (stream ``DOCS-DOCTEST-GATE``):

* it puts the ``tests`` directory on ``sys.path`` so the gate's helper module
  ``tests/_doctest_select.py`` is importable as ``_doctest_select`` from anywhere
  pytest collects (the same flat-import convention ``tests/conftest.py`` already
  uses for ``_changed_select`` / ``_strategies``); and
* it exposes the shared doctest namespace under pytest's standard
  ``doctest_namespace`` fixture, so if a run ever opts into pytest's native
  doctest collection (``--doctest-modules`` / ``--doctest-glob``) the readable,
  import-light examples resolve the same reader-facing names (``np``, ``ts``,
  every built-in system, every public analysis function) the dedicated harness
  injects.  The harness in ``tests/test_doctests.py`` does not depend on this
  fixture — it is provided so the two doctest entry points share one namespace.

It deliberately does **not** re-register the ``--changed`` / ``--changed-since``
options (those belong to ``tests/conftest.py``); pytest forbids registering an
option twice, so this file only adds, never duplicates.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

# Make the flat test-helper modules (``_doctest_select`` and friends) importable
# regardless of the collection root, matching the convention tests/conftest.py
# relies on.
_TESTS_DIR = Path(__file__).resolve().parent / "tests"
if _TESTS_DIR.is_dir() and str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))


@pytest.fixture
def doctest_namespace() -> dict[str, Any]:
    """Seed pytest's native doctest globals with the reader-facing names.

    Returns the same namespace the dedicated harness injects (NumPy as ``np``,
    the package as ``ts``, every built-in system class and every public
    top-level analysis name), so ``--doctest-modules`` runs the import-light
    examples as written.
    """
    from _doctest_select import doctest_namespace as build_namespace

    return build_namespace()
