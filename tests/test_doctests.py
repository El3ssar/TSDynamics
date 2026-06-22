"""Executable-documentation gate: docstring doctests + doc-page python fences.

Stream ``DOCS-DOCTEST-GATE``.  This is the CI harness that keeps the
documentation *runnable*: every curated docstring example and every curated doc
page's ```python``` block is executed under the suite-wide
``filterwarnings = error`` (see ``pyproject.toml``), so a doc that drifts out of
sync with the code fails the build.

Tiers
-----
* **default (fast)** — the curated, verified-fast docstring modules
  (:data:`tests._doctest_select.CURATED_MODULES`) and the curated clean doc pages
  (:data:`tests._doctest_select.CURATED_PAGES`).  All pass clean.
* **``-m full``** — additionally the heavy-simulation modules
  (:data:`tests._doctest_select.FULL_ONLY_MODULES`); the nightly sweep.

The list of *what* is gated lives in ``tests/_doctest_select.py`` (one file per
thing); this module is just the parametrize wrappers.  The forward contract new
pages must satisfy to join the curated set is ``docs/contributing/page-template.md``.
"""

from __future__ import annotations

import pytest
from _doctest_select import (
    CURATED_MODULES,
    CURATED_PAGES,
    FULL_ONLY_MODULES,
    iter_python_fences,
    run_module_doctests,
    run_page_fences,
)

pytestmark = pytest.mark.doctest


@pytest.mark.parametrize("module_name", CURATED_MODULES)
def test_docstring_examples_fast(module_name: str) -> None:
    """Every curated (fast) module's docstring examples run clean."""
    failures = run_module_doctests(module_name)
    assert not failures, "\n\n".join(f.message for f in failures)


@pytest.mark.full
@pytest.mark.parametrize("module_name", FULL_ONLY_MODULES)
def test_docstring_examples_full(module_name: str) -> None:
    """Heavy-simulation docstring examples (nightly ``-m full`` tier)."""
    failures = run_module_doctests(module_name)
    assert not failures, "\n\n".join(f.message for f in failures)


@pytest.mark.parametrize("page", CURATED_PAGES)
def test_doc_page_fences(page: str) -> None:
    """Every ```python``` fence in a curated doc page executes without raising."""
    run_page_fences(page)


def test_curated_lists_are_disjoint_and_unique() -> None:
    """Guard: no module is in both tiers and no list has duplicates."""
    assert len(set(CURATED_MODULES)) == len(CURATED_MODULES)
    assert len(set(FULL_ONLY_MODULES)) == len(FULL_ONLY_MODULES)
    assert len(set(CURATED_PAGES)) == len(CURATED_PAGES)
    assert not (set(CURATED_MODULES) & set(FULL_ONLY_MODULES))


def test_fence_extractor_skips_markers_and_transcripts() -> None:
    """The page extractor yields runnable scripts and skips fragments.

    Asserts the two opt-outs the page-template contract relies on: a fence
    carrying the ``# skip-doctest`` marker, and a ``>>>`` doctest transcript
    (handled by the module path), are both excluded; a plain block is yielded.
    """
    page = (
        "intro\n"
        "```python\n"
        "import tsdynamics as ts\n"
        "ts.Lorenz()\n"
        "```\n"
        "fragment:\n"
        "```python\n"
        "# skip-doctest\n"
        "result = some_undefined_helper()\n"
        "```\n"
        "transcript:\n"
        "```python\n"
        ">>> 1 + 1\n"
        "2\n"
        "```\n"
    )
    blocks = list(iter_python_fences(page))
    assert len(blocks) == 1
    assert "ts.Lorenz()" in blocks[0]
    assert all("skip-doctest" not in b for b in blocks)
    assert all(">>>" not in b for b in blocks)
