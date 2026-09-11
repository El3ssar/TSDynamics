"""Executable-documentation gate: docstring doctests + doc-page python fences.

Stream ``DOCS-DOCTEST-GATE``.  This is the CI harness that keeps the
documentation *runnable*: every gated docstring example and every gated doc
page's ```python``` block is executed under the suite-wide
``filterwarnings = error`` (see ``pyproject.toml``), so a doc that drifts out of
sync with the code fails the build.

What is gated
-------------
**Everything, by default.**  The subjects are *discovered*, not listed: any
module under ``src/tsdynamics`` containing a ``>>>`` and any ``docs`` page
containing a runnable ```python``` fence is gated the moment it is written.
Leaving the gate requires a named entry with a written reason in
``tests/_doctest_select.py`` (:data:`~tests._doctest_select.EXEMPT_MODULES` /
:data:`~tests._doctest_select.EXEMPT_PAGES`).

This replaced an allow-list that named 20 modules and 22 pages — while 12 of the
25 modules it did not name were failing, and 3 of the pages it did name had no
runnable fence at all and so passed vacuously.

Tiers
-----
* **default (fast)** — every gated module and page.  All pass clean.
* **``-m full``** — additionally the heavy-simulation modules/pages
  (:data:`~tests._doctest_select.SLOW_MODULES` /
  :data:`~tests._doctest_select.SLOW_PAGES`), plus the two guards that re-run the
  exemptions and fail when one starts passing.

The list of *what* is gated lives in ``tests/_doctest_select.py`` (one file per
thing); this module is the parametrize wrappers and the anti-rot guards.  The
forward contract a page must satisfy is "The page-fence contract" in that
module's docstring.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from _doctest_select import (
    EXEMPT_MODULES,
    EXEMPT_PAGES,
    SLOW_MODULES,
    SLOW_PAGES,
    discover_doc_pages,
    discover_doctest_modules,
    full_tier_modules,
    full_tier_pages,
    gated_modules,
    gated_pages,
    iter_python_fences,
    run_module_doctests,
    run_page_fences,
)

pytestmark = pytest.mark.doctest

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# The gate itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_name", gated_modules())
def test_docstring_examples_fast(module_name: str) -> None:
    """Every gated (fast) module's docstring examples run clean."""
    failures = run_module_doctests(module_name)
    assert not failures, "\n\n".join(f.message for f in failures)


@pytest.mark.full
@pytest.mark.parametrize("module_name", full_tier_modules())
def test_docstring_examples_full(module_name: str) -> None:
    """Heavy-simulation docstring examples (nightly ``-m full`` tier)."""
    failures = run_module_doctests(module_name)
    assert not failures, "\n\n".join(f.message for f in failures)


@pytest.fixture
def _scratch_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run a page's fences in a throwaway directory.

    Doc pages call ``spec.save("lorenz.json")`` / ``.save("orbit.gif")`` with
    bare relative names, which land in pytest's cwd — the repository root. With
    41 pages gated (and the ``full`` tier rendering gifs and mp4s) that dropped
    untracked artifacts into the working tree on every run, where this repo's
    git auto-sync daemon can pick them up. No gated page *reads* a relative
    path, so redirecting cwd is free.
    """
    monkeypatch.chdir(tmp_path)


@pytest.mark.usefixtures("_scratch_cwd")
@pytest.mark.parametrize("page", gated_pages())
def test_doc_page_fences(page: str) -> None:
    """Every ```python``` fence in a gated doc page executes without raising."""
    run_page_fences(page)


@pytest.mark.full
@pytest.mark.usefixtures("_scratch_cwd")
@pytest.mark.parametrize("page", full_tier_pages())
def test_doc_page_fences_full(page: str) -> None:
    """Heavy doc pages (animation / composition) — nightly ``-m full`` tier."""
    run_page_fences(page)


# ---------------------------------------------------------------------------
# Anti-rot guards: the exemption lists cannot quietly go stale
# ---------------------------------------------------------------------------


def test_no_stale_module_exemptions() -> None:
    """Every exempt / slow module still exists and still has doctests.

    An entry for a module that was deleted or whose examples were removed is
    dead weight that hides how much is actually exempt.
    """
    found = set(discover_doctest_modules())
    stale = sorted((set(EXEMPT_MODULES) | set(SLOW_MODULES)) - found)
    assert not stale, (
        "these modules are named in EXEMPT_MODULES/SLOW_MODULES but no longer "
        f"have doctests (remove them from tests/_doctest_select.py): {stale}"
    )


def test_no_stale_page_exemptions() -> None:
    """Every exempt / slow page still exists and still has a runnable fence."""
    found = set(discover_doc_pages())
    stale = sorted((set(EXEMPT_PAGES) | set(SLOW_PAGES)) - found)
    assert not stale, (
        "these pages are named in EXEMPT_PAGES/SLOW_PAGES but no longer have a "
        f"runnable python fence (remove them): {stale}"
    )


def test_every_exemption_carries_a_reason() -> None:
    """An exemption without a written reason is just a silent hole."""
    for name, reason in {**EXEMPT_MODULES, **EXEMPT_PAGES}.items():
        assert reason and len(reason) > 40, (
            f"exemption {name!r} needs a reason naming the actual defect, got {reason!r}"
        )


@pytest.mark.full
@pytest.mark.parametrize("module_name", sorted(EXEMPT_MODULES))
def test_exempt_modules_still_fail(module_name: str) -> None:
    """A fixed exemption must be REMOVED, not left behind.

    Exemptions are self-cleaning: if the docstring defect has been repaired this
    turns red and asks for the entry to be deleted, so the list can only shrink.
    """
    failures = run_module_doctests(module_name)
    assert failures, (
        f"{module_name} is in EXEMPT_MODULES but its doctests now PASS — "
        "delete the entry from tests/_doctest_select.py so the module is gated."
    )


@pytest.mark.full
@pytest.mark.parametrize("page", sorted(EXEMPT_PAGES))
def test_exempt_pages_still_fail(page: str) -> None:
    """A fixed page exemption must be removed, not left behind."""
    with pytest.raises(Exception):  # noqa: B017 — any failure keeps it exempt
        run_page_fences(page)


# ---------------------------------------------------------------------------
# Publication guard: the mkdocs exclusion list cannot silently grow
# ---------------------------------------------------------------------------

#: The complete set of ``exclude_docs`` entries in ``mkdocs.yml``.  Publishing
#: is the default; every entry here is a deliberate, reviewed omission.
#:
#: This exists because a single blanket tree-drop (``theory/``) once removed
#: ~1.2k lines of accurate documentation from the site — including a page a
#: shipping source docstring links to — and nothing failed.  Widening the
#: exclusion now means editing this set on purpose.
EXPECTED_DOC_EXCLUSIONS: frozenset[str] = frozenset(
    {
        # tooling / non-page assets
        "_tooling/",
        "_static/",
        "css/",
        "hero-preview.html",
        # pre-generation stubs replaced by hooks/docs_autogen.py
        "systems/continuous/index.md",
        "systems/delay/index.md",
        "systems/discrete/index.md",
    }
)


def _mkdocs_exclusions() -> set[str]:
    """Parse the ``exclude_docs:`` block out of ``mkdocs.yml`` textually.

    Read as text rather than via ``yaml.safe_load`` so the guard stays
    independent of the config's other tags and plugins.
    """
    text = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    m = re.search(r"^exclude_docs:\s*\|\s*$", text, re.M)
    assert m, "mkdocs.yml has no `exclude_docs: |` block"
    entries: set[str] = set()
    for line in text[m.end() :].splitlines()[1:]:
        if line.strip() and not line.startswith(("  ", "\t")):
            break  # dedented out of the block
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            entries.add(stripped)
    return entries


def test_doc_exclusions_are_exactly_the_reviewed_set() -> None:
    """``exclude_docs`` matches the reviewed set — no silent growth."""
    actual = _mkdocs_exclusions()
    added = sorted(actual - EXPECTED_DOC_EXCLUSIONS)
    removed = sorted(EXPECTED_DOC_EXCLUSIONS - actual)
    assert not added, (
        "mkdocs.yml now excludes pages that were not reviewed. Publishing is "
        "the default; if these really must be dropped, add them to "
        f"EXPECTED_DOC_EXCLUSIONS with a reason: {added}"
    )
    assert not removed, f"these exclusions are gone from mkdocs.yml — update the guard: {removed}"


def test_no_blanket_tree_exclusions_over_content() -> None:
    """No content directory is dropped wholesale.

    A trailing-slash pattern over a content tree is the failure mode this whole
    guard exists to prevent: it silently swallows every page added under it
    later.  Only the non-page asset directories may be excluded that way.
    """
    asset_dirs = {"_tooling/", "_static/", "css/"}
    tree_drops = {e for e in _mkdocs_exclusions() if e.endswith("/")} - asset_dirs
    assert not tree_drops, (
        "these exclude whole content trees; list the individual pages instead "
        f"so a new page under them is published by default: {sorted(tree_drops)}"
    )


# ---------------------------------------------------------------------------
# Catalogue-count guard: a number in prose cannot drift from the registry
# ---------------------------------------------------------------------------

#: Files whose catalogue counts are checked against the live registry.
_COUNTED_FILES = ("README.md", "CLAUDE.md", "mkdocs.yml")

#: ``{'ode': 142, 'dde': 6, ...}`` — a literal `registry.families()` result.
_FAMILIES_LITERAL = re.compile(r"\{\s*(?:'(?:ode|dde|sde|map)':\s*\d+\s*,?\s*)+\}")
_FAMILY_PAIR = re.compile(r"'(ode|dde|sde|map)':\s*(\d+)")

#: ``177 built-in`` / ``(over) all 177 systems`` — prose asserting the catalogue
#: size *now*.  Deliberately narrow: a historical measurement ("slower on 3 of
#: 136 systems", "over 2x97 systems") is a record of what was measured then, not
#: a claim about today's catalogue, and must not be rewritten by a later count.
_PROSE_COUNT = re.compile(r"(\d{2,4})[- ]built-in|\ball\s+(\d{2,4})\s+systems\b")


def _counted_sources() -> list[tuple[str, str]]:
    """(label, text) for every file whose catalogue counts are gated."""
    out = [(f, (REPO_ROOT / f).read_text(encoding="utf-8")) for f in _COUNTED_FILES]
    out += [
        (str(p.relative_to(REPO_ROOT)), p.read_text(encoding="utf-8"))
        for p in sorted((REPO_ROOT / "docs").rglob("*.md"))
    ]
    return out


def test_documented_family_counts_match_the_registry() -> None:
    """Every literal ``registry.families()`` dict in the docs is the real one.

    Three of these had rotted to a stale ``'ode': 136`` while the catalogue grew
    to 142 — invisibly, because the value sits in a *comment* beside a fence the
    doctest gate executes happily. Executing a block proves the code runs, not
    that the number written next to it is true.
    """
    from tsdynamics import registry

    actual = dict(registry.families())
    wrong: list[str] = []
    for label, text in _counted_sources():
        for line_no, line in enumerate(text.splitlines(), 1):
            for literal in _FAMILIES_LITERAL.findall(line):
                claimed = {k: int(v) for k, v in _FAMILY_PAIR.findall(literal)}
                if claimed != actual:
                    wrong.append(f"{label}:{line_no} claims {claimed}, registry says {actual}")
    assert not wrong, "stale registry.families() literals in the docs:\n" + "\n".join(wrong)


def test_documented_catalogue_totals_are_real_counts() -> None:
    """No prose count (``177 built-in``, ``all 177 systems``) is invented.

    Each must be either the catalogue total or one family's count, so a stale
    ``171``/``154`` fails and a catalogue change forces the prose to be updated
    rather than quietly becoming a lie on the PyPI landing page and in the
    site's ``site_description`` meta tag.
    """
    from tsdynamics import registry

    families = dict(registry.families())
    legal = {sum(families.values())} | set(families.values())
    wrong: list[str] = []
    for label, text in _counted_sources():
        for line_no, line in enumerate(text.splitlines(), 1):
            hits = (g for m in _PROSE_COUNT.findall(line) for g in m if g)
            for claimed in (int(h) for h in hits):
                if claimed not in legal:
                    wrong.append(
                        f"{label}:{line_no} says {claimed}; real counts are "
                        f"{sorted(legal, reverse=True)}"
                    )
    assert not wrong, "stale catalogue counts in the docs:\n" + "\n".join(wrong)


# ---------------------------------------------------------------------------
# Harness self-tests
# ---------------------------------------------------------------------------


def test_gate_covers_everything_not_explicitly_exempt() -> None:
    """Discovery + exemptions partition the subjects, with nothing dropped."""
    modules = set(discover_doctest_modules())
    assert modules == set(gated_modules()) | set(EXEMPT_MODULES) | set(SLOW_MODULES)
    pages = set(discover_doc_pages())
    assert pages == set(gated_pages()) | set(EXEMPT_PAGES) | set(SLOW_PAGES)


def test_curated_lists_are_disjoint_and_unique() -> None:
    """Guard: no subject is in two tiers and no list has duplicates."""
    assert len(set(gated_modules())) == len(gated_modules())
    assert len(set(gated_pages())) == len(gated_pages())
    assert not (set(gated_modules()) & set(EXEMPT_MODULES))
    assert not (set(gated_modules()) & set(SLOW_MODULES))
    assert not (set(EXEMPT_MODULES) & set(SLOW_MODULES))
    assert not (set(EXEMPT_PAGES) & set(SLOW_PAGES))


def test_fence_extractor_skips_markers_and_transcripts() -> None:
    """The page extractor yields runnable scripts and skips fragments.

    Asserts the two opt-outs the page-fence contract relies on: a fence
    carrying the ``# skip-doctest`` marker, and a ``>>>`` doctest transcript
    (handled by the module path), are both excluded; a plain block is yielded.
    """
    page = (
        "intro\n"
        "```python\n"
        "import tsdynamics as ts\n"
        "ts.systems.Lorenz()\n"
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
    assert "ts.systems.Lorenz()" in blocks[0]
    assert all("skip-doctest" not in b for b in blocks)
    assert all(">>>" not in b for b in blocks)
