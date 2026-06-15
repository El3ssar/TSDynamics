"""MkDocs hook: citation / competitor lint for the published documentation.

The project rule (see ``CLAUDE.md`` and the v3 program notes) is absolute:
**never name a competitor dynamical-systems ecosystem in shipped docs** — cite
the *original* literature instead.  Ideas may be absorbed; attribution goes to
the papers, never to another library.

This hook enforces that mechanically at build time.  It scans the markdown of
every page MkDocs renders — hand-written prose *and* the auto-generated system
catalogue (so a stray competitor name in a ``reference`` ClassVar is caught
too) — and aborts the build with a :class:`~mkdocs.exceptions.PluginError`
listing every hit.  Because it fails in ``on_post_build`` rather than per page,
one build reports *all* violations at once.

Design notes
------------
* Matching is case-insensitive and token-aware.  The bare word *Julia* is
  intentionally **not** blocked: "Julia set" (Gaston Julia, complex dynamics)
  and people named Julia are legitimate.  Only the ``*.jl`` package ecosystem
  and named competitor libraries are.
* The blocklist is deliberately small and explicit.  Add a pattern here (with a
  comment) rather than scattering ad-hoc checks; this is the single source of
  truth for "names that may not ship".
"""

from __future__ import annotations

import re

from mkdocs.exceptions import PluginError

#: Patterns that must never appear in a published page.  Each entry carries a
#: short rationale; matching is case-insensitive (see ``_PATTERN``).
_BLOCKED: list[str] = [
    # Named competitor libraries — cite the original paper, not the tool.
    r"\bnolds\b",
    r"\bpyunicorn\b",
    # The competitor ecosystem named as such (not the bare language word).
    r"Julia\s+(?:ecosystem|dynamical|package|library| first|implementation)",
    # Any Julia package reference, e.g. DynamicalSystems.jl, ChaosTools.jl,
    # DifferentialEquations.jl, Attractors.jl, DelayDiffEq.jl, … — the catch-all
    # that makes the rule future-proof against new ``*.jl`` packages.
    r"\b[A-Za-z][A-Za-z0-9_]*\.jl\b",
]

_PATTERN = re.compile("|".join(_BLOCKED), re.IGNORECASE)

#: Accumulated across the build; reset per build in :func:`on_pre_build`.
_violations: list[str] = []


def on_pre_build(config):  # noqa: ARG001 (mkdocs hook signature)
    """Reset accumulated violations so ``mkdocs serve`` rebuilds are clean."""
    _violations.clear()


def on_page_markdown(markdown, page, config, files):  # noqa: ARG001
    """Scan one page's markdown for blocked names, recording any hits."""
    for match in _PATTERN.finditer(markdown):
        snippet = match.group(0).strip()
        _violations.append(f"{page.file.src_uri}: {snippet!r}")
    return markdown


def on_post_build(config):  # noqa: ARG001
    """Fail the build if any page named a competitor ecosystem."""
    if not _violations:
        return
    listing = "\n  ".join(sorted(set(_violations)))
    raise PluginError(
        "citation lint failed — competitor / ecosystem names found in the docs "
        "(cite the original literature instead, never a competing library):\n  " + listing
    )
