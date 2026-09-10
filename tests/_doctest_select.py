"""Doctest collection, namespace and tier-split machinery for the docs gate.

This module is the *data + engine* half of the executable-documentation gate
(stream ``DOCS-DOCTEST-GATE``); :mod:`tests.test_doctests` is the thin pytest
wrapper that turns it into parametrized test items.  Keeping the lists and the
runner here (one file per thing) lets the test module stay a handful of
parametrize calls, and lets other tooling reuse the same discovery helpers.

Two surfaces are gated, both executed under the suite-wide
``filterwarnings = error`` (see ``pyproject.toml``):

1. **Library docstrings** — the ``>>> …`` examples in ``src/tsdynamics/**``
   module/function/class docstrings, run with :mod:`doctest`.
2. **Documentation pages** — the fenced ```python``` blocks in ``docs/**.md``,
   executed top-to-bottom as a script (a page shares one namespace, so block
   *k+1* sees the names block *k* bound — the way a reader runs a tutorial).

Why a shared injected namespace
--------------------------------
The library's doctests are written for a *reader*: they use the short names a
user would have in scope — ``np`` (NumPy), ``ts`` (the package), every built-in
system class (``Lorenz``, ``Henon`` …) and every public analysis function
(``lyapunov_spectrum`` …) — without repeating ``import`` lines in every block.
That convention keeps the rendered docs readable, so the harness honours it by
seeding each doctest's globals with :func:`doctest_namespace` *on top of* the
module's own ``__dict__`` (so the documented object itself is always in scope).

Gate everything, exempt on purpose
----------------------------------
This gate is **inverted**: it discovers its own subjects rather than reading an
allow-list.  Every module under ``src/tsdynamics`` containing a ``>>>`` and every
``docs`` page containing a runnable ```python``` fence is gated *by default*
(:func:`discover_doctest_modules`, :func:`discover_doc_pages`).

That matters because the previous design was an allow-list of 20 modules and 22
pages — and 12 of the 25 modules it did *not* name were failing.  A gate whose
green tick certifies the half that was already clean is not a gate; worse, a new
module with a broken example joined nothing and so broke nothing.  Under
discovery the default is the opposite: a new example is gated the moment it is
written, and leaving the gate costs a **named entry with a written reason** in
:data:`EXEMPT_MODULES` / :data:`EXEMPT_PAGES`.

Exemptions are self-cleaning.  Two guards in :mod:`tests.test_doctests` keep the
list honest: every exemption must still be discovered (no entries for deleted
modules), and — in the ``full`` tier — every exempt subject must **still fail**,
so an exemption whose defect has been fixed turns red and asks to be removed.

Tier split (the ``full`` marker)
---------------------------------
The default (fast) tier runs every gated module and page, and **all of them
pass** under ``filterwarnings = error``.  A handful run genuinely heavy
simulations (a 600-point logistic orbit-diagram sweep; the animation and
composition pages) — those are named in :data:`SLOW_MODULES` / :data:`SLOW_PAGES`
and only run under ``-m full`` (the nightly sweep), so the inner loop stays
quick.

RuntimeWarning allowlist
------------------------
Under ``filterwarnings = error`` a stray ``RuntimeWarning`` (a benign
``log(0)`` / ``0/0`` inside an estimator's intermediate arithmetic) would turn a
correct doctest into a failure.  Modules whose *documented* numerics legitimately
trip such a warning are listed in :data:`RUNTIME_WARNING_MODULES`; for those the
runner downgrades ``RuntimeWarning`` to a non-error during execution.  Every
other warning category — and every other module — stays a hard error.

The page-fence contract
-----------------------
A page is gated automatically, and stays green once **every** one of its fences
either runs clean top-to-bottom as a script (fences on a page share one
namespace, so a later block may use names an earlier one bound) or opts out by
carrying the ``# skip-doctest`` marker (:data:`SKIP_MARKER`) — the marker is for
deliberately-illustrative fragments referencing a placeholder the reader
supplies.  Fences containing ``>>>`` are doctest transcripts and are likewise
not executed as a script (the module-doctest path covers those).

Provenance
----------
Every entry in :data:`EXEMPT_MODULES` was produced by *running* that module's
doctests under ``filterwarnings = error`` and recording the exact failure, which
is what its reason string quotes.  They are all defects in ``src/tsdynamics``
docstrings (missing expected-output lines, a comment on a *want* line, a name the
docstring never binds) rather than defects in the code they document.
:data:`EXEMPT_PAGES` is empty: every discovered page executes clean.
"""

from __future__ import annotations

import doctest
import importlib
import re
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Repo root = parent of the ``tests`` directory this file lives in.
REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"

#: doctest option flags applied to every example.
OPTIONFLAGS = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE

# ---------------------------------------------------------------------------
# Discovery (the docstring half of the gate)
#
# Everything under ``src/tsdynamics`` that contains a ``>>>`` example is gated.
# There is no allow-list: a module joins the gate by having an example, and the
# only way out is a named entry in EXEMPT_MODULES with a written reason.
# ---------------------------------------------------------------------------

SRC_ROOT = REPO_ROOT / "src"
PACKAGE_ROOT = SRC_ROOT / "tsdynamics"


def discover_doctest_modules() -> tuple[str, ...]:
    """Every importable ``tsdynamics`` module whose source contains ``>>>``.

    The scan is textual on purpose: it finds the examples without importing
    anything, so a module that fails to import is a *gate failure* rather than a
    silently-skipped module.
    """
    names: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        if ">>>" not in path.read_text(encoding="utf-8"):
            continue
        rel = path.relative_to(SRC_ROOT).with_suffix("")
        name = ".".join(rel.parts)
        if name.endswith(".__init__"):
            name = name[: -len(".__init__")]
        names.append(name)
    return tuple(names)


# ---------------------------------------------------------------------------
# Exemptions — the ONLY escape from the gate, and each costs a written reason.
#
# Every entry is a *defect in a docstring example*, diagnosed by running it.
# These live in ``src/tsdynamics/**`` and are owned by the code, not the docs,
# so they are recorded here rather than silently dropped.  A ``full``-tier guard
# (``test_exempt_modules_still_fail``) re-runs each one and FAILS when it starts
# passing, so a fixed example cannot linger here forgotten.
# ---------------------------------------------------------------------------

#: module -> the exact defect, as measured.
EXEMPT_MODULES: dict[str, str] = {
    "tsdynamics": (
        "The package-docstring tour shows `traj['x']` and `Lorenz()."
        "lyapunov_spectrum()` with the value in a trailing comment and no "
        "expected-output line, so doctest sees unexpected output. Either bind "
        "the results to names or write the repr as the expected output."
    ),
    "tsdynamics.analysis.chaos.expansion": (
        "`expansion_entropy(...)` expects `0.69...  # ln 2, exact` — the "
        "trailing comment sits on the *want* line, so it is part of the "
        "expected output and never matches. Move the comment to the `>>>` line."
    ),
    "tsdynamics.analysis.chaos.gali": (
        "`gali(Lorenz(), k=2, final_time=25.0)` expects `0.0...` but returns "
        "3.78e-11, which formats in scientific notation and cannot match a "
        "`0.0`-prefixed ellipsis. Wrap in `round(..., 6)` or expect `...e-...`."
    ),
    "tsdynamics.analysis.fixedpoints.fixed": (
        "Three `fixed_points(...)` examples print `FixedPointSet(N items)` but "
        "declare no expected output."
    ),
    "tsdynamics.analysis.fixedpoints.periodic": (
        "Four examples (`estimate_period`, `periodic_orbit`, `periodic_orbits`) "
        "print a repr but declare no expected output. Note these reference "
        "`VanDerPol`, which now EXISTS — the examples run, they just do not "
        "declare what they print."
    ),
    "tsdynamics.analysis.lyapunov": (
        "Three `lyapunov_spectrum`/`max_lyapunov` examples print a "
        "`LyapunovSpectrum(...)` repr with no expected-output line."
    ),
    "tsdynamics.analysis.orbits.poincare": (
        "`poincare_section(traj, plane=('z', 25.0))` uses a name `traj` that no "
        "earlier line in the docstring binds -> NameError."
    ),
    "tsdynamics.data.trajectory": (
        "The `Trajectory` class example opens with `traj = lor.integrate(...)` "
        "but never binds `lor`, so all six following lines NameError."
    ),
    "tsdynamics.derived.tangent": (
        "Two `TangentSystem` examples print a repr with no expected output."
    ),
    "tsdynamics.families.base": (
        "`p.unknown = 5.0  # raises AttributeError` actually raises, so doctest "
        "needs a `Traceback (most recent call last): ... AttributeError` block "
        "rather than a comment."
    ),
    "tsdynamics.families.continuous": (
        "`Henon().run(n=5000)` prints a Trajectory repr with no expected "
        "output, and `sol.meta['t_events'][0].shape` expects `(... ,)` "
        "(with a space) but gets `(70,)`."
    ),
    "tsdynamics.families.discrete": (
        "Two `DiscreteMap` examples print a repr with no expected output."
    ),
}

#: module -> why it is nightly-only.  Verified clean, but the example runs a
#: heavy simulation, so it stays out of the change-scoped inner loop.
SLOW_MODULES: dict[str, str] = {
    "tsdynamics.analysis.orbits.orbit_diagram": (
        "the documented example sweeps a 600-point logistic orbit diagram"
    ),
}


def gated_modules() -> tuple[str, ...]:
    """Discovered modules minus the exempt and the nightly-only ones."""
    skip = set(EXEMPT_MODULES) | set(SLOW_MODULES)
    return tuple(m for m in discover_doctest_modules() if m not in skip)


def full_tier_modules() -> tuple[str, ...]:
    """The nightly-only modules that still exist."""
    found = set(discover_doctest_modules())
    return tuple(m for m in SLOW_MODULES if m in found)


#: Modules whose documented numerics legitimately emit a ``RuntimeWarning``
#: (benign intermediate ``log(0)``/``0/0``).  Downgraded to a non-error *only*
#: for these; every other category stays a hard error everywhere.
RUNTIME_WARNING_MODULES: frozenset[str] = frozenset(
    {
        "tsdynamics.analysis.orbits.orbit_diagram",
    }
)

# ---------------------------------------------------------------------------
# Discovery (the page-fence half of the gate)
#
# Every ``docs/**.md`` page with at least one runnable ```python``` fence is
# gated.  As with modules there is no allow-list: a page joins by having a
# runnable fence.  A fence that is a signature listing, a calling pattern, or a
# deliberate demonstration of what *raises* opts out in place with the
# ``# skip-doctest`` marker — visible to the reader of the page, unlike a name
# buried in a list over here.
# ---------------------------------------------------------------------------


#: page -> the exact defect.  Empty is the goal, and currently the truth: every
#: discovered page executes clean.  Kept as the documented escape hatch so a
#: genuinely un-runnable page has somewhere to go *with a reason* instead of
#: being quietly dropped from a curated list.
EXEMPT_PAGES: dict[str, str] = {}

#: page -> why it is nightly-only.  These pass, but each runs minutes of
#: simulation, so they stay out of the change-scoped inner loop.
SLOW_PAGES: dict[str, str] = {
    "visualization/composition.md": (
        "~240 s: builds many multi-panel composites, several of them 3-D"
    ),
    "visualization/animation.md": ("~105 s: renders animations, including an mp4/gif encode"),
}


def discover_doc_pages() -> tuple[str, ...]:
    """Every ``docs`` page with at least one runnable python fence.

    Uses :func:`iter_python_fences`, which is defined further down this module —
    fine because the name is resolved when this is *called*, not when it is
    defined.  Nothing may call it at import time for that reason.
    """
    pages: list[str] = []
    for path in sorted(DOCS_DIR.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        if next(iter_python_fences(text), None) is not None:
            pages.append(str(path.relative_to(DOCS_DIR)))
    return tuple(pages)


def gated_pages() -> tuple[str, ...]:
    """Discovered pages minus the exempt and the nightly-only ones."""
    skip = set(EXEMPT_PAGES) | set(SLOW_PAGES)
    return tuple(p for p in discover_doc_pages() if p not in skip)


def full_tier_pages() -> tuple[str, ...]:
    """The nightly-only pages that still exist."""
    found = set(discover_doc_pages())
    return tuple(p for p in SLOW_PAGES if p in found)


# A fenced block carrying this marker is a deliberately-illustrative fragment
# (pseudo-code or a snippet the reader completes) and is skipped by the page
# executor.  See "The page-fence contract" in this module's docstring.
SKIP_MARKER = "# skip-doctest"

# Matches an opening python code fence (```python / ```py / ```pycon, any
# backtick run length).  Material/pymdown also allows ``{.python}`` attr lists.
_FENCE_OPEN = re.compile(r"^(`{3,})\s*(?:python|py|pycon)\b.*$")


# ---------------------------------------------------------------------------
# Namespace
# ---------------------------------------------------------------------------


def doctest_namespace() -> dict[str, Any]:
    """Build the shared globals seeded into every doctest / page block.

    Contains ``np`` (NumPy), ``ts`` (the package), every built-in system class
    and every public top-level name (analysis functions, result types, derived
    wrappers).  These are the names a reader has in scope, so the readable,
    import-light examples in the docstrings and pages run as written.
    """
    import numpy as np

    import tsdynamics as ts
    import tsdynamics.systems as systems

    ns: dict[str, Any] = {"np": np, "ts": ts}
    for name in dir(systems):
        obj = getattr(systems, name)
        if isinstance(obj, type):
            ns[name] = obj
    for name in dir(ts):
        if name.startswith("_"):
            continue
        ns.setdefault(name, getattr(ts, name))
    return ns


# ---------------------------------------------------------------------------
# Module doctests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoctestFailure:
    """One failed doctest example, rendered for an assertion message."""

    module: str
    message: str


def run_module_doctests(module_name: str) -> list[DoctestFailure]:
    """Run a module's doctests with the injected namespace; return failures.

    Each example's globals are ``{**module.__dict__, **doctest_namespace()}`` so
    the documented object is always in scope *and* the reader-facing short names
    resolve.  ``RuntimeWarning`` is downgraded to a non-error only for modules in
    :data:`RUNTIME_WARNING_MODULES`; everything else runs under the suite-wide
    ``error`` filter (inherited from the pytest config).
    """
    module = importlib.import_module(module_name)
    extras = doctest_namespace()
    finder = doctest.DocTestFinder()
    tests = [
        t
        for t in finder.find(module, module_name, globs={**module.__dict__, **extras})
        if t.examples
    ]
    failures: list[DoctestFailure] = []
    runner = _RecordingRunner(optionflags=OPTIONFLAGS)
    with warnings.catch_warnings():
        if module_name in RUNTIME_WARNING_MODULES:
            warnings.simplefilter("ignore", RuntimeWarning)
        for test in tests:
            runner.failures_text = []
            runner.run(test, clear_globs=False)
            for msg in runner.failures_text:
                failures.append(DoctestFailure(module_name, msg))
    return failures


class _RecordingRunner(doctest.DocTestRunner):
    """A doctest runner that captures failure text instead of printing it."""

    failures_text: list[str]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.failures_text = []

    def report_failure(
        self,
        out: Any,
        test: doctest.DocTest,
        example: doctest.Example,
        got: str,
    ) -> None:
        self.failures_text.append(
            f"{test.name}:{(test.lineno or 0) + example.lineno + 1}\n"
            f"  >>> {example.source.strip()}\n"
            f"  expected: {example.want.strip()!r}\n"
            f"  got:      {got.strip()!r}"
        )

    def report_unexpected_exception(
        self,
        out: Any,
        test: doctest.DocTest,
        example: doctest.Example,
        exc_info: Any,
    ) -> None:
        exc = exc_info[1]
        self.failures_text.append(
            f"{test.name}:{(test.lineno or 0) + example.lineno + 1}\n"
            f"  >>> {example.source.strip()}\n"
            f"  raised:   {type(exc).__name__}: {exc}"
        )


# ---------------------------------------------------------------------------
# Documentation-page fences
# ---------------------------------------------------------------------------


def iter_python_fences(text: str) -> Iterator[str]:
    """Yield the body of each ```python``` (``py``/``pycon``) fence in *text*.

    Blocks carrying the :data:`SKIP_MARKER` comment are skipped (illustrative
    fragments).  Blocks containing ``>>>`` are also skipped: those are doctest
    transcripts, handled by the module-doctest path, not executed as a script.
    """
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = _FENCE_OPEN.match(lines[i])
        if not m:
            i += 1
            continue
        ticks = m.group(1)
        close = re.compile(r"^" + ticks + r"\s*$")
        body: list[str] = []
        j = i + 1
        while j < len(lines) and not close.match(lines[j]):
            body.append(lines[j])
            j += 1
        block = "\n".join(body)
        if SKIP_MARKER not in block and ">>>" not in block:
            yield block
        i = j + 1


def run_page_fences(page_rel: str) -> None:
    """Execute every runnable python fence in a doc page as one script.

    All fences on a page share a single namespace seeded from
    :func:`doctest_namespace`, so a later block sees names an earlier block
    bound (the way a reader runs a tutorial sequentially).  Raises the original
    exception (with the page/block location chained) on the first failing block,
    which pytest renders as the test failure.
    """
    path = DOCS_DIR / page_rel
    text = path.read_text(encoding="utf-8")
    ns = doctest_namespace()
    for k, block in enumerate(iter_python_fences(text)):
        try:
            code = compile(block, f"{page_rel}:block{k}", "exec")
            exec(code, ns)  # noqa: S102 — executing curated docs is the point
        except Exception as exc:  # pragma: no cover - exercised on a broken page
            raise AssertionError(
                f"{page_rel} python fence #{k} failed: {type(exc).__name__}: {exc}"
            ) from exc
