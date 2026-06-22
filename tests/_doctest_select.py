"""Doctest collection, namespace and tier-split machinery for the docs gate.

This module is the *data + engine* half of the executable-documentation gate
(stream ``DOCS-DOCTEST-GATE``); :mod:`tests.test_doctests` is the thin pytest
wrapper that turns it into parametrized test items.  Keeping the lists and the
runner here (one file per thing) lets the test module stay a handful of
parametrize calls, and lets other tooling import the same curated lists.

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

Tier split (the ``full`` marker)
---------------------------------
The default (fast) tier runs the curated, **verified-fast** docstring modules
and the curated clean doc pages, and **all of them pass** under
``filterwarnings = error``.  A handful of doctests run genuinely heavy
simulations (e.g. a 600-point logistic orbit-diagram sweep) — those modules are
in :data:`FULL_ONLY_MODULES` and only run under ``-m full`` (the nightly sweep),
so the inner loop stays quick.

RuntimeWarning allowlist
------------------------
Under ``filterwarnings = error`` a stray ``RuntimeWarning`` (a benign
``log(0)`` / ``0/0`` inside an estimator's intermediate arithmetic) would turn a
correct doctest into a failure.  Modules whose *documented* numerics legitimately
trip such a warning are listed in :data:`RUNTIME_WARNING_MODULES`; for those the
runner downgrades ``RuntimeWarning`` to a non-error during execution.  Every
other warning category — and every other module — stays a hard error.

Provenance
----------
The curated lists below were produced by running every ``src/tsdynamics``
module's doctests and every ``docs`` page's python fences under
``filterwarnings = error`` and keeping the ones that pass clean.  Modules / pages
whose narrative docstrings are not yet self-contained doctests are deliberately
*excluded* (not silenced): closing those gaps is tracked follow-up work, and the
``docs/contributing/page-template.md`` contract is what new pages must satisfy to
join the curated set.
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
# Curated module lists (the docstring half of the gate)
#
# CURATED_MODULES — verified clean *and* fast: the default-tier gate.  Every one
#   passes under filterwarnings=error.
# FULL_ONLY_MODULES — verified clean but slow (heavy simulations in the example);
#   collected only under ``-m full`` so the fast loop stays quick.
# ---------------------------------------------------------------------------

CURATED_MODULES: tuple[str, ...] = (
    "tsdynamics.analysis.chaos.zero_one",
    "tsdynamics.analysis.dimensions.correlation",
    "tsdynamics.analysis.entropy.core",
    "tsdynamics.analysis.entropy.lz",
    "tsdynamics.analysis.entropy.multiscale",
    "tsdynamics.analysis.lyapunov.from_data",
    "tsdynamics.analysis.orbits.return_map",
    "tsdynamics.data.sampling",
    "tsdynamics.derived.ensemble",
    "tsdynamics.derived.poincare",
    "tsdynamics.derived.projected",
    "tsdynamics.derived.stroboscopic",
    "tsdynamics.engine.run",
    "tsdynamics.errors",
    "tsdynamics.families.delay",
    "tsdynamics.families.wrapped",
    "tsdynamics.registry",
    "tsdynamics.transforms.spectral",
    "tsdynamics.utils.grids",
)

#: Verified clean, but the documented example runs a heavy simulation — gated to
#: the ``full`` tier (nightly) so it never slows the change-scoped loop.
FULL_ONLY_MODULES: tuple[str, ...] = ("tsdynamics.analysis.orbits.orbit_diagram",)

#: Modules whose documented numerics legitimately emit a ``RuntimeWarning``
#: (benign intermediate ``log(0)``/``0/0``).  Downgraded to a non-error *only*
#: for these; every other category stays a hard error everywhere.
RUNTIME_WARNING_MODULES: frozenset[str] = frozenset(
    {
        "tsdynamics.analysis.orbits.orbit_diagram",
    }
)

# ---------------------------------------------------------------------------
# Curated documentation-page list (the page-fence half of the gate)
#
# Every page here executes its ```python``` fences top-to-bottom without raising
# under filterwarnings=error.  Pages with intentionally-illustrative fragments
# (referencing a placeholder ``system``/``signal`` the reader supplies) are
# excluded until they adopt the page-template contract (a ``# skip-doctest``
# marker on the fragment fence).
# ---------------------------------------------------------------------------

CURATED_PAGES: tuple[str, ...] = (
    "analysis/chaos.md",
    "analysis/dimensions.md",
    "analysis/embedding.md",
    "analysis/entropy.md",
    "analysis/index.md",
    "analysis/poincare.md",
    "analysis/recurrence.md",
    "analysis/surrogate.md",
    "index.md",
    "project/changelog.md",
    "project/citation.md",
    "project/contributing.md",
    "reference/registry.md",
    "reference/top-level.md",
    "start/first-trajectory.md",
    "start/install.md",
    "systems/delay/index.md",
    "systems/discrete/index.md",
    "systems/index.md",
    "theory/backends.md",
    "theory/compilation.md",
    "theory/solvers.md",
    "tutorials/equations-to-basins.md",
)

# A fenced block carrying this marker is a deliberately-illustrative fragment
# (pseudo-code or a snippet the reader completes) and is skipped by the page
# executor.  Documented in ``docs/contributing/page-template.md``.
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
    import tsdynamics.transforms  # noqa: F401  (populates registry.transforms)

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
