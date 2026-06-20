"""Diff- and registry-aware test selection for fast, change-scoped CI.

The bulk suite is *registry-driven*: most tests are parametrized over every
built-in system (149 today) or every registered analysis/transform, so a run of
the whole ``not full`` tier is thousands of items.  On a PR that only touches one
system or one analysis area, almost all of that work is irrelevant — yet the old
CI ran it all, on a 2×2 matrix, twice.  This module narrows a run to the tests a
diff can actually affect, **biased always toward over-selection**: when in doubt
it escalates to the full suite rather than risk a silent skip.

It is wired through ``conftest.py`` and activated with ``--changed`` (diff vs
``origin/main`` by default; override the base with ``--changed-since=REF`` or the
``TSD_CHANGED_BASE`` env var).  Without the flag nothing here runs, so the
nightly full sweep, the release gate and any plain ``pytest`` invocation are
untouched.

Selection model
---------------
1. Compute the changed files vs the base (committed branch diff ∪ staged ∪
   unstaged ∪ untracked).  If git cannot answer → **full run** (fail-safe).
2. If *any* changed file is **foundational** — the engine, solver, family,
   derived-wrapper, data or utils layers; the registry / package ``__init__`` /
   plugin wiring; a shared test fixture (``conftest``/``_strategies``/
   ``_sampling``/``_engine_marker``); the lockfile or ``pyproject``; *any* Rust
   crate; or *any* CI workflow — selection is **disabled** and the full tier
   runs.  These files can change the behaviour of essentially every test.
3. Otherwise build a scoped selection:
   * a changed ``tests/test_*.py`` → run that file;
   * a changed ``src/tsdynamics/systems/<…>.py`` → restrict the per-system
     sweeps to the systems *defined in that module* (mapped through the live
     registry by module name, so it is correct whether the package is installed
     as a wheel or editable);
   * a changed ``src/tsdynamics/analysis/<area>/…`` → run that area's test
     files (``_AREA_TESTS``);
   * a changed ``transforms``/``viz`` source file → that surface's tests;
   * a cheap set of registry/layout **guard** tests always runs in scoped mode;
   * documentation / planning / tooling paths are ignored (no test impact);
   * **any path that matches none of the above escalates to a full run.**

The result is a :class:`Plan`.  ``conftest`` turns it into deselections and
prints exactly what it kept and why (``[changed-select] …``).

Safety net: this only ever runs on *PRs* and in the local dev loop.  The
complete suite still runs on every push to ``main`` (the release gate) and every
night (``-m full``), so a mis-scoped selection cannot ship a regression — it can
only delay catching it from "this PR" to "the merge / the nightly".  Because of
that, the failure mode we engineer against is *under*-selection, and every
ambiguous case above resolves to running more, never fewer, tests.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

#: Default diff base; overridable via ``--changed-since`` or ``TSD_CHANGED_BASE``.
DEFAULT_BASE = "origin/main"

# ---------------------------------------------------------------------------
# Classification tables — edit these (not the CI YAML) to tune selection.
# ---------------------------------------------------------------------------

#: Directory prefixes whose change forces a full run (cross-cutting layers).
_FOUNDATIONAL_PREFIXES: tuple[str, ...] = (
    "src/tsdynamics/engine/",
    "src/tsdynamics/solvers/",
    "src/tsdynamics/families/",
    "src/tsdynamics/derived/",
    "src/tsdynamics/data/",
    "src/tsdynamics/utils/",
    "crates/",
    ".github/workflows/",
)

#: Individual files whose change forces a full run.
_FOUNDATIONAL_FILES: frozenset[str] = frozenset(
    {
        "src/tsdynamics/registry.py",
        "src/tsdynamics/__init__.py",
        "src/tsdynamics/plugins.py",
        "src/tsdynamics/analysis/__init__.py",
        "src/tsdynamics/analysis/_result.py",
        "src/tsdynamics/transforms/__init__.py",
        "src/tsdynamics/transforms/_common.py",
        "tests/conftest.py",
        "tests/_engine_marker.py",
        "tests/_strategies.py",
        "tests/_sampling.py",
        "tests/_changed_select.py",
        "tests/xval_harness.py",
        "pyproject.toml",
        "uv.lock",
    }
)

#: Analysis leaf area (``src/tsdynamics/analysis/<area>/``) → its test files.
#: A changed area not listed here escalates to a full run (and the guard test
#: ``tests/test_changed_select.py`` flags the omission).
_AREA_TESTS: dict[str, tuple[str, ...]] = {
    "entropy": ("test_entropy.py", "test_property_entropy.py"),
    "dimensions": ("test_dimensions.py", "test_property_dimensions.py"),
    "embedding": ("test_embedding.py", "test_property_embedding.py"),
    "recurrence": ("test_recurrence.py", "test_property_recurrence.py"),
    "surrogate": ("test_surrogate.py", "test_property_surrogate.py"),
    "chaos": ("test_chaos.py",),
    "fixedpoints": ("test_fixed_points.py",),
    "orbits": ("test_orbits.py", "test_orbit_diagram_perf.py", "test_poincare_perf.py"),
    "basins": ("test_basins.py",),
    # Lyapunov is cross-cutting (the spectrum feeds the known-value catalogue),
    # so it pulls its dedicated tests *and* the literature catalogue.
    "lyapunov": (
        "test_lyapunov_from_data.py",
        "test_variational.py",
        "test_dde_lyapunov.py",
        "test_known_values.py",
    ),
}

_TRANSFORM_TESTS: tuple[str, ...] = ("test_transforms.py", "test_property_transforms.py")
_VIZ_TESTS: tuple[str, ...] = (
    "test_plotspec.py",
    "test_to_plot_spec.py",
    "test_renderers_registry.py",
)

#: Cross-cutting analysis tests that exercise functions from *several* areas
#: (the "regular vs random" cross-quantifier gate; the analysis-pack smoke).
#: Added whenever ANY analysis area or transform is touched, since a one-area
#: edit can break the cross-quantifier agreement they assert.
_CROSSCUT_ANALYSIS_TESTS: tuple[str, ...] = (
    "test_known_quantifiers.py",
    "test_analysis.py",
)

#: The registry-driven per-system sweep files. They mix parametrized sweeps
#: (scoped per system) with hand-written, NON-parametrized per-system tests
#: (e.g. ``test_lorenz96_integrates``) that carry no system in their id — so on a
#: system-module change we keep those bespoke tests too (see :func:`keep_item`),
#: while the big parametrized sweeps stay scoped to the changed systems.
_SYSTEM_SWEEP_FILES: frozenset[str] = frozenset(
    {"test_ode_systems.py", "test_map_systems.py", "test_dde_systems.py"}
)

#: Cheap integrity tests that always run in scoped mode — they catch the
#: "added a system/analysis but forgot to wire it" class of mistake.
_ALWAYS_GUARDS: tuple[str, ...] = (
    "test_registry.py",
    "test_analysis_registry.py",
    "test_analysis_layout.py",
)

#: Paths with no bearing on the test suite — ignored, never escalate.
_IGNORE_PREFIXES: tuple[str, ...] = (
    "docs/",
    "benches/",
    "planning/",
    ".claude/",
    "scripts/",
    ".github/ISSUE_TEMPLATE/",
)
_IGNORE_SUFFIXES: tuple[str, ...] = (".md", ".rst", ".txt")
_IGNORE_FILES: frozenset[str] = frozenset(
    {
        "Makefile",
        "mkdocs.yml",
        "LICENSE",
        ".gitignore",
        ".pre-commit-config.yaml",
        "CHANGELOG.md",
        "README.md",
        "CONTRIBUTING.md",
        "ROADMAP.md",
        "STREAMS.md",
    }
)


@dataclass
class Plan:
    """The outcome of classifying a diff."""

    full: bool
    reason: str
    changed: list[str] = field(default_factory=list)
    #: Test-file *basenames* whose every item is kept.
    selected_files: set[str] = field(default_factory=set)
    #: System *names* whose parametrized sweep items are kept.
    systems: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Git plumbing
# ---------------------------------------------------------------------------


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    # core.quotePath=false → git returns raw UTF-8 paths instead of C-quoting
    # non-ASCII names, so a unicode path classifies correctly (not "unrecognized").
    return subprocess.run(
        ["git", "-c", "core.quotePath=false", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def repo_root() -> Path | None:
    """Absolute path of the working tree root, or ``None`` if not in a repo."""
    r = _git(["rev-parse", "--show-toplevel"], cwd=Path.cwd())
    if r.returncode != 0 or not r.stdout.strip():
        return None
    return Path(r.stdout.strip())


def changed_files(base: str, root: Path) -> set[str] | None:
    """Repo-relative POSIX paths changed vs ``base``.

    Union of: the committed branch diff (``merge-base(base, HEAD)..HEAD``), the
    staged and unstaged working-tree changes, and untracked files — so a local
    run reflects edits that are not yet committed.  Returns ``None`` if *any*
    git invocation fails (the caller then runs the full suite).
    """
    mb = _git(["merge-base", base, "HEAD"], cwd=root)
    point = mb.stdout.strip() if mb.returncode == 0 and mb.stdout.strip() else base
    commands = [
        ["diff", "--name-only", point, "HEAD"],
        ["diff", "--name-only"],
        ["diff", "--name-only", "--cached"],
        ["ls-files", "--others", "--exclude-standard"],
    ]
    out: set[str] = set()
    for cmd in commands:
        r = _git(cmd, cwd=root)
        if r.returncode != 0:
            return None
        for line in r.stdout.splitlines():
            line = line.strip()
            if line:
                out.add(PurePosixPath(line).as_posix())
    return out


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _is_foundational(path: str) -> bool:
    return path in _FOUNDATIONAL_FILES or path.startswith(_FOUNDATIONAL_PREFIXES)


def _is_ignored(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return (
        path.startswith(_IGNORE_PREFIXES)
        or path.endswith(_IGNORE_SUFFIXES)
        or name in _IGNORE_FILES
    )


def _path_to_module(path: str) -> str | None:
    """``src/tsdynamics/systems/continuous/x.py`` → ``tsdynamics.systems.continuous.x``."""
    if not path.startswith("src/") or not path.endswith(".py"):
        return None
    mod = path[len("src/") : -len(".py")].replace("/", ".")
    if mod.endswith(".__init__"):
        mod = mod[: -len(".__init__")]
    return mod


def _systems_by_module() -> dict[str, set[str]]:
    """Map each module that defines built-in systems → the system names in it.

    Keyed by the registry's recorded module name (not a filesystem path), so it
    is correct whether ``tsdynamics`` is installed as a wheel (site-packages) or
    editable (``src/``).
    """
    from tsdynamics import registry

    out: dict[str, set[str]] = {}
    for entry in registry.all_systems(builtin=True):
        out.setdefault(entry.module, set()).add(entry.name)
    return out


def classify(changed: set[str] | None) -> Plan:
    """Turn a changed-file set into a :class:`Plan` (full vs scoped selection)."""
    if changed is None:
        return Plan(full=True, reason="could not compute git diff")

    files = sorted(changed)
    if not files:
        # Base == HEAD (e.g. a freshly cut branch). Run only the cheap guards.
        return Plan(
            full=False,
            reason="no changes vs base",
            changed=files,
            selected_files=set(_ALWAYS_GUARDS),
        )

    selected: set[str] = set()
    system_modules: list[str] = []
    escalate_reasons: list[str] = []
    notes: list[str] = []
    analysis_or_transform_touched = False

    for path in files:
        if _is_foundational(path):
            return Plan(full=True, reason=f"foundational change: {path}", changed=files)
        if _is_ignored(path):
            notes.append(f"ignored: {path}")
            continue
        name = path.rsplit("/", 1)[-1]
        if path.startswith("tests/") and name.startswith("test_") and name.endswith(".py"):
            selected.add(name)
            continue
        if path.startswith("src/tsdynamics/systems/"):
            if name == "__init__.py":
                return Plan(
                    full=True,
                    reason=f"systems export wiring changed: {path}",
                    changed=files,
                )
            mod = _path_to_module(path)
            if mod is None:
                escalate_reasons.append(f"unparseable system path: {path}")
            else:
                system_modules.append(mod)
            continue
        if path.startswith("src/tsdynamics/analysis/"):
            parts = path.split("/")
            area = parts[3] if len(parts) > 4 else None  # .../analysis/<area>/file
            tests = _AREA_TESTS.get(area or "")
            if not tests:
                escalate_reasons.append(f"unmapped analysis path: {path}")
            else:
                selected.update(tests)
                analysis_or_transform_touched = True
            continue
        if path.startswith("src/tsdynamics/transforms/"):
            selected.update(_TRANSFORM_TESTS)
            analysis_or_transform_touched = True
            continue
        if path.startswith("src/tsdynamics/viz/"):
            selected.update(_VIZ_TESTS)
            continue
        # Anything unrecognized: be conservative and run everything.
        escalate_reasons.append(f"unrecognized path: {path}")

    systems: set[str] = set()
    if system_modules:
        by_mod = _systems_by_module()
        for mod in system_modules:
            found = by_mod.get(mod, set())
            if not found:
                escalate_reasons.append(f"system module with no registered class: {mod}")
            systems |= found

    if escalate_reasons:
        return Plan(full=True, reason="; ".join(escalate_reasons), changed=files)

    # A one-area edit can break the cross-quantifier "regular vs random" agreement
    # gate and the analysis-pack smoke, which span several areas → always include.
    if analysis_or_transform_touched:
        selected.update(_CROSSCUT_ANALYSIS_TESTS)

    selected |= set(_ALWAYS_GUARDS)
    return Plan(
        full=False,
        reason="scoped selection",
        changed=files,
        selected_files=selected,
        systems=systems,
        notes=notes,
    )


def compute_plan(base: str) -> Plan:
    """Compute the selection :class:`Plan` for the diff vs ``base`` (never raises)."""
    try:
        root = repo_root()
        if root is None:
            return Plan(full=True, reason="not a git repository")
        return classify(changed_files(base, root))
    except Exception as exc:  # pragma: no cover - defensive: any failure → full
        return Plan(full=True, reason=f"selection error ({exc!r})")


# ---------------------------------------------------------------------------
# Per-item keep decision (used by conftest's collection hook)
# ---------------------------------------------------------------------------


_system_names_cache: frozenset[str] | None = None


def _all_system_names() -> frozenset[str]:
    """Names of every built-in system (cached)."""
    global _system_names_cache
    if _system_names_cache is None:
        from tsdynamics import registry

        _system_names_cache = frozenset(e.name for e in registry.all_systems(builtin=True))
    return _system_names_cache


def system_name_of(item: object) -> str | None:
    """The built-in system a parametrized ``item`` is bound to, if any.

    Two parametrize conventions carry a system, and both are recognised
    regardless of the parameter's *name*:

    * a ``SystemEntry`` value — the family fixtures (``ode_entry``/``map_entry``/…)
      and ad-hoc ``parametrize("entry", registry.all_systems(...))`` (the
      known-value catalogue);
    * a plain **string equal to a registered system name** — the by-name sweeps
      (``test_rust_engine`` ``ODE_SYSTEMS``/``MAP_SYSTEMS``, ``test_ode_systems``
      / ``test_xval_catalogue`` ``INTEGRATION_SAMPLE``, ``test_dde_engine``'s DDE
      names, the hand-listed ``["Lorenz", "Rossler", …]`` cases). Missing these
      would silently drop a changed system's engine/integration tests — exactly
      the under-selection this guards against.

    A string that is *not* a system name (``"bdf"``, ``"line"``, a function name)
    returns ``None`` and is governed only by file selection — so no false match.
    """
    from tsdynamics.registry import SystemEntry

    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return None
    values = list(callspec.params.values())
    for value in values:
        if isinstance(value, SystemEntry):
            return value.name
    names = _all_system_names()
    for value in values:
        if isinstance(value, str) and value in names:
            return value
    return None


def keep_item(item: object, plan: Plan) -> bool:
    """Whether ``item`` survives a scoped ``plan``."""
    filename = Path(str(getattr(item, "path", ""))).name
    if filename in plan.selected_files:
        return True
    name = system_name_of(item)
    if name is not None:
        return name in plan.systems
    # Hand-written, non-parametrized per-system tests (e.g. test_lorenz96_integrates)
    # live in the sweep files but carry no system in their id. When any system
    # module changed, keep them — they are few and cheap, and are exactly the
    # bespoke regressions the parametrized sweep cannot reach.
    return bool(plan.systems and filename in _SYSTEM_SWEEP_FILES)
