"""
TSDynamics — compiled dynamical systems: integration, iteration, and chaos analysis.

Quick start
-----------
>>> from tsdynamics.systems import Lorenz, MackeyGlass, Henon
>>> traj = Lorenz().run(final_time=100.0, dt=0.01)
>>> traj.t.shape, traj.y.shape
((10001,), (10001, 3))
>>> traj["x"]                          # named component access
>>> Lorenz().lyapunov_spectrum()       # ≈ [0.91, 0, -14.57]

Beyond integration, the :mod:`~tsdynamics.derived` wrappers re-present any
system through a new lens (Poincaré map, stroboscopic map, tangent dynamics,
ensembles), and :mod:`~tsdynamics.analysis` provides the quantifiers that
consume them (orbit/bifurcation diagrams, Poincaré sections, Lyapunov tools,
fixed points).

Built-in systems live under :mod:`tsdynamics.systems` (``tsdynamics.systems.Lorenz``)
so the top-level namespace stays focused on the base classes, the derived-system
wrappers, the analysis functions, and the submodules.  For backwards
compatibility ``tsdynamics.Lorenz`` (and ``from tsdynamics import Lorenz``) still
resolve lazily.  See :mod:`tsdynamics.registry` for programmatic access.  Internal
helpers (``ParamSet``, ``SystemBase``) live under ``tsdynamics.families``.

Curated top level — twelve names
--------------------------------
``tsdynamics.<TAB>`` shows **only what you type**: the five family bases you
subclass, :class:`Trajectory`, the :func:`plot` front door, and the four
submodules worth typing a dot after (:mod:`~tsdynamics.systems`,
:mod:`~tsdynamics.analysis`, :mod:`~tsdynamics.viz`, :mod:`~tsdynamics.errors`).

The membership rule, and the four corollaries it decomposes into, are written
out above ``__all__`` in this module's source.  In one line: *a name earns a slot
only if a user types it — and nobody should ever have to construct a library
type to make a call, so a name that is exported because a signature demands one
is evidence of a signature bug.*

Everything else stays **fully reachable** — dropped from ``__all__`` and
autocomplete, never removed:

- the analysis functions and result classes: ``ts.correlation_dimension``,
  ``from tsdynamics import correlation_dimension`` and
  ``ts.analysis.dimensions.correlation_dimension`` all resolve;
- the state-space regions — ``ts.Box`` is ``ts.data.Box``.  No call requires
  one: per-axis bounds reach every door that takes a region
  (``ts.basins_of_attraction(vdp, [(-3, 3), (-3, 3)])``);
- the derived wrappers — ``ts.PoincareMap`` is ``ts.derived.PoincareMap``.  Each
  has a verb on the system it wraps: ``sys.poincare("y", 0.0)``,
  ``sys.stroboscope(period)``, ``sys.tangent(k=2)``, ``sys.project(0, 2)``,
  ``sys.copies(states)``;
- the **machinery submodules** — :mod:`~tsdynamics.engine` (the Rust-facing
  compile/run seam), :mod:`~tsdynamics.solvers` (the solver registry),
  :mod:`~tsdynamics.registry` (the system/analysis/renderer registries),
  :mod:`~tsdynamics.families` (``SystemBase`` / ``ParamSet`` / the ``System``
  protocol), :mod:`~tsdynamics.utils`, :mod:`~tsdynamics.data` and
  :mod:`~tsdynamics.derived`.  ``ts.engine`` still resolves and
  ``from tsdynamics.engine import run`` still imports.

:mod:`~tsdynamics.viz` (and :func:`plot`, and the demoted ``ts.T``) resolves
**lazily** via ``__getattr__``, so a plain ``import tsdynamics`` pulls in no
plotting machinery.

Canonical homes for the data primitives (each has exactly one defining module;
the rest are convenience re-exports): :class:`Trajectory`, :class:`Box`,
:class:`Ball`, :class:`Grid` live in :mod:`tsdynamics.data`; :class:`WrappedSystem`
lives with the family bases in :mod:`tsdynamics.families`.
"""

from typing import Any

from . import (
    analysis,
    errors,
    systems,
)

# Machinery submodules: bound eagerly so ``ts.engine`` / ``ts.registry`` resolve
# and ``from tsdynamics.engine import run`` imports, but kept OFF ``__all__`` /
# ``dir()`` — see ``_INTERNAL_SUBMODULES`` below.  The redundant ``as`` form marks
# them as deliberate re-exports rather than unused imports.
from . import (
    data as data,
)
from . import (
    derived as derived,
)
from . import (
    engine as engine,
)
from . import (
    families as families,
)
from . import (
    registry as registry,
)
from . import (
    solvers as solvers,
)
from . import (
    utils as utils,
)
from .analysis import (
    Attractor as Attractor,
)
from .analysis import (
    AttractorSet as AttractorSet,
)
from .analysis import (
    BasinEntropy as BasinEntropy,
)
from .analysis import (
    BasinFractions as BasinFractions,
)
from .analysis import (
    BasinsResult as BasinsResult,
)
from .analysis import (
    ContinuationResult as ContinuationResult,
)
from .analysis import (
    DimensionResult as DimensionResult,
)
from .analysis import (
    EmbeddingDimension as EmbeddingDimension,
)
from .analysis import (
    ExpansionEntropyResult as ExpansionEntropyResult,
)
from .analysis import (
    FixedPoint as FixedPoint,
)
from .analysis import (
    FixedPointSet as FixedPointSet,
)
from .analysis import (
    GALIResult as GALIResult,
)
from .analysis import (
    LyapunovFromData as LyapunovFromData,
)
from .analysis import (
    LyapunovSpectrum as LyapunovSpectrum,
)
from .analysis import (
    OrbitDiagram as OrbitDiagram,
)
from .analysis import (
    OrbitSet as OrbitSet,
)
from .analysis import (
    PeriodicOrbit as PeriodicOrbit,
)
from .analysis import (
    PoincareSection as PoincareSection,
)
from .analysis import (
    RecurrenceMatrix as RecurrenceMatrix,
)
from .analysis import (
    ReturnMap as ReturnMap,
)
from .analysis import (
    RQAResult as RQAResult,
)
from .analysis import (
    UncertaintyExponent as UncertaintyExponent,
)
from .analysis import (
    WadaResult as WadaResult,
)
from .analysis import (
    WindowedRQA as WindowedRQA,
)
from .analysis import (
    autocorrelation as autocorrelation,
)
from .analysis import (
    basin_entropy as basin_entropy,
)
from .analysis import (
    basin_fractions as basin_fractions,
)
from .analysis import (
    basins_of_attraction as basins_of_attraction,
)
from .analysis import (
    box_counting_dimension as box_counting_dimension,
)
from .analysis import (
    cao_dimension as cao_dimension,
)
from .analysis import (
    continuation as continuation,
)
from .analysis import (
    correlation_dimension as correlation_dimension,
)
from .analysis import (
    correlation_sum as correlation_sum,
)
from .analysis import (
    dimension_spectrum as dimension_spectrum,
)
from .analysis import (
    embed as embed,
)
from .analysis import (
    embedding_dimension as embedding_dimension,
)
from .analysis import (
    estimate_period as estimate_period,
)
from .analysis import (
    expansion_entropy as expansion_entropy,
)
from .analysis import (
    false_nearest_neighbors as false_nearest_neighbors,
)
from .analysis import (
    find_attractors as find_attractors,
)
from .analysis import (
    fixed_mass_dimension as fixed_mass_dimension,
)
from .analysis import (
    fixed_points as fixed_points,
)
from .analysis import (
    gali as gali,
)
from .analysis import (
    generalized_dimension as generalized_dimension,
)
from .analysis import (
    information_dimension as information_dimension,
)
from .analysis import (
    kaplan_yorke_dimension as kaplan_yorke_dimension,
)
from .analysis import (
    lyapunov_from_data as lyapunov_from_data,
)
from .analysis import (
    lyapunov_spectrum as lyapunov_spectrum,
)
from .analysis import (
    max_lyapunov as max_lyapunov,
)
from .analysis import (
    mutual_information as mutual_information,
)
from .analysis import (
    optimal_delay as optimal_delay,
)
from .analysis import (
    orbit_diagram as orbit_diagram,
)
from .analysis import (
    periodic_orbit as periodic_orbit,
)
from .analysis import (
    periodic_orbits as periodic_orbits,
)
from .analysis import (
    poincare_section as poincare_section,
)
from .analysis import (
    recurrence_matrix as recurrence_matrix,
)
from .analysis import (
    resilience as resilience,
)
from .analysis import (
    return_map as return_map,
)
from .analysis import (
    rqa as rqa,
)
from .analysis import (
    tipping_points as tipping_points,
)
from .analysis import (
    uncertainty_exponent as uncertainty_exponent,
)
from .analysis import (
    wada_property as wada_property,
)
from .analysis import (
    windowed_rqa as windowed_rqa,
)
from .analysis import (
    zero_one_test as zero_one_test,
)
from .data import (
    Ball as Ball,
)
from .data import (
    Box as Box,
)
from .data import (
    Grid as Grid,
)
from .data import (
    grid_points as grid_points,
)
from .data import (
    sampler as sampler,
)
from .data import (
    set_distance as set_distance,
)

# The derived wrappers: demoted from ``__all__`` (each has a verb on the system
# it wraps — ``sys.poincare(...)`` / ``.stroboscope`` / ``.tangent`` /
# ``.copies`` / ``.project``), still bound so ``ts.PoincareMap`` resolves.  The
# redundant ``as`` form marks them as deliberate re-exports.
from .derived import (
    EnsembleSystem as EnsembleSystem,
)
from .derived import (
    PoincareMap as PoincareMap,
)
from .derived import (
    ProjectedSystem as ProjectedSystem,
)
from .derived import (
    StroboscopicMap as StroboscopicMap,
)
from .derived import (
    TangentSystem as TangentSystem,
)
from .families import (
    ContinuousSystem,
    DelaySystem,
    DiscreteMap,
    StochasticSystem,
    Trajectory,
    WrappedSystem,
)

# ``basins`` — a short alias of ``basins_of_attraction`` — used to be bound here
# and advertised in ``__all__``.  It is GONE, not demoted, because it broke the
# rule that a name must resolve to what it advertises: ``ts.basins`` was a
# function while ``ts.analysis.basins`` is the basins **subpackage**, so one
# word named two different objects in two namespaces one dot apart.  The docs
# never typed it (0 uses); ``basins_of_attraction`` is what they type, and it is
# still flat re-exported above.  ``ts.basins`` now answers with the redirect in
# :data:`_RENAMED_IN_V6`.
#
# ``bifurcation_diagram`` was DELETED in v6 (C3, one concept one spelling): it
# was the same object as ``orbit_diagram`` under a second name, so every shared
# error message named a function half its callers had never typed.  Guessing it
# is answered by ``_RENAMED_IN_V6``.

# Single source of truth for the package version; rewritten by python-semantic-release.
__version__ = "5.4.0"

# Built-in system classes are NOT bound into this namespace — that would bury the
# submodules (``analysis``, ``data``, ``systems``, …) under ~150 model names in
# ``dir()`` / autocomplete. The canonical path is ``tsdynamics.systems.<Name>``
# (e.g. ``tsdynamics.systems.Lorenz``). For backwards compatibility, ``tsd.Lorenz``
# and ``from tsdynamics import Lorenz`` still resolve, lazily, via ``__getattr__``
# below — but the names stay out of ``__all__`` and ``__dir__`` so they don't clutter
# the top-level surface.
_SYSTEM_NAMES = frozenset(systems._SYSTEM_NAMES)

# The two plotting names promoted to the curated top level (30 -> 32). They live
# in ``tsdynamics.viz.transforms`` and resolve lazily through ``__getattr__``, so
# advertising them costs nothing at import time:
#
#   ts.plot(traj)                                  # what ts.viz.plot does
#   ts.plot(traj, "delay_embedding", delay=7)      # ... plus a named transform
#   ts.plot(vdp, "flow_speed", "streamlines", "nullclines")   # ... several, one subject
#   ts.plot(vdp, ts.T("flow_speed", log=True), ts.T("streamlines", color="w"))
#
# Every name in an example here is a REGISTERED transform (``ts.viz.compatibility()``
# lists them).  It reads like a detail; it is not.  These four lines were the first
# thing a new user copied, and they used to name ``"basins"`` / ``"trajectory"`` —
# transforms that do not exist — so the documented one-liner answered with
# ``InvalidParameterError: unknown plot transform 'basins'``.
#
# Everything else in the plotting surface (``spec`` / ``geometry`` / ``draw`` /
# ``compatibility``) stays under ``ts.viz``, so the curated namespace does not
# drift back towards a flat dump.
_VIZ_FRONT_DOOR = frozenset({"plot", "T"})

#: Submodules that are bound eagerly (so ``ts.engine`` resolves and
#: ``from tsdynamics.engine import run`` imports) but kept **off** ``__all__`` /
#: ``dir()``.  Two reasons, both from the same rule — a tab slot is spent only on
#: something a user will reach for:
#:
#: * ``engine`` / ``solvers`` / ``registry`` / ``families`` / ``utils`` are
#:   machinery — the Rust-facing compile/run seam, the solver table, the system
#:   and plugin registries, the ``SystemBase``/``ParamSet`` internals, and the
#:   shared grid/tolerance constants;
#: * ``data`` / ``derived`` hold the state-space regions (``Box`` / ``Ball`` /
#:   ``Grid``) and the derived wrappers (``PoincareMap`` and friends).  They are
#:   **not** promoted, and the reason is the exact opposite of the one that used
#:   to be written here.  This comment used to say they were redundant *because
#:   everything a user needs from them is already bound on the top level* — which
#:   stopped being true the moment those names were demoted, and which was the
#:   wrong reasoning even while it was true.  The right reason is C1: **nothing
#:   in them is required to make a call.**  ``ts.basins_of_attraction(vdp,
#:   [(-3, 3), (-3, 3)])`` takes plain bounds, ``sys.poincare("y", 0.0)`` builds
#:   the section, ``sys.tangent(k=2)`` the tangent system.  A house whose
#:   contents no call demands does not earn a tab slot; promoting it would only
#:   move the toll up one level.
#:
#: This tuple is the single source of truth for that decision — the namespace
#: gate (``tests/test_namespace_curation.py``) reads it, so demoting or promoting
#: a submodule is a one-line edit here.
_INTERNAL_SUBMODULES = (
    "data",
    "derived",
    "engine",
    "families",
    "registry",
    "solvers",
    "utils",
)

# ---------------------------------------------------------------------------
# The curated top level — the membership rule, then the twelve names
# ---------------------------------------------------------------------------
#
# THE RULE.  A name earns a top-level slot only if a user TYPES it in ordinary
# work.  And no user should ever have to construct a library type to make a
# call — so a name that is here *because a signature demands it* is evidence of
# a signature bug, not of a needed export.
#
# Four corollaries, applied mechanically, so the next person to add a name has
# to argue against a written rule rather than against a list:
#
#   C1  THE TOLL RULE.  Plain Python — tuples, lists, strings, numbers, arrays —
#       reaches every front door.  Fix the signature, then demote the name;
#       never the reverse.  (``ts.basins(vdp, [(-3, 3), (-3, 3)])`` already
#       worked, which is why ``Box`` / ``Ball`` / ``Grid`` never needed to be
#       here.  See the note on ``_INTERNAL_SUBMODULES``.)
#   C2  RECEIVED IS NOT TYPED.  A type you get *back* is not a type you type.
#       It lives at its real home — unless it is the return value of the
#       library's single most common call and users annotate it, which admits
#       exactly one name: ``Trajectory``.
#   C3  ONE CONCEPT, ONE SPELLING.  Two grammars for the same argument is not
#       flexibility; it is the silent-wrong-answer defect.
#   C4  A NAME MUST RESOLVE TO WHAT IT ADVERTISES.  No ``__all__`` entry may be
#       shadowed by a submodule of the same name (``ts.basins`` the function
#       against ``ts.analysis.basins`` the package).
#
# Everything demoted stays FULLY REACHABLE — ``ts.correlation_dimension``,
# ``from tsdynamics import correlation_dimension`` and
# ``ts.analysis.dimensions.correlation_dimension`` all resolve; ``ts.Box`` is
# ``ts.data.Box``; ``ts.PoincareMap`` is ``ts.derived.PoincareMap``.  Demotion
# is never removal.
__all__ = [
    "__version__",
    # ── The family bases: what you SUBCLASS to define a system. ──
    # Each is the sole spelling of a capability (there is no verb, and no
    # plain-Python path, that defines a delay system for you), which is the
    # "there is no other way" test rather than "someone might want it".
    "ContinuousSystem",
    "DelaySystem",
    "DiscreteMap",
    "StochasticSystem",
    "WrappedSystem",
    # ── The one received type that is also a typed one (C2). ──
    # Every run returns it, users annotate it and ``isinstance`` it, and since
    # v6 they CONSTRUCT it from measured data (``ts.Trajectory(t, y)``).
    "Trajectory",
    # ── Plotting: the front door. ──
    # ``ts.plot(anything)``.  Resolves lazily (see ``__getattr__``) so naming it
    # here still costs ``import tsdynamics`` no plotting machinery.  ``T`` is
    # demoted: a plain ``("name", {options})`` pair is a transform call now.
    "plot",
    # ── The four submodules worth typing a dot after. ──
    # The derived wrappers, the state-space regions and the analysis functions
    # all live behind one of these; none of them is required to make a call.
    "systems",
    "analysis",
    "viz",
    "errors",
]


#: Names the v6 scope surgery removed outright, mapped to the reason.  Curating
#: the namespace hides ~230 reachable names from autocomplete, so this
#: ``AttributeError`` becomes the *only* feedback a user gets when they guess a
#: spelling — it has to teach, not just refuse.  A v5 user typing
#: ``ts.permutation_entropy`` should learn the scope changed, not read "no
#: attribute" and assume the install is broken.
_REMOVED_IN_V6 = {
    "entropy": "entropy estimators",
    "permutation_entropy": "entropy estimators",
    "dispersion_entropy": "entropy estimators",
    "sample_entropy": "entropy estimators",
    "multiscale_entropy": "entropy estimators",
    "lz76_complexity": "entropy estimators",
    "surrogates": "surrogate-data tests",
    "surrogate_test": "surrogate-data tests",
    "SurrogateTest": "surrogate-data tests",
    "time_reversal_asymmetry": "surrogate-data tests",
    "nonlinear_prediction_error": "surrogate-data tests",
    "transforms": "signal transforms (PSD, detrend, filters, feature extraction)",
    "power_spectrum": "signal transforms (PSD, detrend, filters, feature extraction)",
    "detrend": "signal transforms (PSD, detrend, filters, feature extraction)",
    "extract_features": "signal transforms (PSD, detrend, filters, feature extraction)",
}


#: Names the v6 namespace curation removed **in favour of another spelling that
#: still exists**, mapped to the line to type instead.  Distinct from
#: :data:`_REMOVED_IN_V6`, where the capability itself left the library: here
#: nothing was lost, so the answer is a redirect, not an explanation.
#:
#: ``basins`` is the whole table today.  It broke C4 — ``ts.basins`` was a
#: function while ``ts.analysis.basins`` is the subpackage — and the docs never
#: typed it.
_RENAMED_IN_V6 = {
    "bifurcation_diagram": (
        'ts.analysis.orbit_diagram(system, "r", values)',
        "it was a second name for orbit_diagram, and a shared implementation can "
        "name only one of its spellings in an error — so half of all callers were "
        "sent to look up a function they had never typed",
    ),
    "basins": (
        "ts.basins_of_attraction(system, region)",
        "the short alias collided with the ts.analysis.basins subpackage, so one "
        "word named two different objects one dot apart",
    ),
}


#: The public homes a curated top level sends people to.  Demotion only works if
#: guessing the short name teaches the qualified one, so these are searched — by
#: exact name first — before any fuzzy match.
_PUBLIC_HOMES = ("data", "derived", "analysis", "viz")


def _home_of(name: str) -> str | None:
    """Return ``"ts.<home>.<name>"`` if ``name`` is public in one of the homes.

    Curation hides ~230 reachable names from autocomplete, so an
    ``AttributeError`` is the only feedback a user gets when they guess.  Most
    demoted names are also bound here and never reach this path; the ones that
    are **not** (``ts.data.Region`` / ``ts.data.region``, everything in
    ``ts.viz``, the result base classes) used to fall through to a fuzzy match
    that answered a real question with a wrong object — ``ts.region`` suggested
    ``ts.systems.Oregonator()``.  An exact hit in a public ``__all__`` is a
    certainty; it must outrank any guess.
    """
    import importlib

    if name.startswith("__") and name.endswith("__"):
        # A protocol probe (``__wrapped__``, ``__path__``, a notebook canary) is
        # not a user typing a name; answering it must not import anything.
        return None
    for home in _PUBLIC_HOMES:
        try:
            module = importlib.import_module(f"{__name__}.{home}")
        except ImportError:  # pragma: no cover - an optional home is still absent
            continue
        if name in getattr(module, "__all__", ()):
            return f"ts.{home}.{name}"
    return None


def _attribute_error(name: str) -> AttributeError:
    """Build an ``AttributeError`` that names the line to type, not the mistake.

    Five cases, in the order a user is likely to hit them: a name the v6 scope
    surgery removed, a name v6 *renamed* (the capability is still here), a name
    that is simply **at a different address** (public in ``ts.data`` /
    ``ts.derived`` / ``ts.analysis`` / ``ts.viz`` — an exact hit, so it outranks
    every guess below), a near-miss on something still here (a built-in system,
    which autocomplete deliberately hides, or a demoted analysis), and a genuine
    miss — answered with the two listings worth tab-completing.
    """
    import difflib

    if name in _RENAMED_IN_V6:
        line, why = _RENAMED_IN_V6[name]
        return AttributeError(
            f"tsdynamics has no attribute {name!r}: it was renamed in v6 ({why}).\n"
            f"Same function, one spelling:\n    {line}"
        )

    if name in _REMOVED_IN_V6:
        return AttributeError(
            f"tsdynamics has no attribute {name!r}: the generic time-series layer "
            f"({_REMOVED_IN_V6[name]}) was removed in v6.\n"
            "TSDynamics is scoped to phase-space methods now. What stayed, and the "
            "closest thing to reach for:\n"
            "    ts.analysis.recurrence   # recurrence plots / RQA\n"
            "    ts.analysis.embedding    # delay embedding (data -> phase space)\n"
            "    ts.analysis.lyapunov     # lyapunov_from_data"
        )

    qualified = _home_of(name)
    if qualified is not None:
        return AttributeError(
            f"module 'tsdynamics' has no attribute {name!r}: the top level is curated, "
            f"and this one lives at its own address.\n    {qualified}"
        )

    system_hit = difflib.get_close_matches(name, sorted(_SYSTEM_NAMES), n=1, cutoff=0.6)
    if system_hit:
        return AttributeError(
            f"module 'tsdynamics' has no attribute {name!r}. Built-in systems live "
            f"under ts.systems — you probably want:\n    ts.systems.{system_hit[0]}()"
        )

    reachable = sorted(set(__all__) | set(globals()) | set(getattr(analysis, "__all__", [])))
    hits = difflib.get_close_matches(name, reachable, n=3, cutoff=0.6)
    if hits:
        lines = "\n".join(f"    ts.{h}" for h in hits)
        return AttributeError(
            f"module 'tsdynamics' has no attribute {name!r}. Did you mean:\n{lines}"
        )

    # Counted live: a hardcoded total goes stale every time a system is added
    # (the docs already say 171 where the registry says 177).
    return AttributeError(
        f"module 'tsdynamics' has no attribute {name!r}.\n"
        "Tab-complete the two catalogues to find it:\n"
        f"    ts.systems.<TAB>    # the {len(_SYSTEM_NAMES)} built-in systems\n"
        "    ts.analysis.<TAB>   # the quantifiers, by capability"
    )


def __getattr__(name: str) -> Any:
    """Lazily resolve the ``viz`` submodule and built-in system classes.

    Two lazy resolutions live here:

    * ``tsdynamics.viz`` — imported on first access (and cached) so a plain
      ``import tsdynamics`` pulls in no plotting/IR machinery; ``viz`` still shows
      in ``__all__`` / ``dir()`` for discoverability.
    * ``tsdynamics.plot`` / ``tsdynamics.T`` — the plotting front door, resolved
      from :mod:`tsdynamics.viz.transforms` on first access (and cached), for the
      same reason: naming them in ``__all__`` must not make ``import tsdynamics``
      import a plotting layer.
    * ``tsdynamics.Lorenz`` and friends — the ~150 built-in system classes,
      resolved from :mod:`tsdynamics.systems` (the canonical path) instead of
      binding all of them into the namespace, which keeps ``dir()`` / autocomplete
      focused on the curated public API.
    """
    if name == "viz":
        import importlib

        # import_module loads the submodule through the import machinery without
        # re-entering this __getattr__ (a plain ``from . import viz`` would recurse).
        _viz = importlib.import_module(f"{__name__}.viz")
        globals()["viz"] = _viz  # cache: subsequent access skips __getattr__
        return _viz
    if name in _VIZ_FRONT_DOOR:
        import importlib

        _transforms = importlib.import_module(f"{__name__}.viz.transforms")
        for attr in _VIZ_FRONT_DOOR:
            globals()[attr] = getattr(_transforms, attr)  # cache both at once
        return globals()[name]
    if name in _SYSTEM_NAMES:
        return getattr(systems, name)
    raise _attribute_error(name)


def __dir__() -> list[str]:
    """Top-level surface = the public API in ``__all__`` (models live under ``systems``).

    Machinery submodules (:data:`_INTERNAL_SUBMODULES`) and the ~150 built-in
    system classes stay resolvable but out of this listing, so ``ts.<TAB>`` is the
    mental model rather than a dump.
    """
    return sorted(__all__)
