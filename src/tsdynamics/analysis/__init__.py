"""The quantifiers.  This docstring is REPLACED at import — see ``_build_doc``.

The flat re-exports below use the redundant-alias form (``x as x``) because
``__all__`` here is **generated** from the registry at the bottom of the file:
a static checker cannot see it, and every one of these names is deliberately
bound.
"""

from __future__ import annotations

from typing import Any

from .. import registry as _registry
from ..plugins import ANALYSES_GROUP, register_entry_points
from . import _discovery

# The capability subpackages self-register their analyses at import.  They are
# DELETED from this module's dict at the bottom of the file (CONTRACT §5.4): a
# submodule of a public package must not answer a plausible verb guess with
# ``TypeError: 'module' object is not callable``.  They stay importable through
# ``sys.modules`` — ``import tsdynamics.analysis.lyapunov`` and
# ``from tsdynamics.analysis.lyapunov import lyapunov_spectrum`` both work.
from . import basins as basins
from . import chaos as chaos
from . import dimensions as dimensions
from . import embedding as embedding
from . import fixedpoints as fixedpoints
from . import lyapunov as lyapunov
from . import orbits as orbits
from . import planar as planar
from . import recurrence as recurrence

# The 32 result classes live at ``ts.analysis.results`` (C2: a type you only ever
# get *back* has one importable address and appears in no ``__all__``).  They
# stay bound here as module attributes; only the tab surface shrinks.
from . import results as results
from . import sampling as sampling
from ._discovery import AnalysisList as AnalysisList
from ._discovery import register as register
from ._public import AREA_SUBPACKAGES, PLANAR_ANALYSES, VERBS
from ._result import AnalysisResult as AnalysisResult
from ._result import ArrayResult as ArrayResult
from ._result import CollectionResult as CollectionResult
from ._result import CountResult as CountResult
from ._result import ScalarResult as ScalarResult
from ._result import ScalingResult as ScalingResult
from ._result import VisualizationNotInstalled as VisualizationNotInstalled
from .basins import Attractor as Attractor
from .basins import AttractorSet as AttractorSet
from .basins import BasinEntropy as BasinEntropy
from .basins import BasinFractions as BasinFractions
from .basins import BasinsResult as BasinsResult
from .basins import ContinuationResult as ContinuationResult
from .basins import UncertaintyExponent as UncertaintyExponent
from .basins import WadaResult as WadaResult
from .basins import attractors as attractors
from .basins import basin_entropy as basin_entropy
from .basins import basin_fractions as basin_fractions
from .basins import basins as _basins_fn
from .basins import continuation as continuation
from .basins import resilience as resilience
from .basins import tipping_points as tipping_points
from .basins import uncertainty_exponent as uncertainty_exponent
from .basins import wada_property as wada_property
from .chaos import ExpansionEntropyResult as ExpansionEntropyResult
from .chaos import GALIResult as GALIResult
from .chaos import expansion_entropy as expansion_entropy
from .chaos import gali as gali
from .chaos import zero_one_test as zero_one_test
from .dimensions import DimensionResult as DimensionResult
from .dimensions import box_counting_dimension as box_counting_dimension
from .dimensions import correlation_dimension as correlation_dimension
from .dimensions import correlation_sum as correlation_sum
from .dimensions import dimension_spectrum as dimension_spectrum
from .dimensions import fixed_mass_dimension as fixed_mass_dimension
from .dimensions import generalized_dimension as generalized_dimension
from .dimensions import information_dimension as information_dimension
from .embedding import Embedding as Embedding
from .embedding import EmbeddingDimension as EmbeddingDimension
from .embedding import autocorrelation as autocorrelation
from .embedding import cao_dimension as cao_dimension
from .embedding import embed as embed
from .embedding import embedding_dimension as embedding_dimension
from .embedding import false_nearest_neighbors as false_nearest_neighbors
from .embedding import mutual_information as mutual_information
from .embedding import optimal_delay as optimal_delay
from .fixedpoints import FixedPoint as FixedPoint
from .fixedpoints import FixedPointSet as FixedPointSet
from .fixedpoints import OrbitSet as OrbitSet
from .fixedpoints import PeriodicOrbit as PeriodicOrbit
from .fixedpoints import estimate_period as estimate_period
from .fixedpoints import fixed_points as fixed_points
from .fixedpoints import periodic_orbits as periodic_orbits
from .lyapunov import LyapunovFromData as LyapunovFromData
from .lyapunov import LyapunovSpectrum as LyapunovSpectrum
from .lyapunov import kaplan_yorke_dimension as kaplan_yorke_dimension
from .lyapunov import lyapunov_from_data as lyapunov_from_data
from .lyapunov import lyapunov_spectrum as lyapunov_spectrum
from .lyapunov import max_lyapunov as max_lyapunov
from .orbits import OrbitDiagram as OrbitDiagram
from .orbits import PoincareSection as PoincareSection
from .orbits import ReturnMap as ReturnMap
from .orbits import orbit_diagram as orbit_diagram
from .orbits import poincare_section as poincare_section
from .orbits import return_map as return_map
from .planar import escape_time_field as escape_time_field
from .planar import flow_field as flow_field
from .planar import ftle_field as ftle_field
from .planar import invariant_density as invariant_density
from .planar import nullclines as nullclines
from .planar import streamlines as streamlines
from .planar import trace_determinant as trace_determinant
from .planar import transient_time_field as transient_time_field
from .recurrence import RecurrenceMatrix as RecurrenceMatrix
from .recurrence import RQAResult as RQAResult
from .recurrence import WindowedRQA as WindowedRQA
from .recurrence import recurrence_matrix as recurrence_matrix
from .recurrence import rqa as rqa
from .recurrence import windowed_rqa as windowed_rqa
from .sampling import estimate_dt_from_sagitta as estimate_dt_from_sagitta
from .sampling import sagitta_profile as sagitta_profile

# The remaining result classes reach this namespace only through ``results``
# (``MutualInformation`` and ``ZeroOneResult`` were returned by exported
# functions and were in NO ``__all__`` before v6).  Bind every one of the 32, so
# "reachable, just not on the tab surface" is true for all of them, not most.
for _cls in results.__all__:
    globals().setdefault(_cls, getattr(results, _cls))
del _cls

#: ``basins`` the ANALYSIS wins the name over ``basins`` the subpackage — that is
#: what §5.4's unbinding is for.  The import above is aliased because the
#: subpackage has to import (and self-register) before the function can shadow it.
globals()["basins"] = _basins_fn

# ---------------------------------------------------------------------------
# The analyses whose definition site is not a subpackage that can self-register.
# ---------------------------------------------------------------------------

# The eight planar / field analyses.  Before v6 they had no docs-nav page and no
# API-reference entry, so a user who wanted FTLE *numbers* (rather than a
# picture) had no door at all.
for _name, _subjects, _area, _keywords in PLANAR_ANALYSES:
    register(
        getattr(planar, _name), subjects=_subjects, area=_area, keywords=_keywords, replace=True
    )
del _name, _subjects, _area, _keywords

# ``set_distance`` is state-space geometry (``tsdynamics.data``), and it is the
# measurement ``continuation`` matches attractors with, so it belongs on the
# quantifier surface as well as at its own address.
from ..data.sampling import set_distance  # noqa: E402

register(
    set_distance,
    subjects=("trajectory", "array"),
    area="geometry",
    keywords="hausdorff distance point sets matching attractors",
    replace=True,
)


#: Computed by :func:`_refresh_surface` from the registry — never hand-written.
__all__: list[str] = []


def discover_plugins(*, strict: bool = False) -> list[str]:
    """Load out-of-tree analysis plugins into :data:`tsdynamics.registry.analyses`.

    Walks the ``tsdynamics.analyses`` entry-point group and registers each loaded
    object under its entry-point name (see
    :func:`tsdynamics.plugins.register_entry_points`).  Called once at import;
    safe to re-invoke after installing a plugin.

    Parameters
    ----------
    strict : bool, default False
        Re-raise the first plugin load failure instead of warning and skipping.

    Returns
    -------
    list[str]
        The names newly registered by this call.
    """
    loaded = register_entry_points(_registry.analyses, ANALYSES_GROUP, strict=strict)
    if loaded:
        # A plugin that lands in ``__all__`` with no module attribute makes
        # ``from tsdynamics.analysis import *`` raise — measured, and for exactly
        # the audience the registry exists to serve.  Bind it and rebuild.
        _refresh_surface()
    return loaded


def _refresh_surface() -> None:
    """Recompute ``__all__`` and ``__doc__`` from the registry.

    ``__all__`` is **generated** here, never hand-written: the four hand lists this
    replaces were already measurably wrong (``ts.LyapunovSpectrum`` resolved and
    ``ts.Embedding`` did not).  A registered analysis therefore cannot be missing
    from the tab surface, and a listed name cannot be missing from the registry.
    """
    global __all__, __doc__
    for entry in _registry.analyses.all():
        globals().setdefault(entry.name, entry.obj)
    __all__ = sorted([*_registry.analyses.names(), *VERBS])
    __doc__ = _build_doc()


def find(what: Any = None, /) -> AnalysisList:
    """Return the analyses that answer this question, or take this subject.

    ``what`` is a **string** (free-text search over the name, area, keywords and
    summary), or a **subject** — a system, a :class:`~tsdynamics.data.Trajectory`,
    an array, a result, or any of their classes — or omitted, for all of them.
    Returns a plain ``list`` of the analysis *functions*, whose repr is the
    grouped table.

    Examples
    --------
    >>> import tsdynamics as ts
    >>> sorted(f.__name__ for f in ts.analysis.find("multistability"))[:3]
    ['attractors', 'basin_entropy', 'basin_fractions']
    >>> len(ts.analysis.find(ts.systems.Henon()))    # a map has no vector field
    14
    >>> len(ts.analysis.find(ts.systems.Lorenz()))
    21
    """
    entries = _registry.analyses.all()
    if what is None:
        return AnalysisList(entries, f"{len(entries)} analyses")
    if isinstance(what, str):
        hits = _discovery.search(entries, what)
        if not hits:
            return AnalysisList(
                (),
                f"nothing matches {what!r}. ts.analysis.find() lists all {len(entries)}.",
            )
        return AnalysisList(hits, f"{len(hits)} analyses match {what!r}")
    tokens = _discovery.subject_tokens(what)
    if tokens is None:
        from tsdynamics.errors import InvalidInputError

        raise InvalidInputError(
            f"find() takes a question or a subject, and {type(what).__name__} is "
            f"neither.\n    ts.analysis.find('chaotic')   # a question\n"
            f"    ts.analysis.find(traj)        # a subject\n"
            f"    ts.analysis.find()            # everything"
        )
    hits = [e for e in entries if set(tokens) & set(e.metadata.get("subjects", ()))]
    subject = what.__name__ if isinstance(what, type) else type(what).__name__
    return AnalysisList(hits, f"{len(hits)} analyses take a {subject}")


def _build_doc() -> str:
    """Render this namespace's docstring — the grouped map — from the registry.

    Grouped by **what you are holding**, because that is a question a user can
    answer about themselves without knowing our vocabulary — and the same
    question the wrong-subject errors answer.  A flat sort cannot answer "is this
    chaotic?": the five analyses that do sit at scattered alphabetical positions
    and one of them (``zero_one_test``) contains no word a newcomer would search.
    """
    entries = _registry.analyses.all()
    return f"""The quantifiers.  Every analysis is a free function whose FIRST argument is
the thing it is about — a system, a trajectory, or a result you already have::

    ts.analysis.lyapunov_spectrum(lorenz)          # a property of the equations
    ts.analysis.correlation_dimension(traj)        # a property of a point set
    ts.analysis.kaplan_yorke_dimension(spectrum)   # a property of the answer above

Two ways in::

    ts.analysis.find(traj)          # what can I measure on THIS?
    ts.analysis.find("chaotic")     # who answers THIS question?

Names are flat and sorted (``ts.analysis.<TAB>``); the map below is the same
{len(entries)} analyses grouped by what you have to hold to call them.

{_discovery.grouped_map(entries)}
"""


#: Names this namespace used to export, and the one spelling that replaced each.
#: Guessing a removed name is how a user discovers the rename, so it must answer
#: with the replacement rather than a bare ``AttributeError``.
_RENAMED_IN_V6: dict[str, tuple[str, str]] = {
    "basins_of_attraction": (
        "ts.analysis.basins(system, region)",
        "basins is the noun a user types, and the collision with the "
        "implementation package is gone now that the package is unbound",
    ),
    "bifurcation": (
        'ts.analysis.orbit_diagram(system, "r", values)',
        "a bifurcation diagram is an orbit diagram of a flow's discrete view",
    ),
    "bifurcation_diagram": (
        'ts.analysis.orbit_diagram(system, "r", values)',
        "it was a second name for orbit_diagram, and a shared implementation can "
        "name only one of its spellings in an error",
    ),
    "find_attractors": (
        "ts.analysis.attractors(system, region)",
        "the noun a user types; every other analysis here is named for what it "
        "returns, not for the act of looking",
    ),
    "periodic_orbit": (
        "ts.analysis.periodic_orbits(system, period_guess, ic=x0)",
        "one verb, one return type: a flow's limit cycle comes back as an "
        "OrbitSet of one, exactly like a map's orbits",
    ),
}

#: Which registry ``area`` each implementation subpackage holds.  ``planar`` is
#: the one whose directory name and area word differ.
_AREA_OF: dict[str, str] = {name: name for name in AREA_SUBPACKAGES}
_AREA_OF["planar"] = "fields"


def __getattr__(name: str) -> Any:
    """Answer a name that is not here — a rename, a subpackage, or a guess.

    Three ordered cases, split by *kind of hit*.  An exact hit in a redirect
    table raises :class:`~tsdynamics.errors.MovedInV6`, an ``ImportError``,
    because a module ``__getattr__`` that raises ``AttributeError`` has its
    message **discarded** by ``from tsdynamics.analysis import X``.  A guess
    stays an ``AttributeError``, so ``hasattr`` keeps answering ``False`` for
    every name in the universe except the enumerated dead ones.
    """
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)

    entry = _RENAMED_IN_V6.get(name)
    if entry is not None:
        from tsdynamics.errors import MovedInV6

        line, why = entry
        raise MovedInV6(
            f"tsdynamics.analysis has no attribute {name!r}: it was renamed in v6 "
            f"({why}).\nSame capability, one spelling:\n    {line}"
        )

    if name in AREA_SUBPACKAGES:
        # An ``AttributeError``, deliberately, and NOT ``MovedInV6``: the import
        # machinery falls back to ``sys.modules`` for a submodule only when the
        # parent's ``__getattr__`` raises ``AttributeError``, and §5.4 keeps
        # ``import tsdynamics.analysis.lyapunov`` working.  Nothing moved — the
        # package is exactly where it was; it just stopped answering a verb guess
        # with ``TypeError: 'module' object is not callable``.
        area = _AREA_OF[name]
        members = sorted(e.name for e in _registry.analyses.all() if e.metadata.get("area") == area)
        shown = "\n".join(f"    ts.analysis.{m}(...)" for m in members[:3])
        raise AttributeError(
            f"{name!r} is the implementation package, not a verb. The analyses in "
            f"it are free functions:\n{shown}\n"
            f'    ts.analysis.find("{area}")   # all {len(members)}'
        )

    close = _discovery.near_miss(name, _registry.analyses.names())
    hint = f" Did you mean {close!r}?" if close else ""
    raise AttributeError(
        f"module 'tsdynamics.analysis' has no attribute {name!r}.{hint}\n"
        f"    ts.analysis.find({name!r})   # search by what you want to measure"
    )


def __dir__() -> list[str]:
    """``dir()`` mirrors ``__all__`` — every analysis, plus ``find``/``register``/``results``.

    This is the **leaf** namespace: its entire job is to enumerate the
    quantifiers, so it enumerates them (``numpy.linalg`` does the same).  The 32
    result classes are not here — a type you only ever get *back* lives at one
    address (``ts.analysis.results``) and appears in no ``__all__``.
    """
    return sorted(__all__)


# Populate the registry from out-of-tree plugins, then compute the surface.
discover_plugins()
_refresh_surface()

# ---------------------------------------------------------------------------
# Unbind the implementation subpackages (CONTRACT §5.4).  MUST be last: any
# ``from .x import y`` below would re-bind ``x`` as a side effect of the import
# machinery.
# ---------------------------------------------------------------------------
for _sub in AREA_SUBPACKAGES:
    globals().pop(_sub, None)
del _sub
globals()["basins"] = _basins_fn  # ...and `basins` the ANALYSIS keeps the name.
del _basins_fn
