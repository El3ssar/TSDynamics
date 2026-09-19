"""``ts.analysis.results`` — the result types, behind one dot.

Every registered analysis returns an :class:`AnalysisResult` subclass.  There are
32 of them, and a user **constructs none**: they are what comes *back*, so under
the v6 rule (a name earns a top-level slot only if a user types it in ordinary
work) they do not belong on the flat ``ts.analysis`` tab surface, where they used
to be 32 of 84 names — 38 % of the listing, none of it an analysis.

They are still ordinary, importable, ``isinstance``-able classes.  Everything
below keeps working, and nothing about the objects themselves changed:

.. code-block:: python

    import tsdynamics as ts

    spectrum = ts.analysis.lyapunov_spectrum(ts.systems.Lorenz())
    isinstance(spectrum, ts.analysis.results.LyapunovSpectrum)     # True

    from tsdynamics.analysis.results import DimensionResult        # works
    from tsdynamics.analysis.lyapunov import LyapunovSpectrum      # still works

This module is a *namespace*, not a home: each class still lives in the
subpackage that produces it, and is re-exported here.  ``dir()`` mirrors
``__all__``, so the listing is exactly the result hierarchy, sorted.

The hierarchy
-------------
- :class:`AnalysisResult` — the base every result shares: ``.meta``, the repr
  that *is* the answer, ``.to_dict()``, ``.to_frame()``, ``.plot``.
- :class:`ScalarResult` / :class:`CountResult` — one number, and a complete
  drop-in for it.
- :class:`ArrayResult` — one array, and a complete drop-in for it.
- :class:`CollectionResult` — a sequence of sub-results, indexed by position.
- :class:`ScalingResult` — a quantity read off the slope of a scaling curve,
  carrying the curve.

Everything else is one of those five with domain names on it.
"""

from __future__ import annotations

from tsdynamics.analysis._result import (
    AnalysisResult,
    ArrayResult,
    CollectionResult,
    CountResult,
    ScalarResult,
    ScalingResult,
)
from tsdynamics.analysis.basins.attractors import Attractor, AttractorSet
from tsdynamics.analysis.basins.basins import BasinFractions, BasinsResult
from tsdynamics.analysis.basins.continuation import ContinuationResult
from tsdynamics.analysis.basins.metrics import BasinEntropy, UncertaintyExponent, WadaResult
from tsdynamics.analysis.chaos.expansion import ExpansionEntropyResult
from tsdynamics.analysis.chaos.gali import GALIResult
from tsdynamics.analysis.chaos.zero_one import ZeroOneResult
from tsdynamics.analysis.dimensions._common import DimensionResult
from tsdynamics.analysis.embedding.delay import MutualInformation
from tsdynamics.analysis.embedding.dimension import EmbeddingDimension
from tsdynamics.analysis.embedding.embed import Embedding
from tsdynamics.analysis.fixedpoints.fixed import FixedPoint, FixedPointSet
from tsdynamics.analysis.fixedpoints.periodic import OrbitSet, PeriodicOrbit
from tsdynamics.analysis.lyapunov import LyapunovSpectrum
from tsdynamics.analysis.lyapunov.from_data import LyapunovFromData
from tsdynamics.analysis.orbits.orbit_diagram import OrbitDiagram
from tsdynamics.analysis.orbits.return_map import ReturnMap
from tsdynamics.analysis.recurrence.matrix import RecurrenceMatrix
from tsdynamics.analysis.recurrence.rqa import RQAResult
from tsdynamics.analysis.recurrence.windowed import WindowedRQA

__all__ = [
    "AnalysisResult",
    "ArrayResult",
    "Attractor",
    "AttractorSet",
    "BasinEntropy",
    "BasinFractions",
    "BasinsResult",
    "CollectionResult",
    "ContinuationResult",
    "CountResult",
    "DimensionResult",
    "Embedding",
    "EmbeddingDimension",
    "ExpansionEntropyResult",
    "FixedPoint",
    "FixedPointSet",
    "GALIResult",
    "LyapunovFromData",
    "LyapunovSpectrum",
    "MutualInformation",
    "OrbitDiagram",
    "OrbitSet",
    "PeriodicOrbit",
    "RQAResult",
    "RecurrenceMatrix",
    "ReturnMap",
    "ScalarResult",
    "ScalingResult",
    "UncertaintyExponent",
    "WadaResult",
    "WindowedRQA",
    "ZeroOneResult",
]


def __dir__() -> list[str]:
    """``dir()`` mirrors ``__all__`` — exactly the 32 result classes, sorted."""
    return sorted(__all__)
