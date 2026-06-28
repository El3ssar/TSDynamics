"""Shared base class for the analysis layer's result objects.

Every analysis in TSDynamics returns a *self-describing result object* rather
than a bare ``float``/``ndarray``/``list``.  :class:`AnalysisResult` is the one
base those results share.  It is purely additive — it adds a uniform surface
without changing any existing return value — so the analysis subpackages can be
reparented onto it one at a time.

The surface every result inherits
---------------------------------
- ``meta`` — a provenance mapping (system, params, version, run settings),
  built from :meth:`tsdynamics.families.base.SystemBase._provenance` at the
  call site via :meth:`AnalysisResult.build_meta`.
- ``__repr__`` — a compact, ``_repr_fields``-driven one-liner.
- ``_repr_html_`` — a small table for Jupyter / IPython.
- :meth:`summary` — a human-readable multi-line readout plus an optional
  interpretation line (subclasses supply the interpretation).
- :meth:`to_dict` — a stdlib, JSON-friendly mapping (arrays become lists).
- :meth:`to_frame` — a :mod:`pandas` ``DataFrame`` (``pandas`` is a soft
  dependency, imported lazily, with an install hint if it is missing).
- ``plot`` — the visualization seam: an accessor that is both callable
  (``result.plot()``) and a namespace of typed kind methods
  (``result.plot.scaling()``).  The in-tree backends seed themselves on first
  use, so it renders out of the box with a plotting library installed; with none
  the seam raises :class:`VisualizationNotInstalled`.

Subclassing contract
--------------------
:class:`AnalysisResult` is a *frozen* dataclass.  A subclass must re-apply the
decorator and stay frozen::

    @dataclass(frozen=True)
    class LyapunovSpectrum(AnalysisResult):
        _repr_fields = ("exponents", "kaplan_yorke")
        exponents: np.ndarray = field(repr=False, compare=False)
        kaplan_yorke: float

        def _interpretation(self) -> str:
            n_pos = int((self.exponents > 0).sum())
            return "chaotic" if n_pos else "regular"

``meta`` is keyword-only on the base, so subclasses are free to declare their own
positional fields without tripping the "non-default argument follows default
argument" rule.

Array-valued fields **must** be declared ``field(compare=False)``.  A frozen
dataclass derives ``__eq__`` / ``__hash__`` from a tuple of its fields, and a
NumPy array is neither boolean-comparable (``arr == arr`` is an array, so a
generated ``__eq__`` would raise ``ValueError``) nor hashable.  Keeping arrays
out of equality and hashing lets two results compare on their scalar summary
fields; ``meta`` (provenance) is already excluded the same way on the base.

The scaling-curve family
------------------------
A large family of estimators reads a single number off the slope of a log--log
(or semi-log) scaling curve: every fractal dimension, the Lyapunov exponent from
a measured time series, expansion entropy, and the Cao / false-nearest-neighbour
embedding-dimension diagnostics.  :class:`ScalingResult` gives that whole family
**one** canonical schema — ``estimate`` / ``stderr`` / ``abscissa`` /
``ordinate`` / ``fit_region`` / ``intercept`` (plus the ``local_slopes`` and
``scaling_window`` diagnostics) — so a single ``result.plot.scaling()`` renders
any of them.  It is additive: existing results are *reparented* onto it by a
later stream, not changed here.

Module layout (de-dup split)
----------------------------
This module is the **re-exporting facade** for the result hierarchy.  The classes
and helpers live in sibling private modules so the hierarchy is navigable and the
shared pieces are de-duplicated:

- :mod:`tsdynamics.analysis._result_base` — :class:`AnalysisResult`.
- :mod:`tsdynamics.analysis._result_scaling` — :class:`ScalingResult`.
- :mod:`tsdynamics.analysis._result_scalar` — :class:`ScalarResult` /
  :class:`CountResult` (+ the ``_NumericOps`` mixin).
- :mod:`tsdynamics.analysis._result_array` — :class:`ArrayResult`.
- :mod:`tsdynamics.analysis._result_collection` — :class:`CollectionResult`.
- :mod:`tsdynamics.analysis._result_viz` — the ``.plot`` seam
  (:class:`VisualizationNotInstalled`, ``_PlotAccessor``).
- :mod:`tsdynamics.analysis._result_json` — the ``to_dict`` / repr helpers
  (``_jsonify`` & co.).

The shared "resolve the semantic plot kind" one-liner the transform results also
open with lives once in :func:`tsdynamics._result_common.resolve_plot_kind`.

Every public name (and the ``_jsonify`` / ``_PlotAccessor`` helpers some tests
import directly) is re-exported here, so ``from tsdynamics.analysis._result
import <X>`` keeps resolving exactly as before the split.
"""

from __future__ import annotations

from tsdynamics.analysis._result_array import ArrayResult
from tsdynamics.analysis._result_base import AnalysisResult
from tsdynamics.analysis._result_collection import CollectionResult

# Private helpers re-exported (redundant ``as`` alias = intentional re-export) for
# the test suite and intra-package consumers that import them from this module's
# historical path:
#   _jsonify          (tests/test_result_to_dict_json.py)
#   _PlotAccessor     (tests/test_plot_accessor_kinds.py)
#   _is_frame_scalar / _NumericOps  (kept reachable for symmetry with pre-split)
from tsdynamics.analysis._result_json import _is_frame_scalar as _is_frame_scalar
from tsdynamics.analysis._result_json import _jsonify as _jsonify
from tsdynamics.analysis._result_scalar import CountResult, ScalarResult
from tsdynamics.analysis._result_scalar import _NumericOps as _NumericOps
from tsdynamics.analysis._result_scaling import ScalingResult
from tsdynamics.analysis._result_viz import VisualizationNotInstalled
from tsdynamics.analysis._result_viz import _PlotAccessor as _PlotAccessor

__all__ = [
    "AnalysisResult",
    "ArrayResult",
    "CollectionResult",
    "CountResult",
    "ScalarResult",
    "ScalingResult",
    "VisualizationNotInstalled",
]


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
