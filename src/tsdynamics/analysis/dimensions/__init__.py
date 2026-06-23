r"""
Fractal dimensions — stream **A-DIM**.

Three complementary estimators of the dimension of an attractor or point set,
each reading a slope off a log--log plot in an automatically selected scaling
region:

- :func:`correlation_dimension` — Grassberger--Procaccia :math:`D_2` from the
  correlation sum (:func:`correlation_sum`), with a Theiler window and
  tree-assisted pair counting.
- :func:`generalized_dimension` — the Rényi spectrum :math:`D_q` by box
  counting, with :func:`box_counting_dimension` (:math:`D_0`),
  :func:`information_dimension` (:math:`D_1`) and the multifractal
  :func:`dimension_spectrum` as convenience wrappers.
- :func:`fixed_mass_dimension` — the nearest-neighbour (fixed-mass) estimator,
  robust into the sparse tails of an attractor.

Every estimator returns a :class:`DimensionResult` — it behaves as the dimension
number in arithmetic (``float(result)``) while carrying the log--log curve, the
selected scaling window, and the :attr:`~DimensionResult.local_slopes`
diagnostic.  All accept a :class:`~tsdynamics.data.Trajectory` or a raw
``(N, dim)`` array interchangeably.

The headline estimators self-register into :data:`tsdynamics.registry.analyses`
so they are discoverable by name alongside out-of-tree analysis plugins.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ... import registry as _registry
from ._common import DimensionResult
from ._scaling import ScalingFit, fit_scaling_region, local_slopes
from .correlation import correlation_dimension, correlation_sum
from .fixedmass import fixed_mass_dimension
from .generalized import (
    box_counting_dimension,
    dimension_spectrum,
    dimension_spectrum_plot_spec,
    generalized_dimension,
    information_dimension,
)

__all__ = [
    "DimensionResult",
    "ScalingFit",
    "box_counting_dimension",
    "correlation_dimension",
    "correlation_sum",
    "dimension_spectrum",
    "dimension_spectrum_plot_spec",
    "fit_scaling_region",
    "fixed_mass_dimension",
    "generalized_dimension",
    "information_dimension",
    "local_slopes",
]

# Self-register the headline estimators (D4 / §4e: in-tree analyses register from
# their own subpackage).  Idempotent across re-imports — `register` keeps the
# same object under the same name.
_registrations: tuple[tuple[str, Callable[..., Any], dict[str, Any]], ...] = (
    (
        "correlation_dimension",
        correlation_dimension,
        {"needs": "trajectory", "family": "dimensions"},
    ),
    (
        "generalized_dimension",
        generalized_dimension,
        {"needs": "trajectory", "family": "dimensions"},
    ),
    (
        "box_counting_dimension",
        box_counting_dimension,
        {"needs": "trajectory", "family": "dimensions"},
    ),
    (
        "information_dimension",
        information_dimension,
        {"needs": "trajectory", "family": "dimensions"},
    ),
    ("fixed_mass_dimension", fixed_mass_dimension, {"needs": "trajectory", "family": "dimensions"}),
)
for _name, _fn, _meta in _registrations:
    _registry.analyses.register(_name, _fn, **_meta)
del _name, _fn, _meta, _registrations


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
