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

from .._discovery import register as _register
from ._common import DimensionResult, UnembeddedSeriesWarning
from ._scaling import ScalingFit, fit_scaling_region, local_slopes
from .correlation import correlation_dimension, correlation_sum
from .fixedmass import fixed_mass_dimension
from .generalized import (
    box_counting_dimension,
    dimension_spectrum,
    generalized_dimension,
    information_dimension,
)

# ``dimension_spectrum_plot_spec`` is viz plumbing (it builds the ``PlotSpec`` the
# ``.plot`` accessor renders), not a dimension estimator.  It stays importable —
# ``from tsdynamics.analysis.dimensions import dimension_spectrum_plot_spec`` —
# but a spec builder has no business in an estimator namespace's autocomplete.
from .generalized import (
    dimension_spectrum_plot_spec as dimension_spectrum_plot_spec,
)

__all__ = [
    "DimensionResult",
    "UnembeddedSeriesWarning",
    "ScalingFit",
    "box_counting_dimension",
    "correlation_dimension",
    "correlation_sum",
    "dimension_spectrum",
    "fit_scaling_region",
    "fixed_mass_dimension",
    "generalized_dimension",
    "information_dimension",
    "local_slopes",
]

# Self-register the estimators: the definition site is the registration site
# (CONTRACT §7.7), through the public ``ts.analysis.register`` door.
_DATA = ("trajectory", "array")
_register(
    correlation_dimension,
    subjects=_DATA,
    area="dimensions",
    returns=DimensionResult,
    keywords="fractal attractor scaling grassberger procaccia",
    cite="Grassberger & Procaccia (1983), Physica D 9, 189",
    doi="10.1016/0167-2789(83)90298-1",
)
_register(
    correlation_sum,
    subjects=_DATA,
    area="dimensions",
    keywords="fractal pair counting grassberger correlation integral",
    cite="Grassberger & Procaccia (1983), Physica D 9, 189",
    doi="10.1016/0167-2789(83)90298-1",
)
_register(
    generalized_dimension,
    subjects=_DATA,
    area="dimensions",
    returns=DimensionResult,
    keywords="fractal multifractal renyi boxcount",
    cite="Hentschel & Procaccia (1983), Physica D 8, 435",
    doi="10.1016/0167-2789(83)90235-X",
)
_register(
    box_counting_dimension,
    subjects=_DATA,
    area="dimensions",
    returns=DimensionResult,
    keywords="fractal capacity boxcount attractor",
    cite="Hentschel & Procaccia (1983), Physica D 8, 435",
    doi="10.1016/0167-2789(83)90235-X",
)
_register(
    information_dimension,
    subjects=_DATA,
    area="dimensions",
    returns=DimensionResult,
    keywords="fractal shannon entropy attractor",
    cite="Hentschel & Procaccia (1983), Physica D 8, 435",
    doi="10.1016/0167-2789(83)90235-X",
)
_register(
    dimension_spectrum,
    subjects=_DATA,
    area="dimensions",
    keywords="fractal multifractal renyi spectrum",
    cite="Hentschel & Procaccia (1983), Physica D 8, 435",
    doi="10.1016/0167-2789(83)90235-X",
)
_register(
    fixed_mass_dimension,
    subjects=_DATA,
    area="dimensions",
    returns=DimensionResult,
    keywords="fractal nearest neighbour sparse attractor",
    cite="Badii & Politi (1985), J. Stat. Phys. 40, 725",
    doi="10.1007/BF01009897",
)


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
