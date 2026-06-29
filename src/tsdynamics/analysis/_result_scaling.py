"""The :class:`ScalingResult` canonical scaling-curve result.

Split out of ``analysis/_result.py``; see that module's facade docstring for the
scaling-curve family rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

import numpy as np

from tsdynamics.analysis._result_base import AnalysisResult
from tsdynamics.analysis._result_scalar import _NumericOps


@dataclass(frozen=True, eq=False)
class ScalingResult(_NumericOps, AnalysisResult):
    r"""A quantity read off the slope of a scaling curve, with that curve.

    Many estimators in TSDynamics share one shape: build a scaling curve, fit a
    straight line over its linear (scaling) region, and report the slope.  Every
    fractal dimension (slope of :math:`\log C(r)` against :math:`\log r`), the
    maximal Lyapunov exponent from a measured series (slope of the mean
    log-divergence against time), expansion entropy, and the Cao /
    false-nearest-neighbour embedding diagnostics all fit this mould.

    :class:`ScalingResult` is the *one* schema for that whole family, so a single
    generic ``result.plot.scaling()`` renders any of them and any consumer can
    find "the curve" and "the fit" without knowing which estimator produced it.

    The number is :attr:`estimate`; ``float(result)`` returns it, and the full
    numeric protocol is mixed in from :class:`_NumericOps` (the same mixin
    :class:`~tsdynamics.analysis._result_scalar.ScalarResult` uses), so a
    :class:`ScalingResult` drops straight into arithmetic and comparisons —
    ``dim > 2.0``, ``0.5 * (d1 + d2)`` and ``abs(result)`` all work without
    unwrapping it.  An operand that is not float-convertible yields
    :data:`NotImplemented`, so ``result == pytest.approx(x)`` still resolves.

    Attributes
    ----------
    estimate : float
        The estimated quantity — the fitted slope of the scaling region (a
        dimension, a Lyapunov exponent, an entropy, …).  ``float(self)`` returns
        this value.
    stderr : float
        Standard error of :attr:`estimate` from the line fit.
    abscissa : numpy.ndarray
        The horizontal scaling coordinate at every point of the curve (e.g.
        :math:`\log r`, or time).  Same length as :attr:`ordinate`.
    ordinate : numpy.ndarray
        The vertical coordinate at every point of the curve (e.g.
        :math:`\log C(r)`, or mean log-divergence).
    fit_region : tuple of int
        The ``(lo, hi)`` inclusive index bounds (into :attr:`abscissa` /
        :attr:`ordinate`) of the scaling region the line was fitted over.
    intercept : float
        Intercept of the fitted line, so ``ordinate ≈ intercept + estimate *
        abscissa`` over the fit region — what a renderer draws the fit line from.

    Notes
    -----
    Construct one with the canonical names::

        ScalingResult(
            estimate=2.05, stderr=0.03,
            abscissa=log_r, ordinate=log_C,
            fit_region=(8, 24), intercept=-1.2,
            meta=AnalysisResult.build_meta(system, ...),
        )

    The class is declared ``@dataclass(frozen=True, eq=False)`` so the
    dataclass machinery does **not** regenerate ``__eq__`` / ``__hash__`` over
    the curve arrays (which are neither boolean-comparable nor hashable); the
    numeric ``__eq__`` / ``__hash__`` from :class:`_NumericOps` (comparing on
    ``float(self)``) are used instead, matching
    :class:`~tsdynamics.analysis._result_scalar.ScalarResult`.  The two curve
    arrays still carry ``field(compare=False)`` for clarity.  Subclasses (e.g.
    :class:`~tsdynamics.analysis.dimensions._common.DimensionResult`) must
    re-apply ``@dataclass(frozen=True, eq=False)`` so the numeric dunders are not
    shadowed by a dataclass-generated ``__eq__``.
    """

    _repr_fields: ClassVar[tuple[str, ...]] = ("estimate", "stderr")

    estimate: float = 0.0
    stderr: float = 0.0
    abscissa: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False, compare=False)
    ordinate: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False, compare=False)
    fit_region: tuple[int, int] = (0, 0)
    intercept: float = 0.0

    # -- the value -------------------------------------------------------

    @property
    def value(self) -> float:
        """The estimate, under the name :class:`_NumericOps` reads.

        :class:`_NumericOps` resolves the numeric protocol off ``self.value``;
        for a scaling result the value *is* :attr:`estimate` (the fitted slope),
        so this property bridges the two.
        """
        return float(self.estimate)

    def __float__(self) -> float:
        """Return :attr:`estimate`, so the result drops into arithmetic."""
        return float(self.estimate)

    # -- scaling diagnostics --------------------------------------------

    @property
    def local_slopes(self) -> np.ndarray:
        r"""Pointwise local slope ``d(ordinate)/d(abscissa)`` of the curve.

        Centered differences (one-sided at the ends, via
        :func:`numpy.gradient`), so non-uniform spacing is handled correctly.
        The plateau of this curve *is* the scaling region; inspecting it is the
        standard sanity check on any reported scaling estimate.  Returns an
        all-``nan`` array of the same shape when there are fewer than two points.

        Returns
        -------
        numpy.ndarray
            Local slope at every point of the curve.
        """
        x = np.asarray(self.abscissa, dtype=float)
        y = np.asarray(self.ordinate, dtype=float)
        if x.size < 2:
            return np.full(x.shape, np.nan)
        return cast("np.ndarray", np.gradient(y, x))

    @property
    def scaling_window(self) -> tuple[float, float]:
        """Return the abscissa span ``(lo, hi)`` the fit was taken over.

        The :attr:`abscissa` values at the two endpoints of :attr:`fit_region`
        — the actual coordinate window of the scaling region (not the index
        bounds).

        Returns
        -------
        tuple of float
            ``(abscissa[lo], abscissa[hi])``.
        """
        lo, hi = self.fit_region
        x = np.asarray(self.abscissa, dtype=float)
        return float(x[lo]), float(x[hi])

    def _interpretation(self) -> str | None:
        """Report the estimate with its scaling-window width and point count."""
        lo, hi = self.fit_region
        n_fit = hi - lo + 1
        return f"estimate = {float(self.estimate):.4g} ± {float(self.stderr):.2g}  (fit over {n_fit} points)"

    # -- visualization ---------------------------------------------------

    def to_plot_spec(self, kind: str | None = None) -> Any:
        """Describe this scaling result as a backend-agnostic :class:`PlotSpec`.

        Builds a ``SCALING_FIT`` spec — the curve as a scatter layer, the fitted
        scaling region highlighted, and the fit line drawn from
        :attr:`intercept` and :attr:`estimate` — so any registered backend can
        render it identically.  The :mod:`tsdynamics.viz.spec` import is lazy, so
        building a result (or importing :mod:`tsdynamics`) never pulls a plot
        library; the spec itself carries no rendering code.

        Parameters
        ----------
        kind : str, optional
            An override for the semantic spec kind (the closed
            :class:`~tsdynamics.viz.spec.PlotKind` vocabulary).  ``None`` (the
            default) uses ``SCALING_FIT``; the ``.plot.scaling()`` seam passes
            ``"scaling_fit"`` explicitly, which resolves to the same kind.

        Returns
        -------
        PlotSpec
        """
        from . import _plotbuilder as pb

        return pb.scaling_fit(
            kind,
            self.abscissa,
            self.ordinate,
            fit_region=self.fit_region,
            slope=self.estimate,
            intercept=self.intercept,
            xlabel="abscissa",
            ylabel="ordinate",
            title=type(self).__name__,
            meta=self.meta,
        )
