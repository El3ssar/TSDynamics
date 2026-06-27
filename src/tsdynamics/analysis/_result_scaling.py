"""The :class:`ScalingResult` canonical scaling-curve result.

Split out of ``analysis/_result.py``; see that module's facade docstring for the
scaling-curve family rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

import numpy as np

from tsdynamics._result_common import resolve_plot_kind
from tsdynamics.analysis._result_base import AnalysisResult


@dataclass(frozen=True)
class ScalingResult(AnalysisResult):
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

    The number is :attr:`estimate`; ``float(result)`` returns it, so a
    :class:`ScalingResult` drops straight into arithmetic and comparisons.

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

    The two curve arrays are declared ``field(compare=False)`` per the
    :class:`AnalysisResult` subclassing contract (a frozen dataclass derives
    ``__eq__`` / ``__hash__`` from its fields, and NumPy arrays are neither
    boolean-comparable nor hashable), so two scaling results compare on their
    scalar summary fields.
    """

    _repr_fields: ClassVar[tuple[str, ...]] = ("estimate", "stderr")

    estimate: float = 0.0
    stderr: float = 0.0
    abscissa: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False, compare=False)
    ordinate: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False, compare=False)
    fit_region: tuple[int, int] = (0, 0)
    intercept: float = 0.0

    # -- the value -------------------------------------------------------

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
        from tsdynamics.viz.spec import Axis, Layer, PlotKind, PlotSpec

        spec_kind = resolve_plot_kind(kind, PlotKind.SCALING_FIT)
        x = np.asarray(self.abscissa, dtype=float)
        y = np.asarray(self.ordinate, dtype=float)
        lo, hi = self.fit_region

        layers = [
            Layer(PlotKind.SCATTER, {"x": x, "y": y}, label="curve"),
        ]
        if x.size and hi >= lo:
            layers.append(
                Layer(
                    PlotKind.MARKERS,
                    {"x": x[lo : hi + 1], "y": y[lo : hi + 1]},
                    label="fit region",
                )
            )
            fit_x = np.array([x[lo], x[hi]], dtype=float)
            layers.append(
                Layer(
                    PlotKind.LINE,
                    {"x": fit_x, "y": self.intercept + self.estimate * fit_x},
                    label=f"slope = {self.estimate:.3g}",
                )
            )

        return PlotSpec(
            kind=spec_kind,
            ndim=2,
            title=type(self).__name__,
            x=Axis(label="abscissa"),
            y=Axis(label="ordinate"),
            layers=layers,
            meta=dict(self.meta) if self.meta else {},
        )
