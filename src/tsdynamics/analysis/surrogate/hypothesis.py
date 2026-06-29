r"""
Surrogate-data hypothesis test (stream **A-SURR**).

:func:`surrogate_test` ties the three pieces together — a generator
(:mod:`.generators`), a discriminating statistic (:mod:`.statistics`), and the
rank arithmetic (:mod:`._common`) — into the classical test of Theiler et al.
(*Physica D* **58**, 77, 1992): evaluate a nonlinearity statistic on the data and
on an ensemble of surrogates that share the data's *linear* properties, and reject
the linear null when the data statistic is an outlier of that ensemble.  The
result is returned as a :class:`SurrogateTest`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._result import AnalysisResult
from ._common import _as_series, _gaussian_significance, empirical_pvalue
from .generators import surrogates
from .statistics import STATISTICS, nonlinear_prediction_error

__all__ = ["SurrogateTest", "surrogate_test"]


@dataclass(frozen=True)
class SurrogateTest(AnalysisResult):
    r"""Outcome of a surrogate-data nonlinearity test.

    An :class:`~tsdynamics.analysis._result.AnalysisResult`, so it carries
    ``.meta`` / ``.summary()`` / ``.to_dict()`` / the ``.plot`` seam alongside the
    test outcome.

    Attributes
    ----------
    data_statistic : float
        The discriminating statistic evaluated on the data.
    surrogate_statistics : numpy.ndarray
        The same statistic on each surrogate (shape ``(n_surrogates,)``).
    p_value : float
        The rank-based surrogate p-value for the chosen ``tail``.
    z_score : float
        Significance of the data statistic in surrogate standard deviations (the
        classic "number of sigmas"); positive when the data lies above the mean.
    rejected : bool
        Whether the linear null is rejected at ``alpha`` (``p_value <= alpha``);
        the boundary is inclusive because Theiler's :math:`M = 2/\alpha - 1`
        ensemble makes the most-extreme attainable p-value exactly ``alpha``.
    statistic : str
        Name of the statistic (``"<callable>"`` for a user-supplied function).
    method : str
        Surrogate method used.
    n_surrogates : int
        Number of surrogates drawn.
    tail : str
        The rejection tail (``"two"`` / ``"greater"`` / ``"less"``).
    alpha : float
        The significance level the rejection decision used.
    """

    data_statistic: float = 0.0
    surrogate_statistics: np.ndarray = field(
        default_factory=lambda: np.empty(0), repr=False, compare=False
    )
    p_value: float = 1.0
    z_score: float = 0.0
    rejected: bool = False
    statistic: str = ""
    method: str = ""
    n_surrogates: int = 0
    tail: str = "two"
    alpha: float = 0.05

    def _rejection_tail(self) -> list[Any]:
        """Build the shaded rejection-tail ``"span"`` annotation(s) for the plot.

        The rejection region is the ``alpha``-quantile tail(s) of the surrogate
        null at the test's :attr:`tail`: the upper tail (``"greater"``), the lower
        tail (``"less"``), or both half-``alpha`` tails (``"two"``).  The data
        statistic falls inside one of these spans exactly when the null is
        rejected — the visual read of the test.  Empty when there are no
        surrogates to quantile.
        """
        from .. import _plotbuilder as pb

        surrogate = np.asarray(self.surrogate_statistics, dtype=float)
        surrogate = surrogate[np.isfinite(surrogate)]
        if surrogate.size == 0:
            return []
        lo, hi = float(surrogate.min()), float(surrogate.max())
        pad = (hi - lo) * 0.05 + (abs(hi) + abs(lo)) * 1e-3 + 1.0
        style = {"color": "lightcoral", "alpha": 0.2}

        def _span(span: tuple[float, float]) -> Any:
            return pb.span(span[0], span[1], axis="x", text="reject", style=style)

        if self.tail == "greater":
            return [_span((float(np.quantile(surrogate, 1.0 - self.alpha)), hi + pad))]
        if self.tail == "less":
            return [_span((lo - pad, float(np.quantile(surrogate, self.alpha))))]
        # two-sided: a half-alpha tail at each end.
        half = self.alpha / 2.0
        return [
            _span((lo - pad, float(np.quantile(surrogate, half)))),
            _span((float(np.quantile(surrogate, 1.0 - half)), hi + pad)),
        ]

    def to_plot_spec(self, kind: str | None = None) -> Any:
        """Describe this surrogate test as a backend-agnostic :class:`PlotSpec`.

        Builds a ``HISTOGRAM_NULL`` spec — the surrogate-statistic ensemble as a
        histogram (the null distribution) with the data statistic marked as a
        vertical reference line and the ``alpha``-quantile **rejection tail(s)
        shaded** (a ``"span"`` annotation per tail, picked by :attr:`tail`) — so
        the rejection reads off as the data line landing inside a shaded tail.
        The :mod:`tsdynamics.viz.spec` import is lazy, so building a spec never
        pulls a plotting library.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind (e.g. ``"histogram_null"``).  ``None`` uses
            ``HISTOGRAM_NULL``.

        Returns
        -------
        PlotSpec
        """
        from .. import _plotbuilder as pb

        surrogate = np.asarray(self.surrogate_statistics, dtype=float)
        verdict = "reject" if self.rejected else "fail to reject"
        annotations: list[Any] = [
            pb.vline(float(self.data_statistic), text="data", style={"color": "lightcoral"})
        ]
        annotations.extend(self._rejection_tail())
        return pb.spec(
            kind,
            "histogram_null",
            layers=[pb.histogram(surrogate, label=f"{self.method} surrogates")],
            xlabel=self.statistic,
            ylabel="count",
            title=f"{self.statistic} surrogate test (p = {self.p_value:.3g}, {verdict})",
            annotations=annotations,
        )

    def __repr__(self) -> str:  # noqa: D105
        verdict = "reject linear null" if self.rejected else "fail to reject"
        return (
            f"SurrogateTest({self.statistic!r} vs {self.n_surrogates}×{self.method!r}: "
            f"stat={self.data_statistic:.4g}, p={self.p_value:.4g}, "
            f"z={self.z_score:+.2f}σ, {verdict} @ α={self.alpha:g})"
        )


def surrogate_test(
    data: Any,
    statistic: str | Callable[..., float] = "time_reversal",
    method: str = "iaaft",
    n: int = 39,
    *,
    tail: str = "auto",
    alpha: float = 0.05,
    seed: int | None = None,
    component: int | str | None = None,
    statistic_kwargs: dict[str, Any] | None = None,
    **surrogate_kwargs: Any,
) -> SurrogateTest:
    r"""Test a series for nonlinearity against linear surrogates.

    Evaluates ``statistic`` on ``data`` and on ``n`` surrogates drawn by
    ``method``, then reports the rank p-value and significance of the data
    statistic within the surrogate ensemble.  The default — time-reversal asymmetry
    against IAAFT surrogates — rejects the linear-Gaussian null for a dissipative
    chaotic flow such as Lorenz.

    Parameters
    ----------
    data : array-like or Trajectory
        The series under test (a component of a multi-component input is selected
        with ``component=``).
    statistic : str or callable, default "time_reversal"
        ``"time_reversal"`` or ``"prediction_error"`` (see :mod:`.statistics`), or
        any callable mapping a 1-D array to a float.
    method : str, default "iaaft"
        Surrogate method — ``"shuffle"``, ``"ft"``, ``"aaft"`` or ``"iaaft"`` (see
        :func:`~tsdynamics.analysis.surrogate.generators.surrogates`).
    n : int, default 39
        Number of surrogates.  Theiler's :math:`M = 2/\alpha - 1` makes ``39`` the
        smallest ensemble able to reach a two-sided ``α = 0.05``.
    tail : {"auto", "two", "greater", "less"}, default "auto"
        Rejection tail.  ``"auto"`` resolves to ``"less"`` for the prediction-error
        statistic — whether named ``"prediction_error"`` or passed as the
        :func:`~tsdynamics.analysis.surrogate.statistics.nonlinear_prediction_error`
        callable (determinism makes the data *more* predictable, so it rejects in
        the lower tail) — and ``"two"`` for every other statistic.
    alpha : float, default 0.05
        Significance level for the ``rejected`` decision.
    seed : int, optional
        Seed for the surrogate ensemble (makes the whole test reproducible).
    component : int or str, optional
        Component to select from multi-component input.
    statistic_kwargs : dict, optional
        Extra keyword arguments forwarded to the statistic.
    **surrogate_kwargs
        Extra keyword arguments forwarded to the generator (e.g. ``max_iter``).

    Returns
    -------
    SurrogateTest
        The test outcome.

    Raises
    ------
    ValueError
        If ``statistic`` or ``method`` is an unknown name, if ``tail`` is not one
        of ``{"auto", "two", "greater", "less"}``, or if ``data`` cannot be coerced
        to a finite 1-D series (wrong shape, fewer than three samples, non-finite
        values, or an ambiguous multi-component input with no ``component=``).
    TypeError
        If a non-integer ``component`` is given for a plain 2-D array.

    References
    ----------
    .. [1] J. Theiler, S. Eubank, A. Longtin, B. Galdrikian, and J. D. Farmer,
       "Testing for nonlinearity in time series: the method of surrogate data,"
       Physica D 58, 77-94 (1992).

    Examples
    --------
    >>> import numpy as np
    >>> from tsdynamics.analysis.surrogate import surrogate_test
    >>> rng = np.random.default_rng(0)
    >>> noise = rng.standard_normal(2000)  # a linear-Gaussian null
    >>> res = surrogate_test(noise, "time_reversal", n=39, seed=0)
    >>> bool(res.rejected)
    False
    """
    series = _as_series(data, component)
    stat_kw = statistic_kwargs or {}

    if callable(statistic):
        stat_fn: Callable[..., Any] = statistic
        stat_name = getattr(statistic, "__name__", "<callable>")
    else:
        key = statistic.lower()
        if key not in STATISTICS:
            raise ValueError(
                f"unknown statistic {statistic!r}; use {sorted(STATISTICS)} or pass a callable."
            )
        stat_fn = STATISTICS[key]
        stat_name = key

    if tail == "auto":
        # Prediction error is one-sided: determinism makes the data *more*
        # predictable, so it rejects in the LOWER tail.  Resolve it both by the
        # registry key ("prediction_error") and by identity, so passing the
        # callable ``nonlinear_prediction_error`` (whose ``__name__`` differs
        # from the key) still gets the correct one-sided ``"less"`` tail.
        predictive = stat_name == "prediction_error" or stat_fn is nonlinear_prediction_error
        tail = "less" if predictive else "two"

    data_statistic = float(stat_fn(series, **stat_kw))
    ensemble = surrogates(series, method, int(n), seed=seed, **surrogate_kwargs)
    surrogate_statistics = np.array([float(stat_fn(s, **stat_kw)) for s in ensemble], dtype=float)

    p_value = empirical_pvalue(data_statistic, surrogate_statistics, tail)
    z_score = _gaussian_significance(data_statistic, surrogate_statistics)
    return SurrogateTest(
        data_statistic=data_statistic,
        surrogate_statistics=surrogate_statistics,
        p_value=p_value,
        z_score=z_score,
        rejected=p_value <= alpha,
        statistic=stat_name,
        method=method,
        n_surrogates=int(n),
        tail=tail,
        alpha=alpha,
        meta={
            "analysis": "surrogate_test",
            "statistic": stat_name,
            "method": method,
            "n_surrogates": int(n),
        },
    )


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
