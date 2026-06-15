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

from ._common import as_series, empirical_pvalue, gaussian_significance
from .generators import surrogates
from .statistics import STATISTICS

__all__ = ["SurrogateTest", "surrogate_test"]


@dataclass(frozen=True)
class SurrogateTest:
    r"""Outcome of a surrogate-data nonlinearity test.

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

    data_statistic: float
    surrogate_statistics: np.ndarray = field(repr=False)
    p_value: float
    z_score: float
    rejected: bool
    statistic: str
    method: str
    n_surrogates: int
    tail: str
    alpha: float

    def __repr__(self) -> str:  # noqa: D105
        verdict = "reject linear null" if self.rejected else "fail to reject"
        return (
            f"SurrogateTest({self.statistic!r} vs {self.n_surrogates}×{self.method!r}: "
            f"stat={self.data_statistic:.4g}, p={self.p_value:.4g}, "
            f"z={self.z_score:+.2f}σ, {verdict} @ α={self.alpha:g})"
        )


def surrogate_test(
    x: Any,
    statistic: str | Callable[..., float] = "time_reversal",
    method: str = "iaaft",
    n: int = 39,
    *,
    tail: str = "auto",
    alpha: float = 0.05,
    seed: int | None = None,
    component: int | None = None,
    statistic_kwargs: dict[str, Any] | None = None,
    **surrogate_kwargs: Any,
) -> SurrogateTest:
    r"""Test a series for nonlinearity against linear surrogates.

    Evaluates ``statistic`` on ``x`` and on ``n`` surrogates drawn by ``method``,
    then reports the rank p-value and significance of the data statistic within the
    surrogate ensemble.  The default — time-reversal asymmetry against IAAFT
    surrogates — rejects the linear-Gaussian null for a dissipative chaotic flow
    such as Lorenz.

    Parameters
    ----------
    x : array-like or Trajectory
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
        Rejection tail.  ``"auto"`` resolves to ``"less"`` for ``"prediction_error"``
        (determinism makes the data *more* predictable) and ``"two"`` otherwise.
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
        If ``statistic``, ``method`` or ``tail`` is unknown.
    """
    series = as_series(x, component)
    stat_kw = statistic_kwargs or {}

    if callable(statistic):
        stat_fn: Callable[..., float] = statistic
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
        tail = "less" if stat_name == "prediction_error" else "two"

    data_statistic = float(stat_fn(series, **stat_kw))
    ensemble = surrogates(series, method, int(n), seed=seed, **surrogate_kwargs)
    surrogate_statistics = np.array([float(stat_fn(s, **stat_kw)) for s in ensemble], dtype=float)

    p_value = empirical_pvalue(data_statistic, surrogate_statistics, tail)
    z_score = gaussian_significance(data_statistic, surrogate_statistics)
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
    )
