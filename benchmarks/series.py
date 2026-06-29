"""Shared input generators for the *from-data* benchmark tasks.

The data-driven estimators (correlation dimension, Lyapunov-from-data, and the
data-based maximal-Lyapunov variants) must all see the **identical** input
series, otherwise the comparison measures the input, not the estimator. So this
module builds the canonical inputs *independently of any benchmarked library* —
the Lorenz ``x(t)`` scalar series via SciPy's ``solve_ivp`` and the Hénon orbit
via a two-line NumPy loop — using the frozen parameters in :mod:`config`.

These generators are **not timed**: they are input preparation, paid once and
reused by every adapter (results are memoised per process). The integration
*speed* tasks deliberately do NOT use these — there each library integrates with
its own engine, which is the whole point of those rows.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import config
import numpy as np


def _lorenz_rhs(params: dict[str, float]) -> Any:
    sigma, rho, beta = params["sigma"], params["rho"], params["beta"]

    def rhs(_t: float, u: np.ndarray) -> list[float]:
        x, y, z = u
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    return rhs


@lru_cache(maxsize=1)
def lorenz_series() -> np.ndarray:
    """Return the canonical Lorenz ``x(t)`` scalar series (SciPy-integrated).

    Transient dropped and length capped per :mod:`config`. Float64, contiguous.

    Returns
    -------
    numpy.ndarray
        1-D Lorenz ``x`` component on a uniform ``SERIES_DT`` grid.
    """
    from scipy.integrate import solve_ivp

    t_end = config.SERIES_FINAL_TIME
    dt = config.SERIES_DT
    n = int(round(t_end / dt)) + 1
    t_eval = np.linspace(0.0, t_end, n)
    sol = solve_ivp(
        _lorenz_rhs(config.LORENZ_PARAMS),
        (0.0, t_end),
        config.LORENZ_IC,
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-10,
    )
    x = np.ascontiguousarray(sol.y[0], dtype=float)
    x = x[config.SERIES_TRANSIENT_SAMPLES :]
    if x.size > config.SERIES_MAX_SAMPLES:
        x = x[: config.SERIES_MAX_SAMPLES]
    return np.ascontiguousarray(x)


@lru_cache(maxsize=1)
def henon_series(n: int = 10_000) -> np.ndarray:
    """Return a Hénon ``x`` orbit (NumPy loop) for from-data Lyapunov estimates.

    Parameters
    ----------
    n : int
        Number of iterations to keep (after a 1000-step transient).

    Returns
    -------
    numpy.ndarray
        1-D Hénon ``x`` component, float64, contiguous.
    """
    a, b = config.HENON_PARAMS["a"], config.HENON_PARAMS["b"]
    x, y = config.HENON_IC
    burn = 1000
    out = np.empty(n, dtype=float)
    for i in range(burn + n):
        x, y = 1.0 - a * x * x + y, b * x
        if i >= burn:
            out[i - burn] = x
    return np.ascontiguousarray(out)


def delay_embed(series: np.ndarray, *, dim: int, delay: int) -> np.ndarray:
    """Delay-embed a scalar series into an ``(N, dim)`` point set.

    Parameters
    ----------
    series : numpy.ndarray
        1-D scalar series.
    dim : int
        Embedding dimension.
    delay : int
        Embedding delay in samples.

    Returns
    -------
    numpy.ndarray
        The ``(N - (dim-1)*delay, dim)`` reconstructed point set.
    """
    series = np.asarray(series, dtype=float)
    n = series.size - (dim - 1) * delay
    if n <= 0:
        raise ValueError("series too short for the requested embedding")
    return np.ascontiguousarray(
        np.column_stack([series[i * delay : i * delay + n] for i in range(dim)])
    )


@lru_cache(maxsize=1)
def white_noise_series(n: int = 8000, seed: int = 0) -> np.ndarray:
    """Return i.i.d. Gaussian white noise — the known-exponent DFA/Hurst input.

    White noise has DFA scaling exponent α = 0.5 and rescaled-range Hurst H = 0.5
    (no long-range correlation), so it is the textbook ground truth for the DFA
    and Hurst tasks: every library should recover ≈0.5.

    Parameters
    ----------
    n : int
        Series length.
    seed : int
        RNG seed (fixed → identical input for every library).

    Returns
    -------
    numpy.ndarray
        1-D white-noise series, float64, contiguous.
    """
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.standard_normal(n), dtype=float)
