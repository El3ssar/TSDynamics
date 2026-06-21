r"""
Surrogate-data generators (stream **A-SURR**).

Four progressively constrained nulls, from "destroy everything" to "keep both the
amplitude distribution and the power spectrum" (Theiler, Eubank, Longtin,
Galdrikian & Farmer, *Physica D* **58**, 77, 1992; Schreiber & Schmitz,
*Phys. Rev. Lett.* **77**, 635, 1996):

- :func:`random_shuffle` — a random permutation of the samples.  Preserves the
  amplitude distribution and nothing else; the null is "i.i.d. noise with this
  histogram".
- :func:`fourier_surrogate` — phase randomisation.  Preserves the power spectrum
  (linear autocorrelation); the null is "a stationary linear Gaussian process".
- :func:`aaft_surrogate` — amplitude-adjusted Fourier transform.  Preserves the
  amplitude distribution *and* (approximately) the power spectrum; the null is
  "a monotonic nonlinear re-scaling of a linear Gaussian process".
- :func:`iaaft_surrogate` — the iterative refinement of AAFT, matching the
  amplitude distribution exactly and the power spectrum to high accuracy.

:func:`surrogates` is the by-name dispatcher every test goes through; each
generator returns an ``(n, N)`` ensemble and takes a ``seed`` for reproducibility.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._common import _as_series, _phase_randomize

__all__ = [
    "aaft_surrogate",
    "fourier_surrogate",
    "iaaft_surrogate",
    "random_shuffle",
    "surrogates",
]

#: Canonical method name → alias set, resolved by :func:`surrogates`.
_METHOD_ALIASES: dict[str, str] = {
    "shuffle": "shuffle",
    "random": "shuffle",
    "random_shuffle": "shuffle",
    "permutation": "shuffle",
    "ft": "ft",
    "fourier": "ft",
    "phase": "ft",
    "rp": "ft",
    "aaft": "aaft",
    "iaaft": "iaaft",
}


def _ranks(x: np.ndarray) -> np.ndarray:
    """Return the rank (0-based ordinal position) of each sample of ``x``."""
    return np.argsort(np.argsort(x, kind="stable"), kind="stable")


def random_shuffle(data: Any, n: int = 1, *, seed: int | None = None, component: int | None = None):
    """Random-permutation surrogates (the constrained-realisation i.i.d. null).

    Each surrogate is an independent random permutation of the samples, so it
    keeps the amplitude distribution exactly and destroys every temporal
    correlation — the appropriate null for "are successive samples independent?".

    Parameters
    ----------
    data : array-like or Trajectory
        The source series (see :func:`~tsdynamics.analysis.surrogate._common._as_series`).
    n : int, default 1
        Number of surrogates to draw.
    seed : int, optional
        Seed for reproducibility.
    component : int or str, optional
        Component to select from multi-component input.

    Returns
    -------
    numpy.ndarray, shape (n, N)
        The surrogate ensemble.
    """
    series = _as_series(data, component)
    rng = np.random.default_rng(seed)
    N = series.size
    out = np.empty((int(n), N), dtype=float)
    for i in range(int(n)):
        out[i] = series[rng.permutation(N)]
    return out


def fourier_surrogate(
    data: Any, n: int = 1, *, seed: int | None = None, component: int | None = None
):
    """Phase-randomised (Fourier transform) surrogates of a linear Gaussian null.

    Keeps the magnitude spectrum — and therefore the power spectrum and the linear
    autocorrelation — of ``data`` exactly while randomising the Fourier phases
    (Theiler et al., 1992).  The surrogate amplitude distribution drifts toward
    Gaussian; use :func:`aaft_surrogate` / :func:`iaaft_surrogate` when the
    distribution must be preserved too.

    Parameters
    ----------
    data : array-like or Trajectory
        The source series.
    n : int, default 1
        Number of surrogates to draw.
    seed : int, optional
        Seed for reproducibility.
    component : int or str, optional
        Component to select from multi-component input.

    Returns
    -------
    numpy.ndarray, shape (n, N)
        The surrogate ensemble.
    """
    series = _as_series(data, component)
    rng = np.random.default_rng(seed)
    return _phase_randomize(series, int(n), rng)


def aaft_surrogate(data: Any, n: int = 1, *, seed: int | None = None, component: int | None = None):
    """Amplitude-adjusted Fourier-transform surrogates (Theiler et al., 1992).

    Tests the null of a *static monotonic nonlinearity acting on a linear Gaussian
    process*: the data is mapped to Gaussian by rank, phase-randomised, then mapped
    back through the inverse rank transform.  The result reproduces the amplitude
    distribution exactly and the power spectrum approximately (the rank remap
    distorts the spectrum slightly — :func:`iaaft_surrogate` removes that bias).

    Parameters
    ----------
    data : array-like or Trajectory
        The source series.
    n : int, default 1
        Number of surrogates to draw.
    seed : int, optional
        Seed for reproducibility.
    component : int or str, optional
        Component to select from multi-component input.

    Returns
    -------
    numpy.ndarray, shape (n, N)
        The surrogate ensemble.
    """
    series = _as_series(data, component)
    rng = np.random.default_rng(seed)
    N = series.size
    ranks_x = _ranks(series)
    sorted_x = np.sort(series)

    out = np.empty((int(n), N), dtype=float)
    for i in range(int(n)):
        # Gaussian series sharing data's rank order, then phase-randomised.
        gaussian = np.sort(rng.standard_normal(N))[ranks_x]
        gaussian_surrogate = _phase_randomize(gaussian, 1, rng)[0]
        # Re-impose data's amplitude distribution by matching ranks.
        out[i] = sorted_x[_ranks(gaussian_surrogate)]
    return out


def iaaft_surrogate(
    data: Any,
    n: int = 1,
    *,
    seed: int | None = None,
    component: int | None = None,
    max_iter: int = 1000,
):
    r"""Refine AAFT surrogates iteratively (IAAFT; Schreiber & Schmitz, 1996).

    Alternates two projections until the sample ordering stops changing: a
    *spectral* step that restores the exact magnitude spectrum of ``data`` (keeping
    the current phases), and an *amplitude* step that restores ``data``'s exact
    sorted values by rank.  Because it ends on the amplitude step, the surrogate's
    amplitude distribution is exact and its power spectrum matches to high accuracy
    — the standard improvement over plain :func:`aaft_surrogate`.

    Parameters
    ----------
    data : array-like or Trajectory
        The source series.
    n : int, default 1
        Number of surrogates to draw.
    seed : int, optional
        Seed for reproducibility.
    component : int or str, optional
        Component to select from multi-component input.
    max_iter : int, default 1000
        Cap on refinement iterations per surrogate (convergence is detected when
        the rank order repeats; the cap only bites on hard cases).

    Returns
    -------
    numpy.ndarray, shape (n, N)
        The surrogate ensemble.
    """
    series = _as_series(data, component)
    rng = np.random.default_rng(seed)
    N = series.size
    target_magnitude = np.abs(np.fft.rfft(series))
    sorted_x = np.sort(series)

    out = np.empty((int(n), N), dtype=float)
    for i in range(int(n)):
        s = series[rng.permutation(N)]  # random-shuffle start
        prev_ranks: np.ndarray | None = None
        for _ in range(int(max_iter)):
            # Spectral projection: keep phases, impose the target magnitudes.
            phases = np.angle(np.fft.rfft(s))
            s = np.fft.irfft(target_magnitude * np.exp(1j * phases), n=N)
            # Amplitude projection: impose data's exact values by rank order.
            ranks = _ranks(s)
            s = sorted_x[ranks]
            if prev_ranks is not None and np.array_equal(ranks, prev_ranks):
                break
            prev_ranks = ranks
        out[i] = s
    return out


_GENERATORS = {
    "shuffle": random_shuffle,
    "ft": fourier_surrogate,
    "aaft": aaft_surrogate,
    "iaaft": iaaft_surrogate,
}


def surrogates(
    data: Any,
    method: str = "iaaft",
    n: int = 1,
    *,
    seed: int | None = None,
    component: int | None = None,
    **kwargs: Any,
):
    """Generate surrogate series by name — the dispatcher every test goes through.

    Parameters
    ----------
    data : array-like or Trajectory
        The source series.
    method : str, default "iaaft"
        One of ``"shuffle"`` (aliases ``"random"``/``"permutation"``), ``"ft"``
        (``"fourier"``/``"phase"``), ``"aaft"``, or ``"iaaft"``.
    n : int, default 1
        Number of surrogates to draw.
    seed : int, optional
        Seed for reproducibility.
    component : int or str, optional
        Component to select from multi-component input.
    **kwargs
        Forwarded to the generator (e.g. ``max_iter`` for ``"iaaft"``).

    Returns
    -------
    numpy.ndarray, shape (n, N)
        The surrogate ensemble.

    Raises
    ------
    ValueError
        If ``method`` is unknown.
    """
    key = _METHOD_ALIASES.get(method.lower())
    if key is None:
        raise ValueError(
            f"unknown surrogate method {method!r}; use 'shuffle', 'ft', 'aaft' or 'iaaft'."
        )
    return _GENERATORS[key](data, n, seed=seed, component=component, **kwargs)


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
