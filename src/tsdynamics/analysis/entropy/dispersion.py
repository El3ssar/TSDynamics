r"""
Dispersion entropy.

References
----------
Rostaghi, M. & Azami, H. (2016). Dispersion entropy: a measure for time-series
analysis. *IEEE Signal Process. Lett.* **23**, 610–614.
"""

from __future__ import annotations

from typing import Any

from .core import Dispersion, Shannon, entropy

__all__ = ["dispersion_entropy"]


def dispersion_entropy(
    data: Any,
    c: int = 6,
    dimension: int = 2,
    delay: int = 1,
    *,
    base: float = 2.0,
    normalize: bool = True,
    component: int | str | None = None,
) -> float:
    r"""
    Dispersion entropy of a time series (Rostaghi & Azami 2016).

    The series is mapped through the normal CDF onto ``c`` amplitude classes,
    embedded with order ``dimension`` and delay ``τ``, and the Shannon entropy of
    the resulting *dispersion patterns* is returned.  Unlike permutation entropy
    it keeps amplitude information and is markedly faster than sample entropy,
    while remaining robust to noise.

    Parameters
    ----------
    data : array-like or Trajectory
        Scalar time series (or a component of a multivariate one).
    c : int, default 6
        Number of amplitude classes.  ``4 ≤ c ≤ 8`` is typical; ``n ≫
        c**dimension`` is recommended.
    dimension : int, default 2
        Embedding order.
    delay : int, default 1
        Embedding delay.
    base : float, default 2.0
        Logarithm base (``2`` → bits).
    normalize : bool, default True
        Divide by ``log_base(c**dimension)`` so the result lies in ``[0, 1]``.
    component : int or str, optional
        Component selector for multi-component input.

    Returns
    -------
    float
        Dispersion entropy.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> dispersion_entropy(rng.random(10000))         # white noise → ≈ 1
    0.99...
    """
    return entropy(
        data,
        outcomes=Dispersion(c, dimension, delay),
        measure=Shannon(base),
        normalize=normalize,
        component=component,
    )


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
