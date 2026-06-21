r"""
Composable entropy estimation: *outcome space* × *probability estimator* × *measure*.

An entropy estimate of a time series is three orthogonal choices stacked on top
of one another:

1. an **outcome space** — how a real-valued series is turned into a finite set of
   discrete *outcomes* and counted (ordinal patterns, dispersion patterns,
   amplitude bins, raw symbols);
2. a **probability estimator** — how outcome counts become a probability vector
   (maximum likelihood / relative frequency, additive smoothing);
3. an **information measure** — the functional applied to that probability vector
   (Shannon, Rényi, Tsallis).

:func:`entropy` composes the three; the named convenience functions
(:func:`~tsdynamics.analysis.entropy.permutation.permutation_entropy`,
:func:`~tsdynamics.analysis.entropy.dispersion.dispersion_entropy`, …) are thin
wrappers that fix the outcome space.

References
----------
Bandt, C. & Pompe, B. (2002). Permutation entropy: a natural complexity measure
for time series. *Phys. Rev. Lett.* **88**, 174102.

Rostaghi, M. & Azami, H. (2016). Dispersion entropy: a measure for time-series
analysis. *IEEE Signal Process. Lett.* **23**, 610–614.

Rényi, A. (1961). On measures of entropy and information. *Proc. 4th Berkeley
Symp. Math. Stat. Probab.* **1**, 547–561.

Tsallis, C. (1988). Possible generalization of Boltzmann–Gibbs statistics.
*J. Stat. Phys.* **52**, 479–487.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy.stats import norm

__all__ = [
    "AddConstant",
    "AmplitudeBinning",
    "Dispersion",
    "InformationMeasure",
    "MLE",
    "OrdinalPatterns",
    "OutcomeSpace",
    "ProbabilityEstimator",
    "Renyi",
    "Shannon",
    "Tsallis",
    "UniqueValues",
    "as_series",
    "entropy",
    "probabilities",
]


# ---------------------------------------------------------------------------
# Input coercion
# ---------------------------------------------------------------------------
def as_series(x: Any, component: int | str | None = None) -> np.ndarray:
    """
    Coerce ``x`` into a 1-D float array (a single scalar time series).

    Accepts a 1-D array-like, a 2-D array (a column is selected by
    ``component``), or a :class:`~tsdynamics.data.Trajectory` (a component is
    selected by index or, when the system declares ``variables``, by name).

    Parameters
    ----------
    x : array-like or Trajectory
        The data.  A 1-D sequence is used directly.
    component : int or str, optional
        Which column/component to extract from 2-D input or a trajectory.
        Required when the input has more than one component; for a 1-D input it
        must be left ``None`` (or ``0``).

    Returns
    -------
    numpy.ndarray
        A contiguous 1-D ``float64`` array.
    """
    # Trajectory: defer to its named-component machinery without importing it
    # (duck-typed to avoid a hard dependency cycle through families/data).
    if hasattr(x, "y") and hasattr(x, "component") and not isinstance(x, np.ndarray):
        if component is None:
            if x.y.ndim == 1 or x.y.shape[1] == 1:
                return np.asarray(x.y, dtype=float).ravel()
            raise ValueError(
                "trajectory has multiple components; pass component= to select one "
                f"(e.g. component=0 or a name from {getattr(x, 'variables', None)})."
            )
        return np.asarray(x.component(component), dtype=float).ravel()

    # A System (something that steps but is not yet data) is the common mistake
    # here.  Coercing it with ``np.asarray(..., dtype=float)`` would leak a bare
    # ``TypeError: float() argument must be ... not 'Lorenz'`` with no domain
    # framing — name what was passed and how to fix it instead.
    if hasattr(x, "step") and hasattr(x, "reinit") and not isinstance(x, np.ndarray):
        from tsdynamics.errors import InvalidInputError

        raise InvalidInputError(
            f"expected a measured series (1-D/2-D array or Trajectory), "
            f"got a System ({type(x).__name__}); run it first, e.g. "
            f"`series = {type(x).__name__.lower()}.run(...)['x']`."
        )

    try:
        arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as err:
        from tsdynamics.errors import InvalidInputError

        raise InvalidInputError(
            f"could not read {type(x).__name__} as a numeric series; "
            f"expected a 1-D/2-D array-like or a Trajectory."
        ) from err
    if arr.ndim == 1:
        if component not in (None, 0):
            raise ValueError("component= is meaningless for a 1-D series.")
        return arr
    if arr.ndim == 2:
        if component is None:
            if arr.shape[1] == 1:
                return arr[:, 0]
            raise ValueError("2-D input has multiple columns; pass component= to select one.")
        if not isinstance(component, (int, np.integer)):
            raise TypeError("component must be an integer index for a plain 2-D array.")
        return arr[:, int(component)]
    raise ValueError(f"expected a 1-D or 2-D series, got {arr.ndim}-D input.")


# ---------------------------------------------------------------------------
# Outcome spaces
# ---------------------------------------------------------------------------
class OutcomeSpace(ABC):
    """
    A scheme that maps a series to counts over a finite set of *outcomes*.

    Subclasses implement :meth:`counts`, returning a length-:attr:`cardinality`
    integer vector (zeros for outcomes that never occurred), so downstream
    estimators and the normalisation step both know the full outcome alphabet.
    """

    @property
    @abstractmethod
    def cardinality(self) -> int:
        """Total number of distinguishable outcomes (the alphabet size)."""

    @abstractmethod
    def counts(self, x: np.ndarray) -> np.ndarray:
        """Return the integer count of every outcome for series ``x``."""

    def encode(self, x: np.ndarray) -> np.ndarray:  # noqa: D401
        """Return the integer outcome label of each window (override if cheap)."""
        raise NotImplementedError


class OrdinalPatterns(OutcomeSpace):
    r"""
    Ordinal (permutation) patterns of an embedded series (Bandt & Pompe 2002).

    Each length-``m`` window ``(x_i, x_{i+τ}, …, x_{i+(m-1)τ})`` is encoded by the
    permutation that sorts it — its *ordinal pattern*.  There are ``m!`` patterns.

    Parameters
    ----------
    m : int
        Embedding (pattern) order, ``m ≥ 2``.
    tau : int
        Embedding delay, ``τ ≥ 1``.
    """

    def __init__(self, m: int = 3, tau: int = 1) -> None:
        if m < 2:
            raise ValueError("ordinal order m must be ≥ 2.")
        if tau < 1:
            raise ValueError("delay tau must be ≥ 1.")
        self.m = int(m)
        self.tau = int(tau)
        # Stable mapping permutation-tuple -> dense index over all m! patterns.
        self._index = {p: i for i, p in enumerate(itertools.permutations(range(self.m)))}

    @property
    def cardinality(self) -> int:  # noqa: D102
        return len(self._index)

    def _patterns(self, x: np.ndarray) -> np.ndarray:
        n = x.size
        span = (self.m - 1) * self.tau
        if n <= span:
            raise ValueError(
                f"series too short: need > {span} samples for m={self.m}, tau={self.tau}."
            )
        n_windows = n - span
        # Build the (n_windows, m) embedding then argsort each row.
        idx = np.arange(n_windows)[:, None] + np.arange(self.m)[None, :] * self.tau
        windows = x[idx]
        # argsort is stable → ties broken by order of appearance (Bandt–Pompe).
        return np.argsort(windows, axis=1, kind="stable")

    def counts(self, x: np.ndarray) -> np.ndarray:  # noqa: D102
        patterns = self._patterns(x)
        labels = np.fromiter(
            (self._index[tuple(row)] for row in patterns), dtype=np.intp, count=patterns.shape[0]
        )
        return np.bincount(labels, minlength=self.cardinality)

    def window_variance(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(labels, per-window variance)`` — the weighting for WPE."""
        n = x.size
        span = (self.m - 1) * self.tau
        n_windows = n - span
        idx = np.arange(n_windows)[:, None] + np.arange(self.m)[None, :] * self.tau
        windows = x[idx]
        patterns = np.argsort(windows, axis=1, kind="stable")
        labels = np.fromiter(
            (self._index[tuple(row)] for row in patterns), dtype=np.intp, count=n_windows
        )
        weights = windows.var(axis=1)
        return labels, weights


class Dispersion(OutcomeSpace):
    r"""
    Dispersion patterns (Rostaghi & Azami 2016).

    The series is mapped through the normal CDF (fitted to the sample mean/std)
    onto ``c`` amplitude classes, then embedded with order ``m`` and delay ``τ``;
    each window becomes a *dispersion pattern* over ``c**m`` possibilities.

    Parameters
    ----------
    c : int
        Number of amplitude classes, ``c ≥ 2``.
    m : int
        Embedding order, ``m ≥ 2``.
    tau : int
        Embedding delay, ``τ ≥ 1``.
    """

    def __init__(self, c: int = 6, m: int = 2, tau: int = 1) -> None:
        if c < 2:
            raise ValueError("number of classes c must be ≥ 2.")
        if m < 2:
            raise ValueError("embedding order m must be ≥ 2.")
        if tau < 1:
            raise ValueError("delay tau must be ≥ 1.")
        self.c = int(c)
        self.m = int(m)
        self.tau = int(tau)

    @property
    def cardinality(self) -> int:  # noqa: D102
        return self.c**self.m

    def _classes(self, x: np.ndarray) -> np.ndarray:
        std = x.std()
        if std == 0.0:
            # Degenerate (constant) series → every sample in the middle class.
            return np.full(x.size, (self.c + 1) // 2, dtype=np.intp)
        y = norm.cdf((x - x.mean()) / std)
        # Map (0,1) onto equal-width classes 1..c: round(c·y + 0.5) ≡ ⌊c·y⌋ + 1
        # (Rostaghi & Azami 2016, eq. 3), so class k ⟺ c·y ∈ [k-1, k).
        z = np.floor(self.c * y).astype(np.intp) + 1
        return np.clip(z, 1, self.c)

    def counts(self, x: np.ndarray) -> np.ndarray:  # noqa: D102
        z = self._classes(x)
        n = z.size
        span = (self.m - 1) * self.tau
        if n <= span:
            raise ValueError(
                f"series too short: need > {span} samples for m={self.m}, tau={self.tau}."
            )
        n_windows = n - span
        idx = np.arange(n_windows)[:, None] + np.arange(self.m)[None, :] * self.tau
        windows = z[idx] - 1  # to 0-based digits
        # Mixed-radix encode the m digits (base c) into a single label.
        powers = self.c ** np.arange(self.m - 1, -1, -1)
        labels = windows @ powers
        return np.bincount(labels, minlength=self.cardinality)


class AmplitudeBinning(OutcomeSpace):
    """
    Histogram outcome space: bin samples into ``bins`` equal-width amplitude bins.

    Parameters
    ----------
    bins : int
        Number of bins, ``bins ≥ 2``.
    range : tuple(float, float), optional
        Lower/upper edge; defaults to the data min/max.
    """

    def __init__(self, bins: int = 10, range: tuple[float, float] | None = None) -> None:  # noqa: A002
        if bins < 2:
            raise ValueError("bins must be ≥ 2.")
        self.bins = int(bins)
        self.range = range

    @property
    def cardinality(self) -> int:  # noqa: D102
        return self.bins

    def counts(self, x: np.ndarray) -> np.ndarray:  # noqa: D102
        lo, hi = self.range if self.range is not None else (float(x.min()), float(x.max()))
        if hi <= lo:
            hi = lo + 1.0  # degenerate constant series → one occupied bin
        edges = np.linspace(lo, hi, self.bins + 1)
        labels = np.clip(np.digitize(x, edges[1:-1]), 0, self.bins - 1)
        return np.bincount(labels, minlength=self.bins)


class UniqueValues(OutcomeSpace):
    """
    Treat the (already symbolic / discrete) samples themselves as outcomes.

    The cardinality is the number of distinct values observed — a data-dependent
    alphabet, so :meth:`cardinality` is only valid after a :meth:`counts` call.
    """

    def __init__(self) -> None:
        self._cardinality = 0

    @property
    def cardinality(self) -> int:  # noqa: D102
        return self._cardinality

    def counts(self, x: np.ndarray) -> np.ndarray:  # noqa: D102
        _, counts = np.unique(x, return_counts=True)
        self._cardinality = counts.size
        return counts


# ---------------------------------------------------------------------------
# Probability estimators
# ---------------------------------------------------------------------------
class ProbabilityEstimator(ABC):
    """Turn an integer count vector into a probability vector."""

    @abstractmethod
    def probabilities(self, counts: np.ndarray) -> np.ndarray:
        """Return probabilities summing to 1."""


class MLE(ProbabilityEstimator):
    """Maximum-likelihood (relative-frequency / plug-in) estimator."""

    def probabilities(self, counts: np.ndarray) -> np.ndarray:  # noqa: D102
        total = counts.sum()
        if total == 0:
            raise ValueError("no outcomes observed — series too short for this outcome space.")
        return counts / total


class AddConstant(ProbabilityEstimator):
    r"""
    Additive-smoothing estimator: ``p_i = (n_i + a) / (N + a·K)``.

    ``a = 1`` is Laplace smoothing; ``a = 0.5`` is the Krichevsky–Trofimov
    estimator.  Smoothing assigns non-zero probability to unobserved outcomes,
    which regularises entropy estimates from short series.

    Parameters
    ----------
    a : float
        Pseudocount added to every outcome, ``a > 0``.
    """

    def __init__(self, a: float = 1.0) -> None:
        if a <= 0:
            raise ValueError("smoothing constant a must be > 0.")
        self.a = float(a)

    def probabilities(self, counts: np.ndarray) -> np.ndarray:  # noqa: D102
        k = counts.size
        return (counts + self.a) / (counts.sum() + self.a * k)


# ---------------------------------------------------------------------------
# Information measures
# ---------------------------------------------------------------------------
class InformationMeasure(ABC):
    """An entropy functional ``p ↦ H(p)`` with a known uniform maximum."""

    @abstractmethod
    def apply(self, p: np.ndarray) -> float:
        """Evaluate the measure on a probability vector."""

    def maximum(self, n_outcomes: int) -> float:
        """Value on the uniform distribution over ``n_outcomes`` (the maximum)."""
        if n_outcomes <= 1:
            return 0.0
        return self.apply(np.full(n_outcomes, 1.0 / n_outcomes))


class Shannon(InformationMeasure):
    r"""Shannon entropy ``H = -∑ p log_b p`` (Shannon 1948); ``base`` sets ``b``."""

    def __init__(self, base: float = 2.0) -> None:
        if base <= 1:
            raise ValueError("log base must be > 1.")
        self.base = float(base)

    def apply(self, p: np.ndarray) -> float:  # noqa: D102
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)) / np.log(self.base))


class Renyi(InformationMeasure):
    r"""
    Rényi entropy of order ``q`` (Rényi 1961).

    ``H_q = (1/(1-q)) log_b ∑ p^q``; ``q → 1`` recovers Shannon.

    Parameters
    ----------
    q : float
        Order, ``q ≥ 0``, ``q ≠ 1`` (use :class:`Shannon` for ``q = 1``).
    base : float
        Logarithm base.
    """

    def __init__(self, q: float = 2.0, base: float = 2.0) -> None:
        if q < 0:
            raise ValueError("Rényi order q must be ≥ 0.")
        if base <= 1:
            raise ValueError("log base must be > 1.")
        self.q = float(q)
        self.base = float(base)

    def apply(self, p: np.ndarray) -> float:  # noqa: D102
        p = p[p > 0]
        if abs(self.q - 1.0) < 1e-12:
            return float(-np.sum(p * np.log(p)) / np.log(self.base))
        return float(np.log(np.sum(p**self.q)) / ((1.0 - self.q) * np.log(self.base)))


class Tsallis(InformationMeasure):
    r"""
    Tsallis entropy of order ``q`` (Tsallis 1988).

    ``S_q = (1 - ∑ p^q) / (q - 1)``; ``q → 1`` recovers Shannon (nats).

    Parameters
    ----------
    q : float
        Entropic index, ``q ≠ 1`` (use :class:`Shannon` for ``q = 1``).
    """

    def __init__(self, q: float = 2.0) -> None:
        self.q = float(q)

    def apply(self, p: np.ndarray) -> float:  # noqa: D102
        p = p[p > 0]
        if abs(self.q - 1.0) < 1e-12:
            return float(-np.sum(p * np.log(p)))
        return float((1.0 - np.sum(p**self.q)) / (self.q - 1.0))


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------
def probabilities(
    x: Any,
    outcomes: OutcomeSpace,
    *,
    prob: ProbabilityEstimator | None = None,
    component: int | str | None = None,
) -> np.ndarray:
    """
    Probability vector of ``x`` under an outcome space and estimator.

    Parameters
    ----------
    x : array-like or Trajectory
        The series (see :func:`as_series`).
    outcomes : OutcomeSpace
        How the series is symbolised and counted.
    prob : ProbabilityEstimator, optional
        Defaults to :class:`MLE` (relative frequency).
    component : int or str, optional
        Component selector for multi-component input.

    Returns
    -------
    numpy.ndarray
        Probabilities over the outcome space (summing to 1).
    """
    series = as_series(x, component)
    prob = prob if prob is not None else MLE()
    return prob.probabilities(outcomes.counts(series))


def entropy(
    data: Any,
    *,
    outcomes: OutcomeSpace,
    prob: ProbabilityEstimator | None = None,
    measure: InformationMeasure | None = None,
    normalize: bool = False,
    component: int | str | None = None,
) -> float:
    """
    Entropy of ``data`` — the composable entry point.

    Symbolise with ``outcomes``, estimate probabilities with ``prob``, then
    apply the information ``measure``.  With ``normalize=True`` the result is
    divided by the measure's maximum over the outcome alphabet, mapping it to
    ``[0, 1]``.

    Parameters
    ----------
    data : array-like or Trajectory
        The series (see :func:`as_series`).
    outcomes : OutcomeSpace
        Symbolisation scheme (e.g. :class:`OrdinalPatterns`, :class:`Dispersion`).
    prob : ProbabilityEstimator, optional
        Defaults to :class:`MLE`.
    measure : InformationMeasure, optional
        Defaults to :class:`Shannon` (base 2, i.e. bits).
    normalize : bool, default False
        Divide by the maximum (uniform) value over the alphabet.
    component : int or str, optional
        Component selector for multi-component input.

    Returns
    -------
    float
        The (optionally normalised) entropy.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> entropy(rng.random(5000), outcomes=OrdinalPatterns(3), normalize=True)  # ≈ 1
    0.99...
    """
    series = as_series(data, component)
    prob = prob if prob is not None else MLE()
    measure = measure if measure is not None else Shannon()
    counts = outcomes.counts(series)
    p = prob.probabilities(counts)
    h = measure.apply(p)
    if normalize:
        hmax = measure.maximum(outcomes.cardinality)
        return h / hmax if hmax > 0 else 0.0
    return h


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
