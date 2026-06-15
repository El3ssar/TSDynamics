"""
Shared Hypothesis strategies and deterministic signal builders for the
property-test harness (stream I-QA).

The property tests under ``tests/test_property_*.py`` assert *mathematical
invariants* of the analysis and transform layer (e.g. ``permutation_entropy``
is in ``[0, 1]`` when normalised, a Fourier surrogate preserves the power
spectrum, ``detrend`` removes a linear trend).  To do that robustly, they need
two kinds of input:

- **Structured signals** — sinusoids, AR(1) noise, deterministic-map orbits.
  These are far better estimator inputs than raw float arrays: they exercise the
  numeric routines on realistic data while staying reproducible.  The builders
  here take plain parameters (so Hypothesis can drive them) and return finite,
  non-degenerate ``float64`` arrays.
- **Constrained raw arrays** — for the pure bound/shape invariants (PSD ``>= 0``,
  length preservation), a finite, bounded, non-constant 1-D array is enough.
  :func:`finite_signals` is that strategy.

Everything here is deterministic given its inputs; the only randomness is
seeded explicitly (``np.random.default_rng(seed)``), so a failing example always
reproduces.  This module is a helper (underscore-prefixed) and is **not**
collected as a test module.
"""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

__all__ = [
    "ar1",
    "finite_signals",
    "henon_series",
    "lengths",
    "logistic_series",
    "seeds",
    "sinusoid",
    "white_noise",
]

# ---------------------------------------------------------------------------
# Scalar strategies
# ---------------------------------------------------------------------------

#: Reproducible RNG seeds.
seeds = st.integers(min_value=0, max_value=2**31 - 1)


def lengths(min_value: int = 64, max_value: int = 512) -> st.SearchStrategy[int]:
    """Series lengths for estimator inputs (long enough to be meaningful)."""
    return st.integers(min_value=min_value, max_value=max_value)


def finite_signals(
    min_size: int = 32,
    max_size: int = 512,
    *,
    elements_bound: float = 1.0e6,
    min_std: float = 1.0e-3,
) -> st.SearchStrategy[np.ndarray]:
    """
    A 1-D ``float64`` array that is finite, bounded and non-constant.

    Suitable for the pure bound/shape invariants that must hold for *any* real
    signal.  The ``min_std`` filter drops degenerate (near-constant) draws that
    several estimators legitimately reject — the semantic tests use the
    structured builders below instead.
    """
    base = arrays(
        dtype=np.float64,
        shape=st.integers(min_value=min_size, max_value=max_size),
        elements=st.floats(
            min_value=-elements_bound,
            max_value=elements_bound,
            allow_nan=False,
            allow_infinity=False,
            width=64,
        ),
    )
    return base.filter(lambda a: float(np.std(a)) > min_std)


# ---------------------------------------------------------------------------
# Deterministic signal builders (plain parameters → array)
# ---------------------------------------------------------------------------


def sinusoid(
    n: int = 512,
    *,
    freq: float = 0.05,
    phase: float = 0.0,
    amplitude: float = 1.0,
    offset: float = 0.0,
) -> np.ndarray:
    """A pure cosine of ``n`` samples (unit sample spacing)."""
    t = np.arange(int(n), dtype=float)
    return offset + amplitude * np.cos(2.0 * np.pi * freq * t + phase)


def white_noise(n: int = 512, *, seed: int = 0, scale: float = 1.0) -> np.ndarray:
    """Seeded Gaussian white noise."""
    return np.random.default_rng(int(seed)).normal(0.0, float(scale), int(n))


def ar1(n: int = 512, *, phi: float = 0.7, seed: int = 0, scale: float = 1.0) -> np.ndarray:
    """
    A seeded AR(1) process ``x_{t+1} = phi x_t + e_t`` (linear-stochastic null).

    ``|phi| < 1`` keeps it stationary; this is the canonical *linear* signal a
    surrogate/nonlinearity test must NOT reject.
    """
    rng = np.random.default_rng(int(seed))
    noise = rng.normal(0.0, float(scale), int(n))
    x = np.empty(int(n), dtype=float)
    x[0] = noise[0]
    for i in range(1, int(n)):
        x[i] = phi * x[i - 1] + noise[i]
    return x


def logistic_series(
    n: int = 512, *, r: float = 4.0, x0: float = 0.4, burn: int = 100
) -> np.ndarray:
    """
    A logistic-map orbit ``x_{t+1} = r x_t (1 - x_t)``.

    ``r = 4`` is fully chaotic (the headline deterministic-nonlinear signal);
    lower ``r`` gives periodic orbits.  A short burn-in lands on the attractor.
    """
    x = float(x0)
    for _ in range(int(burn)):
        x = r * x * (1.0 - x)
    out = np.empty(int(n), dtype=float)
    for i in range(int(n)):
        out[i] = x
        x = r * x * (1.0 - x)
    return out


def henon_series(
    n: int = 512,
    *,
    a: float = 1.4,
    b: float = 0.3,
    x0: float = 0.1,
    y0: float = 0.1,
    burn: int = 100,
    component: int = 0,
) -> np.ndarray:
    """
    One coordinate of a Hénon-map orbit (a deterministic-chaotic series).

    ``component=0`` returns the ``x`` series, ``1`` the ``y`` series.  Used as a
    cheap, compile-free chaotic source for the fast-tier property tests.
    """
    x, y = float(x0), float(y0)
    for _ in range(int(burn)):
        x, y = 1.0 - a * x * x + y, b * x
    out = np.empty(int(n), dtype=float)
    for i in range(int(n)):
        out[i] = x if component == 0 else y
        x, y = 1.0 - a * x * x + y, b * x
    return out
