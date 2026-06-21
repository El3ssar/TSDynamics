"""Lyapunov-based quantifiers: spectra, maximal exponent, Kaplan–Yorke dimension."""

from __future__ import annotations

from typing import Any

import numpy as np

from tsdynamics.families import DelaySystem

from ... import registry as _registry
from .from_data import LyapunovFromData, lyapunov_from_data

__all__ = [
    "LyapunovFromData",
    "kaplan_yorke_dimension",
    "lyapunov_from_data",
    "lyapunov_spectrum",
    "max_lyapunov",
]


def kaplan_yorke_dimension(spectrum: Any) -> float:
    """
    Kaplan–Yorke (Lyapunov) dimension from a Lyapunov spectrum.

    ``D_KY = j + (λ₁ + ... + λ_j) / |λ_{j+1}|`` where ``j`` is the largest
    index with a non-negative cumulative sum (Kaplan & Yorke 1979).

    Parameters
    ----------
    spectrum : array-like
        Lyapunov exponents (any order; sorted descending internally).

    Returns
    -------
    float
        0.0 when every exponent is negative; ``len(spectrum)`` when the
        cumulative sum never turns negative (spectrum incomplete).

    Examples
    --------
    >>> kaplan_yorke_dimension([0.906, 0.0, -14.57])   # Lorenz
    2.062...
    """
    s = np.sort(np.asarray(spectrum, dtype=float))[::-1]
    if s.size == 0 or s[0] < 0.0:
        return 0.0
    cum = np.cumsum(s)
    nonneg = np.nonzero(cum >= 0.0)[0]
    j = int(nonneg[-1])
    if j == s.size - 1:
        return float(s.size)  # spectrum doesn't close — dimension saturates
    return float(j + 1 + cum[j] / abs(s[j + 1]))


def lyapunov_spectrum(
    system: Any,
    *,
    k: int | None = None,
    final_time: float | None = None,
    n: int | None = None,
    transient: float | None = None,
    dt: float | None = None,
    ic: Any | None = None,
    method: str | None = None,
) -> np.ndarray:
    """
    Lyapunov spectrum of any system — the uniform, documented entry point.

    Dispatches to the family implementation (QR tangent dynamics for maps, the
    extended variational system on the engine for ODEs, the engine
    function-space estimator for DDEs), translating this one signature to each
    family's native keywords.

    Parameters
    ----------
    system : System
        A flow (ODE/DDE) or a discrete map.
    k : int, optional
        Number of exponents to compute (was ``n_exp``).  Defaults to
        ``system.dim`` for flows/maps; a DDE may request more than ``dim`` (its
        tangent space is the infinite-dimensional history).
    final_time : float, optional
        Averaging-window length for a **flow** (after the transient).  Mutually
        exclusive with ``n``; a flow uses ``final_time``.
    n : int, optional
        Number of iterations for a **map**.  Mutually exclusive with
        ``final_time``; a map uses ``n``.
    transient : float, optional
        Amount discarded before averaging (a flow burn-in **time**).  Maps
        reorthonormalise from the initial condition and take no transient here.
    dt : float, optional
        Sampling / integration step (flows only).
    ic : array-like, optional
        Initial condition.  Falls back to ``system.ic``, then random.
    method : str, optional
        Solver kernel (continuous flows only).

    Returns
    -------
    ndarray, shape ``(k,)``
        Lyapunov exponents ordered from largest to smallest.

    Examples
    --------
    >>> lyapunov_spectrum(Lorenz(), final_time=300.0)   # [0.91, ~0, -14.57]
    >>> lyapunov_spectrum(Henon(), k=2, n=5000)         # [0.42, -1.62]
    """
    method_fn = getattr(system, "lyapunov_spectrum", None)
    if method_fn is None:
        raise TypeError(
            f"{type(system).__name__} has no lyapunov_spectrum implementation. "
            f"For derived wrappers, compute the spectrum on the underlying system."
        )
    if k is not None and k <= 0:
        raise ValueError(f"k (number of exponents) must be a positive integer, got {k!r}.")

    fwd: dict[str, Any] = {}
    if k is not None:
        fwd["n_exp"] = k
    if ic is not None:
        fwd["ic"] = ic

    if getattr(system, "is_discrete", False):
        # Maps: horizon is `n` (iterations); no time, solver or burn-in concept.
        if final_time is not None:
            raise ValueError("lyapunov_spectrum: final_time is for flows; a map uses n.")
        if dt is not None:
            raise ValueError("lyapunov_spectrum: dt has no meaning for a discrete map.")
        if transient is not None:
            raise ValueError(
                "lyapunov_spectrum: transient is not supported for a map spectrum "
                "(the QR iteration reorthonormalises from the initial condition)."
            )
        if method is not None:
            raise ValueError("lyapunov_spectrum: a map spectrum has no solver method.")
        if n is not None:
            fwd["steps"] = n
    else:
        # Flows (ODE/DDE): horizon is `final_time`; transient is a burn-in time.
        if n is not None:
            raise ValueError("lyapunov_spectrum: n is for maps; a flow/DDE uses final_time.")
        if final_time is not None:
            fwd["final_time"] = final_time
        if transient is not None:
            fwd["burn_in"] = transient
        if dt is not None:
            fwd["dt"] = dt
        if method is not None:
            if isinstance(system, DelaySystem):
                raise ValueError(
                    "lyapunov_spectrum: a DDE selects its engine via backend, not method."
                )
            fwd["method"] = method
    return method_fn(**fwd)


def max_lyapunov(
    system: Any,
    *,
    d0: float = 1e-9,
    n: int = 400,
    steps_per: int = 5,
    dt: float | None = None,
    transient: int = 500,
    ic: Any | None = None,
    seed: int | None = None,
) -> float:
    """
    Maximal Lyapunov exponent by two-trajectory rescaling (Benettin et al. 1976).

    Runs a reference and a perturbed copy of the system in lockstep through
    the :class:`~tsdynamics.families.System` protocol — no Jacobian needed, so it
    works for any ODE or map (including ones with non-smooth right-hand
    sides).  Not available for DDEs (their state cannot be ``set_state``-ed);
    use ``DelaySystem.lyapunov_spectrum`` instead.

    Parameters
    ----------
    system : System
        ODE or map.
    d0 : float
        Perturbation size restored at every rescaling.
    n : int
        Number of rescaling cycles (more → better averaging).
    steps_per : int
        Protocol steps between rescalings.
    dt : float, optional
        Step size for continuous systems (default: the system's step default).
    transient : int
        Protocol steps discarded before measuring.
    ic : array-like, optional
        Initial condition for the reference trajectory.
    seed : int, optional
        Seed for the random perturbation direction.

    Returns
    -------
    float
        Estimated maximal exponent (per unit time / per iteration).

    Examples
    --------
    >>> max_lyapunov(Lorenz(ic=[1.0, 1.0, 1.0]), dt=0.05)   # ≈ 0.91
    """
    if isinstance(system, DelaySystem):
        raise NotImplementedError(
            "max_lyapunov needs set_state, which delay systems cannot support — "
            "use DelaySystem.lyapunov_spectrum (the engine estimator) instead."
        )
    if system.is_discrete and dt is not None:
        raise ValueError(
            "dt has no meaning for discrete maps — omit it (every step is one iteration)."
        )

    rng = np.random.default_rng(seed)
    ref = system.copy()
    ref.reinit(ic)
    for _ in range(transient):
        ref.step(dt)

    pert = system.copy()
    direction = rng.normal(size=system.dim)
    direction *= d0 / np.linalg.norm(direction)
    pert.reinit(ref.state() + direction)

    t_start = ref.time()
    log_sum = 0.0
    for _ in range(n):
        for _ in range(steps_per):
            ref.step(dt)
            pert.step(dt)
        delta = pert.state() - ref.state()
        d = float(np.linalg.norm(delta))
        if d == 0.0 or not np.isfinite(d):
            raise RuntimeError(
                "max_lyapunov: trajectories collapsed or diverged — "
                "try a larger d0 or smaller steps_per."
            )
        log_sum += np.log(d / d0)
        pert.set_state(ref.state() + (d0 / d) * delta)

    if system.is_discrete:
        elapsed = float(n * steps_per)
    else:
        # Normalize by the *actual* elapsed integration time, read from the
        # reference trajectory's clock — robust to whatever per-step advance the
        # system makes when ``dt`` is ``None`` (built-in flows step by their own
        # ``_default_step_dt``; a continuous ``WrappedSystem`` steps by its
        # ``default_dt``). Guessing a step-size attribute name silently rescales
        # the exponent whenever the guess misses the real per-step advance.
        elapsed = float(ref.time() - t_start)
        if elapsed <= 0.0 or not np.isfinite(elapsed):
            raise RuntimeError(
                "max_lyapunov: the reference clock did not advance — a continuous "
                "system must report elapsed time through time(); pass an explicit dt."
            )
    return log_sum / elapsed


# Self-register the headline Lyapunov quantifiers (D4 / §4e: in-tree analyses
# register from their own subpackage).  Idempotent across re-imports — `register`
# keeps the same object under the same name.
for _name, _fn, _meta in (
    ("lyapunov_spectrum", lyapunov_spectrum, {"needs": "system", "family": "lyapunov"}),
    ("max_lyapunov", max_lyapunov, {"needs": "system", "family": "lyapunov"}),
    ("lyapunov_from_data", lyapunov_from_data, {"needs": "series", "family": "lyapunov"}),
    ("kaplan_yorke_dimension", kaplan_yorke_dimension, {"needs": "spectrum", "family": "lyapunov"}),
):
    _registry.analyses.register(_name, _fn, **_meta)
del _name, _fn, _meta


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
