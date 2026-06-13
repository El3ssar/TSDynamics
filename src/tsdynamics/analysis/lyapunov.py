"""Lyapunov-based quantifiers: spectra, maximal exponent, Kaplan–Yorke dimension."""

from __future__ import annotations

from typing import Any

import numpy as np

from tsdynamics.base import DelaySystem

__all__ = ["kaplan_yorke_dimension", "lyapunov_spectrum", "max_lyapunov"]


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


def lyapunov_spectrum(sys: Any, **kwargs: Any) -> np.ndarray:
    """
    Lyapunov spectrum of any system — uniform entry point.

    Dispatches to the family implementation (QR tangent dynamics for maps,
    compiled variational equations for ODEs, ``jitcdde_lyap`` for DDEs).
    Keyword arguments are forwarded (``steps=`` for maps; ``final_time=``,
    ``dt=``, ``burn_in=``, ... for flows).
    """
    method = getattr(sys, "lyapunov_spectrum", None)
    if method is None:
        raise TypeError(
            f"{type(sys).__name__} has no lyapunov_spectrum implementation. "
            f"For derived wrappers, compute the spectrum on the underlying system."
        )
    return method(**kwargs)


def max_lyapunov(
    sys: Any,
    *,
    d0: float = 1e-9,
    n_rescale: int = 400,
    steps_per: int = 5,
    dt: float | None = None,
    transient: int = 500,
    ic: Any | None = None,
    seed: int | None = None,
) -> float:
    """
    Maximal Lyapunov exponent by two-trajectory rescaling (Benettin et al. 1976).

    Runs a reference and a perturbed copy of the system in lockstep through
    the :class:`~tsdynamics.base.System` protocol — no Jacobian needed, so it
    works for any ODE or map (including ones with non-smooth right-hand
    sides).  Not available for DDEs (their state cannot be ``set_state``-ed);
    use ``DelaySystem.lyapunov_spectrum`` instead.

    Parameters
    ----------
    sys : System
        ODE or map.
    d0 : float
        Perturbation size restored at every rescaling.
    n_rescale : int
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
    if isinstance(sys, DelaySystem):
        raise NotImplementedError(
            "max_lyapunov needs set_state, which delay systems cannot support — "
            "use DelaySystem.lyapunov_spectrum (jitcdde_lyap) instead."
        )

    rng = np.random.default_rng(seed)
    ref = sys.copy()
    ref.reinit(ic)
    for _ in range(transient):
        ref.step(dt)

    pert = sys.copy()
    direction = rng.normal(size=sys.dim)
    direction *= d0 / np.linalg.norm(direction)
    pert.reinit(ref.state() + direction)

    log_sum = 0.0
    for _ in range(n_rescale):
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

    if sys.is_discrete:
        elapsed_per_cycle = float(steps_per)
    else:
        step_dt = dt if dt is not None else getattr(sys, "_default_step_dt", 0.01)
        elapsed_per_cycle = steps_per * float(step_dt)
    return log_sum / (n_rescale * elapsed_per_cycle)
