"""Output-grid helpers shared by every family and the engine seam.

The one definition of the uniform output grid an ``integrate`` call samples on.
Both the family base classes (:mod:`tsdynamics.families`) and the engine run
layer (:mod:`tsdynamics.engine.run`) build the same grid, so it lives here in
the leaf ``utils`` package (it imports only NumPy) and both layers consume it —
rather than each carrying a byte-identical private copy that could silently
drift apart.
"""

from __future__ import annotations

import numpy as np

__all__ = ["make_output_grid"]


def make_output_grid(t0: float, tf: float, dt: float) -> np.ndarray:
    """Build a uniform output grid from ``t0`` to ``tf`` (inclusive).

    The grid is ``arange(t0, tf, dt)`` with ``tf`` appended when the last
    sample would otherwise fall short of it — so the final time is always
    sampled exactly, regardless of whether ``dt`` divides ``tf - t0``.

    This is the one chokepoint every flow family (ODE / DDE / SDE) and the
    engine run layer build their grid through, so it is also where the two
    silent-footgun horizons are caught early with a domain message: a
    non-positive ``dt`` (which used to surface as a bare ``ZeroDivisionError``
    from this helper) and a window that does not run forward in time (which used
    to yield a one-sample garbage trajectory).

    Parameters
    ----------
    t0, tf : float
        Start and end of the window.
    dt : float
        Output sampling interval.  Must be strictly positive.

    Returns
    -------
    ndarray
        The output times, with ``t_arr[0] == t0`` and ``t_arr[-1] == tf``.

    Raises
    ------
    tsdynamics.errors.InvalidParameterError
        If ``dt`` is not strictly positive, or if ``tf`` is not strictly after
        ``t0`` (an empty / backwards window).  Both subclass :class:`ValueError`,
        so ``except ValueError`` still catches them.

    Examples
    --------
    >>> make_output_grid(0.0, 1.0, 0.5)
    array([0. , 0.5, 1. ])
    """
    from tsdynamics.errors import invalid_value

    if not dt > 0:
        raise invalid_value("dt", dt, rule="must be > 0 (the output sampling interval)")
    if not tf > t0:
        raise invalid_value(
            "final_time",
            tf,
            rule=f"must run forward in time (be > the start time {t0!r})",
            hint="check the sign and that final_time exceeds t0",
        )
    t_arr = np.arange(t0, tf, dt)
    if t_arr.size == 0 or t_arr[-1] < tf - 1e-12:
        t_arr = np.append(t_arr, tf)
    return t_arr


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
