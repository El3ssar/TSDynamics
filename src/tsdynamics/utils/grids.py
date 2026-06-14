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

    Parameters
    ----------
    t0, tf : float
        Start and end of the window.
    dt : float
        Output sampling interval.

    Returns
    -------
    ndarray
        The output times, with ``t_arr[0] == t0`` and ``t_arr[-1] == tf``.
    """
    t_arr = np.arange(t0, tf, dt)
    if t_arr.size == 0 or t_arr[-1] < tf - 1e-12:
        t_arr = np.append(t_arr, tf)
    return t_arr
