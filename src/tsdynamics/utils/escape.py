"""Did this orbit leave the building?  The one escape test, shared by every layer.

A run that blows up *without* reaching the engine's hard ``1e150`` guard comes
back as an ordinary, finite, entirely meaningless trajectory: measured, a Chua
run from ``ic=[500, 0, 0]`` returns ``max|y| = 1.6e9`` with no exception and
``isnan`` all ``False``, and every downstream answer computed from it — a
Poincaré section reporting "300 crossings", a Lyapunov spectrum reporting
``λ = [0.3057, 0.3054, -6.068]`` — is typographically identical to an honest
one.

This module is the single definition of "escaped", so the trajectory repr, the
section repr and the Lyapunov verdict cannot drift apart about it.  It lives in
:mod:`tsdynamics.utils` — the leaf package — because ``data``, ``families``,
``derived`` and ``analysis`` all have to agree on the answer.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

__all__ = ["ESCAPE_GROWTH", "ESCAPE_SCALE", "Unbounded", "detect_unbounded", "escaped"]

#: The magnitude an orbit has to reach before it counts as an escape, **and** the
#: factor by which it has to have grown over the run.  Both conditions,
#: deliberately: a model whose states are natively ~1e10 (a population, a stellar
#: mass) is not diverging just because it is large, and a growth factor alone
#: would flag an ordinary transient settling onto an attractor from a tiny start.
#: The engine's own hard guard is ``1e150`` (``√f64::MAX``) — it *raises* there —
#: so this is the band below the raise where a finite, returned and meaningless
#: orbit used to look exactly like an answer.
ESCAPE_SCALE = 1e8
ESCAPE_GROWTH = 1e3


class Unbounded(NamedTuple):
    """What an escaping orbit reached — :attr:`tsdynamics.data.Trajectory.unbounded`.

    Attributes
    ----------
    peak : float
        The largest ``|state|`` (max-norm) anywhere on the trajectory.
    start : float
        ``|state|`` at the first sample — the scale it grew *from*.
    growth : float
        ``peak / max(start, 1)``, how far it climbed.
    first_sample : int
        The first sample at or above :data:`ESCAPE_SCALE`.  (Not ``index``:
        ``tuple.index`` is a method, and a ``NamedTuple`` field may not shadow
        it.)
    non_finite : bool
        Whether any sample is ``inf`` / ``nan``.
    """

    peak: float
    start: float
    growth: float
    first_sample: int
    non_finite: bool

    def __str__(self) -> str:
        """Render the one line that says the orbit left the building."""
        what = "became non-finite" if self.non_finite else f"reached {self.peak:.3g}"
        return (
            f"⚠ unbounded — the state {what} "
            f"({self.growth:.3g}x the start, first at sample {self.first_sample}); "
            f"this orbit is not on an attractor"
        )


def detect_unbounded(y: np.ndarray) -> Unbounded | None:
    """Return the escape record for state block ``y``, or ``None`` if it stayed put.

    Parameters
    ----------
    y : ndarray, shape (T, dim)
        The states of one trajectory, one row per sample.

    Returns
    -------
    Unbounded or None
    """
    block = np.asarray(y)
    if block.size == 0 or block.ndim != 2 or not np.issubdtype(block.dtype, np.number):
        return None
    mag = np.abs(block.astype(np.float64, copy=False))
    finite = np.isfinite(mag)
    non_finite = not bool(finite.all())
    peak = float(mag[finite].max()) if finite.any() else float("inf")
    start = float(mag[0][finite[0]].max()) if finite[0].any() else 0.0
    growth = peak / max(start, 1.0)
    if not non_finite and not (peak >= ESCAPE_SCALE and growth >= ESCAPE_GROWTH):
        return None
    over = np.nonzero(~finite.all(axis=1) | (mag.max(axis=1) >= ESCAPE_SCALE))[0]
    index = int(over[0]) if over.size else int(mag.shape[0])
    return Unbounded(
        peak=peak, start=start, growth=growth, first_sample=index, non_finite=non_finite
    )


def escaped(peak: float | None) -> bool:
    """Whether a recorded state magnitude is past :data:`ESCAPE_SCALE`.

    The one-number form, for a caller that holds only the landed state (a
    Lyapunov estimator) rather than the whole orbit.  ``None`` — nothing was
    recorded — is *not* an escape: a family that cannot answer says nothing.
    """
    if peak is None:
        return False
    value = float(peak)
    return not np.isfinite(value) or value >= ESCAPE_SCALE
