"""
Private helpers for the viz module.

Not part of the public API. Do not import from outside tsdynamics.viz.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes

# ---------------------------------------------------------------------------
# Axes resolution
# ---------------------------------------------------------------------------


def _resolve_ax(
    ax: matplotlib.axes.Axes | None,
    *,
    projection: str | None = None,
    figsize: tuple[float, float] = (6.0, 4.0),
) -> matplotlib.axes.Axes:
    """
    Return *ax* if given, otherwise create a new figure and axes.

    Parameters
    ----------
    ax : Axes or None
        Existing axes to draw into. If None a new figure is created.
    projection : str, optional
        Matplotlib projection string, e.g. ``"3d"``. Ignored when *ax* is given —
        the caller is responsible for passing a compatible axes type.
    figsize : (float, float)
        Figure size used only when creating a new axes.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        subplot_kw = {"projection": projection} if projection else {}
        _, new_ax = plt.subplots(figsize=figsize, subplot_kw=subplot_kw)
        return new_ax
    return ax


# ---------------------------------------------------------------------------
# Axis labelling
# ---------------------------------------------------------------------------


def _label_dims(
    ax: matplotlib.axes.Axes,
    dims: tuple[int, ...],
    prefix: str = "x",
) -> None:
    """
    Set x, y (and optionally z) axis labels from a dims tuple.

    Parameters
    ----------
    ax : Axes
    dims : tuple of int
        Dimension indices; ``dims[0]`` → x-label, ``dims[1]`` → y-label,
        ``dims[2]`` → z-label (only for 3D axes).
    prefix : str
        Label prefix character (default ``"x"``).
    """
    ax.set_xlabel(f"${prefix}_{{{dims[0]}}}$")
    ax.set_ylabel(f"${prefix}_{{{dims[1]}}}$")
    if len(dims) >= 3:
        # Axes3D has set_zlabel; guard for safety
        if hasattr(ax, "set_zlabel"):
            ax.set_zlabel(f"${prefix}_{{{dims[2]}}}$")


# ---------------------------------------------------------------------------
# Forward-seam Protocols (not enforced at runtime yet)
# ---------------------------------------------------------------------------


@runtime_checkable
class _HasLyapunov(Protocol):
    """
    Structural protocol satisfied by DynSys, DynSysDelay, and DynMap.

    When system-instance support is activated in plotters, functions will
    accept ``exponents: np.ndarray | _HasLyapunov`` and auto-compute via
    ``isinstance(arg, _HasLyapunov)``.
    """

    def lyapunov_spectrum(self) -> np.ndarray:  # pragma: no cover
        ...


@runtime_checkable
class _HasTrajectory(Protocol):
    """
    Structural protocol for systems that can produce a ``(t, X)`` trajectory.

    DynSys and DynSysDelay satisfy this via ``.integrate()``.
    DynMap will satisfy it once an ``.integrate()`` alias for ``.iterate()``
    is added to ``map_base.py``.
    """

    def integrate(self, **kwargs) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
        ...
