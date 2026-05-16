"""
Event-driven analysis: zero-crossings, Poincaré sections, and return maps.

Three layers, one algorithm, one return type
--------------------------------------------

Every operation in this module reduces to:

1. Pick a scalar condition ``g(t, y(t))`` along a trajectory.
2. Find the times where ``g = 0`` (with sign discipline).
3. Refine each crossing sub-sample via cubic Hermite + Brent root-finding.
4. Return a :class:`~tsdynamics.base.Trajectory` whose ``t`` are the
   refined crossing times and whose ``y`` is the (possibly reduced)
   state at those crossings.

The three public ops differ only in **what they put in the result**:

- :func:`detect_events` — ``y`` is the full state at each crossing.
- :func:`poincare_section` — same as :func:`detect_events`, but defaults
  to ``direction="up"`` (canonical Poincaré).
- :func:`return_map` — ``y`` is reduced to a single scalar observable
  per crossing.  Plot the standard return map with
  ``rmap.to_dataspec(kind="return_map", step=1)``.

All three accept a condition object **or** the shortcut kwargs
``axis=``, ``value=``, ``direction=`` (which build a :class:`Plane` for
you) **or** a bare callable ``fn(t, y) -> float``.  Pick whichever reads
better at the call site::

    traj.detect_events(axis=2, value=27.0, direction="up")
    traj.poincare_section(Plane(axis=2, value=27.0))
    traj.return_map(axis=2, value=27.0, observable=0)
    traj.detect_events(lambda t, y: np.linalg.norm(y) - 1.0)

Built-in conditions
-------------------

- :class:`Plane`        — axis-aligned hyperplane ``y[axis] == value``.
- :class:`LinearPlane`  — general hyperplane ``<normal, y> == offset``.
- :class:`EventCondition` (protocol) — anything with ``direction`` and a
  ``detect(t, y, *, rtol)`` method.

Bracket convention
------------------

A bracket ``k → k+1`` is "up" iff ``g[k] < 0`` and ``g[k+1] >= 0``,
"down" iff ``g[k] > 0`` and ``g[k+1] <= 0``.  The asymmetric ``< / >=``
means a sample with ``g == 0`` closes the previous bracket rather than
opening a new one — this avoids double-counting when ``g`` passes
through zero exactly at a sample.  The first sample is special-cased
(no previous bracket).

After N2 the cubic-Hermite interpolant will be replaced by the
integrator's native dense output; the public API does not change.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from scipy.optimize import brentq

from ._registry import trajectory_op

Direction = Literal["up", "down", "either"]
_DIRECTIONS: tuple[str, ...] = ("up", "down", "either")


# ---------------------------------------------------------------------------
# EventCondition protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class EventCondition(Protocol):
    """
    Anything that knows how to find events along a trajectory.

    Conditions only need to expose a ``direction`` attribute and a
    ``detect(t, y, *, rtol)`` method that returns refined crossings as
    ``(t_events, y_events)``.  The built-in :class:`Plane` and
    :class:`LinearPlane` inherit a default ``detect`` from
    :class:`_ZeroCrossingCondition`; custom conditions can either subclass
    it or implement ``detect`` directly.
    """

    direction: Direction

    def detect(  # noqa: D102 - protocol stub
        self,
        t: np.ndarray,
        y: np.ndarray,
        *,
        rtol: float = 1e-8,
    ) -> tuple[np.ndarray, np.ndarray]: ...


# ---------------------------------------------------------------------------
# Cubic-Hermite helpers
# ---------------------------------------------------------------------------


def _hermite_state(
    s: float,
    y_a: np.ndarray,
    y_b: np.ndarray,
    m_a: np.ndarray,
    m_b: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Cubic-Hermite interpolation of ``y(t)`` on a single bracket."""
    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    return h00 * y_a + h10 * dt * m_a + h01 * y_b + h11 * dt * m_b


def _hermite_slope(
    s: float,
    y_a: np.ndarray,
    y_b: np.ndarray,
    m_a: np.ndarray,
    m_b: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Time-derivative of the cubic-Hermite interpolant (quadratic in ``s``)."""
    h00p = 6.0 * s * s - 6.0 * s
    h10p = 3.0 * s * s - 4.0 * s + 1.0
    h01p = -6.0 * s * s + 6.0 * s
    h11p = 3.0 * s * s - 2.0 * s
    return (h00p * y_a + h01p * y_b) / dt + h10p * m_a + h11p * m_b


def _compute_slopes(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``dy/dt`` via central differences on the interior, one-sided at edges."""
    return np.gradient(y, t, axis=0)


# ---------------------------------------------------------------------------
# Sign-change bracket detection
# ---------------------------------------------------------------------------


def _bracket_mask(g: np.ndarray, direction: Direction) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(bracket_indices, crossing_direction)`` for sign-change brackets.

    See module docstring for the asymmetric ``< / >=`` convention.
    """
    if direction not in _DIRECTIONS:
        raise ValueError(f"direction must be one of {_DIRECTIONS}, got {direction!r}")
    g = np.asarray(g, dtype=float)
    if g.size < 2:
        empty = np.empty(0, dtype=int)
        return empty, empty
    g_lo = g[:-1]
    g_hi = g[1:]
    up = (g_lo < 0.0) & (g_hi >= 0.0)
    down = (g_lo > 0.0) & (g_hi <= 0.0)
    if g[0] == 0.0:
        if g[1] > 0.0:
            up[0] = True
        elif g[1] < 0.0:
            down[0] = True
    if direction == "up":
        mask = up
    elif direction == "down":
        mask = down
    else:
        mask = up | down
    idx = np.flatnonzero(mask)
    direc = np.where(up[idx], 1, np.where(down[idx], -1, 0)).astype(int)
    return idx, direc


# ---------------------------------------------------------------------------
# _ZeroCrossingCondition base class (shared algorithm for all conditions)
# ---------------------------------------------------------------------------


class _ZeroCrossingCondition:
    """
    Mixin for conditions defined as ``g(t, y) = 0``.

    Subclasses implement :meth:`evaluate` for a single sample and
    (optionally) :meth:`evaluate_along` for vectorised evaluation.  The
    base :meth:`detect` finds sign-change brackets and refines each via
    Brent root-finding on the cubic-Hermite interpolant of ``y(t)``.
    """

    direction: Direction = "either"

    def evaluate(self, t: float, y: np.ndarray) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def evaluate_along(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Vectorised default — loops over samples; subclasses may override."""
        return np.array(
            [self.evaluate(float(tk), yk) for tk, yk in zip(t, y, strict=True)],
            dtype=float,
        )

    def detect(
        self,
        t: np.ndarray,
        y: np.ndarray,
        *,
        rtol: float = 1e-8,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find sign-change brackets of ``evaluate_along`` and refine each."""
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim != 2:
            raise ValueError(f"y must be 2-D (T, dim), got shape {y.shape}")
        if t.shape[0] != y.shape[0]:
            raise ValueError(f"t and y first axis mismatch: {t.shape[0]} vs {y.shape[0]}")
        if t.shape[0] < 2:
            return np.empty(0, dtype=float), np.empty((0, y.shape[1]), dtype=float)

        g = self.evaluate_along(t, y).astype(float, copy=False)
        idx, _ = _bracket_mask(g, self.direction)
        if idx.size == 0:
            return np.empty(0, dtype=float), np.empty((0, y.shape[1]), dtype=float)

        slopes = _compute_slopes(t, y)
        ts: list[float] = []
        ys: list[np.ndarray] = []
        for k in idx:
            t_a = float(t[k])
            t_b = float(t[k + 1])
            dt = t_b - t_a
            if dt <= 0.0:
                continue
            y_a = y[k]
            y_b = y[k + 1]
            m_a = slopes[k]
            m_b = slopes[k + 1]

            def g_of_s(
                s: float,
                t_a: float = t_a,
                dt: float = dt,
                y_a: np.ndarray = y_a,
                y_b: np.ndarray = y_b,
                m_a: np.ndarray = m_a,
                m_b: np.ndarray = m_b,
            ) -> float:
                y_s = _hermite_state(s, y_a, y_b, m_a, m_b, dt)
                return float(self.evaluate(t_a + s * dt, y_s))

            g_lo = g_of_s(0.0)
            g_hi = g_of_s(1.0)
            if g_lo == 0.0:
                s_star = 0.0
            elif g_hi == 0.0:
                s_star = 1.0
            elif g_lo * g_hi > 0.0:
                # Endpoints disagree with the sampled bracket — pick the side
                # closer to zero so we still return *something* sensible.
                s_star = 0.0 if abs(g_lo) < abs(g_hi) else 1.0
            else:
                xtol = max(rtol * dt, 1e-15)
                s_star = brentq(g_of_s, 0.0, 1.0, xtol=xtol, rtol=rtol, maxiter=100)

            ts.append(t_a + s_star * dt)
            ys.append(_hermite_state(s_star, y_a, y_b, m_a, m_b, dt))

        if not ts:
            return np.empty(0, dtype=float), np.empty((0, y.shape[1]), dtype=float)
        return np.asarray(ts, dtype=float), np.asarray(ys, dtype=float)


# ---------------------------------------------------------------------------
# Public conditions
# ---------------------------------------------------------------------------


@dataclass
class Plane(_ZeroCrossingCondition):
    """
    Axis-aligned hyperplane ``y[axis] == value``.

    Parameters
    ----------
    axis : int
        State-space component defining the plane.
    value : float
        Hyperplane offset on that axis.
    direction : {"up", "down", "either"}, default "either"
        Crossing-direction filter.

    Examples
    --------
    >>> p = Plane(axis=2, value=27.0, direction="up")
    >>> p.evaluate(0.0, np.array([1.0, 2.0, 28.0]))
    1.0

    Notes
    -----
    For one-axis threshold queries, prefer the shortcut form on the op:
    ``traj.detect_events(axis=0, value=0.5, direction="up")`` builds the
    :class:`Plane` for you, without an import.
    """

    axis: int
    value: float
    direction: Direction = "either"

    def evaluate(self, t: float, y: np.ndarray) -> float:
        """Signed distance from ``y`` to the hyperplane along ``axis``."""
        return float(y[self.axis] - self.value)

    def evaluate_along(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`evaluate`."""
        return y[:, self.axis] - self.value


@dataclass
class LinearPlane(_ZeroCrossingCondition):
    """
    General hyperplane ``<normal, y> == offset``.

    Parameters
    ----------
    normal : ndarray, shape (dim,)
        Plane normal.  Need not be unit-length.
    offset : float, default 0.0
        Plane offset along the normal direction.
    direction : {"up", "down", "either"}, default "either"

    Examples
    --------
    >>> lp = LinearPlane(normal=np.array([1.0, 1.0, 0.0]), offset=2.0)
    >>> lp.evaluate(0.0, np.array([1.5, 1.5, 9.0]))
    1.0
    """

    normal: np.ndarray
    offset: float = 0.0
    direction: Direction = "either"

    def __post_init__(self) -> None:
        """Validate the normal vector."""
        self.normal = np.asarray(self.normal, dtype=float).ravel()
        if self.normal.size == 0:
            raise ValueError("LinearPlane: 'normal' must be non-empty")
        if not np.any(self.normal):
            raise ValueError("LinearPlane: 'normal' must not be the zero vector")

    def evaluate(self, t: float, y: np.ndarray) -> float:
        """Signed distance from ``y`` to the hyperplane (un-normalised)."""
        return float(np.dot(self.normal, y) - self.offset)

    def evaluate_along(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`evaluate`."""
        return y @ self.normal - self.offset


# Internal wrapper for users passing a bare callable instead of an
# EventCondition object.  Not exported.
class _CallableCondition(_ZeroCrossingCondition):
    def __init__(self, fn: Callable[[float, np.ndarray], float], direction: Direction) -> None:
        self.fn = fn
        self.direction = direction

    def evaluate(self, t: float, y: np.ndarray) -> float:
        """Delegate to the user-supplied callable."""
        return float(self.fn(t, y))


# ---------------------------------------------------------------------------
# Condition resolution: condition object | callable | shortcut kwargs
# ---------------------------------------------------------------------------


def _resolve_condition(
    condition: EventCondition | Callable[[float, np.ndarray], float] | None,
    op_default_direction: Direction | None,
    kwargs: dict[str, Any],
) -> tuple[EventCondition, dict[str, Any]]:
    """
    Build an :class:`EventCondition` from one of three call styles.

    Returns ``(condition, leftover_kwargs)``.  Pops the shortcut kwargs
    (``axis``, ``value``, ``direction``, ``normal``, ``offset``) and
    leaves any caller-specific kwargs (``rtol`` etc.) in place.

    ``op_default_direction`` is the per-op default — ``"either"`` for
    :func:`detect_events`, ``"up"`` for :func:`poincare_section` and
    :func:`return_map` (canonical Poincaré).  Pass ``None`` here to
    leave the direction completely up to the caller / condition object.
    """
    axis = kwargs.pop("axis", None)
    value = kwargs.pop("value", None)
    normal = kwargs.pop("normal", None)
    offset = kwargs.pop("offset", 0.0)
    direction = kwargs.pop("direction", op_default_direction)
    has_shortcut = axis is not None or value is not None or normal is not None or "offset" in kwargs

    # Style 1: shortcut kwargs only (no condition arg)
    if condition is None:
        if not has_shortcut:
            raise TypeError(
                "Need a condition (Plane / LinearPlane / callable) or shortcut "
                "kwargs: axis= and value= (axis-aligned), or normal= (linear)."
            )
        if normal is not None:
            return (
                LinearPlane(normal=normal, offset=offset, direction=direction or "either"),
                kwargs,
            )
        if axis is not None and value is not None:
            return (
                Plane(axis=axis, value=value, direction=direction or "either"),
                kwargs,
            )
        raise TypeError(
            "Axis-aligned shortcut needs BOTH axis= and value=; got "
            f"axis={axis!r}, value={value!r}."
        )

    # Style 2: an EventCondition object (anything with .detect)
    if hasattr(condition, "detect"):
        if direction is not None and getattr(condition, "direction", None) != direction:
            # Override the condition's direction for this call.  We rebuild
            # built-in conditions cleanly; for other condition types we
            # honour what the user supplied as-is (their .detect is the
            # source of truth).
            if isinstance(condition, Plane):
                condition = Plane(axis=condition.axis, value=condition.value, direction=direction)
            elif isinstance(condition, LinearPlane):
                condition = LinearPlane(
                    normal=np.asarray(condition.normal, dtype=float).copy(),
                    offset=condition.offset,
                    direction=direction,
                )
        return condition, kwargs

    # Style 3: a bare callable fn(t, y) -> float
    if callable(condition):
        return (
            _CallableCondition(condition, direction=direction or "either"),
            kwargs,
        )

    raise TypeError(
        "condition must be an EventCondition (with .detect), a callable "
        f"fn(t, y) -> float, or None with shortcut kwargs; got {type(condition).__name__}."
    )


# ---------------------------------------------------------------------------
# Public ops
# ---------------------------------------------------------------------------


@trajectory_op(returns="trajectory")
def detect_events(
    t: np.ndarray,
    y: np.ndarray,
    condition: EventCondition | Callable[[float, np.ndarray], float] | None = None,
    *,
    rtol: float = 1e-8,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Locate refined zero-crossings of a condition along the trajectory.

    Three equivalent call styles::

        # 1. Shortcut kwargs (axis-aligned plane)
        traj.detect_events(axis=2, value=27.0, direction="up")

        # 2. Explicit condition object
        traj.detect_events(Plane(axis=2, value=27.0, direction="up"))

        # 3. Bare callable
        traj.detect_events(lambda t, y: np.linalg.norm(y) - 1.0,
                           direction="up")

    Parameters
    ----------
    condition : EventCondition, callable, or None
        See call styles above.  If ``None``, ``axis=`` and ``value=`` (or
        ``normal=`` and ``offset=``) must be supplied as kwargs.
    rtol : float, default ``1e-8``
        Relative tolerance for the bracketed root-finder; the absolute
        time tolerance is ``rtol * (t_{k+1} - t_k)``.
    **kwargs
        Shortcut kwargs (see :func:`_resolve_condition`).

    Returns
    -------
    Trajectory
        ``t`` = refined event times; ``y`` = state at each event,
        cubic-Hermite interpolated between samples.

    Examples
    --------
    >>> t = np.linspace(0.0, 2 * np.pi, 1000)
    >>> y = np.column_stack([np.sin(t), np.cos(t)])
    >>> ev = detect_events((t, y), axis=0, value=0.0, direction="up")
    """
    cond, _ = _resolve_condition(condition, op_default_direction=None, kwargs=kwargs)
    return cond.detect(t, y, rtol=rtol)


@trajectory_op(returns="trajectory")
def poincare_section(
    t: np.ndarray,
    y: np.ndarray,
    condition: EventCondition | Callable[[float, np.ndarray], float] | None = None,
    *,
    rtol: float = 1e-8,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Poincaré section of a trajectory through a hyperplane.

    Canonical Poincaré convention is *upward crossings only* — the
    default ``direction`` is ``"up"`` regardless of what the supplied
    condition object says.  Pass ``direction="either"`` (or
    ``direction=None`` to honour the condition's own direction) to opt
    out.

    Three call styles, same as :func:`detect_events`::

        traj.poincare_section(axis=2, value=27.0)
        traj.poincare_section(Plane(axis=2, value=27.0))
        traj.poincare_section(LinearPlane(normal=[1, 1, 0], offset=0))

    Parameters
    ----------
    condition, **kwargs
        See :func:`detect_events`.
    rtol : float, default ``1e-8``

    Returns
    -------
    Trajectory
        ``t`` = refined crossing times; ``y`` = full state at each
        crossing.  The section axis is *kept* — project it away with
        :meth:`Trajectory.project` if you want a ``(dim - 1)``-dim view.
    """
    cond, _ = _resolve_condition(condition, op_default_direction="up", kwargs=kwargs)
    return cond.detect(t, y, rtol=rtol)


@trajectory_op(returns="trajectory")
def return_map(
    t: np.ndarray,
    y: np.ndarray,
    condition: EventCondition | Callable[[float, np.ndarray], float] | None = None,
    observable: int | Callable[[float, np.ndarray], float] = 0,
    *,
    rtol: float = 1e-8,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a return map from a section by sampling a scalar observable.

    Equivalent to :func:`poincare_section` followed by extracting a
    single scalar from the state at each crossing.  The result is a
    :class:`~tsdynamics.base.Trajectory` whose ``y`` has shape
    ``(K, 1)`` — the observable's value at each successive crossing.

    To plot the canonical "x_{k+1} vs x_k" return map::

        rmap = traj.return_map(axis=2, value=27.0, observable=0)
        spec = rmap.to_dataspec(kind="return_map", step=1)
        plt.scatter(spec["x"], spec["y"])

    Three call styles, same as :func:`detect_events`::

        traj.return_map(axis=2, value=27.0, observable=0)
        traj.return_map(Plane(axis=2, value=27.0), observable=0)
        traj.return_map(lambda t, y: y[0] - y[1], observable=2)

    Parameters
    ----------
    condition, **kwargs
        See :func:`detect_events`.  Canonical direction default is
        ``"up"``.
    observable : int or callable, default ``0``
        Scalar reduction of the state at each crossing.  If ``int``, picks
        a component; if callable, must be ``fn(t, y) -> float``.

    Returns
    -------
    Trajectory
        ``t`` = crossing times; ``y`` = observable values, shape
        ``(K, 1)``.  The "step=N return map" is the windowed pair view
        ``(y[:-N, 0], y[N:, 0])`` — see :meth:`Trajectory.to_dataspec`
        with ``kind="return_map"``.
    """
    cond, _ = _resolve_condition(condition, op_default_direction="up", kwargs=kwargs)
    t_events, y_events = cond.detect(t, y, rtol=rtol)
    obs_vals = _evaluate_observable(observable, t_events, y_events)
    return t_events, obs_vals.reshape(-1, 1)


def _evaluate_observable(
    observable: int | Callable[[float, np.ndarray], float],
    t: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Reduce ``y`` at each crossing to a 1-D scalar series."""
    if isinstance(observable, int | np.integer):
        if y.shape[0] == 0:
            return np.empty(0, dtype=float)
        dim = y.shape[1]
        c = int(observable)
        if not (-dim <= c < dim):
            raise IndexError(f"return_map: observable component {c} out of range for dim {dim}")
        return y[:, c].astype(float, copy=False)
    if callable(observable):
        if y.shape[0] == 0:
            return np.empty(0, dtype=float)
        return np.array(
            [float(observable(float(tk), yk)) for tk, yk in zip(t, y, strict=True)],
            dtype=float,
        )
    raise TypeError(
        f"return_map: observable must be int or callable, got {type(observable).__name__}"
    )


__all__ = [
    "Direction",
    "EventCondition",
    "LinearPlane",
    "Plane",
    "detect_events",
    "poincare_section",
    "return_map",
]
