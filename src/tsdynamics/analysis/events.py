"""
Event & section detection (M2).

Locate the times at which a scalar event function ``g(t, y(t))`` crosses zero
along a discrete trajectory, with sign discipline and bracketed sub-sample
refinement. This is the primitive backing Poincaré sections, return maps,
threshold crossings, transit times, and a few flavours of period detection.

The public surface is intentionally small:

- :class:`EventCondition` — the protocol every condition implements.
- :class:`Plane`, :class:`LinearPlane`, :class:`Threshold`,
  :class:`LocalExtremum`, :class:`Custom` — the built-in conditions.
- :class:`EventResult` — refined-event container.
- :func:`detect_events` — driver that dispatches to the condition.

Detection algorithm for the zero-crossing conditions (Plane, LinearPlane,
Threshold, Custom):

1. Evaluate ``g_k = condition.evaluate(t_k, y_k)`` for each sample.
2. Find sign-change indices ``k`` where ``(g_k, g_{k+1})`` brackets a zero
   *and* the sign change matches ``direction``.
3. Build a cubic Hermite interpolant of ``y(t)`` between samples ``k`` and
   ``k+1`` using slopes from central differences (``np.gradient``).
4. Find the refined root of ``s ↦ condition.evaluate(t_a + s·Δt, H(s))`` on
   ``[0, 1]`` via Brent's method (falling back to the nearer endpoint if
   the bracket has degenerated to a single boundary zero).

:class:`LocalExtremum` uses the cubic Hermite *derivative* (a quadratic in
``s``) to refine — see its docstring.

After N2 (true dense output from the Rust ODE stepper) the public API stays
the same but the cubic Hermite interpolant will be replaced by the
integrator's native dense output for higher accuracy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from scipy.optimize import brentq

from ._registry import trajectory_op

Direction = Literal["up", "down", "either"]
_DIRECTIONS: tuple[str, ...] = ("up", "down", "either")


# ---------------------------------------------------------------------------
# EventResult
# ---------------------------------------------------------------------------


@dataclass
class EventResult:
    """
    Refined events located along a trajectory.

    Attributes
    ----------
    t : ndarray, shape (K,)
        Refined event times, monotonically increasing.
    y : ndarray, shape (K, dim)
        Refined state at each event (cubic-Hermite interpolated between
        samples — exact at the samples).
    indices : ndarray, shape (K,)
        For each event, the index ``k`` of the sample immediately *before*
        the crossing in the source trajectory.  ``t[k] <= event_t < t[k+1]``.
    direction : ndarray, shape (K,)
        ``+1`` for an upward crossing, ``-1`` for a downward crossing,
        ``0`` for a degenerate bracket (e.g. boundary zero).

    The class also unpacks as a ``(t, y)`` tuple to mirror :class:`Trajectory`.
    """

    t: np.ndarray
    y: np.ndarray
    indices: np.ndarray
    direction: np.ndarray
    condition: EventCondition | None = field(default=None, repr=False)

    def __iter__(self):
        return iter((self.t, self.y))

    def __len__(self) -> int:
        return int(self.t.size)

    def __repr__(self) -> str:
        return (
            f"EventResult(n={len(self)}, t=[{self.t[0]:.4g}, {self.t[-1]:.4g}])"
            if len(self)
            else "EventResult(n=0)"
        )


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class EventCondition(Protocol):
    """
    Protocol every event condition implements.

    Conditions used with :func:`detect_events` only need to expose a
    ``direction`` attribute and a ``detect(t, y, *, rtol)`` method.  The
    built-in zero-crossing conditions inherit a default ``detect`` from
    :class:`_ZeroCrossingCondition`; users adding a custom condition can
    either inherit from that or implement ``detect`` directly.
    """

    direction: Direction

    def detect(  # noqa: D102 - protocol stub
        self,
        t: np.ndarray,
        y: np.ndarray,
        *,
        rtol: float = 1e-8,
    ) -> EventResult: ...


# ---------------------------------------------------------------------------
# Cubic Hermite helpers
# ---------------------------------------------------------------------------


def _hermite_state(
    s: float,
    y_a: np.ndarray,
    y_b: np.ndarray,
    m_a: np.ndarray,
    m_b: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Cubic Hermite interpolation of ``y(t)`` on a single bracket.

    ``s = (t - t_a) / (t_b - t_a)`` is the local parameter in ``[0, 1]``;
    ``m_a``, ``m_b`` are the trajectory's time-slopes at ``t_a`` and ``t_b``.
    """
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
    """Time-derivative of the cubic Hermite interpolant (a quadratic in ``s``)."""
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

    A bracket ``k → k+1`` is "up" iff ``g[k] < 0`` and ``g[k+1] >= 0``, and
    "down" iff ``g[k] > 0`` and ``g[k+1] <= 0``.  The asymmetric ``< / >=``
    means a sample with ``g == 0`` closes the previous bracket rather than
    opening a new one, which avoids double-counting when ``g`` passes
    through zero exactly at a sample.  The exception is the very first
    sample (``g[0] == 0``): no previous bracket exists there, so we count
    it explicitly.

    ``crossing_direction`` has values ``+1`` for upward and ``-1`` for downward.
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
# Base classes for the built-in conditions
# ---------------------------------------------------------------------------


class _ZeroCrossingCondition:
    """
    Mixin for conditions defined as ``g(t, y) = 0``.

    Subclasses implement :meth:`evaluate` for a single sample and (optionally)
    :meth:`evaluate_along` for vectorised evaluation along the whole trajectory.
    The base :meth:`detect` then finds sign-change brackets and refines each
    via Brent root-finding on the cubic Hermite interpolant of ``y(t)``.
    """

    direction: Direction = "either"

    def evaluate(self, t: float, y: np.ndarray) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def evaluate_along(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Vectorised default — loops over samples; subclasses can override."""
        return np.array(
            [self.evaluate(float(tk), yk) for tk, yk in zip(t, y, strict=True)],
            dtype=float,
        )

    def detect(self, t: np.ndarray, y: np.ndarray, *, rtol: float = 1e-8) -> EventResult:
        """Find sign-change brackets of ``evaluate_along`` and refine each."""
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim != 2:
            raise ValueError(f"y must be 2-D (T, dim), got shape {y.shape}")
        if t.shape[0] != y.shape[0]:
            raise ValueError(f"t and y first axis mismatch: {t.shape[0]} vs {y.shape[0]}")
        if t.shape[0] < 2:
            return EventResult(
                np.empty(0, dtype=float),
                np.empty((0, y.shape[1]), dtype=float),
                np.empty(0, dtype=int),
                np.empty(0, dtype=int),
                self,
            )

        g = self.evaluate_along(t, y).astype(float, copy=False)
        idx, direc = _bracket_mask(g, self.direction)
        if idx.size == 0:
            return EventResult(
                np.empty(0, dtype=float),
                np.empty((0, y.shape[1]), dtype=float),
                idx,
                direc,
                self,
            )

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

            t_star = t_a + s_star * dt
            y_star = _hermite_state(s_star, y_a, y_b, m_a, m_b, dt)
            ts.append(t_star)
            ys.append(y_star)

        if not ts:
            return EventResult(
                np.empty(0, dtype=float),
                np.empty((0, y.shape[1]), dtype=float),
                np.empty(0, dtype=int),
                np.empty(0, dtype=int),
                self,
            )
        return EventResult(
            np.asarray(ts, dtype=float),
            np.asarray(ys, dtype=float),
            np.asarray(idx, dtype=int),
            np.asarray(direc, dtype=int),
            self,
        )


# ---------------------------------------------------------------------------
# Built-in conditions
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
        Crossing direction filter.

    Examples
    --------
    >>> p = Plane(axis=2, value=27.0, direction="up")
    >>> p.evaluate(0.0, np.array([1.0, 2.0, 28.0]))
    1.0
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
        Plane normal vector.  Need not be unit-length.
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


@dataclass
class Threshold(_ZeroCrossingCondition):
    """
    Scalar threshold on one component: ``y[component] - value == 0``.

    Equivalent to :class:`Plane` with a different default ``direction``
    (``"up"`` rather than ``"either"``) — kept as a distinct class because
    threshold-style queries usually want a one-sided crossing.

    Examples
    --------
    >>> th = Threshold(component=0, value=0.5)
    >>> th.direction
    'up'
    """

    component: int
    value: float
    direction: Direction = "up"

    def evaluate(self, t: float, y: np.ndarray) -> float:
        """``y[component] - value``."""
        return float(y[self.component] - self.value)

    def evaluate_along(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`evaluate`."""
        return y[:, self.component] - self.value


@dataclass
class Custom(_ZeroCrossingCondition):
    """
    Arbitrary user-supplied scalar function ``fn(t, y) -> float``.

    Use this for anything that doesn't fit the built-in plane / threshold
    moulds — e.g. ``r(t) = ||y|| - r0`` for radial crossings.

    Parameters
    ----------
    fn : callable
        ``fn(t, y) -> float``.  Should be vectorisable but the default
        :meth:`evaluate_along` loops over samples; subclass and override if
        speed matters.
    direction : {"up", "down", "either"}, default "either"

    Examples
    --------
    >>> radial = Custom(lambda t, y: np.linalg.norm(y) - 1.0)
    >>> radial.evaluate(0.0, np.array([0.6, 0.8]))
    0.0
    """

    fn: Callable[[float, np.ndarray], float]
    direction: Direction = "either"

    def evaluate(self, t: float, y: np.ndarray) -> float:
        """Delegate to the user-supplied ``fn``."""
        return float(self.fn(t, y))


# ---------------------------------------------------------------------------
# LocalExtremum: detection via sign change of dy/dt
# ---------------------------------------------------------------------------


@dataclass
class LocalExtremum:
    """
    Local maxima or minima of one state component, refined via Hermite.

    For each sample ``k`` where the discrete derivative
    ``d_k = (np.gradient(y[:, component], t))_k`` changes sign in the
    requested direction, refine the extremum's time by solving the
    cubic Hermite *derivative* (a quadratic in the local parameter ``s``)
    for its root on ``[0, 1]``.  The full state at that time is then
    obtained from the cubic Hermite interpolant.

    Parameters
    ----------
    component : int
        Which state component to look for extrema in.
    kind : {"max", "min"}, default "max"
    direction : {"up", "down", "either"}
        Implied by ``kind`` and not user-settable; here for
        :class:`EventCondition` protocol compliance.

    Examples
    --------
    >>> ex = LocalExtremum(component=0, kind="max")
    >>> ex.direction
    'down'
    """

    component: int
    kind: Literal["max", "min"] = "max"

    @property
    def direction(self) -> Direction:
        """Crossing direction of ``dy/dt`` implied by ``kind``.

        For a maximum ``dy/dt`` goes ``+ → −`` (downward); for a minimum it
        goes ``− → +`` (upward).  Exposed for protocol compliance and
        introspection.
        """
        return "down" if self.kind == "max" else "up"

    def detect(self, t: np.ndarray, y: np.ndarray, *, rtol: float = 1e-8) -> EventResult:
        """Locate extrema via sign change of ``dy[component]/dt``, refined via Hermite."""
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim != 2:
            raise ValueError(f"y must be 2-D (T, dim), got shape {y.shape}")
        if t.shape[0] != y.shape[0]:
            raise ValueError(f"t and y first axis mismatch: {t.shape[0]} vs {y.shape[0]}")
        if self.kind not in ("max", "min"):
            raise ValueError(f"LocalExtremum.kind must be 'max' or 'min', got {self.kind!r}")
        if t.shape[0] < 3:
            return EventResult(
                np.empty(0, dtype=float),
                np.empty((0, y.shape[1]), dtype=float),
                np.empty(0, dtype=int),
                np.empty(0, dtype=int),
                self,
            )

        comp = self.component
        dim = y.shape[1]
        if not (-dim <= comp < dim):
            raise IndexError(f"LocalExtremum: component {comp} out of range for dim {dim}")

        slopes = _compute_slopes(t, y)
        d = slopes[:, comp]
        idx, _ = _bracket_mask(d, self.direction)
        if idx.size == 0:
            return EventResult(
                np.empty(0, dtype=float),
                np.empty((0, dim), dtype=float),
                idx,
                np.empty(0, dtype=int),
                self,
            )

        ts: list[float] = []
        ys: list[np.ndarray] = []
        kept_idx: list[int] = []
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

            def dydt_of_s(
                s: float,
                dt: float = dt,
                y_a: np.ndarray = y_a,
                y_b: np.ndarray = y_b,
                m_a: np.ndarray = m_a,
                m_b: np.ndarray = m_b,
                comp: int = comp,
            ) -> float:
                return float(_hermite_slope(s, y_a, y_b, m_a, m_b, dt)[comp])

            d_lo = dydt_of_s(0.0)
            d_hi = dydt_of_s(1.0)
            if d_lo == 0.0:
                s_star = 0.0
            elif d_hi == 0.0:
                s_star = 1.0
            elif d_lo * d_hi > 0.0:
                # Bracket has degenerated; pick the side closer to zero.
                s_star = 0.0 if abs(d_lo) < abs(d_hi) else 1.0
            else:
                xtol = max(rtol * dt, 1e-15)
                s_star = brentq(dydt_of_s, 0.0, 1.0, xtol=xtol, rtol=rtol, maxiter=100)

            t_star = t_a + s_star * dt
            y_star = _hermite_state(s_star, y_a, y_b, m_a, m_b, dt)
            ts.append(t_star)
            ys.append(y_star)
            kept_idx.append(int(k))

        if not ts:
            return EventResult(
                np.empty(0, dtype=float),
                np.empty((0, dim), dtype=float),
                np.empty(0, dtype=int),
                np.empty(0, dtype=int),
                self,
            )
        # All extrema of one kind share the same crossing direction by definition.
        sign = -1 if self.kind == "max" else 1
        return EventResult(
            np.asarray(ts, dtype=float),
            np.asarray(ys, dtype=float),
            np.asarray(kept_idx, dtype=int),
            np.full(len(ts), sign, dtype=int),
            self,
        )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@trajectory_op(returns="passthrough")
def detect_events(
    t: np.ndarray,
    y: np.ndarray,
    condition: EventCondition,
    *,
    rtol: float = 1e-8,
) -> EventResult:
    """
    Locate refined events of ``condition`` along the trajectory.

    Available equivalently as a free function and as a
    :class:`Trajectory` method — pick whichever reads better at the call
    site::

        detect_events(traj, condition)
        detect_events((t, y), condition)
        detect_events(t, y, condition)
        traj.detect_events(condition)

    Parameters
    ----------
    condition : EventCondition
        Any of :class:`Plane`, :class:`LinearPlane`, :class:`Threshold`,
        :class:`LocalExtremum`, :class:`Custom`, or a user-supplied object
        implementing the :class:`EventCondition` protocol.
    rtol : float, default ``1e-8``
        Relative tolerance for the bracketed root-finder.  The absolute
        time tolerance is ``rtol * (t_{k+1} - t_k)``.

    Returns
    -------
    EventResult

    Examples
    --------
    >>> t = np.linspace(0.0, 2 * np.pi, 1000)
    >>> y = np.column_stack([np.sin(t), np.cos(t)])
    >>> evs = detect_events(t, y, Threshold(component=0, value=0.0, direction="up"))
    >>> evs.t.size
    1
    """
    if not hasattr(condition, "detect"):
        raise TypeError(
            f"condition must implement .detect(t, y, rtol=...); got {type(condition).__name__}"
        )
    return condition.detect(t, y, rtol=rtol)


__all__ = [
    "Custom",
    "Direction",
    "EventCondition",
    "EventResult",
    "LinearPlane",
    "LocalExtremum",
    "Plane",
    "Threshold",
    "detect_events",
]
