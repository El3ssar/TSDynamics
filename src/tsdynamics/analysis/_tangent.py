r"""
Shared tangent-dynamics primitives for the analysis subpackages.

The fixed-point/periodic-orbit (**A-FP**) and chaos-indicator (**A-CHAOS**)
streams each need the same self-contained tangent-dynamics building blocks: a
``(step, jac)`` pair for a :class:`~tsdynamics.families.DiscreteMap` (compiled
``_step`` / ``_jacobian``), a ``(rhs, jac)`` pair for a
:class:`~tsdynamics.families.ContinuousSystem` (SymEngine-lambdified numeric RHS
/ Jacobian), and the classic / augmented (state ⊕ fundamental matrix) RK4 steps.

These were previously copy-pasted into both ``fixedpoints/_common.py`` and
``chaos/_common.py`` (drifting independently); they live here once.  Each
subpackage re-exports them under its existing names so callers are unchanged.
All primitives stay engine-free (no tape lowering), so the detectors keep
running in the fast tier.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from tsdynamics.families import ContinuousSystem, DiscreteMap
    from tsdynamics.families.base import ParamSet

# ── state coercion ────────────────────────────────────────────────────────────


def to_native(x: np.ndarray, dim: int) -> float | np.ndarray:
    """Present the state to a compiled ``_step``/``_jacobian`` in its native form.

    One-dimensional maps are written for a scalar argument (``x = X``);
    higher-dimensional maps unpack an array (``x, y = X[0], X[1]``).
    """
    a = np.asarray(x, dtype=float).ravel()
    return float(a[0]) if dim == 1 else a


# ── tangent dynamics: maps ───────────────────────────────────────────────────


def map_fns(
    system: DiscreteMap,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Return ``(step, jac)`` callables for a discrete map.

    ``step(x) -> ndarray (dim,)`` advances one iteration; ``jac(x) -> ndarray
    (dim, dim)`` is the Jacobian at ``x``.  Both wrap the class's compiled
    ``_step`` / ``_jacobian`` (the convention
    :class:`~tsdynamics.derived.TangentSystem` uses).  A map without an analytic
    ``_jacobian`` falls back to a forward finite difference of ``_step``.
    """
    cls = type(system)
    dim = int(cast("int", system.dim))  # dim resolved at construction
    # ``_step`` / ``_jacobian`` accept the native scalar-or-array form from
    # ``to_native`` (1-D maps take a float); treat them as untyped callables.
    step_raw: Callable[..., Any] = cls._step
    jac_raw: Callable[..., Any] | None = getattr(cls, "_jacobian", None)
    params = tuple(cast("ParamSet", system.params).as_tuple())

    def step(x: np.ndarray) -> np.ndarray:
        return np.asarray(step_raw(to_native(x, dim), *params), dtype=float).ravel()

    if jac_raw is not None:

        def jac(x: np.ndarray) -> np.ndarray:
            j = np.asarray(jac_raw(to_native(x, dim), *params), dtype=float)
            return np.atleast_2d(j).reshape(dim, dim)

    else:

        def jac(x: np.ndarray) -> np.ndarray:
            return finite_diff_jac(step, x, dim)

    return step, jac


def finite_diff_jac(
    step: Callable[[np.ndarray], np.ndarray], x: np.ndarray, dim: int, eps: float = 1e-7
) -> np.ndarray:
    """Forward finite-difference Jacobian of a map step (fallback only)."""
    x = np.asarray(x, dtype=float).ravel()
    base = step(x)
    out = np.empty((dim, dim))
    for j in range(dim):
        xp = x.copy()
        h = eps * (1.0 + abs(xp[j]))
        xp[j] += h
        out[:, j] = (step(xp) - base) / h
    return out


# ── tangent dynamics: flows ──────────────────────────────────────────────────


def flow_fns(
    system: ContinuousSystem,
) -> tuple[Callable[..., np.ndarray], Callable[..., np.ndarray]]:
    """Return ``(rhs, jac)`` for a flow: ``rhs(u, t)`` and ``jac(u, t)``.

    Both come from the SymEngine-lambdified numeric forms
    (:meth:`ContinuousSystem._rhs_numeric` / :meth:`ContinuousSystem.jacobian`),
    so the shooting/monodromy/variational integrator is self-contained (no
    engine tape lowering) and runs in the fast tier.  Parameters are captured at
    call time.
    """
    rhs = system._rhs_numeric()

    def jac(u: np.ndarray, t: float = 0.0) -> np.ndarray:
        return np.asarray(system.jacobian(u, t), dtype=float)

    return rhs, jac


def rk4_state(rhs: Callable[..., np.ndarray], x: np.ndarray, t: float, h: float) -> np.ndarray:
    """One classic RK4 step of ``dx/dt = rhs(x, t)`` (state only)."""
    k1 = rhs(x, t)
    k2 = rhs(x + 0.5 * h * k1, t + 0.5 * h)
    k3 = rhs(x + 0.5 * h * k2, t + 0.5 * h)
    k4 = rhs(x + h * k3, t + h)
    return cast("np.ndarray", x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))


def rk4_variational(
    rhs: Callable[..., np.ndarray],
    jac: Callable[..., np.ndarray],
    x: np.ndarray,
    m: np.ndarray,
    t: float,
    h: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""One RK4 step of the augmented system ``dx/dt = f``, ``dM/dt = J(x) M``.

    ``M`` holds the tangent vectors as columns (shape ``(dim, ncol)``); started
    at the identity it accumulates the monodromy matrix.  The same
    fundamental-matrix flow drives the shooting monodromy, GALI and expansion
    entropy.
    """
    k1x = rhs(x, t)
    k1m = jac(x, t) @ m
    x2, m2, t2 = x + 0.5 * h * k1x, m + 0.5 * h * k1m, t + 0.5 * h
    k2x = rhs(x2, t2)
    k2m = jac(x2, t2) @ m2
    x3, m3 = x + 0.5 * h * k2x, m + 0.5 * h * k2m
    k3x = rhs(x3, t2)
    k3m = jac(x3, t2) @ m3
    x4, m4, t4 = x + h * k3x, m + h * k3m, t + h
    k4x = rhs(x4, t4)
    k4m = jac(x4, t4) @ m4
    x_new = x + (h / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    m_new = m + (h / 6.0) * (k1m + 2.0 * k2m + 2.0 * k3m + k4m)
    return x_new, m_new
