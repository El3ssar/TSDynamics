r"""
Generalized Alignment Index — GALI\ :sub:`k` (Skokos, Bountis & Antonopoulos 2007).

GALI\ :sub:`k` is the volume of the parallelepiped spanned by ``k`` *unit*
deviation (tangent) vectors,

.. math::

    \mathrm{GALI}_k(t) = \lVert \hat w_1(t) \wedge \cdots \wedge \hat w_k(t) \rVert
        = \prod_{i=1}^{k} \sigma_i,

the product of the singular values :math:`\sigma_i` of the matrix whose columns
are the normalised deviation vectors.  For a **chaotic** orbit the vectors align
along the most-expanding direction and the index decays exponentially,

.. math::  \mathrm{GALI}_k(t) \sim e^{-[(\lambda_1-\lambda_2)+\cdots+(\lambda_1-\lambda_k)]\,t},

while for a **regular** (quasi-periodic) orbit it stays constant or decays only
by a power law — so the decay law cleanly separates order from chaos and, for
chaos, *measures* the leading Lyapunov-exponent gaps.

Supported systems are finite-tangent-space ones: discrete maps
(:class:`~tsdynamics.families.DiscreteMap`) and flows
(:class:`~tsdynamics.families.ContinuousSystem`).  The deviation vectors are
evolved with the exact tangent map (maps) or an RK4 variational step (flows) and
renormalised per column each step — which bounds their magnitude without
changing the directions GALI reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tsdynamics.families import ContinuousSystem, DiscreteMap

from . import _common as _c

__all__ = ["GALIResult", "gali"]


@dataclass(frozen=True)
class GALIResult:
    r"""A GALI\ :sub:`k` time series with the tools to read order vs chaos off it.

    ``float(result)`` is the final value — :math:`\approx 1` for a regular orbit,
    :math:`\to 0` for a chaotic one — so the result drops straight into a
    threshold test.

    Attributes
    ----------
    k : int
        Number of deviation vectors.
    times : ndarray
        Time (flows) or iteration index (maps) at each sample.
    values : ndarray
        GALI\ :sub:`k` at each sample.
    is_discrete : bool
        Whether the underlying system is a map.
    """

    k: int
    times: np.ndarray = field(repr=False)
    values: np.ndarray = field(repr=False)
    is_discrete: bool

    def __float__(self) -> float:
        """Return the final GALI value (the order/chaos summary)."""
        return float(self.values[-1])

    @property
    def final(self) -> float:
        """The last GALI value."""
        return float(self.values[-1])

    def decay_rate(self, *, floor: float = 1e-12, t_min: float | None = None) -> float:
        r"""Exponential decay rate of GALI\ :sub:`k` (a positive number for chaos).

        Fits a line to :math:`\ln \mathrm{GALI}_k` over the samples that lie
        above ``floor`` (i.e. before the index reaches the floating-point noise
        floor) and returns ``-slope``.  For a chaotic orbit this estimates the
        Lyapunov gap sum :math:`(\lambda_1-\lambda_2)+\cdots+(\lambda_1-\lambda_k)`
        (Skokos et al. 2008); for a regular orbit it is ``~0``.

        Parameters
        ----------
        floor : float, default 1e-12
            Ignore samples at or below this value (numerical underflow region).
        t_min : float, optional
            Also ignore samples before this time/iteration (skip the initial
            alignment transient).

        Returns
        -------
        float
            Estimated decay rate; ``0.0`` if too few usable samples remain.
        """
        t, v = self.times, self.values
        mask = v > floor
        if t_min is not None:
            mask &= t >= t_min
        if int(np.count_nonzero(mask)) < 2:
            return 0.0
        slope, _, _ = _c._linfit(t[mask], np.log(v[mask]))
        return float(-slope)

    def is_chaotic(self, *, threshold: float = 1e-6) -> bool:
        """Whether the final GALI value collapsed below ``threshold`` (chaotic)."""
        return self.final < threshold

    def to_plot_spec(self, kind: str | None = None) -> Any:
        r"""Describe the GALI\ :sub:`k` curve as a backend-agnostic :class:`PlotSpec`.

        Builds a ``DIAGNOSTIC_CURVE`` of GALI\ :sub:`k` against time (or iteration
        index for a map) on a **log y-axis** — the recommended scale, since
        GALI\ :sub:`k` decays exponentially for a chaotic orbit and saturates for
        a regular one, so a log axis reads the decay rate off as a slope.  The
        :mod:`tsdynamics.viz.spec` import is lazy, so building a spec never pulls a
        plotting library.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind (e.g. ``"diagnostic_curve"``).  ``None``
            uses ``DIAGNOSTIC_CURVE``.

        Returns
        -------
        PlotSpec
        """
        from tsdynamics.viz.spec import Axis, Layer, PlotKind, PlotSpec

        spec_kind = PlotKind(kind) if kind is not None else PlotKind.DIAGNOSTIC_CURVE
        xlabel = "iteration" if self.is_discrete else "time"
        return PlotSpec(
            kind=spec_kind,
            ndim=2,
            title=f"GALI$_{self.k}$",
            x=Axis(label=xlabel),
            y=Axis(label=f"GALI$_{self.k}$", scale="log"),
            layers=[
                Layer(
                    PlotKind.LINE,
                    {"x": np.asarray(self.times), "y": np.asarray(self.values)},
                    label=f"GALI$_{self.k}$",
                )
            ],
        )

    def __repr__(self) -> str:  # noqa: D105
        kind = "map" if self.is_discrete else "flow"
        return f"GALIResult(k={self.k}, {kind}, final={self.final:.3g}, n={self.values.size})"


def gali(
    system: Any,
    k: int = 2,
    *,
    steps: int | None = None,
    final_time: float | None = None,
    dt: float | None = None,
    ic: Any | None = None,
    transient: float | None = None,
    seed: int | None = 0,
    n_internal: int = 10,
) -> GALIResult:
    r"""Compute the GALI\ :sub:`k` time series of a map or flow.

    Parameters
    ----------
    system : DiscreteMap or ContinuousSystem
        The system whose tangent dynamics to track.  Delay/stochastic systems
        are not supported (their tangent space is not finite-dimensional here).
    k : int, default 2
        Number of deviation vectors, ``2 <= k <= system.dim``.  (``k = 1`` is
        trivially ``1``.)
    steps : int, optional
        Number of iterations (maps).  Default 1000.
    final_time : float, optional
        Integration time (flows).  Default 100.0.
    dt : float, optional
        Sampling/recording step for flows.  Default 0.1.  Not valid for maps.
    ic : array-like, optional
        Initial condition (defaults to the system's resolved IC).
    transient : float, optional
        Burn-in discarded before tracking (iterations for maps, time for flows).
        Defaults: 500 iterations / 20.0 time units.
    seed : int, optional
        Seed for the random orthonormal initial deviation frame.
    n_internal : int, default 10
        RK4 sub-steps per ``dt`` for flows (controls variational accuracy).

    Returns
    -------
    GALIResult

    Raises
    ------
    NotImplementedError
        If ``system`` is neither a discrete map nor a continuous flow.
    ValueError
        If ``k`` is outside ``[2, dim]``; if ``dt`` is passed for a map (or
        ``steps`` for a flow); or if the orbit diverges to a non-finite state
        from every tried initial condition after the retry budget — in which
        case pass an ``ic`` from a known basin point (or shorten ``transient``).

    Examples
    --------
    >>> gali(Henon(), k=2, steps=60).is_chaotic()      # exponential collapse
    True
    >>> float(gali(Lorenz(), k=2, final_time=25.0))    # ~0: Lorenz is chaotic
    0.0...

    References
    ----------
    Skokos, Bountis & Antonopoulos (2007), *Physica D* 231, 30--54.
    Skokos, Bountis & Antonopoulos (2008), *Eur. Phys. J. Spec. Top.* 165, 5--14.
    """
    if isinstance(system, DiscreteMap):
        mode = "map"
    elif isinstance(system, ContinuousSystem):
        mode = "flow"
    else:
        raise NotImplementedError(
            f"gali supports discrete maps and continuous flows, not "
            f"{type(system).__name__} (its tangent space is not finite-dimensional here)."
        )

    dim = int(system.dim)
    k = int(k)
    if not 2 <= k <= dim:
        raise ValueError(f"k must satisfy 2 <= k <= dim ({dim}); got {k}.")

    rng = np.random.default_rng(seed)
    w0 = _c._orthonormal_frame(dim, k, rng)

    if mode == "map":
        if dt is not None:
            raise ValueError("dt has no meaning for a discrete map — omit it.")
        n_steps = 1000 if steps is None else int(steps)
        n_burn = 500 if transient is None else int(transient)
        run_args: tuple = (n_steps, n_burn)
        discrete = True
    else:
        if steps is not None:
            raise ValueError("steps applies to maps; use final_time/dt for a flow.")
        t_end = 100.0 if final_time is None else float(final_time)
        step_dt = 0.1 if dt is None else float(dt)
        t_burn = 20.0 if transient is None else float(transient)
        run_args = (t_end, step_dt, t_burn, int(n_internal))
        discrete = False

    # Track the tangent dynamics from a point that stays on the attractor.  An
    # orbit that escapes the basin (common when the IC is random — many systems
    # carry no ``default_ic``) blows up to a non-finite state and frame, so the
    # runner soft-fails and we retry from a fresh seeded random IC, matching the
    # map convention in :class:`~tsdynamics.derived.TangentSystem` (the seed
    # keeps the retries reproducible).
    max_retries = 10
    for attempt in range(max_retries):
        x = (
            np.asarray(system.resolve_ic(ic), dtype=float).ravel()
            if attempt == 0
            else rng.random(dim)
        )
        w = w0.copy()
        result = (
            _gali_map(system, x, w, *run_args)
            if mode == "map"
            else _gali_flow(system, x, w, *run_args)
        )
        if result is not None:
            times, values = result
            return GALIResult(k=k, times=times, values=values, is_discrete=discrete)

    raise ValueError(
        f"gali: {type(system).__name__} orbit diverges from every tried IC after "
        f"{max_retries} attempts — pass an `ic` from a known basin point, or shorten "
        "the burn-in via `transient`."
    )


def _gali_map(
    system: Any, x: np.ndarray, w: np.ndarray, n_steps: int, n_burn: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """GALI for a map: ``W <- J(x) W`` (Jacobian at the pre-image), per-column normalise.

    Returns ``None`` (soft failure) if the orbit or the tangent frame diverges to
    a non-finite value, so the caller can retry from a fresh IC.
    """
    step, jac = _c._map_fns(system)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(n_burn):
            x = step(x)
            if not np.all(np.isfinite(x)):
                return None
        values = np.empty(n_steps)
        times = np.arange(1, n_steps + 1, dtype=float)
        for i in range(n_steps):
            j = jac(x)  # Jacobian at the pre-image x_n
            x = step(x)
            if not np.all(np.isfinite(j)) or not np.all(np.isfinite(x)):
                return None
            w = j @ w
            if not np.all(np.isfinite(w)):
                return None
            w = _c._normalize_columns(w)
            values[i] = _c.gali_volume(w)
    return times, values


def _gali_flow(
    system: Any,
    x: np.ndarray,
    w: np.ndarray,
    final_time: float,
    dt: float,
    transient: float,
    n_internal: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """GALI for a flow: RK4-evolve the variational system, per-column normalise per ``dt``.

    Returns ``None`` (soft failure) on a non-finite (diverged) orbit or frame, so
    the caller can retry from a fresh IC.
    """
    rhs, jac = _c._flow_fns(system)
    h_burn = dt / max(1, n_internal)
    t = 0.0
    n_burn = int(round(max(0.0, transient) / h_burn))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(n_burn):
            x = _c._rk4_state(rhs, x, t, h_burn)
            t += h_burn
            if not np.all(np.isfinite(x)):
                return None

        n_steps = int(round(final_time / dt))
        h = dt / max(1, n_internal)
        values = np.empty(n_steps)
        times = np.empty(n_steps)
        for i in range(n_steps):
            for _ in range(n_internal):
                x, w = _c._rk4_variational(rhs, jac, x, w, t, h)
                t += h
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(w)):
                return None
            w = _c._normalize_columns(w)
            values[i] = _c.gali_volume(w)
            times[i] = t
    return times, values
