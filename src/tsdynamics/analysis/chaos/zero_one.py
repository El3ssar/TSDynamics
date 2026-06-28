r"""
The 0--1 test for chaos (Gottwald & Melbourne 2004, 2009).

A binary order/chaos diagnostic that acts directly on a *scalar observable*
:math:`\phi_j` of the dynamics — no phase-space reconstruction, no Jacobian.
For a frequency :math:`c` it drives the skew translation

.. math::

    p_c(n) = \sum_{j=1}^{n} \phi_j \cos(jc), \qquad
    q_c(n) = \sum_{j=1}^{n} \phi_j \sin(jc),

and measures the asymptotic growth of the mean-square displacement of
:math:`(p_c, q_c)`.  Regular dynamics keep :math:`(p_c, q_c)` bounded (the
displacement is bounded, growth rate :math:`K_c \approx 0`); chaotic dynamics
make it diffuse like a random walk (linear growth, :math:`K_c \approx 1`).  The
returned :math:`K` is the median of :math:`K_c` over many frequencies, obtained
by the regularised mean-square displacement + correlation method of the 2009
paper (more robust than fitting a growth exponent).

The observable should be sampled so successive values are not strongly
correlated (every iterate for a map; a suitable stride for a flow).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from tsdynamics.errors import InvalidParameterError

from .._result import AnalysisResult, ScalarResult
from . import _common as _c

__all__ = ["ZeroOneResult", "zero_one_test"]


@dataclass(frozen=True, eq=False)
class ZeroOneResult(ScalarResult):
    r"""The 0--1-test indicator :math:`K`, carrying the translation plane it was read from.

    A :class:`~tsdynamics.analysis._result.ScalarResult`, so it is a drop-in for
    the bare ``K`` value — ``float(result)`` is :math:`K`, and ``result > 0.9``
    and every arithmetic / comparison operator work — while it also carries the
    skew-translation variables :math:`(p_c, q_c)` at a representative frequency
    :math:`c`.  Those variables stay **bounded** for regular dynamics and
    **diffuse** like a random walk for chaotic dynamics (Gottwald & Melbourne
    2004), so their plane is the test's diagnostic figure; :meth:`to_plot_spec`
    renders it as a phase portrait.

    Attributes
    ----------
    value : float
        The median growth indicator :math:`K` (``~0`` regular, ``~1`` chaotic).
    p, q : numpy.ndarray
        The cumulative translation components :math:`p_c(n)` / :math:`q_c(n)` at a
        representative frequency — a bounded blob (regular) or a diffusing cloud
        (chaotic).  Empty when the plane was not captured.
    """

    _repr_fields: ClassVar[tuple[str, ...]] = ("value",)

    p: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False, compare=False)
    q: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False, compare=False)

    def to_plot_spec(self, kind: str | None = None) -> Any:
        r"""Describe the translation plane :math:`(p_c, q_c)` as a :class:`PlotSpec`.

        Builds a ``PHASE_PORTRAIT_2D`` of the skew-translation trajectory: a
        ``LINE`` through :math:`(p_c(n), q_c(n))`, equal-aspect so the bounded
        (regular) vs diffusive (chaotic) geometry reads off directly.  Falls back
        to the one-point scalar spec when no plane was captured.  The
        :mod:`tsdynamics.viz.spec` import is lazy, so building a spec never pulls a
        plotting library.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind (e.g. ``"phase_portrait_2d"``).  ``None``
            uses ``PHASE_PORTRAIT_2D``.

        Returns
        -------
        PlotSpec
        """
        p = np.asarray(self.p, dtype=float)
        q = np.asarray(self.q, dtype=float)
        if p.size == 0 or q.size == 0:
            return super().to_plot_spec(kind=kind)

        from tsdynamics.viz.spec import Axis, Layer, PlotKind, PlotSpec

        spec_kind = PlotKind(kind) if kind is not None else PlotKind.PHASE_PORTRAIT_2D
        return PlotSpec(
            kind=spec_kind,
            ndim=2,
            aspect="equal",
            title=f"0--1 test translation plane ($K$ = {float(self):.3g})",
            x=Axis(label="$p_c$"),
            y=Axis(label="$q_c$"),
            layers=[Layer(PlotKind.LINE, {"x": p, "y": q}, label="$(p_c, q_c)$")],
            meta=dict(self.meta) if self.meta else {},
        )


def _observable(
    system: Any,
    component: int | None,
    *,
    final_time: float | None,
    n: int | None,
    dt: float | None,
    transient: float | None,
    ic: Any | None,
) -> np.ndarray:
    """Resolve the scalar observable from a System (integrate/iterate it) or data.

    A System produces its own decorrelated series — every iteration for a map or
    discrete view (Poincaré / stroboscopic), the ``dt``-grid for a flow.  A
    measured :class:`~tsdynamics.data.Trajectory` / ``ndarray`` is read directly
    (the ``data`` overload); the horizon keywords then do not apply.
    """
    is_system = hasattr(system, "is_discrete") and (
        hasattr(system, "trajectory")
        or hasattr(system, "iterate")
        or hasattr(system, "integrate")
        or hasattr(system, "_step")
    )
    if not is_system:
        if any(v is not None for v in (final_time, n, dt, transient, ic)):
            raise InvalidParameterError(
                "zero_one_test: final_time/n/dt/transient/ic apply only when the first "
                "argument is a System; a measured series / Trajectory is used as-is."
            )
        return _c._as_observable(system, component)
    if system.is_discrete:
        skip = int(transient) if transient is not None else 0
        count = int(n) if n is not None else 5000
        kw: dict[str, Any] = {"transient": skip}
        if ic is not None and hasattr(system, "iterate"):
            kw["ic"] = ic
        return _c._as_observable(system.trajectory(count, **kw), component)
    # continuous flow — sample on the dt grid (successive samples must be
    # decorrelated for the test to be meaningful; a coarse dt, or a Poincaré /
    # stroboscopic view passed as ``system``, gives the cleanest K).
    if n is not None:
        raise InvalidParameterError(
            "zero_one_test: n is for maps/discrete views; a flow uses final_time."
        )
    horizon = float(final_time) if final_time is not None else 1000.0
    burn = float(transient) if transient is not None else 0.0
    step = float(dt) if dt is not None else 0.1
    traj = system.integrate(final_time=horizon + burn, dt=step, ic=ic)
    if burn:
        traj = traj.after(burn)
    return _c._as_observable(traj, component)


def zero_one_test(
    system: Any,
    *,
    component: int | None = None,
    final_time: float | None = None,
    n: int | None = None,
    dt: float | None = None,
    transient: float | None = None,
    ic: Any | None = None,
    n_c: int = 100,
    c_range: tuple[float, float] = (np.pi / 5.0, 4.0 * np.pi / 5.0),
    n_cut: int | None = None,
    seed: int | None = 0,
    return_distribution: bool = False,
) -> ZeroOneResult | tuple[ZeroOneResult, np.ndarray]:
    r"""Run the 0--1 test for chaos on a system or a measured observable.

    Parameters
    ----------
    system : System, Trajectory, or array-like
        A dynamical system (integrated / iterated internally to produce the
        observable, like :func:`~tsdynamics.analysis.chaos.gali`), or a measured
        1-D series / :class:`~tsdynamics.data.Trajectory` used directly (the
        ``data`` overload).  For a flow pass a coarse ``dt`` — or a Poincaré /
        stroboscopic view as ``system`` — so successive samples are decorrelated.
    component : int, optional
        Column to use when a multi-component system / trajectory is passed.
    final_time : float, optional
        Integration horizon for a flow (system input).  Default 1000.0.
    n : int, optional
        Number of iterations for a map / discrete view (system input).
        Default 5000.
    dt : float, optional
        Sampling / integration step for a flow (system input).  Default 0.1.
    transient : float, optional
        Discarded before recording — a flow burn-in **time**, a map / discrete
        burn-in in **steps** (system input).
    ic : array-like, optional
        Initial condition (system input).
    n_c : int, default 100
        Number of random frequencies :math:`c` drawn from ``c_range``.
    c_range : (float, float), default ``(pi/5, 4*pi/5)``
        Interval the frequencies are drawn from.  The default avoids the
        resonances near :math:`c = 0, \pi` (Gottwald & Melbourne 2009).
    n_cut : int, optional
        Largest displacement lag used in the mean-square displacement.  Default
        ``N // 10`` (the rule of thumb: stay well below the series length).
    seed : int, optional
        Seed for the frequency draw (makes :math:`K` reproducible).
    return_distribution : bool, default False
        If true, also return the per-frequency :math:`K_c` array.

    Returns
    -------
    ZeroOneResult or (ZeroOneResult, ndarray)
        The median correlation growth indicator :math:`K` (``~0`` regular, ``~1``
        chaotic) as a drop-in for its ``float`` value (``result > 0.9`` and
        ``float(result)`` work) carrying ``.meta`` and the translation plane
        :math:`(p_c, q_c)` (``result.plot()`` renders it); with
        ``return_distribution`` also the ``K_c`` values.  The correlation method
        returns a Pearson
        coefficient, so :math:`K \in [-1, 1]` in principle (a regular orbit can
        give a small negative :math:`K`); it concentrates near ``0`` (regular) or
        ``1`` (chaotic), so ``K > 0.5`` is the usual chaos threshold.

    Raises
    ------
    InvalidParameterError
        If the observable is shorter than 200 points (too short for the test to
        be meaningful); if ``n_c < 1``; if horizon keywords are passed for a
        measured-series input; or if ``n`` is passed for a flow.

    Examples
    --------
    >>> zero_one_test(Logistic(params={"r": 4.0}), n=5000) > 0.9     # chaotic
    True
    >>> x = Logistic(params={"r": 4.0}).iterate(steps=5000).component("x")
    >>> zero_one_test(x) > 0.9          # the data overload
    True

    References
    ----------
    Gottwald & Melbourne, "A new test for chaos in deterministic systems",
    *Proc. R. Soc. Lond. A* **460** (2004) 603--611.

    Gottwald & Melbourne, "On the implementation of the 0--1 test for chaos",
    *SIAM J. Appl. Dyn. Syst.* **8** (2009) 129--145.
    """
    phi = _observable(
        system, component, final_time=final_time, n=n, dt=dt, transient=transient, ic=ic
    )
    n_pts = phi.size
    if n_pts < 200:
        raise InvalidParameterError(
            f"the 0-1 test needs a long series to be meaningful; got {n_pts} points (need >= 200)."
        )
    if n_cut is None:
        n_cut = n_pts // 10
    n_cut = int(max(1, min(n_cut, n_pts - 1)))
    if n_c < 1:
        raise InvalidParameterError(f"n_c must be >= 1, got {n_c}.")

    rng = np.random.default_rng(seed)
    c_values = rng.uniform(c_range[0], c_range[1], size=int(n_c))

    j = np.arange(1, n_pts + 1, dtype=float)
    lags = np.arange(1, n_cut + 1, dtype=float)
    mean_phi_sq = float(np.mean(phi)) ** 2

    # Drive all frequencies at once: the skew-translation sums for every ``c``
    # are the columns of ``p``/``q`` (shape ``(n_pts, n_c)``). Batching the
    # ``cos``/``sin``/``cumsum`` across frequencies removes the per-frequency
    # Python work; each column is byte-identical to the per-``c`` cumsum.
    phase = np.outer(j, c_values)  # j*c for every (sample, frequency)
    phi_col = phi[:, None]
    p_all = np.cumsum(phi_col * np.cos(phase), axis=0)  # (n_pts, n_c)
    q_all = np.cumsum(phi_col * np.sin(phase), axis=0)

    # Mean-square displacement at each lag, vectorised over all frequencies at
    # once: the per-lag difference ``p[lag:] - p[:-lag]`` and the ``np.mean`` over
    # samples are unchanged (same elements, same reduction axis) — only the
    # per-frequency Python loop is gone.
    msd = np.empty((n_cut, int(n_c)))
    for li in range(n_cut):
        lag = li + 1
        dp = p_all[lag:] - p_all[:-lag]
        dq = q_all[lag:] - q_all[:-lag]
        msd[li] = np.mean(dp * dp + dq * dq, axis=0)

    # Regularised mean-square displacement: subtract the oscillatory term so only
    # the (diffusive) trend remains (Gottwald & Melbourne 2009, eq. 2.6).
    osc = mean_phi_sq * (1.0 - np.cos(np.outer(lags, c_values))) / (1.0 - np.cos(c_values))
    d_all = msd - osc  # (n_cut, n_c)
    k_c = np.array([_c._pearson(lags, d_all[:, idx]) for idx in range(int(n_c))])

    k = float(np.median(k_c))
    # Capture the skew-translation plane (p_c, q_c) at the most representative
    # frequency — the one whose K_c is closest to the reported median K — purely
    # for the diagnostic figure (it does not enter K).  The representative
    # column already lives in the batched ``p_all``/``q_all`` (byte-identical to a
    # standalone ``cumsum`` for that single c), so slice it instead of recomputing.
    idx_rep = int(np.argmin(np.abs(k_c - k)))
    p_rep = p_all[:, idx_rep]
    q_rep = q_all[:, idx_rep]
    result = ZeroOneResult(
        value=k,
        p=p_rep,
        q=q_rep,
        meta=AnalysisResult.build_meta(system, analysis="zero_one_test"),
    )
    return (result, k_c) if return_distribution else result


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
