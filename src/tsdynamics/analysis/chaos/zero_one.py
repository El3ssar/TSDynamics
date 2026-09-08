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

The observable must be sampled so successive values are not strongly correlated
— every iterate for a map, a coarse stride for a flow.  This is not a nicety: an
oversampled flow makes :math:`(p_c, q_c)` drift smoothly instead of diffusing and
the test then reports a *chaotic* orbit as regular.  :func:`zero_one_test`
therefore measures the sampling density and decimates by default; see its
``oversampling`` argument and :class:`OversamplingWarning`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from tsdynamics.errors import InvalidParameterError

from .._result import AnalysisResult, ScalarResult
from . import _common as _c

__all__ = ["OversamplingWarning", "ZeroOneResult", "zero_one_test"]

# ── oversampling guard ───────────────────────────────────────────────────────
# The 0-1 test needs an observable whose successive values are not strongly
# correlated: Gottwald & Melbourne (2009, §"Choice of sampling time") show that
# an oversampled flow makes the skew translation (p_c, q_c) trace a smooth
# ballistic curve instead of a random walk, and the correlation method then
# reports K ~ 0 — "regular" — for a plainly chaotic orbit.
#
# The condition is measured as the mean number of samples per oscillation of the
# observable (twice the mean spacing of its mean-crossings).  Measured on Lorenz
# (lambda_1 = 0.906, K must be ~1): at 308 samples/oscillation K = -0.018, at 138
# K = -0.023, at 62 K = +0.61, at 33 K = +0.998.  Decimating toward ~5
# samples/oscillation is the fix Gottwald & Melbourne prescribe and it is
# uniformly safe here: Rössler at c = 5.7 needs it (K = -0.004 at 286
# samples/oscillation, +0.97 at 3.9), a periodic/quasi-periodic orbit keeps
# K ~ 0 either way, and a map already sits near 4 so it is untouched.
_OVERSAMPLED_ABOVE = 10.0
_TARGET_SAMPLES_PER_OSCILLATION = 5.0
# Never decimate below this many samples: a short record is worth more than a
# perfectly-decorrelated one (the test itself refuses below 200 points).
_MIN_KEPT_SAMPLES = 250


class OversamplingWarning(UserWarning):
    r"""The 0--1 test was handed an observable sampled far too finely.

    The Gottwald--Melbourne test requires roughly one sample per oscillation of
    the observable; an oversampled flow makes the skew translation drift
    smoothly rather than diffuse, and the test then reports a *chaotic* orbit as
    regular (:math:`K \approx 0`).  :func:`zero_one_test` normally repairs this
    by decimating (``oversampling="resample"``, the default); the warning is
    emitted when it is asked not to, or when the record is too short to decimate
    far enough to fix the problem.
    """


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

        from .. import _plotbuilder as pb

        return pb.spec(
            kind,
            "phase_portrait_2d",
            layers=[pb.line(p, q, label="$(p_c, q_c)$")],
            aspect="equal",
            xlabel="$p_c$",
            ylabel="$q_c$",
            title=f"0--1 test translation plane ($K$ = {float(self):.3g})",
            meta=self.meta,
        )


def _samples_per_oscillation(phi: np.ndarray) -> float:
    """Mean samples per oscillation of ``phi``, from its mean-crossing rate.

    A full oscillation crosses the mean twice, so the mean spacing between sign
    changes of ``phi - mean(phi)`` is half a period.  Chosen over an FFT peak
    because a chaotic flow's spectrum is broadband (the FFT estimate scatters by
    an order of magnitude on Lorenz, the crossing rate does not) and over the
    lag-1 autocorrelation because that saturates near 1 and cannot say *how far*
    to decimate.  ``inf`` when the observable never crosses its mean (a constant
    or monotone record — nothing to fix).
    """
    y = np.asarray(phi, dtype=float)
    if y.size < 2:  # nothing to oscillate (guard before ``mean`` of an empty slice)
        return float("inf")
    y = y - y.mean()
    sign = np.signbit(y)
    crossings = int(np.count_nonzero(sign[1:] != sign[:-1]))
    if crossings == 0:
        return float("inf")
    return 2.0 * float(y.size - 1) / crossings


def _apply_oversampling_guard(phi: np.ndarray, policy: str) -> tuple[np.ndarray, int, float]:
    """Detect (and by default repair) an oversampled observable.

    Returns the observable to test, the stride applied, and the measured
    samples-per-oscillation of the *input*.  Under ``"resample"`` the series is
    decimated toward :data:`_TARGET_SAMPLES_PER_OSCILLATION`, never below
    :data:`_MIN_KEPT_SAMPLES` points; when that cap leaves it still oversampled,
    an :class:`OversamplingWarning` says so rather than letting a chaotic orbit
    be reported as regular.
    """
    spo = _samples_per_oscillation(phi)
    if policy == "ignore" or not np.isfinite(spo) or spo <= _OVERSAMPLED_ABOVE:
        return phi, 1, spo
    advice = (
        f"the observable is oversampled (~{spo:.0f} samples per oscillation): the 0-1 "
        "test needs roughly one sample per oscillation, and an oversampled series "
        "drives K toward 0 even for a chaotic orbit"
    )
    if policy == "warn":
        warnings.warn(
            f"zero_one_test: {advice}. Sample the flow more coarsely (a larger dt), "
            "pass a Poincare/stroboscopic view as `system`, or use the default "
            "oversampling='resample'.",
            OversamplingWarning,
            stacklevel=3,
        )
        return phi, 1, spo
    wanted = max(1, int(round(spo / _TARGET_SAMPLES_PER_OSCILLATION)))
    stride = min(wanted, max(1, phi.size // _MIN_KEPT_SAMPLES))
    out = phi[::stride] if stride > 1 else phi
    if _samples_per_oscillation(out) > _OVERSAMPLED_ABOVE:
        warnings.warn(
            f"zero_one_test: {advice}, and the record is too short to decimate far "
            f"enough to fix it (stride {stride} of the {wanted} needed leaves "
            f"{out.size} points). K is biased toward 0 — use a longer record, or "
            "sample the flow more coarsely to begin with.",
            OversamplingWarning,
            stacklevel=3,
        )
    return out, stride, spo


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
        # Honor an explicit ``ic`` for *any* discrete System — a map *or* a
        # discrete view (Poincaré / stroboscopic), whose ``trajectory`` accepts
        # ``ic`` via ``**kwargs`` even though it has no ``iterate``.  Silently
        # substituting the wrapper's default IC would hand back a K for an orbit
        # the caller never asked about (the 0-1 test characterises a *specific*
        # orbit), exactly the footgun ``gali`` guards against.
        if ic is not None:
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
    oversampling: str = "resample",
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
    oversampling : {"resample", "warn", "ignore"}, default "resample"
        What to do when the observable is **oversampled** — the classic misuse of
        this test, and one it fails silently: an observable sampled many times
        per oscillation makes :math:`(p_c, q_c)` drift smoothly instead of
        diffusing, and :math:`K` collapses toward ``0`` for a chaotic orbit
        (Lorenz at ``dt = 0.02`` gives :math:`K = -0.02`).  ``"resample"``
        decimates the observable to about five samples per oscillation — the
        remedy Gottwald & Melbourne prescribe — and records the stride in
        ``result.meta``; if the record is too short to decimate that far it
        decimates as far as it can and warns.  ``"warn"`` leaves the observable
        alone and raises an :class:`OversamplingWarning`.  ``"ignore"`` disables
        the guard entirely (you are then on your own).
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

    Warns
    -----
    OversamplingWarning
        When the observable is oversampled and the guard could not (or was asked
        not to) fix it — see ``oversampling``.  :math:`K` is then biased toward
        ``0`` and a chaotic orbit may be reported as regular.

    Raises
    ------
    InvalidParameterError
        If the observable is shorter than 200 points (too short for the test to
        be meaningful); if ``n_c < 1``; if ``oversampling`` is not one of
        ``"resample"`` / ``"warn"`` / ``"ignore"``; if horizon keywords are
        passed for a measured-series input; or if ``n`` is passed for a flow.

    Examples
    --------
    >>> zero_one_test(Logistic(params={"r": 4.0}), n=5000) > 0.9     # chaotic
    True
    >>> x = Logistic(params={"r": 4.0}).iterate(steps=5000).component("x")
    >>> zero_one_test(x) > 0.9          # the data overload
    True
    >>> lorenz = Lorenz(ic=[1.0, 1.0, 1.0])                  # a flow, sampled fine
    >>> zero_one_test(lorenz, component=0, dt=0.02, final_time=600.0) > 0.9
    True

    References
    ----------
    Gottwald & Melbourne, "A new test for chaos in deterministic systems",
    *Proc. R. Soc. Lond. A* **460** (2004) 603--611.

    Gottwald & Melbourne, "On the implementation of the 0--1 test for chaos",
    *SIAM J. Appl. Dyn. Syst.* **8** (2009) 129--145.
    """
    if oversampling not in {"resample", "warn", "ignore"}:
        raise InvalidParameterError(
            f"oversampling must be 'resample', 'warn' or 'ignore', got {oversampling!r}."
        )
    phi = _observable(
        system, component, final_time=final_time, n=n, dt=dt, transient=transient, ic=ic
    )
    phi, stride, spo = _apply_oversampling_guard(phi, oversampling)
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
    meta = AnalysisResult.build_meta(system, analysis="zero_one_test")
    meta["samples_per_oscillation"] = float(spo)
    meta["stride"] = int(stride)
    meta["n_samples"] = int(n_pts)
    result = ZeroOneResult(value=k, p=p_rep, q=q_rep, meta=meta)
    return (result, k_c) if return_distribution else result


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
