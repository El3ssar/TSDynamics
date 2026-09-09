r"""Spectrum and convergence plot transforms — the quantitative diagnostics.

Five registered transforms.  What they have in common is that each turns a
*number the library already reports* into the picture a referee asks for before
believing it:

``lyapunov_spectrum``
    The exponents as an ordered stem plot against the :math:`\lambda = 0` line,
    with the Kaplan--Yorke dimension in the title.  Chaos is "a stem crosses
    zero", read directly.
``lyapunov_convergence``
    The **running** Benettin estimate against time.  This is how you *show* that
    an exponent converged instead of asserting it: the curve flattens, or it
    does not.
``gali_curves``
    :math:`\mathrm{GALI}_k` against time on log axes with the analytic
    power-law reference drawn, so order (a curve tracking a power law, or
    holding constant) separates from chaos (a curve plunging away from it).
``zero_one_pq_plane``
    The :math:`(p_c, q_c)` translation plane of the Gottwald--Melbourne test.
    The reason the test is trusted is that the answer is *visible*: a bounded
    blob is regular, a Brownian sprawl is chaotic.
``scaling_fit``
    The log--log curve of any :class:`~tsdynamics.analysis.ScalingResult` with
    the fitted window **delimited**, plus two inspection views
    (``view="residuals"`` and ``view="local_slopes"``) — because a scaling fit
    whose region you cannot see is exactly how a wrong dimension gets believed.

Log axes live in the *data*
---------------------------
:class:`~tsdynamics.viz.transforms.Geometry` carries axis labels and limits but
no axis **scale**, so a transform cannot ask for a log axis.  ``gali_curves``
therefore plots :math:`\log_{10}` of the quantities and says so on the axis
labels.  That is not merely a workaround: it makes the power-law references
straight lines with an exact, stated slope, which is what one actually reads.

Every transform here is ``source="data"``: each accepts the already-computed
result object (a :class:`~tsdynamics.analysis.LyapunovSpectrum`, a
:class:`~tsdynamics.analysis.GALIResult`, a
:class:`~tsdynamics.analysis.ScalingResult`) and, per the registry's rule, also
accepts a system — because a model gives you data for free.  None of them owns
any numerics: the estimators are :mod:`tsdynamics.analysis`'s, cited there.

References
----------
.. [1] Benettin, G., Galgani, L., Giorgilli, A. & Strelcyn, J.-M. (1980).
   "Lyapunov Characteristic Exponents for smooth dynamical systems ... Part 2."
   *Meccanica*, 15, 21-30.  (The running time-average this plots.)
.. [2] Kaplan, J. L. & Yorke, J. A. (1979). "Chaotic behavior of
   multidimensional difference equations."  LNM 730, 204-227.
.. [3] Skokos, Ch., Bountis, T. C. & Antonopoulos, Ch. (2007). "Geometrical
   properties of local dynamics in Hamiltonian systems: the Generalized
   Alignment Index (GALI) method."  *Physica D*, 231, 30-54.  (The
   :math:`t^{-(k-s)}` power law drawn as the reference.)
.. [4] Gottwald, G. A. & Melbourne, I. (2004). "A new test for chaos in
   deterministic systems."  *Proc. R. Soc. A*, 460, 603-611.
.. [5] Grassberger, P. & Procaccia, I. (1983). "Characterization of Strange
   Attractors."  *Phys. Rev. Lett.*, 50, 346-349.  (The log--log scaling region
   this delimits.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .._frames import FrameSpace, OverlayRole
from ..spec import PlotKind
from ._base import Geometry, Part, Presentation, make_frame
from ._registry import plot_transform

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

__all__ = [
    "gali_curves",
    "lyapunov_convergence",
    "lyapunov_spectrum",
    "scaling_fit",
    "zero_one_pq_plane",
]


# ---------------------------------------------------------------------------
# Shared style / helpers
# ---------------------------------------------------------------------------

#: Reference geometry (a zero line, a power-law guide, a fit window): visible,
#: never the subject.
_REFERENCE_STYLE: dict[str, Any] = {
    "color": "gray",
    "linewidth": 1.0,
    "linestyle": "dashed",
    "alpha": 0.75,
}

#: The stems of the Lyapunov spectrum: they carry the sign, not the identity, so
#: they are thin and unlabelled after the first.
_STEM_STYLE: dict[str, Any] = {"color": "gray", "linewidth": 1.5, "alpha": 0.9}

#: Highlight for the points a scaling fit was actually taken over.
_FIT_REGION_STYLE: dict[str, Any] = {"marker": "circle", "markersize": 9.0, "alpha": 0.9}

#: The smallest value plotted on a log axis.  ``GALI`` underflows to exactly 0
#: for a chaotic orbit, and ``log10(0)`` is ``-inf``; clipping keeps the plunge
#: visible instead of dropping the samples that show it.
_LOG_FLOOR = 1e-16


def _is_system(subject: Any) -> bool:
    """Whether ``subject`` looks like a dynamical system rather than data."""
    return hasattr(subject, "dim") and (
        hasattr(subject, "_equations") or hasattr(subject, "_step") or hasattr(subject, "_drift")
    )


def _given(**options: Any) -> dict[str, Any]:
    """Keep only the options the caller actually set (drop every ``None``).

    Transform signatures name the estimator options they forward **explicitly**
    rather than taking ``**kwargs``: the composition front door routes a shared
    keyword only to a transform whose signature names it, so a ``**kwargs``
    catch-all would make ``ts.plot(sys, "zero_one_pq_plane", component=0)``
    *silently drop* the argument — a plot that draws fine and answers a
    different question.  Defaulting each to ``None`` and forwarding only what
    was set keeps the estimator's own defaults authoritative.
    """
    return {name: value for name, value in options.items() if value is not None}


def _hline_part(x: np.ndarray, y: float, label: str | None) -> Part:
    """Build a horizontal reference line spanning ``x``, as a pinned ``line`` part.

    A reference *line* rather than an
    :class:`~tsdynamics.viz.spec.Annotation`: an annotation is honored by the
    drawing backends and dropped by the two data-export ones, so the JSON /
    three.js form of the figure would lose the zero line the whole reading
    depends on.
    """
    if x.size:
        lo, hi = float(np.min(x)), float(np.max(x))
    else:  # pragma: no cover - defensive
        lo, hi = 0.0, 1.0
    return Part(
        {"x": np.array([lo, hi]), "y": np.full(2, float(y))},
        label=label,
        style=_REFERENCE_STYLE,
        primitive="line",
    )


def _vline_parts(xs: Sequence[float], y: np.ndarray, label: str | None) -> list[Part]:
    """Vertical delimiters at ``xs`` spanning ``y``, as pinned ``line`` parts."""
    finite = y[np.isfinite(y)] if y.size else np.zeros(1)
    if finite.size == 0:  # pragma: no cover - defensive
        finite = np.zeros(1)
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        lo, hi = lo - 0.5, hi + 0.5
    pad = 0.05 * (hi - lo)
    parts: list[Part] = []
    for i, x in enumerate(xs):
        parts.append(
            Part(
                {"x": np.full(2, float(x)), "y": np.array([lo - pad, hi + pad])},
                label=label if i == 0 else None,
                style=_REFERENCE_STYLE,
                primitive="line",
            )
        )
    return parts


# ---------------------------------------------------------------------------
# lyapunov_spectrum
# ---------------------------------------------------------------------------


def _resolve_exponents(subject: Any, options: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Return ``(exponents, meta)`` from a result, an array, or a system."""
    from tsdynamics.errors import InvalidInputError

    if _is_system(subject):
        from tsdynamics.analysis.lyapunov import lyapunov_spectrum as estimate

        result = estimate(subject, **options)
        return np.asarray(result, dtype=float), dict(result.meta or {})
    if options:
        raise InvalidInputError(
            f"lyapunov_spectrum got option(s) {sorted(options)}, which configure the "
            "*estimator*; they only apply when the subject is a system. The spectrum you "
            "passed is already computed."
        )
    values = getattr(subject, "values", subject)
    exps = np.atleast_1d(np.asarray(values, dtype=float)).ravel()
    if exps.size == 0:
        raise InvalidInputError(
            "lyapunov_spectrum has no exponents to draw. Pass a system, a LyapunovSpectrum, "
            "or a non-empty array of exponents."
        )
    return exps, dict(getattr(subject, "meta", None) or {})


def _example_spectrum(_primitive: str) -> tuple[Any, dict[str, Any]]:
    """Return the gate's subject: the literature Lorenz spectrum, no integration."""
    from tsdynamics.analysis.lyapunov import LyapunovSpectrum

    return LyapunovSpectrum(values=np.array([0.906, 0.0, -14.572])), {}


@plot_transform(
    name="lyapunov_spectrum",
    source="data",
    kind=PlotKind.LYAPUNOV_SPECTRUM,
    frame=FrameSpace.INDEX,
    ndim=1,
    role=OverlayRole.BASE,
    default_primitive="points",
    primitives=("points", "line"),
    analysis="tsdynamics.analysis.lyapunov.lyapunov_spectrum",
    example=_example_spectrum,
    doc="The ordered exponents as a stem plot against the lambda = 0 line.",
)
def lyapunov_spectrum(
    subject: Any,
    *,
    k: int | None = None,
    final_time: float | None = None,
    n: int | None = None,
    transient: float | None = None,
    dt: float | None = None,
    ic: Any | None = None,
) -> Geometry:
    r"""Draw the Lyapunov exponents, ordered and stemmed from the :math:`\lambda = 0` line.

    The one thing a spectrum plot has to make unmistakable is the **sign** of
    each exponent, because that is the whole classification: one positive
    exponent is chaos, two is hyperchaos, none is regular motion.  So every
    exponent is stemmed from zero and the zero line is drawn as geometry (not an
    annotation, so the data-export backends keep it).  The Kaplan--Yorke
    dimension implied by the spectrum is put in the title, where it belongs next
    to the numbers it was computed from.

    Parameters
    ----------
    subject : LyapunovSpectrum, array-like, or System
        An already-computed spectrum, a bare array of exponents, or a system
        (then :func:`~tsdynamics.analysis.lyapunov.lyapunov_spectrum` is run).
    k, final_time, n, transient, dt, ic
        Forwarded to the estimator when ``subject`` is a system; each defaults
        to ``None``, meaning *"leave the estimator's own default alone"*.
        Passing one alongside an already-computed spectrum **raises**, rather
        than being silently ignored.

    Returns
    -------
    Geometry
        An ``index``-framed geometry: the zero line, one stem per exponent, and
        the exponents themselves.

    Raises
    ------
    tsdynamics.errors.InvalidInputError
        If there are no exponents, or estimator options were passed with an
        already-computed spectrum.

    Examples
    --------
    >>> ts.plot(ts.systems.Lorenz(), "lyapunov_spectrum", final_time=300)  # doctest: +SKIP
    """
    from tsdynamics.analysis.lyapunov import kaplan_yorke_dimension

    exps, meta = _resolve_exponents(
        subject,
        _given(k=k, final_time=final_time, n=n, transient=transient, dt=dt, ic=ic),
    )
    order = np.argsort(exps)[::-1]
    exps = exps[order]
    index = np.arange(exps.size, dtype=float)

    parts: list[Part] = [_hline_part(index, 0.0, r"$\lambda = 0$")]
    for i, value in enumerate(exps):
        parts.append(
            Part(
                {"x": np.full(2, float(i)), "y": np.array([0.0, float(value)])},
                label="stems" if i == 0 else None,
                style=_STEM_STYLE,
                primitive="line",
            )
        )
    parts.append(
        Part(
            {"x": index, "y": exps},
            label=r"$\lambda_i$",
            style={"marker": "circle", "markersize": 8.0},
        )
    )

    try:
        dky: float | None = float(kaplan_yorke_dimension(exps))
    except Exception:  # an incomplete / non-finite spectrum has no dimension
        dky = None
    n_pos = int(np.sum(exps > max(1e-6, 1e-3 * float(np.max(np.abs(exps))))))
    title = "Lyapunov spectrum" + (f" ($D_{{KY}}$ = {dky:.4g})" if dky is not None else "")

    return Geometry(
        "lyapunov_spectrum",
        make_frame(FrameSpace.INDEX, 1, ("index $i$",)),
        parts,
        axis_labels=("index $i$", r"$\lambda_i$"),
        title=title,
        legend=True,
        meta={
            **meta,
            "analysis": "lyapunov_spectrum",
            "kaplan_yorke": dky,
            "n_positive": n_pos,
            "exponents": [float(v) for v in exps],
        },
    )


# ---------------------------------------------------------------------------
# lyapunov_convergence
# ---------------------------------------------------------------------------


def _resolve_convergence(
    subject: Any,
    *,
    k: int | None,
    steps: int,
    dt: float | None,
    ic: Any | None,
) -> tuple[np.ndarray, np.ndarray, bool, str, dict[str, Any]]:
    """Return ``(times, estimates, is_discrete, name, meta)`` for the convergence curve."""
    from tsdynamics.errors import InvalidInputError

    # An already-recorded (times, estimates) pair — the pure-data door.
    if isinstance(subject, tuple) and len(subject) == 2:
        times = np.asarray(subject[0], dtype=float).ravel()
        est = np.atleast_2d(np.asarray(subject[1], dtype=float))
        if est.shape[0] != times.size and est.shape[1] == times.size:
            est = est.T
        if times.size == 0 or est.shape[0] != times.size:
            raise InvalidInputError(
                "lyapunov_convergence got a (times, estimates) pair whose shapes do not "
                f"match: times {times.shape}, estimates {est.shape}. Estimates must be "
                "(n_steps, k)."
            )
        return times, est, False, "", {"analysis": "tangent.convergence"}

    from tsdynamics.derived.tangent import TangentSystem

    tangent = subject if isinstance(subject, TangentSystem) else None
    if tangent is None:
        if not _is_system(subject):
            raise InvalidInputError(
                "lyapunov_convergence needs a system, a TangentSystem, or a recorded "
                f"(times, estimates) pair; got a {type(subject).__name__}."
            )
        tangent = TangentSystem(subject, k)
    times, est = tangent.convergence(steps, dt, ic=ic)
    return (
        times,
        est,
        bool(tangent.is_discrete),
        type(tangent.system).__name__,
        {
            "analysis": "tangent.convergence",
            "k": int(tangent.k),
            "steps": int(steps),
            "n_or_dt": dt,
        },
    )


def _settled_range(est: np.ndarray, after: float) -> tuple[float, float] | None:
    """Vertical range chosen from the *settled* part of a convergence record.

    A running time-average opens enormous — the first few steps of a Benettin
    run give estimates orders of magnitude above the answer — so autoscaling on
    the whole record squashes the entire settling story into a line at the
    bottom of the axes.  (Measured on Lorenz: the first sample is ~40x the
    converged value, and the default view showed nothing.)  The transient is
    real information and is still **drawn**; it just does not get to set the
    scale.
    """
    n = est.shape[0]
    start = min(max(int(after * n), 1), max(n - 1, 0))
    tail = est[start:]
    finite = tail[np.isfinite(tail)]
    if finite.size == 0:  # pragma: no cover - defensive
        return None
    lo, hi = float(np.min(finite)), float(np.max(finite))
    pad = 0.08 * (hi - lo) if hi > lo else max(0.1 * abs(hi), 0.1)
    return lo - pad, hi + pad


def _example_convergence(_primitive: str) -> tuple[Any, dict[str, Any]]:
    """Return the gate's subject: a recorded pair, so no system is ever integrated."""
    t = np.linspace(1.0, 200.0, 120)
    settle = np.array([0.906, 0.0, -14.572])
    est = settle[None, :] + (np.array([2.0, 1.0, -3.0])[None, :] / t[:, None])
    return (t, est), {}


@plot_transform(
    name="lyapunov_convergence",
    source="data",
    kind=PlotKind.DIAGNOSTIC_CURVE,
    frame=FrameSpace.SCALING,
    ndim=1,
    role=OverlayRole.BASE,
    default_primitive="line",
    primitives=("line", "points"),
    analysis="tsdynamics.derived.tangent.TangentSystem.convergence",
    example=_example_convergence,
    doc="Each exponent's running Benettin estimate against time — has it settled?",
)
def lyapunov_convergence(
    subject: Any,
    *,
    k: int | None = None,
    steps: int = 2000,
    dt: float | None = None,
    ic: Any | None = None,
    zero_line: bool = True,
    autoscale_after: float = 0.05,
) -> Geometry:
    r"""Draw the running Lyapunov estimates against time — the convergence read-out.

    A reported Lyapunov exponent is a *time average*, and a time average is only
    an answer once it has stopped moving.  This is the figure that shows it has:
    one curve per exponent, each labelled with the value it settled on, so the
    number in the caption and the curve behind it are the same object.

    Parameters
    ----------
    subject : System, TangentSystem, or (times, estimates)
        A system (a :class:`~tsdynamics.derived.tangent.TangentSystem` is built
        over it), an existing tangent system, or an already-recorded pair from
        :meth:`~tsdynamics.derived.tangent.TangentSystem.convergence`.
    k : int, optional
        Number of exponents to track (default: the system's dimension).
    steps : int, optional
        Tangent steps to record.  Default ``2000``.
    dt : float, optional
        Per-step increment — ``dt`` for a flow, an iteration count for a map.
        ``None`` uses the family default.
    ic : array-like, optional
        Initial condition.
    zero_line : bool, optional
        Draw the :math:`\lambda = 0` reference.  Default ``True``: the sign of
        the leading exponent is the reason to look at this plot.
    autoscale_after : float, optional
        Fraction of the record ignored when choosing the **vertical range**
        (default ``0.05``).  The opening estimates of a running average are
        enormous and would otherwise squash the settling into a flat line; the
        transient is still drawn, it just does not set the scale.  Pass ``0.0``
        to scale on everything.

    Returns
    -------
    Geometry
        A ``scaling``-framed geometry over time: the zero line plus one curve
        per exponent, labelled ``$\lambda_i$ -> <settled value>``.

    Raises
    ------
    tsdynamics.errors.InvalidInputError
        If the subject is not a system / tangent system / recorded pair, or the
        recorded pair's shapes disagree.

    Examples
    --------
    >>> ts.plot(ts.systems.Lorenz(), "lyapunov_convergence", steps=1500, dt=0.5)  # doctest: +SKIP
    """
    times, est, is_discrete, name, meta = _resolve_convergence(
        subject, k=k, steps=steps, dt=dt, ic=ic
    )
    parts: list[Part] = []
    if zero_line:
        parts.append(_hline_part(times, 0.0, r"$\lambda = 0$"))
    for i in range(est.shape[1]):
        final = float(est[-1, i])
        parts.append(
            Part(
                {"x": times, "y": est[:, i].astype(float)},
                label=rf"$\lambda_{{{i + 1}}} \to {final:.4g}$",
            )
        )
    axis = "iteration" if is_discrete else "time"
    yrange = _settled_range(est, autoscale_after) if autoscale_after > 0.0 else None
    return Geometry(
        "lyapunov_convergence",
        make_frame(FrameSpace.SCALING, 1, (axis,)),
        parts,
        axis_labels=(axis, "running Lyapunov estimate"),
        axis_limits=(None, yrange),
        title=f"Lyapunov convergence{f' — {name}' if name else ''}",
        legend=True,
        meta={**meta, "settled": [float(v) for v in est[-1]]},
    )


# ---------------------------------------------------------------------------
# gali_curves
# ---------------------------------------------------------------------------


def _gali_results(subject: Any, k: Any, options: dict[str, Any]) -> list[Any]:
    """Return the list of :class:`GALIResult` objects to draw."""
    from tsdynamics.errors import InvalidInputError

    if hasattr(subject, "values") and hasattr(subject, "times") and hasattr(subject, "k"):
        if options:
            raise InvalidInputError(
                f"gali_curves got option(s) {sorted(options)}, which configure the "
                "estimator; they only apply when the subject is a system."
            )
        return [subject]
    if not _is_system(subject):
        members = list(subject) if isinstance(subject, (list, tuple)) else []
        if members and all(hasattr(m, "times") and hasattr(m, "k") for m in members):
            return members
        raise InvalidInputError(
            "gali_curves needs a system, a GALIResult, or a sequence of GALIResults; got a "
            f"{type(subject).__name__}."
        )

    from tsdynamics.analysis.chaos import gali

    ks = [int(k)] if isinstance(k, (int, np.integer)) else [int(v) for v in k]
    return [gali(subject, kk, **options) for kk in ks]


def _example_gali(_primitive: str) -> tuple[Any, dict[str, Any]]:
    """Return the gate's subject: a synthetic GALI result, no variational integration."""
    from tsdynamics.analysis.chaos.gali import GALIResult

    t = np.linspace(1.0, 100.0, 150)
    return GALIResult(k=2, times=t, values=1.0 / t, is_discrete=False), {}


@plot_transform(
    name="gali_curves",
    source="data",
    kind=PlotKind.DIAGNOSTIC_CURVE,
    frame=FrameSpace.SCALING,
    ndim=1,
    role=OverlayRole.BASE,
    default_primitive="line",
    primitives=("line", "points"),
    analysis="tsdynamics.analysis.chaos.gali",
    example=_example_gali,
    doc="GALI_k against time on log axes, with the analytic power-law reference.",
)
def gali_curves(
    subject: Any,
    *,
    k: int | Sequence[int] = 2,
    frequencies: int = 1,
    reference_at: float = 0.5,
    floor: float = _LOG_FLOOR,
    final_time: float | None = None,
    n: int | None = None,
    dt: float | None = None,
    transient: float | None = None,
    ic: Any | None = None,
    seed: int | None = None,
) -> Geometry:
    r"""GALI\ :sub:`k` against time on log axes, with the power-law reference drawn.

    The Generalized Alignment Index separates order from chaos by *how* it
    decays, so the figure has to make the decay law readable.  Both axes are
    :math:`\log_{10}` (in the data — see this module's docstring), which turns
    the regular-orbit prediction

    .. math:: \mathrm{GALI}_k(t) \propto t^{-(k-s)}

    for an orbit on an :math:`s`-frequency torus (Skokos et al. 2007) into a
    straight line of known slope.  That reference is drawn, anchored to the
    start of each curve: a regular orbit tracks it, a chaotic one — which decays
    *exponentially* — plunges away from it and hits the floor.

    Parameters
    ----------
    subject : System, GALIResult, or sequence of GALIResult
        A system (then :func:`~tsdynamics.analysis.chaos.gali` is run for each
        requested ``k``), or already-computed result(s).
    k : int or sequence of int, optional
        Which GALI orders to compute when ``subject`` is a system.  Default
        ``2``; pass ``(2, 3, 4)`` for the usual family.
    frequencies : int, optional
        The number of frequencies :math:`s` the power-law reference assumes.
        Default ``1`` (a periodic orbit / limit cycle), giving slope
        :math:`-(k-1)`.  Pass ``0`` to omit the reference.  **The power law is
        the Hamiltonian regular-orbit prediction**: a dissipative system's
        regular orbit decays exponentially instead, so the guide is a reference
        line, never a fit.  For a 2-degree-of-freedom Hamiltonian orbit on a
        2-torus (Hénon--Heiles) pass ``frequencies=2``.
    reference_at : float, optional
        Where along the record the reference guide touches its curve, as a
        fraction (default ``0.5``, the middle sample).  A guide anchored at the
        first sample sits decades away from the data, because
        :math:`\mathrm{GALI}_k` holds near 1 through the alignment transient
        before the asymptotic law takes over.
    floor : float, optional
        Values at or below this are clipped before taking the logarithm (GALI
        underflows to exactly ``0`` for a chaotic orbit).  Default ``1e-16``.
    final_time, n, dt, transient, ic, seed
        Forwarded to :func:`~tsdynamics.analysis.chaos.gali` when ``subject`` is
        a system; each defaults to ``None``, meaning *"leave ``gali``'s own
        default alone"* (so ``seed=None`` here keeps ``gali``'s deterministic
        ``seed=0`` rather than randomising the deviation frame).

    Returns
    -------
    Geometry
        A ``scaling``-framed geometry over :math:`\log_{10} t`: one curve per
        requested order, plus one dashed power-law reference per curve.

    Raises
    ------
    tsdynamics.errors.InvalidInputError
        If the subject is not a system or GALI result, or estimator options were
        passed with an already-computed result.

    Examples
    --------
    >>> ts.plot(ts.systems.Lorenz(), "gali_curves", k=(2, 3))  # doctest: +SKIP
    """
    results = _gali_results(
        subject,
        k,
        _given(final_time=final_time, n=n, dt=dt, transient=transient, ic=ic, seed=seed),
    )
    discrete = bool(getattr(results[0], "is_discrete", False))
    axis = r"$\log_{10} n$" if discrete else r"$\log_{10} t$"

    parts: list[Part] = []
    orders: list[int] = []
    for res in results:
        t = np.asarray(res.times, dtype=float).ravel()
        v = np.asarray(res.values, dtype=float).ravel()
        good = np.isfinite(t) & (t > 0.0) & np.isfinite(v)
        t, v = t[good], v[good]
        if t.size == 0:
            continue
        order = int(getattr(res, "k", 0))
        orders.append(order)
        logt = np.log10(t)
        logv = np.log10(np.maximum(v, floor))
        parts.append(Part({"x": logt, "y": logv}, label=rf"$\mathrm{{GALI}}_{{{order}}}$"))

        slope = -(order - int(frequencies))
        if frequencies and slope <= 0:
            # Anchor the guide *on* the curve, at the sample ``reference_at`` of
            # the way through the record, so its slope is compared against the
            # data next to it.  Anchoring at the first sample instead pushes the
            # guide decades away from the curve (GALI_k holds near 1 through the
            # alignment transient before the asymptotic law takes over), which is
            # a guide nobody can read a slope off.
            j = min(max(int(reference_at * (logt.size - 1)), 0), logt.size - 1)
            ref = logv[j] + slope * (logt - logt[j])
            # Clip the guide to the curve's own vertical span: an unclipped
            # steep guide runs off the top of the axes and drags the autoscale
            # with it, shrinking the data it was meant to be compared against.
            span = float(np.max(logv)) - float(np.min(logv))
            margin = 0.05 * span if span > 0 else 0.5
            inside = (ref >= float(np.min(logv)) - margin) & (ref <= float(np.max(logv)) + margin)
            label = (
                rf"$\propto t^{{{slope}}}$ ($s={int(frequencies)}$)"
                if slope < 0
                else rf"constant ($k \leq s={int(frequencies)}$)"
            )
            if bool(np.any(inside)):
                parts.append(
                    Part(
                        {"x": logt[inside], "y": ref[inside]},
                        label=label,
                        style=_REFERENCE_STYLE,
                        primitive="line",
                    )
                )

    if not parts:
        from tsdynamics.errors import InvalidInputError

        raise InvalidInputError(
            "gali_curves found no finite samples to draw; the GALI run produced an empty "
            "or non-finite series."
        )

    return Geometry(
        "gali_curves",
        make_frame(FrameSpace.SCALING, 1, (axis,)),
        parts,
        axis_labels=(axis, r"$\log_{10} \mathrm{GALI}_k$"),
        title="GALI" + (f" (k = {', '.join(str(o) for o in orders)})" if orders else ""),
        legend=True,
        meta={
            "analysis": "gali",
            "k": orders,
            "frequencies": int(frequencies),
            "floor": float(floor),
            "final": [float(np.asarray(r.values)[-1]) for r in results if np.size(r.values)],
        },
    )


# ---------------------------------------------------------------------------
# zero_one_pq_plane
# ---------------------------------------------------------------------------


def _example_pq(_primitive: str) -> tuple[Any, dict[str, Any]]:
    """Return the gate's subject: a synthetic bounded translation plane."""
    from tsdynamics.analysis.chaos.zero_one import ZeroOneResult

    n = np.arange(400.0)
    return (
        ZeroOneResult(value=0.02, p=np.cos(0.7 * n), q=np.sin(0.7 * n)),
        {},
    )


@plot_transform(
    name="zero_one_pq_plane",
    source="data",
    kind=PlotKind.PHASE_PORTRAIT_2D,
    frame=FrameSpace.STATE2,
    ndim=2,
    role=OverlayRole.BASE,
    default_primitive="line",
    primitives=("line", "points", "density"),
    presentation=Presentation(aspect="equal"),
    analysis="tsdynamics.analysis.chaos.zero_one_test",
    example=_example_pq,
    doc="The (p, q) translation plane of the 0-1 test: bounded blob or Brownian sprawl.",
)
def zero_one_pq_plane(
    subject: Any,
    *,
    component: int | None = None,
    final_time: float | None = None,
    n: int | None = None,
    dt: float | None = None,
    transient: float | None = None,
    ic: Any | None = None,
    n_cut: int | None = None,
    seed: int | None = None,
    oversampling: str | None = None,
) -> Geometry:
    r"""Draw the Gottwald--Melbourne translation plane :math:`(p_c, q_c)`.

    The 0--1 test drives a skew translation with the observable and asks whether
    the translation variables stay **bounded** (regular) or **diffuse** like a
    Brownian path (chaotic).  The indicator :math:`K` is a number summarising
    that; this is the picture it summarises, and looking at it is the standard
    guard against the test's known failure mode — an oversampled observable,
    whose plane drifts smoothly instead of diffusing.

    Parameters
    ----------
    subject : ZeroOneResult, System, Trajectory, or 1-D array
        An already-run test, or anything
        :func:`~tsdynamics.analysis.chaos.zero_one_test` accepts.
    component, final_time, n, dt, transient, ic, n_cut, seed, oversampling
        Forwarded to :func:`~tsdynamics.analysis.chaos.zero_one_test` when the
        subject is not already a result; each defaults to ``None``, meaning
        *"leave the test's own default alone"*.  ``component=`` is required for
        a multi-component system, and a coarse ``dt=`` is what keeps successive
        flow samples decorrelated.

    Returns
    -------
    Geometry
        A ``state2``, equal-aspect geometry on the ``(p_c, q_c)`` axes — which
        are *not* the state-space axes, so this never silently overlays a phase
        portrait.  The starting point is marked, so the sprawl's extent is read
        against where it began.

    Raises
    ------
    tsdynamics.errors.InvalidInputError
        If the test captured no translation plane.

    Examples
    --------
    ``component=`` is not optional here for a multi-dimensional system — unlike
    the series diagnostics, which default to the first component, this one
    forwards ``None`` to the test, whose refusal to pick an observable for you is
    deliberate.

    >>> ts.plot(ts.systems.Lorenz(), "zero_one_pq_plane", component=0, dt=0.5)  # doctest: +SKIP
    """
    from tsdynamics.errors import InvalidInputError

    test = _given(
        component=component,
        final_time=final_time,
        n=n,
        dt=dt,
        transient=transient,
        ic=ic,
        n_cut=n_cut,
        seed=seed,
        oversampling=oversampling,
    )
    if hasattr(subject, "p") and hasattr(subject, "q"):
        if test:
            raise InvalidInputError(
                f"zero_one_pq_plane got option(s) {sorted(test)}, which configure the test; "
                "they only apply when the subject is a system or a series."
            )
        result = subject
    else:
        from tsdynamics.analysis.chaos import zero_one_test

        result = zero_one_test(subject, **test)

    p = np.asarray(result.p, dtype=float).ravel()
    q = np.asarray(result.q, dtype=float).ravel()
    if p.size == 0 or q.size == 0:
        raise InvalidInputError(
            "zero_one_pq_plane: the 0-1 test captured no translation plane (p/q are empty), "
            "so there is nothing to draw. Re-run zero_one_test on a longer record."
        )

    k_value = float(result)
    verdict = "chaotic" if k_value > 0.5 else "regular"
    parts = [
        Part({"x": p, "y": q}, label="$(p_c, q_c)$"),
        Part(
            {"x": p[:1], "y": q[:1]},
            label="start",
            style={"marker": "x", "markersize": 9.0, "color": "black"},
            primitive="points",
        ),
    ]
    return Geometry(
        "zero_one_pq_plane",
        make_frame(FrameSpace.STATE2, 2, ("$p_c$", "$q_c$")),
        parts,
        axis_labels=("$p_c$", "$q_c$"),
        aspect="equal",
        title=f"0--1 test translation plane ($K$ = {k_value:.3g}, {verdict})",
        legend=True,
        meta={
            **(dict(result.meta) if getattr(result, "meta", None) else {}),
            "analysis": "zero_one_test",
            "K": k_value,
            "verdict": verdict,
        },
    )


# ---------------------------------------------------------------------------
# scaling_fit
# ---------------------------------------------------------------------------

#: Per-estimator axis labels for a scaling curve, keyed by ``DimensionResult.kind``.
_SCALING_LABELS: dict[str, tuple[str, str]] = {
    "correlation": (r"$\log r$", r"$\log C(r)$"),
    "generalized": (r"$\log \epsilon$", "partition ordinate"),
    "fixed_mass": (r"$\langle \log r_k \rangle$", r"$\psi(k)$"),
}

#: The three inspection views of one fit.
_SCALING_VIEWS: tuple[str, ...] = ("fit", "residuals", "local_slopes")


def _scaling_pieces(subject: Any) -> tuple[np.ndarray, np.ndarray, int, int, float, float, float]:
    """Return ``(x, y, lo, hi, slope, intercept, stderr)`` from a scaling result."""
    from tsdynamics.errors import InvalidInputError

    x = np.asarray(getattr(subject, "abscissa", ()), dtype=float).ravel()
    y = np.asarray(getattr(subject, "ordinate", ()), dtype=float).ravel()
    if x.size == 0 or x.size != y.size:
        raise InvalidInputError(
            "scaling_fit needs a ScalingResult carrying its curve (abscissa / ordinate) — "
            f"got {type(subject).__name__} with abscissa {x.shape} and ordinate {y.shape}. "
            "Every dimension estimator, lyapunov_from_data and uncertainty_exponent return "
            "one."
        )
    lo, hi = (int(v) for v in getattr(subject, "fit_region", (0, x.size - 1)))
    lo = max(0, min(lo, x.size - 1))
    hi = max(lo, min(hi, x.size - 1))
    return (
        x,
        y,
        lo,
        hi,
        float(getattr(subject, "estimate", 0.0)),
        float(getattr(subject, "intercept", 0.0)),
        float(getattr(subject, "stderr", 0.0)),
    )


def _example_scaling(_primitive: str) -> tuple[Any, dict[str, Any]]:
    """Return the gate's subject: a synthetic log--log curve with a clean scaling region."""
    from tsdynamics.analysis._result_scaling import ScalingResult

    x = np.linspace(-4.0, 0.0, 40)
    y = 2.05 * x - 1.2 + 0.4 * np.tanh(3.0 * (x + 3.0)) - 0.4
    return (
        ScalingResult(
            estimate=2.05,
            stderr=0.03,
            abscissa=x,
            ordinate=y,
            fit_region=(10, 30),
            intercept=-1.2,
        ),
        {},
    )


@plot_transform(
    name="scaling_fit",
    source="data",
    kind=PlotKind.SCALING_FIT,
    frame=FrameSpace.SCALING,
    ndim=1,
    role=OverlayRole.BASE,
    default_primitive="points",
    primitives=("points", "line"),
    analysis="tsdynamics.analysis._result_scaling.ScalingResult",
    example=_example_scaling,
    doc="A log-log scaling curve with the fitted window delimited (+ residual views).",
)
def scaling_fit(subject: Any, *, view: str = "fit") -> Geometry:
    r"""Draw a scaling curve with its fitted region **delimited**, plus two ways to check it.

    Every fractal dimension, ``lyapunov_from_data`` and ``uncertainty_exponent``
    reports the slope of a straight stretch of a log--log curve.  The number is
    only as good as that stretch, so this figure states it three ways:

    ``view="fit"`` (default)
        The whole curve, the fitted points highlighted, the fit line, **and two
        dashed delimiters at the window edges** — so the region is visible as a
        region, not merely inferable from a marker style.
    ``view="residuals"``
        Curve minus fit against the abscissa, with the zero line and the same
        delimiters.  Structure inside the window (a bow, a step) means the
        "scaling region" is not straight and the slope is not a scaling
        exponent.
    ``view="local_slopes"``
        The pointwise local slope, with the reported estimate as a horizontal
        line.  A genuine scaling region is a *plateau* at that level; this is
        the standard sanity check (Grassberger & Procaccia 1983), and the
        library already computes it as
        :attr:`~tsdynamics.analysis.ScalingResult.local_slopes`.

    Parameters
    ----------
    subject : ScalingResult
        Any result carrying ``abscissa`` / ``ordinate`` / ``fit_region`` /
        ``estimate`` / ``intercept`` — every dimension estimator,
        :func:`~tsdynamics.analysis.lyapunov.lyapunov_from_data`, and
        :func:`~tsdynamics.analysis.basins.uncertainty_exponent`.
    view : {"fit", "residuals", "local_slopes"}, optional
        Which of the three readings to draw.  Default ``"fit"``.

    Returns
    -------
    Geometry
        A ``scaling``-framed geometry whose axes are the estimator's own
        (``log r`` / ``log C(r)`` for a correlation dimension, …).

    Raises
    ------
    tsdynamics.errors.InvalidParameterError
        If ``view`` is not one of the three.
    tsdynamics.errors.InvalidInputError
        If the subject carries no scaling curve.

    Examples
    --------
    >>> d = ts.correlation_dimension(traj)                       # doctest: +SKIP
    >>> ts.plot(d, "scaling_fit")                                # doctest: +SKIP
    >>> ts.plot(d, "scaling_fit", view="local_slopes")           # doctest: +SKIP
    """
    from tsdynamics.errors import InvalidParameterError

    if view not in _SCALING_VIEWS:
        raise InvalidParameterError(
            f"unknown scaling_fit view={view!r}; valid: {', '.join(_SCALING_VIEWS)}."
        )

    x, y, lo, hi, slope, intercept, stderr = _scaling_pieces(subject)
    xlabel, ylabel = _SCALING_LABELS.get(
        str(getattr(subject, "kind", "")), (r"$\log$ scale", r"$\log$ measure")
    )
    n_fit = hi - lo + 1
    slope_label = rf"slope = {slope:.4g} $\pm$ {stderr:.2g}"

    if view == "residuals":
        resid = y - (intercept + slope * x)
        parts = [_hline_part(x, 0.0, "zero")]
        parts.append(Part({"x": x, "y": resid}, label="residual"))
        parts.append(
            Part(
                {"x": x[lo : hi + 1], "y": resid[lo : hi + 1]},
                label=f"fit region ({n_fit} points)",
                style=_FIT_REGION_STYLE,
            )
        )
        parts.extend(_vline_parts((x[lo], x[hi]), resid, "fit window"))
        ylabel = f"residual of {ylabel}"
        title = f"scaling residuals — {slope_label}"
    elif view == "local_slopes":
        local = np.gradient(y, x) if x.size >= 2 else np.full(x.shape, np.nan)
        parts = [_hline_part(x, slope, slope_label)]
        parts.append(Part({"x": x, "y": local}, label="local slope"))
        parts.append(
            Part(
                {"x": x[lo : hi + 1], "y": local[lo : hi + 1]},
                label=f"fit region ({n_fit} points)",
                style=_FIT_REGION_STYLE,
            )
        )
        parts.extend(_vline_parts((x[lo], x[hi]), local, "fit window"))
        ylabel = "local slope"
        title = f"local slopes — plateau at {slope:.4g}?"
    else:
        fit_x = np.array([x[lo], x[hi]])
        parts = [
            Part({"x": x, "y": y}, label="curve"),
            Part(
                {"x": x[lo : hi + 1], "y": y[lo : hi + 1]},
                label=f"fit region ({n_fit} points)",
                style=_FIT_REGION_STYLE,
            ),
            Part(
                {"x": fit_x, "y": intercept + slope * fit_x},
                label=slope_label,
                style={"color": "black", "linewidth": 1.6},
                primitive="line",
            ),
        ]
        parts.extend(_vline_parts((x[lo], x[hi]), y, "fit window"))
        title = f"{type(subject).__name__} — {slope_label}"

    return Geometry(
        "scaling_fit",
        make_frame(FrameSpace.SCALING, 1, (xlabel,)),
        parts,
        axis_labels=(xlabel, ylabel),
        title=title,
        legend=True,
        meta={
            **(dict(subject.meta) if getattr(subject, "meta", None) else {}),
            "analysis": "scaling_fit",
            "view": view,
            "estimate": slope,
            "stderr": stderr,
            "fit_region": (lo, hi),
            "scaling_window": (float(x[lo]), float(x[hi])),
        },
    )


def __dir__() -> list[str]:
    """Expose only the registered transforms to ``dir()`` / autocomplete."""
    return sorted(__all__)
