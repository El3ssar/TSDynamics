r"""Data-capable diagnostic transforms — the curves you read a series *with*.

Seven registered transforms, every one ``source="data"``: they compute from the
samples you already have, and they accept a bare array, a
:class:`~tsdynamics.data.Trajectory` **or** a system (a model gives you data for
free — it integrates one and goes on, recording the choice in ``meta``).

What is here, and why each earns its place:

``psd``
    :math:`S(f)` of a phase-space trajectory.  **Re-admitted** by the owner under
    the rule written into
    :data:`~tsdynamics.viz.transforms._registry.ADMITTED_SERIES_DIAGNOSTICS`:
    *the power spectrum of a phase-space trajectory is a phase-space diagnostic;
    a PSD toolbox with windowing options, detrending and filter design is not.*
    It is how the literature separates **periodic** (discrete lines),
    **quasiperiodic** (incommensurate lines) and **chaotic** (broadband) motion.
    The knob set is therefore deliberately closed — ``method`` and ``nperseg``,
    nothing else — and the gate in ``tests/test_viz_transforms.py`` fails if the
    admitted set grows.
``autocorrelation`` / ``mutual_information``
    The two curves you *choose an embedding delay* from, with the decision each
    one implies drawn on the figure: the :math:`1/e` and first-zero lags for the
    autocorrelation, the first significant local minimum for :math:`I(\\tau)`.
``fnn`` / ``cao``
    The two curves you *choose an embedding dimension* from.  Both estimators
    already computed these and threw them away behind an integer.
``line_lengths``
    :math:`P(l)` and :math:`P(v)` — the diagonal- and vertical-line length
    distributions **DET / L_max / LAM / TT are computed from**.  ``rqa`` reduces
    them to four numbers and discards the distributions; this draws them.
``return_time``
    The first-return / first-passage time distribution of a level set.  Admitted
    under the same rule as the PSD: it is a property of the orbit's visits to a
    region of phase space, not a generic series statistic.

Every one of them is a **thin adapter**: the estimator lives in
:mod:`tsdynamics.analysis` (named in each row's ``analysis=``), or — for the PSD
— in :mod:`scipy.signal`, deliberately *not* re-homed into ``analysis/``, because
a PSD toolbox in this library is exactly what the v6 scope surgery removed.

References
----------
.. [1] Welch, P. D. (1967). "The use of Fast Fourier Transform for the estimation
   of power spectra." *IEEE Trans. Audio Electroacoust.* 15(2), 70-73.
.. [2] Bergé, P., Pomeau, Y. & Vidal, C. (1984). *Order Within Chaos*. Wiley.
   (Chapter on the power spectrum as the periodic / quasiperiodic / chaotic
   discriminant.)
.. [3] Fraser, A. M. & Swinney, H. L. (1986). "Independent coordinates for
   strange attractors from mutual information." *Phys. Rev. A* 33(2), 1134-1140.
.. [4] Kennel, M. B., Brown, R. & Abarbanel, H. D. I. (1992). "Determining
   embedding dimension for phase-space reconstruction using a geometrical
   construction." *Phys. Rev. A* 45(6), 3403-3411.
.. [5] Cao, L. (1997). "Practical method for determining the minimum embedding
   dimension of a scalar time series." *Physica D* 110(1-2), 43-50.
.. [6] Marwan, N., Romano, M. C., Thiel, M. & Kurths, J. (2007). "Recurrence
   plots for the analysis of complex systems." *Physics Reports* 438(5-6),
   237-329.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .._frames import FrameSpace, OverlayRole
from .._visibility import listing_dir
from ..spec import PlotKind
from ._base import Geometry, Part, Presentation, make_frame
from ._data import _component_index, _meta, _split_traj, _title
from ._registry import plot_transform

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

__all__ = [
    "autocorrelation",
    "cao",
    "fnn",
    "line_lengths",
    "mutual_information",
    "psd",
    "return_time",
]

__dir__ = listing_dir(__all__)


# ---------------------------------------------------------------------------
# Subject coercion: an array, a Trajectory, or a system
# ---------------------------------------------------------------------------

#: What a system subject is integrated for when the caller says nothing.  Small
#: on purpose: a diagnostic curve is cheap and a default that silently burns a
#: minute is a worse plot than one that says "pass final_time=".
_DEFAULT_FINAL_TIME = 200.0
_DEFAULT_DT = 0.05
_DEFAULT_STEPS = 5000


def _is_trajectory(subject: Any) -> bool:
    """Whether ``subject`` walks and quacks like a :class:`~tsdynamics.data.Trajectory`."""
    return getattr(subject, "t", None) is not None and getattr(subject, "y", None) is not None


def _trajectory_of(subject: Any, options: dict[str, Any]) -> Any:
    """Return a trajectory for ``subject``, integrating a system if that is what it is.

    A ``data`` transform accepts a system because a model gives you data for
    free.  Every auto-chosen default made here is recorded in the geometry's
    ``meta`` — a plot of half the story with nothing to say so is the failure
    mode this avoids.
    """
    if _is_trajectory(subject):
        return subject
    final_time = options.get("final_time")
    dt = options.get("dt")
    steps = options.get("steps")
    # v6: ``run`` is the one trajectory verb, and ``family`` replaced ``is_discrete``.
    if not hasattr(subject, "run"):
        return None
    if getattr(subject, "family", None) == "map":
        return subject.run(steps=int(steps if steps is not None else _DEFAULT_STEPS))
    return subject.run(
        final_time=float(final_time if final_time is not None else _DEFAULT_FINAL_TIME),
        dt=float(dt if dt is not None else _DEFAULT_DT),
    )


def series_of(
    subject: Any,
    *,
    components: int | str = 0,
    final_time: float | None = None,
    dt: float | None = None,
    steps: int | None = None,
) -> tuple[np.ndarray, float, dict[str, Any], str]:
    """Coerce any accepted subject to ``(values, sample_spacing, meta, title)``.

    Accepts a bare 1-D (or 2-D, column-selected) array, a
    :class:`~tsdynamics.data.Trajectory`, or a **system** — which is integrated
    (or iterated) to produce the samples, with the choice recorded in ``meta``.

    Parameters
    ----------
    subject : ndarray, Trajectory, or System
        What to read the series from.
    components : int or str, optional
        Which component; a name when the source declares ``variables``.
    final_time, dt, steps : optional
        Integration controls, used **only** when ``subject`` is a system.  ``dt``
        additionally overrides the sample spacing inferred from a trajectory's
        time axis (and is the spacing of a bare array, default ``1``).

    Returns
    -------
    values : ndarray
        The 1-D series.
    spacing : float
        The sample spacing in time units (``1.0`` for a bare array or a map).
    meta : dict
        Provenance, including any integration defaults this function chose.
    title : str
        A figure title read off the source, or ``""``.
    """
    from tsdynamics.errors import InvalidInputError

    traj = _trajectory_of(subject, {"final_time": final_time, "dt": dt, "steps": steps})
    if traj is not None:
        t, y, names, is_discrete = _split_traj(traj)
        idx = _component_index(components, names, y.shape[1])
        values = np.asarray(y[:, idx], dtype=float)
        spacing = float(dt) if dt is not None else _spacing_of(t, is_discrete)
        meta = {**_meta(traj), "component": idx, "sample_spacing": spacing}
        if traj is not subject:  # a system was integrated for us
            meta["integrated_for_plot"] = {
                "final_time": float(t[-1]) if t.size else 0.0,
                "n_samples": int(values.size),
            }
        return values, spacing, meta, _title(traj)

    arr = np.asarray(subject, dtype=float)
    if arr.ndim > 2:
        raise InvalidInputError(
            f"expected a 1-D series, a (T, dim) array, a Trajectory or a system; got an "
            f"array of shape {arr.shape}."
        )
    if arr.ndim == 2:
        idx = (
            _component_index(components, None, arr.shape[1])
            if not isinstance(components, str)
            else 0
        )
        values = np.asarray(arr[:, idx], dtype=float)
    else:
        values = arr.ravel()
    spacing = float(dt) if dt is not None else 1.0
    return values, spacing, {"sample_spacing": spacing}, ""


def _spacing_of(t: np.ndarray, is_discrete: bool) -> float:
    """Return the sample spacing of a time axis (``1.0`` for a map or a degenerate axis)."""
    if is_discrete or t.size < 2:
        return 1.0
    step = float(t[1] - t[0])
    return step if np.isfinite(step) and step > 0.0 else 1.0


def _require(values: np.ndarray, minimum: int, what: str) -> np.ndarray:
    """Return ``values`` if it is long enough and finite, else raise."""
    from tsdynamics.errors import InvalidInputError

    if values.size < minimum:
        raise InvalidInputError(f"{what} needs at least {minimum} samples, got {values.size}.")
    if not np.isfinite(values).all():
        finite: np.ndarray = values[np.isfinite(values)]
        if finite.size < minimum:
            raise InvalidInputError(
                f"{what} needs at least {minimum} finite samples, got {finite.size}."
            )
        return finite
    return values


def _marker_part(x: float, lo: float, hi: float, label: str) -> Part:
    """Return a vertical reference segment at ``x``, pinned to the ``line`` primitive.

    The reading of a diagnostic curve *is* the crossing it implies, so the chosen
    lag / dimension is drawn on the figure rather than left in ``meta``.  It is
    pinned because it is an annotation, not data: it stays a line whether the
    curve itself is drawn as a line, points or a staircase.
    """
    return Part(
        {"x": np.array([x, x], dtype=float), "y": np.array([lo, hi], dtype=float)},
        label=label,
        style={"linestyle": "dashed", "linewidth": 1.0},
        primitive="line",
    )


def _span(values: np.ndarray) -> tuple[float, float]:
    """Return a finite ``(lo, hi)`` spanning ``values``, usable as a marker's extent."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:  # pragma: no cover - defensive
        return (0.0, 1.0)
    lo, hi = float(finite.min()), float(finite.max())
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


# ---------------------------------------------------------------------------
# Example subjects for the compatibility gate
# ---------------------------------------------------------------------------


def _demo_series(n: int = 512) -> np.ndarray:
    """Return a deterministic quasi-periodic series (every diagnostic is non-degenerate)."""
    t = np.linspace(0.0, 40.0, n)
    wave = np.sin(t) + 0.6 * np.sin(np.sqrt(2.0) * t) + 0.05 * np.cos(7.0 * t)
    return np.asarray(wave, dtype=float)


def _demo_orbit(n: int = 400) -> np.ndarray:
    """Return a chaotic logistic orbit — a compact recurrence / return-time subject."""
    x = np.empty(n, dtype=float)
    x[0] = 0.4
    for i in range(n - 1):
        x[i + 1] = 3.9 * x[i] * (1.0 - x[i])
    return x


# ---------------------------------------------------------------------------
# psd
# ---------------------------------------------------------------------------


@plot_transform(
    name="psd",
    source="data",
    kind=PlotKind.DIAGNOSTIC_CURVE,
    frame=FrameSpace.SCALING,
    ndim=1,
    role=OverlayRole.BASE,
    default_primitive="line",
    primitives=("line", "points"),
    presentation=Presentation(legend=False),
    example=lambda primitive: (_demo_series(), {}),
    doc="Power spectral density S(f) of one trajectory component.",
)
def psd(
    subject: Any,
    *,
    components: int | str = 0,
    method: str = "welch",
    nperseg: int | None = None,
    final_time: float | None = None,
    dt: float | None = None,
    steps: int | None = None,
) -> Geometry:
    r"""Power spectral density :math:`S(f)` of a phase-space trajectory.

    The figure the nonlinear-dynamics literature reads a regime off: a
    **periodic** orbit gives discrete lines at :math:`f_0` and its harmonics, a
    **quasiperiodic** one gives lines at incommensurate frequencies and their
    integer combinations, and a **chaotic** one gives a broadband continuum.

    Two estimators, and deliberately no more (see the module docstring for the
    admission rule this transform ships under): ``"periodogram"`` — the raw
    :math:`|\hat{x}(f)|^2`, sharpest for resolving discrete lines — and
    ``"welch"`` (the default) — averaged over overlapping Hann-windowed segments,
    which trades line resolution for a stable estimate of the broadband floor.
    The mean is removed before transforming; that is not a *detrending option*,
    it is the removal of the DC spike that would otherwise dominate the plot.

    What correct looks like
    -----------------------
    Feed it a pure sinusoid: the peak sits at that frequency, in **inverse time
    units** (so doubling ``dt`` halves the reported frequency of the same
    oscillation).  A two-tone signal peaks at both tones.  Gate:
    ``tests/test_viz_truth.py::TestASpectrumPeaksWhereTheSignalOscillates``.

    Parameters
    ----------
    subject : ndarray, Trajectory, or System
        The series, the trajectory, or the system to integrate for one.
    components : int or str, optional
        Which component to transform.
    method : {"welch", "periodogram"}, optional
        The estimator.  Default ``"welch"``.
    nperseg : int, optional
        Welch segment length in samples.  ``None`` splits the series into eight
        overlapping segments, which is a reasonable variance/resolution trade for
        a plot.  Ignored by ``"periodogram"``.
    final_time, dt, steps : optional
        Passed to :func:`series_of` (used when ``subject`` is a system; ``dt``
        also sets the sample spacing).

    Returns
    -------
    Geometry
        A ``scaling``-framed geometry with channels ``x`` (frequency, in inverse
        time units) and ``y`` (:math:`S(f)`).

    Raises
    ------
    tsdynamics.errors.InvalidParameterError
        If ``method`` is not one of the two admitted estimators.
    tsdynamics.errors.InvalidInputError
        If the series is shorter than 8 samples.

    Notes
    -----
    The geometry **declares log-log axes**, because a power spectrum on linear
    axes is a spike at ``f = 0`` and a flat line — a picture of nothing, on every
    backend, whatever the data.  It is a default and not a lock: the spec renders
    itself and the tweak runs afterwards, so::

        ts.plot(traj, "psd").rescale(x="linear", y="linear").plot()

    puts them back.

    References
    ----------
    Welch, P. D. (1967). *IEEE Trans. Audio Electroacoust.* 15(2), 70-73.
    """
    from scipy import signal

    from tsdynamics.errors import InvalidParameterError

    values, spacing, meta, title = series_of(
        subject, components=components, final_time=final_time, dt=dt, steps=steps
    )
    values = _require(values, 8, "a power spectrum")
    fs = 1.0 / spacing
    if method == "welch":
        seg = int(nperseg) if nperseg is not None else max(8, values.size // 8)
        seg = min(seg, values.size)
        freq, power = signal.welch(values, fs=fs, nperseg=seg, window="hann", detrend="constant")
        estimator = f"welch(nperseg={seg})"
    elif method == "periodogram":
        freq, power = signal.periodogram(values, fs=fs, detrend="constant")
        estimator = "periodogram"
    else:
        raise InvalidParameterError(
            f"unknown psd method {method!r}; use 'welch' (averaged, stable broadband floor) "
            "or 'periodogram' (raw, sharpest discrete lines). This transform is a "
            "phase-space diagnostic, not a spectral toolbox, so the choice stops there."
        )
    return Geometry(
        "psd",
        make_frame(FrameSpace.SCALING, ("f",)),
        channels={"x": np.asarray(freq, dtype=float), "y": np.asarray(power, dtype=float)},
        label="$S(f)$",
        axis_labels=("frequency $f$", "$S(f)$"),
        axis_scales=("log", "log"),
        title=f"{title} power spectrum".strip(),
        meta={**meta, "psd_method": estimator, "sampling_rate": fs},
    )


# ---------------------------------------------------------------------------
# autocorrelation
# ---------------------------------------------------------------------------


@plot_transform(
    name="autocorrelation",
    source="data",
    kind=PlotKind.DIAGNOSTIC_CURVE,
    frame=FrameSpace.SCALING,
    ndim=1,
    role=OverlayRole.BASE,
    default_primitive="line",
    primitives=("line", "points", "steps"),
    analysis="tsdynamics.analysis.embedding.delay.autocorrelation",
    example=lambda primitive: (_demo_series(), {"max_delay": 40}),
    doc="The autocorrelation C(tau) with its 1/e and first-zero crossings marked.",
)
def autocorrelation(
    subject: Any,
    *,
    components: int | str = 0,
    max_delay: int = 50,
    final_time: float | None = None,
    dt: float | None = None,
    steps: int | None = None,
) -> Geometry:
    r"""Autocorrelation :math:`C(\tau)` with the two lags a delay embedding reads off it.

    Both classical delay criteria are drawn on the figure rather than left for
    the caller to eyeball: the lag where :math:`C` first falls to :math:`1/e`
    (the decorrelation time), and the lag of the first zero crossing.  Both are
    also in ``meta`` as ``tau_1_over_e`` / ``tau_first_zero`` (``None`` when the
    curve does not reach them inside ``max_delay`` — an honest answer, not an
    extrapolation).

    What correct looks like
    -----------------------
    :math:`C(0) = 1` always.  For a sinusoid of period :math:`T` the curve
    peaks again at lag :math:`T` — read on the **sample** axis, that is
    :math:`T/\mathrm{d}t` samples.  White noise falls to the noise floor in one
    sample.  Gate: ``tests/test_viz_truth.py::TestAutocorrelationFindsThePeriod``.

    Parameters
    ----------
    subject : ndarray, Trajectory, or System
        The series, the trajectory, or the system to integrate for one.
    components : int or str, optional
        Which component.
    max_delay : int, optional
        Largest lag drawn, in **samples**.
    final_time, dt, steps : optional
        Passed to :func:`series_of`.

    Returns
    -------
    Geometry
        A ``scaling``-framed geometry on the lag axis ``tau`` — the same frame
        :func:`mutual_information` uses, so the two delay diagnostics
        legitimately overlay.

    References
    ----------
    Fraser, A. M. & Swinney, H. L. (1986). *Phys. Rev. A* 33(2), 1134-1140.
    """
    from tsdynamics.analysis.embedding import autocorrelation as _acf

    values, spacing, meta, title = series_of(
        subject, components=components, final_time=final_time, dt=dt, steps=steps
    )
    values = _require(values, 3, "an autocorrelation")
    curve = np.asarray(_acf(values, max_delay=int(max_delay)), dtype=float)
    lags = np.arange(curve.size, dtype=float)

    tau_e = _first_below(curve, float(np.exp(-1.0)))
    tau_zero = _first_below(curve, 0.0)
    lo, hi = _span(curve)
    parts = [Part({"x": lags, "y": curve}, label=r"$C(\tau)$")]
    if tau_e is not None:
        parts.append(_marker_part(float(tau_e), lo, hi, rf"$1/e$ at $\tau$ = {tau_e}"))
    if tau_zero is not None:
        parts.append(_marker_part(float(tau_zero), lo, hi, rf"first zero at $\tau$ = {tau_zero}"))
    return Geometry(
        "autocorrelation",
        make_frame(FrameSpace.SCALING, ("tau",)),
        parts,
        axis_labels=(r"lag $\tau$ (samples)", r"$C(\tau)$"),
        title=f"{title} autocorrelation".strip(),
        meta={
            **meta,
            "max_delay": int(max_delay),
            "tau_1_over_e": tau_e,
            "tau_first_zero": tau_zero,
            "sample_spacing": spacing,
        },
    )


def _first_below(curve: np.ndarray, level: float) -> int | None:
    """Return the first lag at which ``curve`` is at or below ``level``, else ``None``."""
    hits = np.flatnonzero(curve <= level)
    return int(hits[0]) if hits.size else None


# ---------------------------------------------------------------------------
# mutual_information
# ---------------------------------------------------------------------------


@plot_transform(
    name="mutual_information",
    source="data",
    kind=PlotKind.DIAGNOSTIC_CURVE,
    frame=FrameSpace.SCALING,
    ndim=1,
    role=OverlayRole.BASE,
    default_primitive="line",
    primitives=("line", "points", "steps"),
    analysis="tsdynamics.analysis.embedding.delay.mutual_information",
    example=lambda primitive: (_demo_series(), {"max_delay": 30}),
    doc="Time-delayed mutual information I(tau) with the chosen delay marked.",
)
def mutual_information(
    subject: Any,
    *,
    components: int | str = 0,
    max_delay: int = 50,
    bins: int | None = None,
    final_time: float | None = None,
    dt: float | None = None,
    steps: int | None = None,
) -> Geometry:
    r"""Time-delayed mutual information :math:`I(\tau)`, with the selected delay marked.

    :func:`~tsdynamics.analysis.embedding.optimal_delay` reduces this curve to one
    integer.  The integer is the *decision*; the curve is the *evidence*, and a
    delay read off a curve with no clear first minimum should not be trusted —
    which you can only see by looking.

    Parameters
    ----------
    subject : ndarray, Trajectory, or System
        The series, the trajectory, or the system to integrate for one.
    components : int or str, optional
        Which component.
    max_delay : int, optional
        Largest lag evaluated, in **samples**.
    bins : int, optional
        Histogram bins per axis for the estimator.  ``None`` uses its
        sample-size-dependent default.
    final_time, dt, steps : optional
        Passed to :func:`series_of`.

    Returns
    -------
    Geometry
        On the same ``tau`` lag axis as :func:`autocorrelation`.

    References
    ----------
    Fraser, A. M. & Swinney, H. L. (1986). *Phys. Rev. A* 33(2), 1134-1140.
    """
    from tsdynamics.analysis.embedding import mutual_information as _mi

    values, _, meta, title = series_of(
        subject, components=components, final_time=final_time, dt=dt, steps=steps
    )
    values = _require(values, 8, "mutual information")
    result = _mi(values, max_delay=int(max_delay), bins=bins)
    curve = np.asarray(result.values, dtype=float)
    lags = np.arange(curve.size, dtype=float)
    chosen = int(result.optimal_lag) if curve.size else 0

    lo, hi = _span(curve)
    parts = [
        Part({"x": lags, "y": curve}, label=r"$I(\tau)$"),
        _marker_part(float(chosen), lo, hi, rf"$\tau$ = {chosen}"),
    ]
    return Geometry(
        "mutual_information",
        make_frame(FrameSpace.SCALING, ("tau",)),
        parts,
        axis_labels=(r"delay $\tau$ (samples)", r"$I(\tau)$"),
        title=f"{title} mutual information".strip(),
        meta={**meta, **dict(result.meta), "optimal_lag": chosen},
    )


# ---------------------------------------------------------------------------
# fnn / cao
# ---------------------------------------------------------------------------


@plot_transform(
    name="fnn",
    source="data",
    kind=PlotKind.DIAGNOSTIC_CURVE,
    frame=FrameSpace.SCALING,
    ndim=1,
    role=OverlayRole.BASE,
    default_primitive="line",
    primitives=("line", "points", "steps"),
    analysis="tsdynamics.analysis.embedding.dimension.false_nearest_neighbors",
    example=lambda primitive: (_demo_series(), {"delay": 6, "max_dim": 6}),
    doc="Kennel's false-nearest-neighbour fraction against embedding dimension.",
)
def fnn(
    subject: Any,
    *,
    components: int | str = 0,
    delay: int = 1,
    max_dim: int = 10,
    theiler: int = 0,
    final_time: float | None = None,
    dt: float | None = None,
    steps: int | None = None,
) -> Geometry:
    r"""Draw the false-nearest-neighbour fraction against embedding dimension :math:`d`.

    The curve decays to zero at the dimension where the attractor stops
    self-intersecting; the chosen :math:`m` is marked.  Reading the curve rather
    than the integer is the point — a fraction that decays *gradually* is telling
    you the delay is wrong, and the integer cannot.

    Parameters
    ----------
    subject : ndarray, Trajectory, or System
        The series, the trajectory, or the system to integrate for one.
    components : int or str, optional
        Which component.
    delay : int, optional
        Embedding delay :math:`\tau` in samples.
    max_dim : int, optional
        Largest dimension evaluated.
    theiler : int, optional
        Exclude temporally-close neighbours with :math:`|i-j| \le w`.
    final_time, dt, steps : optional
        Passed to :func:`series_of`.

    Returns
    -------
    Geometry
        A ``scaling``-framed geometry on the dimension axis ``m``.

    References
    ----------
    Kennel, M. B., Brown, R. & Abarbanel, H. D. I. (1992). *Phys. Rev. A* 45, 3403.
    """
    from tsdynamics.analysis.embedding import false_nearest_neighbors

    values, _, meta, title = series_of(
        subject, components=components, final_time=final_time, dt=dt, steps=steps
    )
    values = _require(values, 16, "a false-nearest-neighbour curve")
    result = false_nearest_neighbors(
        values, delay=int(delay), max_dim=int(max_dim), theiler=int(theiler)
    )
    return _dimension_geometry(
        "fnn",
        result,
        curves=[(np.asarray(result.fnn_fraction, dtype=float), "FNN fraction")],
        ylabel="false-neighbour fraction",
        title=f"{title} false nearest neighbours".strip(),
        meta=meta,
    )


@plot_transform(
    name="cao",
    source="data",
    kind=PlotKind.DIAGNOSTIC_CURVE,
    frame=FrameSpace.SCALING,
    ndim=1,
    role=OverlayRole.BASE,
    default_primitive="line",
    primitives=("line", "points", "steps"),
    analysis="tsdynamics.analysis.embedding.dimension.cao_dimension",
    example=lambda primitive: (_demo_series(), {"delay": 6, "max_dim": 6}),
    doc="Cao's averaged-false-neighbour E1(d) / E2(d) against embedding dimension.",
)
def cao(
    subject: Any,
    *,
    components: int | str = 0,
    delay: int = 1,
    max_dim: int = 10,
    theiler: int = 0,
    final_time: float | None = None,
    dt: float | None = None,
    steps: int | None = None,
) -> Geometry:
    r"""Cao's :math:`E_1(d)` and :math:`E_2(d)` against embedding dimension.

    :math:`E_1` saturates at 1 once the embedding unfolds the attractor;
    :math:`E_2` stays near 1 for stochastic data and departs from it for
    deterministic data — which is why *both* curves are drawn, and why an
    estimator that returns only the integer throws away the determinism test.

    The :math:`d = 1` point of both curves is diagnostically unreliable (see
    :func:`~tsdynamics.analysis.embedding.cao_dimension`); it is drawn, because
    silently dropping a data point is worse than a documented caveat.

    Parameters
    ----------
    subject : ndarray, Trajectory, or System
        The series, the trajectory, or the system to integrate for one.
    components : int or str, optional
        Which component.
    delay : int, optional
        Embedding delay :math:`\tau` in samples.
    max_dim : int, optional
        Largest dimension evaluated.
    theiler : int, optional
        Exclude temporally-close neighbours with :math:`|i-j| \le w`.
    final_time, dt, steps : optional
        Passed to :func:`series_of`.

    Returns
    -------
    Geometry

    References
    ----------
    Cao, L. (1997). *Physica D* 110(1-2), 43-50.
    """
    from tsdynamics.analysis.embedding import cao_dimension

    values, _, meta, title = series_of(
        subject, components=components, final_time=final_time, dt=dt, steps=steps
    )
    values = _require(values, 16, "a Cao E1/E2 curve")
    result = cao_dimension(values, delay=int(delay), max_dim=int(max_dim), theiler=int(theiler))
    curves = [(np.asarray(result.afn_e1, dtype=float), "$E_1(d)$")]
    if result.afn_e2 is not None:
        curves.append((np.asarray(result.afn_e2, dtype=float), "$E_2(d)$"))
    return _dimension_geometry(
        "cao",
        result,
        curves=curves,
        ylabel="$E_1$, $E_2$",
        title=f"{title} Cao embedding dimension".strip(),
        meta=meta,
    )


def _dimension_geometry(
    name: str,
    result: Any,
    *,
    curves: Sequence[tuple[np.ndarray, str]],
    ylabel: str,
    title: str,
    meta: dict[str, Any],
) -> Geometry:
    """Assemble the shared ``fnn`` / ``cao`` geometry: the curve(s) plus the chosen ``m``."""
    dims = np.asarray(result.dims, dtype=float)
    parts = [Part({"x": dims, "y": curve}, label=label) for curve, label in curves]
    stacked = np.concatenate([curve for curve, _ in curves]) if curves else np.zeros(1)
    lo, hi = _span(stacked)
    chosen = int(result.dimension)
    parts.append(_marker_part(float(chosen), lo, hi, f"$m$ = {chosen}"))
    return Geometry(
        name,
        make_frame(FrameSpace.SCALING, ("m",)),
        parts,
        axis_labels=("embedding dimension $d$", ylabel),
        title=title,
        meta={**meta, **dict(result.meta), "dimension": chosen, "delay": int(result.delay)},
    )


# ---------------------------------------------------------------------------
# line_lengths
# ---------------------------------------------------------------------------


@plot_transform(
    name="line_lengths",
    source="data",
    kind=PlotKind.DIAGNOSTIC_CURVE,
    frame=FrameSpace.SCALING,
    ndim=1,
    role=OverlayRole.BASE,
    default_primitive="line",
    primitives=("line", "points", "steps"),
    analysis="tsdynamics.analysis.recurrence.rqa.rqa",
    example=lambda primitive: (_demo_orbit(200), {}),
    doc="P(l) and P(v) — the recurrence line-length distributions DET/LAM are read from.",
)
def line_lengths(
    subject: Any,
    *,
    components: int | str | None = None,
    threshold: float | None = None,
    recurrence_rate: float | None = None,
    theiler: int = 0,
    min_diagonal: int = 2,
    min_vertical: int = 2,
    normalize: bool = True,
    final_time: float | None = None,
    dt: float | None = None,
    steps: int | None = None,
) -> Geometry:
    r"""Draw the diagonal and vertical line-length distributions of a recurrence matrix.

    :func:`~tsdynamics.analysis.rqa` computes exactly these two histograms and
    then reduces them to four numbers — DET and :math:`L_{max}` from
    :math:`P(l)`, LAM and TT from :math:`P(v)` — discarding the distributions.
    They are the more informative object: a power-law :math:`P(l)` and an
    exponential one can share a DET, and only the plot tells them apart.

    Parameters
    ----------
    subject : RecurrenceMatrix, ndarray, Trajectory, or System
        A recurrence matrix, or anything one can be built from.
    components : int or str, optional
        Restrict a multi-component source to one component.  ``None`` (the
        default) uses the **full state vector**, which is the phase-space
        recurrence and what :func:`~tsdynamics.analysis.recurrence_matrix` does.
    threshold : float, optional
        Fixed distance threshold :math:`\varepsilon`.
    recurrence_rate : float, optional
        Target recurrence rate, calibrating :math:`\varepsilon` instead.
    theiler : int, optional
        Excluded near-diagonal band :math:`|i-j| \le w`.
    min_diagonal, min_vertical : int, optional
        Shortest line counted in each distribution — the same two knobs, under
        the same two names, that :func:`~tsdynamics.analysis.rqa` computes DET
        and LAM with, so the drawn histogram is the one the numbers come from.
    normalize : bool, optional
        Return probabilities :math:`P(l)` (the default) rather than raw counts.
    final_time, dt, steps : optional
        Passed to :func:`series_of` when ``subject`` is a system.

    Returns
    -------
    Geometry
        Two parts — the diagonal :math:`P(l)` and the vertical :math:`P(v)` —
        on a shared line-length axis ``l``.

    Raises
    ------
    tsdynamics.errors.InvalidInputError
        If the recurrence matrix carries no line of either kind (nothing to
        draw), which means the threshold is far too small.

    Notes
    -----
    Both distributions are heavy-tailed; read them on log axes::

        ts.plot(traj, "line_lengths").rescale(y="log").plot()

    References
    ----------
    Marwan, N. et al. (2007). *Physics Reports* 438(5-6), 237-329.
    """
    from tsdynamics.analysis.recurrence._common import (
        _diagonal_run_lengths,
        _vertical_run_lengths,
    )
    from tsdynamics.errors import InvalidInputError

    matrix, meta, title = _recurrence_of(
        subject,
        components=components,
        threshold=threshold,
        recurrence_rate=recurrence_rate,
        theiler=theiler,
        final_time=final_time,
        dt=dt,
        steps=steps,
    )
    diagonal = _diagonal_run_lengths(matrix.matrix)
    vertical = _vertical_run_lengths(matrix.matrix, theiler=matrix.theiler_window)

    parts: list[Part] = []
    for lengths, minimum, label in (
        (diagonal, int(min_diagonal), "$P(l)$ diagonal"),
        (vertical, int(min_vertical), "$P(v)$ vertical"),
    ):
        centres, weights = _length_histogram(lengths, minimum, bool(normalize))
        if centres.size:
            parts.append(Part({"x": centres, "y": weights}, label=label))
    if not parts:
        raise InvalidInputError(
            "the recurrence matrix carries no diagonal or vertical line of the minimum "
            f"length (min_diagonal={int(min_diagonal)}, min_vertical={int(min_vertical)}), "
            "so there is no distribution to draw; lower them or raise the recurrence rate "
            "(epsilon is almost certainly too small)."
        )
    return Geometry(
        "line_lengths",
        make_frame(FrameSpace.SCALING, ("l",)),
        parts,
        axis_labels=("line length", "$P$" if normalize else "count"),
        title=f"{title} recurrence line lengths".strip(),
        meta={
            **meta,
            "epsilon": float(matrix.epsilon),
            "theiler_window": int(matrix.theiler_window),
            "min_diagonal": int(min_diagonal),
            "min_vertical": int(min_vertical),
            "n_diagonal_lines": int(diagonal.size),
            "n_vertical_lines": int(vertical.size),
        },
    )


def _length_histogram(
    lengths: np.ndarray, min_length: int, normalize: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Bin integer run lengths onto unit-width bins at ``min_length, min_length+1, ...``."""
    lengths = np.asarray(lengths, dtype=np.int64).ravel()
    lengths = lengths[lengths >= max(1, min_length)]
    if lengths.size == 0:
        return np.empty(0), np.empty(0)
    counts = np.bincount(lengths)
    centres = np.arange(counts.size, dtype=float)
    keep = centres >= max(1, min_length)
    centres = centres[keep]
    weights = counts[keep].astype(float)
    if normalize:
        total = weights.sum()
        if total > 0:
            weights = weights / total
    return centres, weights


def _recurrence_of(
    subject: Any,
    *,
    components: int | str | None,
    threshold: float | None,
    recurrence_rate: float | None,
    theiler: int,
    final_time: float | None,
    dt: float | None,
    steps: int | None,
) -> tuple[Any, dict[str, Any], str]:
    """Return ``(RecurrenceMatrix, meta, title)`` for any accepted ``line_lengths`` subject."""
    from tsdynamics.analysis import recurrence_matrix
    from tsdynamics.analysis.recurrence.matrix import RecurrenceMatrix

    if isinstance(subject, RecurrenceMatrix):
        return subject, dict(subject.meta), ""

    if components is None:
        # Phase-space recurrence is of the *state vector*, so the default keeps
        # every coordinate: a trajectory (or system) contributes its whole `y`,
        # and a bare (N, dim) array is already a point set.
        traj = _trajectory_of(subject, {"final_time": final_time, "dt": dt, "steps": steps})
        if traj is not None:
            points = np.asarray(traj.y, dtype=float)
            meta, title = _meta(traj), _title(traj)
        else:
            points = np.atleast_2d(np.asarray(subject, dtype=float))
            if points.shape[0] == 1:
                points = points.T
            meta, title = {}, ""
    else:
        values, _, meta, title = series_of(
            subject,
            components=0 if components is None else components,
            final_time=final_time,
            dt=dt,
            steps=steps,
        )
        points = _require(values, 4, "a recurrence matrix")
    # `recurrence_matrix` requires *exactly one* of the two threshold spellings;
    # with neither given, calibrate to a 10% recurrence rate — the conventional
    # RQA default, and the choice that cannot depend on the data's units.
    if threshold is None and recurrence_rate is None:
        recurrence_rate = 0.1
    matrix = recurrence_matrix(
        points,
        threshold=threshold,
        recurrence_rate=recurrence_rate,
        theiler=int(theiler),
    )
    return matrix, meta, title


# ---------------------------------------------------------------------------
# return_time
# ---------------------------------------------------------------------------


@plot_transform(
    name="return_time",
    source="data",
    kind=PlotKind.DIAGNOSTIC_CURVE,
    frame=FrameSpace.SCALING,
    ndim=1,
    role=OverlayRole.BASE,
    default_primitive="line",
    primitives=("line", "points", "steps"),
    example=lambda primitive: (_demo_orbit(400), {"n_bins": 12}),
    doc="The first-return / first-passage time distribution of a level set.",
)
def return_time(
    subject: Any,
    *,
    components: int | str = 0,
    threshold: float | None = None,
    direction: str = "up",
    n_bins: int = 30,
    normalize: bool = True,
    final_time: float | None = None,
    dt: float | None = None,
    steps: int | None = None,
) -> Geometry:
    r"""Draw the distribution of first-return times to a level set of one observable.

    Successive crossings of :math:`x = c` in the chosen ``direction`` cut the
    orbit into excursions; their durations are the first-return (equivalently,
    first-passage) times, and :math:`P(T)` distinguishes a periodic orbit (one
    spike), a quasiperiodic one (a few sharp peaks) and a chaotic one (a broad,
    often exponential-tailed distribution).

    This is admitted under the same rule as :func:`psd`: it is a property of the
    orbit's *visits to a region of phase space*, not a generic series statistic.
    It computes no estimator — the crossings are read straight out of the input
    samples, which is why it declares no ``analysis`` — and the crossing times
    themselves are returned in ``meta["return_times"]`` for a caller who wants
    the raw inter-event series (it is what feeds a Hilbert plot, for instance).

    What correct looks like
    -----------------------
    A periodic orbit returns after exactly one period, every time: its
    distribution is one spike and ``meta["mean_return_time"]`` is the period,
    in **time units**.  Gate:
    ``tests/test_viz_truth.py::TestReturnTimesOfAPeriodicOrbitAreThePeriod``.

    Parameters
    ----------
    subject : ndarray, Trajectory, or System
        The series, the trajectory, or the system to integrate for one.
    components : int or str, optional
        Which component defines the observable.
    threshold : float, optional
        The level :math:`c`.  ``None`` uses the series **mean**, which is the
        choice that guarantees crossings exist.
    direction : {"up", "down", "both"}, optional
        Which crossings start an excursion.
    n_bins : int, optional
        Bins in the returned distribution.
    normalize : bool, optional
        Return a probability density (default) rather than raw counts.
    final_time, dt, steps : optional
        Passed to :func:`series_of`.

    Returns
    -------
    Geometry
        A ``scaling``-framed geometry on the return-time axis ``T`` (in **time
        units** — sample index times the source's sample spacing).

    Raises
    ------
    tsdynamics.errors.InvalidParameterError
        If ``direction`` is not one of the three spellings.
    tsdynamics.errors.InvalidInputError
        If fewer than two crossings occur, so no return time is defined.
    """
    from tsdynamics.errors import InvalidInputError, InvalidParameterError

    if direction not in ("up", "down", "both"):
        raise InvalidParameterError(f"unknown direction {direction!r}; use 'up', 'down' or 'both'.")
    values, spacing, meta, title = series_of(
        subject, components=components, final_time=final_time, dt=dt, steps=steps
    )
    values = _require(values, 4, "a return-time distribution")
    level = float(np.mean(values)) if threshold is None else float(threshold)
    crossings = _crossing_indices(values, level, direction)
    if crossings.size < 2:
        raise InvalidInputError(
            f"the series crosses {level:.6g} {direction}ward {crossings.size} time(s), so no "
            "return time is defined; lower the threshold, lengthen the series, or use "
            "direction='both'."
        )
    times = np.diff(crossings.astype(float)) * spacing
    # Split rather than passing `density=bool(...)`: numpy's overloads key off the
    # literal, and the two branches genuinely return different dtypes.
    if normalize:
        density, edges = np.histogram(times, bins=int(n_bins), density=True)
        counts = np.asarray(density, dtype=float)
    else:
        tally, edges = np.histogram(times, bins=int(n_bins), density=False)
        counts = np.asarray(tally, dtype=float)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return Geometry(
        "return_time",
        make_frame(FrameSpace.SCALING, ("T",)),
        channels={"x": centres, "y": counts.astype(float)},
        label="$P(T)$",
        axis_labels=("return time $T$", "$P(T)$" if normalize else "count"),
        title=f"{title} return times".strip(),
        meta={
            **meta,
            "threshold": level,
            "direction": direction,
            "n_returns": int(times.size),
            "mean_return_time": float(times.mean()),
            "return_times": times,
        },
    )


def _crossing_indices(values: np.ndarray, level: float, direction: str) -> np.ndarray:
    """Sample indices at which ``values`` crosses ``level`` in ``direction``."""
    above = values > level
    up = np.flatnonzero(~above[:-1] & above[1:]) + 1
    down = np.flatnonzero(above[:-1] & ~above[1:]) + 1
    if direction == "up":
        return up
    if direction == "down":
        return down
    return np.sort(np.concatenate([up, down]))
