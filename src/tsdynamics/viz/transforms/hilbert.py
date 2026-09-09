r"""Space-filling-curve views of a 1-D observable — the ``hilbert*`` transforms.

A long scalar record — 20,000 samples of :math:`x(t)`, a windowed RQA measure, an
inter-event series, a symbol sequence — is a strip a few hundred pixels wide by
the time it reaches a page.  Laying it along a **space-filling curve** instead
turns it into a square image in which samples that are close *in the record* stay
close *in the picture*, so a long-range pattern becomes a two-dimensional
texture.  This is the Hilbert plot of Estévez-Rams *et al.* [1]_, and four views
of it are registered here:

===================== ==========================================================
``hilbert``           the data image itself
``hilbert_fourier``   its centred 2-D power spectrum (periodicity as symmetry)
``hilbert_difference`` the locality-loss field: **where the curve's locality
                      breaks**, i.e. which neighbouring pixels are far apart in
                      the record
``hilbert_labels``    the visit order — the curve's own path, as an image
===================== ==========================================================

The forty curves
----------------
The real Hilbert-type curves come from the optional **hilbertplot** package
(``pip install tsdynamics[hilbert]``), the reference implementation of all forty
two-dimensional Hilbert curves [2]_.  Only its **numpy core** is used —
``image`` / ``fourier`` / ``difference_map`` / ``label_map`` return raw arrays —
and never its ``[plot]`` extra: the two rendering stacks must not meet, or a
Hilbert plot would bypass :class:`~tsdynamics.viz.spec.PlotSpec`, the
:class:`~tsdynamics.viz.style.Theme`, :data:`~tsdynamics.viz.style.STYLE_KEYS`
and three of the four backends.

Without the extra installed, asking for a Hilbert curve **raises**, naming both
the extra and the alternatives.  It never silently substitutes a different
ordering, because the locality property *is* the claim of the plot.  Measured
here (``tests/test_viz_hilbert.py`` pins these numbers) as the record-distance
between every pair of 8-adjacent cells of a :math:`128 \times 128` grid:

=========  ==========  ========  ==============
ordering   median gap  mean gap  P(gap <= 8)
=========  ==========  ========  ==============
Hilbert           3.0     114.3           0.683
Morton (Z)        5.0      94.5           0.656
snake            85.5      96.1           0.275
row-major       127.5      96.1           0.251
=========  ==========  ========  ==============

**Never quote a single locality number**: the median says Hilbert is 43x more
local than row-major, and the mean says it is 19% *worse* — because Hilbert buys
that median by making its few worst jumps much longer.  Both are true, and a
figure that quotes one of them is an advertisement.

Three orderings ship in-tree and need no dependency at all — ``"rowmajor"``,
``"snake"`` (boustrophedon) and ``"morton"`` (Z-order) — so the *plot* always
exists and the extra is an upgrade, not a gate.  They are what the governance
gate draws when hilbertplot is absent.

The pixel-to-sample map, and why it is trustworthy
--------------------------------------------------
An image is only half of a Hilbert plot: the other half is being able to say
*which sample* a pixel is, for a hover read-out or an annotation.  Getting that
wrong is the worst failure mode available here — the image still renders and the
read-out is simply a lie, with no exception raised.  Three independent locks:

1. **The fit is always explicit.**  This module never passes ``fit="auto"``, so
   it never depends on how hilbertplot resolves it.  The grid side follows one
   documented rule (:func:`grid_side`) that lives here.
2. **The map is read from public API**, not reconstructed:
   ``HilbertPlot.label_map()`` returns the visit-order index of every cell, built
   from the *same* ``(granularity, order, fit)`` call as ``image()``, so the two
   are consistent by construction rather than by assumption.
3. **The result is checked**, every call: the returned image's shape must equal
   the side this module computed, and a sample of cells must carry exactly the
   values the map says they do.  A mismatch raises rather than draws.

References
----------
.. [1] Estévez-Rams, E., Lora Serrano, R., Aragón Fernández, B. &
   Brito Reyes, I. (2015). "Visualizing long vectors of measurements by use of
   the Hilbert curve." *Computer Physics Communications* 197, 118-128.
.. [2] Estévez-Rams, E., Pérez-Cruz, J. A. & Rodríguez Hoyos, O. (2017).
   "Hilbert curves in two dimensions." *Revista Cubana de Física* 34(1), 9-14.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._frames import FrameSpace, OverlayRole
from ..spec import PlotKind
from ._base import Geometry, Presentation, make_frame
from ._registry import plot_transform
from .series import series_of

__all__ = [
    "ORDERINGS",
    "curve_names",
    "grid_side",
    "hilbert",
    "hilbert_difference",
    "hilbert_fourier",
    "hilbert_labels",
    "sample_index_map",
]

#: The dependency-free orderings that ship in-tree.  They are **not**
#: substitutes for a Hilbert curve (see the module docstring on locality); they
#: exist so the plot itself never depends on an optional package, and so the
#: compatibility gate has something to draw when the extra is absent.
ORDERINGS: dict[str, str] = {
    "rowmajor": "left to right, bottom to top — no locality, the honest baseline",
    "snake": "boustrophedon: row-major with alternate rows reversed",
    "morton": "Z-order (bit-interleaved); needs a power-of-two side",
}

#: The distribution that provides the forty real Hilbert curves.
_EXTRA = "hilbertplot"


# ---------------------------------------------------------------------------
# Grid sizing — this module's rule, not a guess at somebody else's
# ---------------------------------------------------------------------------


def grid_side(n: int, fit: str, *, any_side: bool) -> int:
    """Return the side of the square grid ``n`` values are laid on.

    The rule is stated here and nowhere else, so nothing in TSDynamics has to
    infer it from another package's behaviour.

    Parameters
    ----------
    n : int
        Number of values.
    fit : {"square", "pad", "truncate"}
        ``"square"`` is the tight ``ceil(sqrt(n))`` grid (only for an ordering
        that accepts any side); ``"pad"`` the smallest enclosing ``2**k`` grid;
        ``"truncate"`` the largest ``2**k`` grid that fits *inside* the data, so
        the tail is dropped.
    any_side : bool
        Whether the ordering can fill a non-power-of-two square.

    Returns
    -------
    int

    Raises
    ------
    tsdynamics.errors.InvalidParameterError
        For an unknown ``fit``, ``n < 1``, or ``fit="square"`` on an ordering
        that only generates power-of-two grids.
    """
    from tsdynamics.errors import InvalidParameterError

    if n < 1:
        raise InvalidParameterError("a space-filling-curve plot needs at least one value.")
    if fit == "square":
        if not any_side:
            raise InvalidParameterError(
                "fit='square' needs an ordering that fills a grid of any side; this one "
                "generates only 2**k grids. Use fit='pad' or fit='truncate'."
            )
        side = int(np.ceil(np.sqrt(n)))
        return max(1, side)
    if fit in ("pad", "truncate"):
        k = 0
        while 4 ** (k + 1) <= n:
            k += 1
        if fit == "pad" and 4**k < n:
            k += 1
        return int(2**k)
    raise InvalidParameterError(
        f"unknown fit {fit!r}; use 'square' (tight ceil(sqrt(n)) grid), 'pad' (smallest "
        "enclosing 2**k grid) or 'truncate' (largest 2**k grid inside the data)."
    )


def _default_fit(any_side: bool) -> str:
    """Return the fit used when the caller names none: tight if possible, else padded."""
    return "square" if any_side else "pad"


# ---------------------------------------------------------------------------
# The in-tree orderings
# ---------------------------------------------------------------------------


def _ordering_points(name: str, side: int) -> np.ndarray:
    """Return the ``(side*side, 2)`` array of ``(x, y)`` cells an in-tree ordering visits."""
    from tsdynamics.errors import InvalidParameterError

    idx = np.arange(side * side, dtype=np.int64)
    if name == "rowmajor":
        return np.column_stack([idx % side, idx // side])
    if name == "snake":
        rows = idx // side
        cols = idx % side
        cols = np.where(rows % 2 == 1, side - 1 - cols, cols)
        return np.column_stack([cols, rows])
    if name == "morton":
        if side & (side - 1):
            raise InvalidParameterError(
                f"the 'morton' ordering needs a power-of-two side, got {side}; use "
                "fit='pad' or fit='truncate'."
            )
        bits = max(1, int(side).bit_length() - 1)
        x = np.zeros_like(idx)
        y = np.zeros_like(idx)
        for b in range(bits):
            x |= ((idx >> (2 * b)) & 1) << b
            y |= ((idx >> (2 * b + 1)) & 1) << b
        return np.column_stack([x, y])
    raise InvalidParameterError(  # pragma: no cover - guarded by the caller
        f"unknown in-tree ordering {name!r}; the registered ones are {sorted(ORDERINGS)}."
    )


def _granulate(values: np.ndarray, granularity: int) -> np.ndarray:
    """Apply the *l*-granularity transform: each block of ``l`` values becomes its mean.

    Same length as the input (a coarsening, not a decimation), matching
    hilbertplot's ``granulate`` so the two paths coarsen identically.
    """
    granularity = int(granularity)
    if granularity <= 1:
        return values
    n = values.size
    out = np.empty(n, dtype=float)
    full = (n // granularity) * granularity
    if full:
        out[:full] = np.repeat(values[:full].reshape(-1, granularity).mean(axis=1), granularity)
    if full < n:
        out[full:] = values[full:].mean()
    return out


def _scatter(points: np.ndarray, values: np.ndarray, side: int) -> np.ndarray:
    """Place ``values`` onto a ``side x side`` grid along ``points``; unfilled cells are NaN."""
    values = values[: side * side]
    image = np.full((side, side), np.nan, dtype=float)
    sel = points[: values.size]
    image[sel[:, 1], sel[:, 0]] = values
    return image


def _labels_from_points(points: np.ndarray, side: int) -> np.ndarray:
    """Return the visit-order index of every cell — the inverse of ``points``."""
    labels = np.zeros((side, side), dtype=np.int64)
    labels[points[:, 1], points[:, 0]] = np.arange(points.shape[0], dtype=np.int64)
    return labels


def _difference_from_labels(labels: np.ndarray) -> np.ndarray:
    """Mean 8-neighbour visit-order distance per cell — the locality-loss field.

    High where the curve's two-dimensional neighbourhood is *not* a
    one-dimensional neighbourhood: the ridges are the ordering's locality
    barriers, and for a row-major layout the whole field is one huge plateau,
    which is the point of drawing it.
    """
    padded = np.pad(labels.astype(float), 1, mode="edge")
    total = np.zeros(labels.shape, dtype=float)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            total += np.abs(
                padded[1 + dy : 1 + dy + labels.shape[0], 1 + dx : 1 + dx + labels.shape[1]]
                - labels
            )
    return total / 8.0


def _fourier_of(image: np.ndarray) -> np.ndarray:
    """Return the centred ``log(1 + |F|)`` spectrum of a data image (NaN cells filled)."""
    filled = np.where(
        np.isfinite(image), image, np.nanmean(image) if np.isfinite(image).any() else 0.0
    )
    filled = filled - filled.mean()
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(filled))))


# ---------------------------------------------------------------------------
# Layout: one entry point, two backends (hilbertplot, or an in-tree ordering)
# ---------------------------------------------------------------------------


def curve_names() -> tuple[str, ...]:
    """Every ordering ``curve=`` accepts right now — in-tree first, then the extra's.

    The list **grows** when the optional ``hilbertplot`` package is installed;
    the three in-tree orderings are always there.

    Returns
    -------
    tuple of str
    """
    names = tuple(sorted(ORDERINGS))
    try:
        import hilbertplot
    except ImportError:
        return names
    return names + tuple(hilbertplot.catalog().names)


def _require_hilbertplot(name: str) -> Any:
    """Import hilbertplot, or raise an error that names the extra and the alternatives."""
    try:
        import hilbertplot
    except ImportError as exc:
        from tsdynamics.analysis._result_viz import VisualizationNotInstalled

        raise VisualizationNotInstalled(
            f"curve={name!r} is one of the forty Hilbert-type curves, which come from the "
            f"optional {_EXTRA!r} package: install it with "
            f"`pip install tsdynamics[hilbert]` (or `pip install {_EXTRA}`).\n"
            f"It is not substituted silently, because the locality of the Hilbert curve is "
            f"the whole claim of this plot. If you want a picture now, ask for one of the "
            f"dependency-free orderings explicitly: curve={sorted(ORDERINGS)}."
        ) from exc
    return hilbertplot


_VIEWS = ("image", "fourier", "difference", "labels")


def _layout(
    values: np.ndarray,
    *,
    curve: str,
    view: str,
    granularity: int,
    fit: str | None,
) -> tuple[np.ndarray, np.ndarray, int, str]:
    """Return ``(field, sample_index, side, resolved_fit)`` for one view.

    ``field`` is the ``(side, side)`` array to draw and ``sample_index`` the
    pixel-to-sample map (the visit order), always consistent with it.
    """
    from tsdynamics.errors import InvalidParameterError

    if view not in _VIEWS:  # pragma: no cover - guarded by the four transforms
        raise InvalidParameterError(f"unknown hilbert view {view!r}; use one of {_VIEWS}.")
    n = int(values.size)
    if isinstance(curve, str) and curve in ORDERINGS:
        any_side = curve != "morton"
        resolved = fit if fit is not None else _default_fit(any_side)
        side = grid_side(n, resolved, any_side=any_side)
        points = _ordering_points(curve, side)
        coarse = _granulate(values, granularity)
        labels = _labels_from_points(points, side)
        field = {
            "image": lambda: _scatter(points, coarse, side),
            "labels": lambda: labels.astype(float),
            "difference": lambda: _difference_from_labels(labels),
            "fourier": lambda: _fourier_of(_scatter(points, coarse, side)),
        }[view]()
        return field, labels, side, resolved

    hp = _require_hilbertplot(curve)
    try:
        obj = hp.curve(curve)
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        raise InvalidParameterError(
            f"unknown curve {curve!r}; use an in-tree ordering {sorted(ORDERINGS)} or one of "
            f"hilbertplot's {len(hp.catalog())} curves (see hilbertplot.catalog().names)."
        ) from exc
    resolved = fit if fit is not None else _default_fit(bool(obj.generalizes))
    side = grid_side(n, resolved, any_side=bool(obj.generalizes))
    plot = hp.hilbert_plot(obj, values)
    kw = {"granularity": int(granularity), "fit": resolved}
    field = {
        "image": lambda: plot.image(**kw),
        "fourier": lambda: plot.fourier(**kw),
        "difference": lambda: plot.difference_map(**kw),
        "labels": lambda: plot.label_map(**kw).astype(float),
    }[view]()
    index = plot.label_map(**kw)
    _verify_layout(field, index, side, curve, resolved, plot, kw, view, n)
    return np.asarray(field, dtype=float), np.asarray(index), side, resolved


def _verify_layout(
    field: np.ndarray,
    index: np.ndarray,
    side: int,
    curve: str,
    fit: str,
    plot: Any,
    kw: dict[str, Any],
    view: str,
    n: int,
) -> None:
    """Fail loudly if the external layout is not the one this module computed.

    The failure this closes has no other symptom: a wrong pixel-to-sample map
    still renders a perfectly plausible image, and only the hover read-out is
    wrong.  Both halves are checked — the grid *size* against
    :func:`grid_side`, and the *mapping* against the image it is supposed to
    describe (sampled, so the check is O(1) rather than O(N)).
    """
    from tsdynamics.errors import BackendError

    if field.shape != (side, side) or index.shape != (side, side):
        raise BackendError(
            f"{_EXTRA} laid {n} values for curve {curve!r} (fit={fit!r}) on a "
            f"{field.shape} grid, but TSDynamics computed side {side}. The pixel-to-sample "
            "map would be wrong with no other symptom, so this raises instead of drawing. "
            f"Please report it against tsdynamics with your {_EXTRA} version."
        )
    if view != "image":
        return
    image = np.asarray(field, dtype=float)
    values = np.asarray(plot.data, dtype=float)
    if kw["granularity"] > 1:
        values = _granulate(values, kw["granularity"])
    ys, xs = np.nonzero(np.isfinite(image))
    if ys.size == 0:  # pragma: no cover - an all-NaN input
        return
    take = np.linspace(0, ys.size - 1, num=min(64, ys.size)).astype(int)
    mapped = index[ys[take], xs[take]]
    if mapped.max() >= values.size or not np.allclose(image[ys[take], xs[take]], values[mapped]):
        raise BackendError(  # pragma: no cover - the invariant this exists to catch
            f"the {_EXTRA} pixel-to-sample map disagrees with the image it describes for "
            f"curve {curve!r} (fit={fit!r}); refusing to draw a plot whose annotations "
            "would be wrong."
        )


def sample_index_map(
    subject: Any,
    *,
    component: int | str = 0,
    curve: str = "Hilbert",
    granularity: int = 1,
    fit: str | None = None,
    final_time: float | None = None,
    dt: float | None = None,
    steps: int | None = None,
) -> np.ndarray:
    """Return the pixel-to-sample map of a Hilbert plot: ``map[y, x]`` is the sample index.

    Kept off the geometry's ``meta`` by default because it is exactly as large as
    the image, and a spec's ``meta`` is serialized by
    :func:`tsdynamics.viz.to_json`.  Ask for it here (or with
    ``with_sample_index=True``) when you want to annotate a pixel with the time
    it came from::

        idx = ts.viz.transforms.hilbert.sample_index_map(traj, component="x")
        t_of_pixel = traj.t[idx]          # same shape as the image

    Parameters
    ----------
    subject, component, curve, granularity, fit, final_time, dt, steps
        As for :func:`hilbert`.

    Returns
    -------
    ndarray
        A ``(side, side)`` integer array.  A cell whose index is ``>= len(series)``
        is padding.
    """
    values, _, _, _ = series_of(
        subject, component=component, final_time=final_time, dt=dt, steps=steps
    )
    _, index, _, _ = _layout(
        np.asarray(values, dtype=float),
        curve=curve,
        view="image",
        granularity=granularity,
        fit=fit,
    )
    return index


# ---------------------------------------------------------------------------
# The four transforms
# ---------------------------------------------------------------------------


def _hilbert_geometry(
    name: str,
    subject: Any,
    *,
    view: str,
    component: int | str,
    curve: str,
    granularity: int,
    fit: str | None,
    with_sample_index: bool,
    final_time: float | None,
    dt: float | None,
    steps: int | None,
    axis_labels: tuple[str, str],
    color_label: str,
    primitives: tuple[str, ...],
) -> Geometry:
    """Build the geometry shared by the four ``hilbert*`` transforms."""
    values, spacing, meta, title = series_of(
        subject, component=component, final_time=final_time, dt=dt, steps=steps
    )
    values = np.asarray(values, dtype=float)
    field, index, side, resolved = _layout(
        values, curve=curve, view=view, granularity=int(granularity), fit=fit
    )
    axis = np.arange(side, dtype=float)
    extra: dict[str, Any] = {
        "curve": str(curve),
        "curve_source": "in-tree" if curve in ORDERINGS else _EXTRA,
        "fit": resolved,
        "granularity": int(granularity),
        "side": int(side),
        "n_samples": int(values.size),
        "n_padding_cells": int(side * side - min(side * side, values.size)),
        "sample_spacing": spacing,
        "view": view,
    }
    if with_sample_index:
        extra["sample_index"] = index
    return Geometry(
        name,
        make_frame(FrameSpace.GRID2, 2, axis_labels),
        channels={"x": axis, "y": axis, "z": np.asarray(field, dtype=float)},
        axis_labels=axis_labels,
        primitives=primitives,
        aspect="equal",
        title=f"{title} {name} ({curve}, {side}x{side})".strip(),
        color_label=color_label,
        meta={**meta, **extra},
    )


def _demo_series(n: int = 256) -> np.ndarray:
    """Return a deterministic series for the compatibility gate (256 = a full 16x16 grid)."""
    t = np.linspace(0.0, 30.0, n)
    return np.asarray(np.sin(t) + 0.4 * np.sin(np.sqrt(3.0) * t), dtype=float)


def _demo_options(_primitive: str) -> tuple[Any, dict[str, Any]]:
    """Return the gate's subject and options: a real Hilbert curve if installed, else in-tree.

    The gate must be able to draw every declared cell on a machine that does not
    have the optional package, and it must exercise the *real* path on one that
    does.  Deciding that here — on the record, next to the transform — keeps the
    test file free of any knowledge about this row.
    """
    try:
        import hilbertplot  # noqa: F401
    except ImportError:
        return _demo_series(), {"curve": "snake"}
    return _demo_series(), {"curve": "Hilbert"}


_PRESENTATION = Presentation(aspect="equal", autocolor=True, cmap="viridis")


@plot_transform(
    name="hilbert",
    source="data",
    kind=PlotKind.SPACETIME,
    frame=FrameSpace.GRID2,
    ndim=2,
    role=OverlayRole.FIELD,
    default_primitive="image",
    primitives=("image", "surface3d"),
    presentation=_PRESENTATION,
    example=_demo_options,
    doc="A 1-D observable laid on a space-filling curve as a 2-D image.",
)
def hilbert(
    subject: Any,
    *,
    component: int | str = 0,
    curve: str = "Hilbert",
    granularity: int = 1,
    fit: str | None = None,
    with_sample_index: bool = False,
    final_time: float | None = None,
    dt: float | None = None,
    steps: int | None = None,
) -> Geometry:
    """Lay a 1-D observable on a space-filling curve and draw it as an image.

    Parameters
    ----------
    subject : ndarray, Trajectory, or System
        The series, the trajectory, or the system to integrate for one.  A bare
        array covers the cases a trajectory does not: a windowed RQA measure
        (``windowed_rqa(...).determinism``), an inter-event series
        (``ts.viz.geometry(traj, "return_time").meta["return_times"]``), or a
        symbolic sequence encoded as integers.
    component : int or str, optional
        Which component of a multi-component source.
    curve : str, optional
        A hilbertplot curve name or index (``"Hilbert"``, ``"Moore"``, ``0`` …,
        forty in all — needs the optional extra) or one of the dependency-free
        in-tree orderings, :data:`ORDERINGS`.  :func:`curve_names` lists what is
        available right now.
    granularity : int, optional
        Coarsen first with the *l*-granularity transform: each block of ``l``
        values is replaced by its mean, which suppresses sample-scale noise and
        leaves the long-range texture.  ``1`` (default) is the faithful view.
    fit : {"square", "pad", "truncate"}, optional
        How to size the grid; see :func:`grid_side`.  ``None`` picks the tight
        square when the ordering allows any side, else the padded power-of-two
        grid.
    with_sample_index : bool, optional
        Put the pixel-to-sample map on the geometry's ``meta``.  Off by default:
        it is as large as the image, and ``meta`` is serialized by
        :func:`tsdynamics.viz.to_json`.  See :func:`sample_index_map`.
    final_time, dt, steps : optional
        Passed to :func:`~tsdynamics.viz.transforms.series.series_of`.

    Returns
    -------
    Geometry
        A ``grid2`` lattice; padding cells are ``NaN`` and render transparent.

    Raises
    ------
    tsdynamics.analysis._result_viz.VisualizationNotInstalled
        If a Hilbert curve is requested without the optional ``hilbertplot``
        package.  It is never silently replaced by an in-tree ordering.

    Examples
    --------
    >>> ts.plot(traj, "hilbert", component="x")                    # doctest: +SKIP
    >>> ts.plot(traj, "hilbert", granularity=16).plot()            # doctest: +SKIP
    """
    return _hilbert_geometry(
        "hilbert",
        subject,
        view="image",
        component=component,
        curve=curve,
        granularity=granularity,
        fit=fit,
        with_sample_index=with_sample_index,
        final_time=final_time,
        dt=dt,
        steps=steps,
        axis_labels=("cell x", "cell y"),
        color_label="value",
        primitives=("image", "surface3d"),
    )


@plot_transform(
    name="hilbert_fourier",
    source="data",
    kind=PlotKind.SPACETIME,
    frame=FrameSpace.GRID2,
    ndim=2,
    role=OverlayRole.FIELD,
    default_primitive="image",
    primitives=("image", "contour", "surface3d"),
    presentation=_PRESENTATION,
    example=_demo_options,
    doc="The centred 2-D power spectrum of a Hilbert plot — periodicity as symmetry.",
)
def hilbert_fourier(
    subject: Any,
    *,
    component: int | str = 0,
    curve: str = "Hilbert",
    granularity: int = 1,
    fit: str | None = None,
    with_sample_index: bool = False,
    final_time: float | None = None,
    dt: float | None = None,
    steps: int | None = None,
) -> Geometry:
    """Draw the **Fourier map**: the centred 2-D spectrum of the Hilbert plot's image.

    A hidden periodicity in the record becomes a bright, symmetric pattern here,
    because the curve maps a period into a repeated 2-D motif.  Returned as
    ``log(1 + |F|)`` so faint structure survives the dynamic range, with the DC
    component removed.

    Parameters
    ----------
    subject, component, curve, granularity, fit, with_sample_index, final_time, dt, steps
        As for :func:`hilbert`.

    Returns
    -------
    Geometry

    References
    ----------
    Estévez-Rams, E. et al. (2015). *Comput. Phys. Commun.* 197, 118-128.
    """
    return _hilbert_geometry(
        "hilbert_fourier",
        subject,
        view="fourier",
        component=component,
        curve=curve,
        granularity=granularity,
        fit=fit,
        with_sample_index=with_sample_index,
        final_time=final_time,
        dt=dt,
        steps=steps,
        axis_labels=("$k_x$", "$k_y$"),
        color_label=r"$\log(1 + |F|)$",
        primitives=("image", "contour", "surface3d"),
    )


@plot_transform(
    name="hilbert_difference",
    source="data",
    kind=PlotKind.SPACETIME,
    frame=FrameSpace.GRID2,
    ndim=2,
    role=OverlayRole.FIELD,
    default_primitive="image",
    primitives=("image", "contour", "surface3d"),
    presentation=Presentation(aspect="equal", autocolor=True, cmap="magma"),
    example=_demo_options,
    doc="The locality-loss field — where the curve's 2-D neighbourhood is not a 1-D one.",
)
def hilbert_difference(
    subject: Any,
    *,
    component: int | str = 0,
    curve: str = "Hilbert",
    granularity: int = 1,
    fit: str | None = None,
    with_sample_index: bool = False,
    final_time: float | None = None,
    dt: float | None = None,
    steps: int | None = None,
) -> Geometry:
    """**Where the locality breaks** — the mean 8-neighbour visit-order distance per cell.

    Every space-filling curve trades locality somewhere; this is the map of
    where.  Bright ridges are the curve's locality barriers: two pixels touching
    across a ridge are far apart in the record, so a texture that straddles one
    is an artefact of the layout, not of the data.  Drawing it is what makes the
    Hilbert plot honest — and comparing it across curves (``curve="rowmajor"``
    versus ``curve="Hilbert"``) is the measurement, not the anecdote.

    It depends on the curve and the grid size only, never on the values.

    Parameters
    ----------
    subject, component, curve, granularity, fit, with_sample_index, final_time, dt, steps
        As for :func:`hilbert`.

    Returns
    -------
    Geometry
    """
    return _hilbert_geometry(
        "hilbert_difference",
        subject,
        view="difference",
        component=component,
        curve=curve,
        granularity=granularity,
        fit=fit,
        with_sample_index=with_sample_index,
        final_time=final_time,
        dt=dt,
        steps=steps,
        axis_labels=("cell x", "cell y"),
        color_label="mean neighbour gap",
        primitives=("image", "contour", "surface3d"),
    )


@plot_transform(
    name="hilbert_labels",
    source="data",
    kind=PlotKind.SPACETIME,
    frame=FrameSpace.GRID2,
    ndim=2,
    role=OverlayRole.FIELD,
    default_primitive="image",
    primitives=("image",),
    presentation=Presentation(aspect="equal", autocolor=True, cmap="twilight"),
    example=_demo_options,
    doc="The curve's own path — each cell coloured by the step at which it is visited.",
)
def hilbert_labels(
    subject: Any,
    *,
    component: int | str = 0,
    curve: str = "Hilbert",
    granularity: int = 1,
    fit: str | None = None,
    with_sample_index: bool = False,
    final_time: float | None = None,
    dt: float | None = None,
    steps: int | None = None,
) -> Geometry:
    """Draw the **visit order**: ``z[y, x]`` is the step at which the curve reaches that cell.

    The curve itself, drawn the same way the data is — so a texture in the data
    image can be read against the path that produced it.  It is also the
    pixel-to-sample map (see :func:`sample_index_map`), which is why it is a view
    and not a debugging aid.

    Parameters
    ----------
    subject, component, curve, granularity, fit, with_sample_index, final_time, dt, steps
        As for :func:`hilbert`.  Only the *length* of the series matters here.

    Returns
    -------
    Geometry
    """
    return _hilbert_geometry(
        "hilbert_labels",
        subject,
        view="labels",
        component=component,
        curve=curve,
        granularity=granularity,
        fit=fit,
        with_sample_index=with_sample_index,
        final_time=final_time,
        dt=dt,
        steps=steps,
        axis_labels=("cell x", "cell y"),
        color_label="visit order",
        primitives=("image",),
    )
