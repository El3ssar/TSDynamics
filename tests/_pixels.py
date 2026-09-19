"""Read the PICTURE, not its description.

Every other viz test in this suite inspects the :class:`~tsdynamics.viz.Plot` —
its kind, its layers, its axis records — and then asserts about *that*.  A spec
is a description of a figure, and a description can be right while the figure is
wrong: a recurrence plot whose 5 % of recurrent points arrives as 14 % of ink, a
3-D attractor that drops the colormap the 2-D one honours, a vector field left on
its old window after ``xlim=`` moved the axes.  All three had a correct spec.

This module is the missing instrument: render to an Agg buffer, read the RGBA
array back, and *measure* it.

Three ideas carry the whole thing.

**The data region, not the figure.**  An empty plot is not an empty canvas —
spines, ticks and labels are ink, and a blank-plot test that counted them would
pass on a plot with nothing drawn in it.  :func:`render` therefore records each
axes' own box (trimmed by :data:`INSET` pixels so the spines fall outside) and
every measurement defaults to the *largest* one — the panel a reader would call
"the plot".  A colorbar's strip collapses under the trim and drops out on its
own, which is what stops a pretty gradient standing in for a drawn curve.

**The background is the modal colour, not white.**  It has to be: a ``density``
image's background is viridis-zero, a dark theme's is near-black, and a basins
image has no white in it at all.  Taking the most common colour in the region
means :func:`ink_fraction` reads "how much of this panel is not the flat
backdrop" on every transform, every primitive and every theme, with no table of
per-plot expectations to maintain.

**Robust properties, never hashes.**  Fonts, matplotlib versions and layout
solvers all move pixels.  Everything here answers one of three questions — *is
there ink*, *did the picture change*, *is the ink roughly this fraction* — and
each survives a font substitution.  A golden-image test would not, and a flaky
pixel test is worse than none.

Small by construction: :data:`SIZE` × :data:`SIZE` inches at :data:`DPI` is a
150 × 150 buffer, 90 kB, and the whole 39-transform sweep renders in ~4 s.
"""

from __future__ import annotations

import collections
import io
from dataclasses import dataclass
from typing import Any

import pytest

# matplotlib is an OPTIONAL extra: the base test job installs the library
# without it and runs the viz suites in a separate job. Skip the whole module
# rather than fail collection — an ImportError here reddens the entire run.
pytest.importorskip("matplotlib")

import matplotlib

# Pinned, but only when it is not already what we want: ``force=True`` switches
# pyplot's backend and closes every open figure as a side effect, and this module
# is imported at collection time — before the suite is in a position to want that.
if matplotlib.get_backend().lower() != "agg":  # pragma: no cover - headless CI is Agg
    matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import AbstractMovieWriter  # noqa: E402

#: Figure edge in inches, and the resolution it is rasterised at.  150 × 150 px
#: is enough for a tick label to be legible ink and small enough that a
#: parametrized sweep over the whole transform registry stays in the fast tier.
SIZE = 3.0
DPI = 50

#: Per-channel distance from the background colour at which a pixel counts as
#: ink.  Well above 8-bit rounding and antialiasing fringe, well below any real
#: mark: a pale-grey gridline on white is ~35 apart, a drawn line far more.
TOL = 12

#: Pixels trimmed off each edge of an axes box before measuring.  Drops the
#: spines and the outermost tick marks, so "is there ink in the plot" is a
#: question about what was *drawn*, not about the frame it was drawn in.
INSET = 3


@dataclass(frozen=True)
class Box:
    """An axes' data region in array coordinates (rows from the top)."""

    left: int
    right: int
    top: int
    bottom: int

    @property
    def area(self) -> int:
        """Pixel count."""
        return max(0, self.right - self.left) * max(0, self.bottom - self.top)


@dataclass(frozen=True)
class Picture:
    """A rendered figure, as pixels plus the boxes its axes occupy.

    ``panels`` is ordered **largest first**, so ``panel(0)`` is the main data
    area of a single-panel plot and the panels of a grid are addressable without
    knowing the layout engine's arithmetic.
    """

    rgba: np.ndarray
    panels: tuple[Box, ...]
    n_axes: int

    # -- regions ---------------------------------------------------------

    def panel(self, index: int = 0) -> np.ndarray:
        """The pixels inside panel ``index`` (largest panel first)."""
        box = self.panels[index]
        return self.rgba[box.top : box.bottom, box.left : box.right]

    @property
    def data(self) -> np.ndarray:
        """The pixels inside the largest axes — "the plot"."""
        return self.panel(0)

    # -- measurements ----------------------------------------------------

    def ink(self, index: int = 0) -> float:
        """Fraction of panel ``index`` that is not the flat background colour."""
        return ink_fraction(self.panel(index))

    def dark(self, index: int = 0) -> float:
        """Fraction of panel ``index`` on the dark side of its own range."""
        return dark_fraction(self.panel(index))

    def __len__(self) -> int:
        return len(self.panels)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def render(plot: Any, *, size: float = SIZE, dpi: float = DPI) -> Picture:
    """Render ``plot`` through matplotlib and read the pixels back.

    The figure is disposed before returning — ``fig.clear()`` **and**
    ``pyplot.close("all")``, because the renderer builds bare
    :class:`~matplotlib.figure.Figure` objects that pyplot has never heard of, so
    closing alone frees nothing.
    """
    plot.size(size, size, dpi=dpi)
    fig = plot.render("matplotlib")
    try:
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
        panels = _panels(fig, rgba.shape[0])
        n_axes = len(fig.axes)
    finally:
        fig.clear()
        plt.close("all")
    if not panels:  # pragma: no cover - a figure with no usable axes
        raise AssertionError("rendered figure exposed no measurable axes region")
    return Picture(rgba=rgba, panels=panels, n_axes=n_axes)


def _panels(fig: Any, height: int) -> tuple[Box, ...]:
    """Each visible axes' trimmed data region, largest first.

    Matplotlib's display coordinates put the origin bottom-left; the RGBA buffer
    puts row 0 at the top.  That flip is the whole of the arithmetic here.
    """
    width = fig.canvas.get_width_height()[0]
    boxes: list[Box] = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        x0, y0, x1, y1 = ax.get_window_extent().extents
        box = Box(
            left=max(0, int(np.ceil(x0)) + INSET),
            right=min(width, int(np.floor(x1)) - INSET),
            top=max(0, height - int(np.floor(y1)) + INSET),
            bottom=min(height, height - int(np.ceil(y0)) - INSET),
        )
        if box.area > 0:
            boxes.append(box)
    boxes.sort(key=lambda b: b.area, reverse=True)
    return tuple(boxes)


# ---------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------


def background(rgba: np.ndarray) -> tuple[int, int, int]:
    """The most common RGB triple — whatever this plot's backdrop happens to be."""
    flat = rgba[..., :3].reshape(-1, 3)
    common, _ = collections.Counter(map(tuple, flat)).most_common(1)[0]
    return (int(common[0]), int(common[1]), int(common[2]))


def ink_fraction(rgba: np.ndarray) -> float:
    """Fraction of pixels more than :data:`TOL` from the background colour."""
    flat = rgba[..., :3].reshape(-1, 3).astype(np.int16)
    if flat.size == 0:  # pragma: no cover - defensive
        return 0.0
    bg = np.array(background(rgba), dtype=np.int16)
    return float((np.abs(flat - bg).max(axis=1) > TOL).mean())


def dark_fraction(rgba: np.ndarray) -> float:
    """Fraction of pixels below the midpoint of this region's own luminance range.

    Colormap-agnostic on purpose: for a binary image (a recurrence matrix drawn
    in ``Greys``) it is exactly "how much of the picture is a recurrent point",
    which is the number the data says it should be.
    """
    lum = rgba[..., :3].astype(np.float64).mean(axis=-1).ravel()
    if lum.size == 0 or lum.max() - lum.min() < 1.0:  # pragma: no cover - flat region
        return 0.0
    return float((lum < 0.5 * (lum.min() + lum.max())).mean())


def _delta(a: Picture | np.ndarray, b: Picture | np.ndarray) -> np.ndarray:
    """Boolean mask of pixels that differ by more than :data:`TOL` in any channel."""
    left = a.rgba if isinstance(a, Picture) else a
    right = b.rgba if isinstance(b, Picture) else b
    if left.shape != right.shape:  # pragma: no cover - guarded by callers
        raise AssertionError(
            f"cannot compare renders of different size: {left.shape} / {right.shape}"
        )
    delta = np.abs(left[..., :3].astype(np.int16) - right[..., :3].astype(np.int16))
    return delta.max(axis=-1) > TOL


def difference(a: Picture | np.ndarray, b: Picture | np.ndarray) -> float:
    """Fraction of pixels that differ between two renders of the same size.

    Thresholded at :data:`TOL`, so antialiasing jitter is not "the picture
    changed" and a dropped keyword is not hidden by it either.
    """
    return float(_delta(a, b).mean())


def changed_pixels(a: Picture | np.ndarray, b: Picture | np.ndarray) -> int:
    """How many pixels differ — a *count*, for effects too small to be a fraction.

    A figure keyword that only edits a short text string moves ~90 pixels of a
    100 kpx figure; one that is silently dropped moves exactly zero.  The gap is
    the whole point, and it reads honestly as a count and misleadingly as a
    percentage.
    """
    return int(_delta(a, b).sum())


# ---------------------------------------------------------------------------
# animation
# ---------------------------------------------------------------------------


class _BufferWriter(AbstractMovieWriter):
    """A movie writer that keeps frames in memory as raw, lossless RGBA.

    Goes through the very ``fig.savefig`` call ``PillowWriter`` / ``FFMpegWriter``
    make, so what it collects is what would have been written — but uncompressed
    and unpalettised, which a ``.gif`` is not.
    """

    def __init__(self, fps: float = 5.0) -> None:
        super().__init__(fps=fps)
        self.frames: list[bytes] = []

    def _supports_transparency(self) -> bool:
        return True

    def setup(self, fig: Any, outfile: Any, dpi: float | None = None) -> None:
        """Bind the figure the frames are grabbed from."""
        super().setup(fig, outfile, dpi=dpi)
        self.frames = []

    def grab_frame(self, **savefig_kwargs: Any) -> None:
        """Rasterise the figure as it currently stands."""
        buf = io.BytesIO()
        self.fig.savefig(buf, format="rgba", dpi=self.dpi, **savefig_kwargs)
        self.frames.append(buf.getvalue())

    def finish(self) -> None:
        """Nothing to flush — the frames are already in hand."""


def frames(plot: Any, *, size: float = SIZE, dpi: float = DPI) -> list[np.ndarray]:
    """Write ``plot``'s animation through the writer protocol and return its frames.

    Each frame comes back as an ``(H, W, 4)`` uint8 array — the same shape
    :func:`render` produces, so the same measurements apply to a movie frame as
    to a still.
    """
    plot.size(size, size, dpi=dpi)
    anim = plot.render("matplotlib")
    if not hasattr(anim, "save"):  # pragma: no cover - guarded by callers
        raise AssertionError(f"{type(anim).__name__} is not an animation; pass animate=")
    writer = _BufferWriter()
    fig = anim._fig
    try:
        anim.save("unused", writer=writer, dpi=dpi)
    finally:
        fig.clear()
        plt.close("all")
    side = int(round(size * dpi))
    return [_as_frame(raw, side) for raw in writer.frames]


def _as_frame(raw: bytes, side: int) -> np.ndarray:
    """Reshape one raw RGBA blob, inferring the square edge when it differs."""
    flat = np.frombuffer(raw, dtype=np.uint8)
    pixels = flat.size // 4
    edge = side if side * side == pixels else int(round(np.sqrt(pixels)))
    return flat.reshape(edge, edge, 4)


def gif_frames(path: Any) -> list[np.ndarray]:
    """Read a written ``.gif`` back as a list of RGB arrays.

    The end-to-end check: not what the animation object believed it would write,
    but what is on disk.
    """
    from PIL import Image

    out: list[np.ndarray] = []
    with Image.open(str(path)) as img:
        for index in range(getattr(img, "n_frames", 1)):
            img.seek(index)
            out.append(np.asarray(img.convert("RGB")))
    return out
