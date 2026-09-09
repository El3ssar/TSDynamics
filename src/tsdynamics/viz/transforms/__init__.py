"""Plot transforms — the extension point for "what to plot".

A **transform** turns a subject (a :class:`~tsdynamics.data.Trajectory`, a
system, an analysis result) into :class:`Geometry`; a **primitive** turns that
geometry into layers; a declared **compatibility row** says which pairs are
legal.  Adding a plot to TSDynamics is one decorated function::

    from tsdynamics.viz.transforms import Geometry, make_frame, plot_transform
    from tsdynamics.viz._frames import FrameSpace

    @plot_transform(
        name="nullclines", source="model",
        frame=FrameSpace.STATE2, ndim=2, kind="phase_portrait_2d", role="base",
        default_primitive="contour", primitives=("contour", "line"),
        analysis="tsdynamics.analysis.fields.nullclines",
        example=lambda primitive: (Duffing(), {"grid": 40}),
        doc="Zero level sets of each RHS component on a 2-D slice.",
    )
    def nullclines(system, *, plane=("x", "y"), at=None, grid=400) -> Geometry:
        ...

Nothing else changes: no renderer edit, no
:class:`~tsdynamics.viz.spec.PlotKind` edit, no ``compose`` edit, no test edit.
``ts.plot(system, "nullclines")`` works, ``ts.viz.compatibility()`` lists the
row, and ``tests/test_viz_compatibility.py`` renders every cell of it.

Two rules the registry enforces rather than merely documents:

- **Exactly two source categories** — ``data`` (computable from the samples you
  have; also accepts a system, because a model gives you data for free) and
  ``model`` (must evaluate or integrate the right-hand side somewhere new).
- **A transform is a thin adapter and owns no new math.**  Point
  :attr:`~tsdynamics.viz.transforms.PlotTransform.analysis` at the estimator; if
  one does not exist, write it in :mod:`tsdynamics.analysis` first, with its
  citation and its tests.  Numerics inside ``viz/`` is how a plotting layer
  quietly becomes an un-reviewed analysis layer.

Out-of-tree transforms arrive through the ``tsdynamics.plot_transforms``
entry-point group and are indistinguishable from in-tree ones.
"""

from __future__ import annotations

# Imported for its **registration side effect**: importing ``_data`` is what puts
# the in-tree transforms into ``registry.plot_transforms``.  (It pulls in
# ``._registry`` and ``._base`` itself, so the order below is irrelevant.)
from . import _data, fields, hilbert, planar, series, spectra, stability  # noqa: F401
from ._base import (
    Channel,
    ChannelType,
    Geometry,
    Part,
    PlotTransform,
    Presentation,
    Primitive,
    make_frame,
)
from ._frontdoor import plot
from ._primitives import PRIMITIVES, RESERVED_PRIMITIVES, get_primitive, primitive_names
from ._registry import (
    ADMITTED_SERIES_DIAGNOSTICS,
    EXCLUDED_SERIES_TOOLBOX,
    T,
    TransformCall,
    build_spec,
    compatibility,
    draw,
    geometry,
    get,
    lower,
    names,
    plot_transform,
    transforms,
)

__all__ = [
    "ADMITTED_SERIES_DIAGNOSTICS",
    "EXCLUDED_SERIES_TOOLBOX",
    "PRIMITIVES",
    "RESERVED_PRIMITIVES",
    "Channel",
    "ChannelType",
    "Geometry",
    "Part",
    "PlotTransform",
    "Presentation",
    "Primitive",
    "T",
    "TransformCall",
    "build_spec",
    "compatibility",
    "draw",
    "geometry",
    "get",
    "get_primitive",
    "lower",
    "make_frame",
    "names",
    "plot",
    "plot_transform",
    "primitive_names",
    "transforms",
]


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
