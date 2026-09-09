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

# ── internals: bound and importable, but off the curated tab surface ────────────
# The redundant ``as`` form marks them as deliberate re-exports rather than unused
# imports.  Nothing is removed — ``from tsdynamics.viz.transforms import lower``
# still works — these are simply not names a user writing a plot ever types:
#
# * ``Channel`` / ``ChannelType`` / ``Part`` — the constituent pieces of a
#   :class:`Geometry`, which transforms build through :func:`make_frame`.
# * ``PRIMITIVES`` / ``RESERVED_PRIMITIVES`` — the primitive tables behind
#   :func:`primitive_names` / :func:`get_primitive`, which are the accessors.
# * ``ADMITTED_SERIES_DIAGNOSTICS`` / ``EXCLUDED_SERIES_TOOLBOX`` — the scope
#   ledger recording which series diagnostics this library does and does not own.
# * ``TransformCall`` / ``build_spec`` / ``lower`` — the transform→PlotSpec
#   lowering pipeline that :func:`plot` drives.
from ._base import (
    Channel as Channel,
)
from ._base import (
    ChannelType as ChannelType,
)
from ._base import (
    Geometry,
    PlotTransform,
    Presentation,
    Primitive,
    make_frame,
)
from ._base import (
    Part as Part,
)
from ._frontdoor import plot
from ._primitives import (
    PRIMITIVES as PRIMITIVES,
)
from ._primitives import (
    RESERVED_PRIMITIVES as RESERVED_PRIMITIVES,
)
from ._primitives import get_primitive, primitive_names
from ._registry import (
    ADMITTED_SERIES_DIAGNOSTICS as ADMITTED_SERIES_DIAGNOSTICS,
)
from ._registry import (
    EXCLUDED_SERIES_TOOLBOX as EXCLUDED_SERIES_TOOLBOX,
)
from ._registry import (
    T,
    compatibility,
    draw,
    geometry,
    get,
    names,
    plot_transform,
    transforms,
)
from ._registry import (
    TransformCall as TransformCall,
)
from ._registry import (
    build_spec as build_spec,
)
from ._registry import (
    lower as lower,
)

__all__ = [
    # The front door and its option carrier.
    "plot",
    "T",
    # Writing a transform.
    "PlotTransform",
    "Geometry",
    "make_frame",
    "plot_transform",
    # Writing / choosing a primitive.
    "Primitive",
    "Presentation",
    "primitive_names",
    "get_primitive",
    # Introspecting the registry.
    "transforms",
    "names",
    "get",
    "compatibility",
    # Raw-array escape hatches.
    "geometry",
    "draw",
]


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete.

    The geometry-constituent types, the primitive tables and the lowering
    pipeline stay bound and importable — see the internals block above.
    """
    return sorted(__all__)
