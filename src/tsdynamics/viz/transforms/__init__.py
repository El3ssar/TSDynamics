"""Plot transforms — the extension point for "what to plot".

A **transform** turns a subject (a :class:`~tsdynamics.data.Trajectory`, a
system, an analysis result) into :class:`Geometry`; a **primitive** turns that
geometry into layers; a declared **compatibility row** says which pairs are
legal.  Adding a plot to TSDynamics is one decorated function, and every name it
needs is public (:mod:`tsdynamics.viz` re-exports all of them)::

    from tsdynamics.viz import PlotKind, plot_transform
    from tsdynamics.analysis import recurrence_matrix

    @plot_transform(
        name="recurrence", source="data",
        frame="grid2", ndim=2, kind=PlotKind.IMAGE, labels=("i", "j"),
        default_primitive="image", primitives=("image",),
        analysis="tsdynamics.analysis.recurrence_matrix",
        example=lambda primitive: (Lorenz().run(20.0, dt=0.05), {}),
        doc="Recurrence plot of a trajectory.",
    )
    def recurrence(traj, *, recurrence_rate=0.05):
        R = recurrence_matrix(traj, recurrence_rate=recurrence_rate).matrix
        i = np.arange(R.shape[0], dtype=float)
        return {"x": i, "y": i, "z": np.asarray(R.todense(), float)}

``compute`` may return a plain **mapping of channels**: the name, the coordinate
space, the axis count and the axis labels are declared on the decorator, so
repeating them in a hand-built :class:`Geometry` would declare each twice and
leave the registry checking the author against themselves.  Build a
:class:`Geometry` (with :func:`make_frame`, :class:`Part`, :class:`FrameSpace`,
all exported here) when the shape or the labels depend on the subject.

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

import sys as _sys
from types import ModuleType

# ── internals: bound and importable, but off the curated tab surface ────────────
# The redundant ``as`` form marks them as deliberate re-exports rather than unused
# imports.  Nothing is removed — ``from tsdynamics.viz.transforms import lower``
# still works — these are simply not names a user writing a plot ever types:
#
# * ``Channel`` / ``ChannelType`` — the typed innards of a :class:`Part`, built
#   for you from the channel mapping a transform returns.
# * ``PRIMITIVES`` / ``RESERVED_PRIMITIVES`` — the primitive tables behind
#   :func:`primitive_names` / :func:`get_primitive`, which are the accessors.
# * ``ADMITTED_SERIES_DIAGNOSTICS`` / ``EXCLUDED_SERIES_TOOLBOX`` — the scope
#   ledger recording which series diagnostics this library does and does not own.
# * ``TransformCall`` / ``build_spec`` / ``lower`` — the transform→PlotSpec
#   lowering pipeline that :func:`plot` drives.
from .._frames import FrameSpace

# Imported for its **registration side effect**: importing ``_data`` is what puts
# the in-tree transforms into ``registry.plot_transforms``.  (It pulls in
# ``._registry`` and ``._base`` itself, so the order below is irrelevant.)
from . import (  # noqa: F401
    _data,
    fields,
    hilbert,
    planar,
    results,
    series,
    spectra,
    stability,
)
from ._base import (
    Channel as Channel,
)
from ._base import (
    ChannelType as ChannelType,
)
from ._base import (
    Geometry,
    Part,
    PlotTransform,
    Presentation,
    Primitive,
    make_frame,
)
from ._frontdoor import plot
from ._primitives import (
    PRIMITIVES as PRIMITIVES,
)
from ._primitives import (
    RESERVED_PRIMITIVES as RESERVED_PRIMITIVES,
)
from ._primitives import get_primitive, primitive_names, register_primitive
from ._registry import (
    ADMITTED_SERIES_DIAGNOSTICS as ADMITTED_SERIES_DIAGNOSTICS,
)
from ._registry import (
    EXCLUDED_SERIES_TOOLBOX as EXCLUDED_SERIES_TOOLBOX,
)
from ._registry import (
    PART_KEYS as PART_KEYS,
)
from ._registry import (
    T,
    allow,
    compatibility,
    draw,
    find,
    geometry,
    get,
    names,
    plot_transform,
    register,
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
from ._registry import (
    part_from_mapping as part_from_mapping,
)
from ._registry import (
    record_for as record_for,
)

__all__ = [
    # The four shared registry verbs (`register` / `names` / `find` / `get`),
    # so `ts.viz.transforms` answers exactly like primitives / renderers / themes.
    "register",
    "names",
    "find",
    "get",
    # The front door and its option carrier.
    "plot",
    "T",
    # Writing a transform: everything the authoring recipe needs, so no
    # transform author has to import from a private module.
    "PlotTransform",
    "Geometry",
    "Part",
    "FrameSpace",
    "make_frame",
    "plot_transform",
    # Writing / choosing a primitive.
    "Primitive",
    "Presentation",
    "primitive_names",
    "get_primitive",
    "register_primitive",
    # Introspecting the registry, and extending a declared row.
    "transforms",
    "compatibility",
    "allow",
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


class _CallableModule(ModuleType):
    """A module that is also callable — so ``ts.viz.transforms()`` lists the names.

    The four registries advertise **one** shape (``register`` / ``names`` /
    ``find`` / ``get``), and three of them are objects you can call to get the
    listing.  ``transforms`` is a *module*, so ``ts.viz.transforms()`` was a
    ``TypeError: 'module' object is not callable`` — one of the four ways "learn
    one, know all four" was false.

    Re-classing the module (rather than replacing it with a registry object)
    keeps every other property intact: ``import tsdynamics.viz.transforms``,
    ``from tsdynamics.viz.transforms import register``, ``isinstance(m,
    ModuleType)`` and the submodule entry in ``sys.modules`` are all unchanged.
    """

    def __call__(self) -> list[str]:
        """Return the registered transform names (the sorted listing)."""
        return names()


_sys.modules[__name__].__class__ = _CallableModule
