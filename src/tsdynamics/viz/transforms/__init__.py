"""Plot transforms — the extension point for "what to plot".

A **transform** turns a subject (a :class:`~tsdynamics.data.Trajectory`, a
system, an analysis result) into :class:`Geometry`; a **primitive** turns that
geometry into layers; a declared **compatibility row** says which pairs are
legal.  Adding a plot to TSDynamics is one decorated function, and every name it
needs is public (:mod:`tsdynamics.viz` re-exports all of them)::

    from tsdynamics.viz import PlotKind
    from tsdynamics.viz.transforms import register
    from tsdynamics.analysis import recurrence_matrix

    @register(
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

The listing here is **five verbs** — ``register`` / ``names`` / ``find`` / ``get``
/ ``allow``.  The authoring types (:class:`Geometry`, :class:`Part`,
:class:`FrameSpace`, :func:`make_frame`, :class:`Presentation`) are listed one
dot away at :mod:`ts.viz.spec <tsdynamics.viz.spec>` and stay importable from
here; the drawing doors (``plot`` / ``draw`` / ``geometry`` / ``compatibility``)
are the ``ts.viz`` names, measured ``is``-identical to these.  See ``__all__``.
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
# * the six front doors (``plot`` ``draw`` ``geometry`` ``compatibility``
#   ``make_frame`` ``T``) — measured ``is``-identical to the ``ts.viz`` spelling,
#   which is the one the documentation teaches; ``plot_transform`` is measured
#   ``is register``; the five IR types are listed at ``ts.viz.spec``; and the
#   four primitive names are the ``ts.viz.primitives`` registry.  See ``__all__``.
from .._frames import FrameSpace as FrameSpace
from .._visibility import listing_dir

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
    Geometry as Geometry,
)
from ._base import (
    Part as Part,
)
from ._base import (
    PlotTransform as PlotTransform,
)
from ._base import (
    Presentation as Presentation,
)
from ._base import (
    Primitive as Primitive,
)
from ._base import (
    make_frame as make_frame,
)
from ._frontdoor import plot as plot
from ._primitives import (
    PRIMITIVES as PRIMITIVES,
)
from ._primitives import (
    RESERVED_PRIMITIVES as RESERVED_PRIMITIVES,
)
from ._primitives import (
    get_primitive as get_primitive,
)
from ._primitives import (
    primitive_names as primitive_names,
)
from ._primitives import (
    register_primitive as register_primitive,
)
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
    T as T,
)
from ._registry import (
    TransformCall as TransformCall,
)
from ._registry import (
    allow,
    find,
    get,
    names,
    register,
)
from ._registry import (
    build_spec as build_spec,
)
from ._registry import (
    compatibility as compatibility,
)
from ._registry import (
    draw as draw,
)
from ._registry import (
    geometry as geometry,
)
from ._registry import (
    lower as lower,
)
from ._registry import (
    part_from_mapping as part_from_mapping,
)
from ._registry import (
    plot_transform as plot_transform,
)
from ._registry import (
    record_for as record_for,
)
from ._registry import (
    transforms as transforms,
)

#: ``ts.viz.transforms`` — **the four shared registry verbs, plus the one verb
#: that is only here** (contract §11.3, T3).
#:
#: This namespace answers the same ``register`` / ``names`` / ``find`` / ``get``
#: shape as ``primitives`` / ``renderers`` / ``themes``, so learning one teaches
#: the rest — and :func:`~tsdynamics.viz.transforms.allow` is the fifth because
#: it exists nowhere else: registering a primitive is only half of adding a way
#: to draw, and ``allow`` is the half that admits it into a shipped row.
#:
#: .. versionchanged:: 6.0
#:    Seventeen names left the listing, and **not one stopped resolving**.  Six
#:    were the front doors, measured ``is``-identical to the ``ts.viz`` spelling
#:    of the same object (``plot`` ``draw`` ``geometry`` ``compatibility``
#:    ``make_frame`` ``T``) — a second address for a thing you have already
#:    found.  ``plot_transform`` is measured ``is register``: one object, two
#:    spellings, which is the C3 defect.  Five are IR types already listed one
#:    dot away at :mod:`ts.viz.spec <tsdynamics.viz.spec>` (``PlotTransform``
#:    ``Geometry`` ``Part`` ``FrameSpace`` ``Presentation``), and the five
#:    primitive names are the ``ts.viz.primitives`` registry under second
#:    spellings (``Primitive`` ``primitive_names`` ``get_primitive``
#:    ``register_primitive``) plus ``transforms``, which is this module naming
#:    itself.  Every one is still bound here and still importable:
#:    ``from tsdynamics.viz.transforms import Geometry`` is unchanged.
__all__ = [
    # The four shared registry verbs (`register` / `names` / `find` / `get`),
    # so `ts.viz.transforms` answers exactly like primitives / renderers / themes.
    "register",
    "names",
    "find",
    "get",
    # ...and the fifth verb, which lives only here: extend a declared row so a
    # primitive YOU registered becomes a legal cell (see `allow`).
    "allow",
]

__dir__ = listing_dir(__all__)


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
