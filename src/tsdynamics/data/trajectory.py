"""
The trajectory — the lingua franca every analysis consumes.

:class:`Trajectory` is the result of integrating or iterating a dynamical
system: a time vector ``t`` and a state array ``y`` of shape ``(T, dim)``,
plus a back-reference to the producing system and a provenance ``meta`` dict.

It lives in :mod:`tsdynamics.data` (not in the families) because it is a *data*
type: the families merely produce it, while the whole analysis layer
(dimensions, embeddings, entropy, recurrence, surrogates, …) consumes it.  It
re-exports through :mod:`tsdynamics.families` and the top-level namespace, so
``from tsdynamics import Trajectory`` and ``from tsdynamics.data import
Trajectory`` resolve to the same object.

The point-set operations (:meth:`Trajectory.minmax`,
:meth:`Trajectory.standardize`, :meth:`Trajectory.neighbors`,
:meth:`Trajectory.set_distance`) build on the geometry primitives in
:mod:`tsdynamics.data.sampling`; the KD-tree backing
:meth:`Trajectory.neighbors` is built lazily and cached per instance.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast

import numpy as np

from tsdynamics._utils.escape import Unbounded, detect_unbounded
from tsdynamics._utils.plot_namespace import plot_namespace as _plot_namespace
from tsdynamics._utils.plot_namespace import plot_seam_error as _plot_seam_error
from tsdynamics.errors import InvalidInputError, remedy
from tsdynamics.errors import taught as _taught

#: What this module *defines*.  ``dir()`` here used to offer ``np``, ``cast``,
#: ``dataclass``, ``NamedTuple``, ``Literal``, ``Callable``, ``Any``,
#: ``TYPE_CHECKING`` and ``annotations`` alongside the four names that matter
#: (``CONTRACT.md`` §11, T4).  ``Unbounded`` / ``detect_unbounded`` are imported
#: from :mod:`tsdynamics._utils.escape` and listed there, not here.
__all__ = ["MinMax", "Neighbors", "Trajectory", "as_trajectory"]

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from scipy.spatial import cKDTree

    from tsdynamics.viz.spec import Animation, Plot


#: How far a ``t`` axis may deviate from perfectly uniform and still report a
#: single :attr:`Trajectory.dt`.  Specified, not guessed: a ``make_output_grid``
#: grid deviates by ~1e-16 relative, and the coarsest *deliberate*
#: non-uniformity in the library (a Poincaré section's crossing times) deviates
#: by O(1) — nine decades of margin on each side.
_DT_UNIFORM_RTOL = 1e-9

#: The ``meta`` keys that describe **the rows in hand** rather than the run that
#: produced them, and so must be re-derived whenever rows are selected.
#:
#: ``meta`` is public and people read it.  Measured before v6 round 8:
#: ``tr[::5].meta["dt"]`` was still the run's ``0.01`` while the object's own
#: axis said ``0.05``, and ``tr[500:].meta["ic"]`` was ``[1, 1, 1]`` for a slice
#: starting at ``t = 5`` from ``[-6.51, -6.97, 23.92]``.  The estimators read
#: :attr:`Trajectory.dt` and were right; a user reading ``meta["dt"]`` to turn a
#: per-sample exponent into a per-unit-time one was off by exactly the
#: decimation factor, silently.  Everything else in ``meta`` — the system, the
#: parameters, the solver, the tolerances — is still true of a slice and is
#: carried verbatim.
#:
#: Each entry is ``key -> f(t, y)``, returning the re-derived value or ``None``
#: to **drop** the key — a slice with no uniform step reports no step at all,
#: rather than the wrong one.  A table rather than three branches, so a fourth
#: row-describing key is one line and cannot be added to the prose alone.
_ROW_DERIVED_META: dict[str, Callable[[np.ndarray, np.ndarray], Any]] = {
    "dt": lambda t, y: _axis_dt(t),
    "t0": lambda t, y: float(t[0]) if t.size else None,
    "ic": lambda t, y: np.array(y[0], dtype=float) if y.size else None,
}

#: The word every plotting door in the library accepts for naming the CURVES.
#: Peeled upstream of this module (by ``ts.plot`` / ``traj.plot`` / ``system.plot``),
#: so it never arrives here as a per-kind option — but it must be *suggestible*
#: and *listed*, or the singular ``label=`` gets answered with ``zlabel=``, an
#: axis name, which is what sent three beta testers into the internals.
_CURVE_NAMING_KEYS = frozenset({"labels"})

#: The four accessor namespaces ruling A2 deleted, and the members of each that
#: a trajectory can actually answer.
_DELETED_TRAJECTORY_ACCESSORS: dict[str, tuple[str, ...]] = {
    "dims": ("correlation_dimension", "generalized_dimension"),
    "lyap": ("lyapunov_from_data",),
    "recurrence": ("recurrence_matrix", "rqa"),
    "chaos": ("zero_one_test",),
}

#: Methods that were a **second spelling** of the bracket and are therefore gone
#: (``name -> (why, the lines to type instead)``).  The error is the migration
#: guide: it names the one spelling and the line runs on the object that was held.
_DELETED_TRAJECTORY_METHODS: dict[str, tuple[str, tuple[str, ...]]] = {
    "sel": (
        "selecting components is what the bracket does, and it returns a Trajectory",
        (
            "traj['x', 'z']      # a 2-column Trajectory (names carried)",
            "traj['x']           # one component, as a plain (T,) array",
        ),
    ),
}


# ---------------------------------------------------------------------------
# __plot_spec__ routing tables (the single-panel front door)
# ---------------------------------------------------------------------------

#: Friendly ``kind=`` spellings → the internal routing key.  A *recipe* like
#: ``"delay"`` is **not** a :class:`~tsdynamics.viz.spec.PlotKind` member (the
#: enum is frozen); it routes to a producer that emits a real semantic kind
#: (a delay embedding is a ``PHASE_PORTRAIT_2D``).  Every other ``kind=`` value
#: passes through unchanged and is resolved against ``PlotKind`` directly.
_KIND_ALIASES: dict[str, str] = {
    "delay": "delay_embedding",
    "delay_embedding": "delay_embedding",
    # ``"field"`` is a *recipe* (not a frozen PlotKind value): it routes to the
    # ``spatial_field`` producer, which emits a real ``SPATIAL_FIELD`` spec — a
    # 1-D profile line or a 2-D heatmap of the system's field, reshaped via its
    # ``_field_shape``.  ``"spatial_field"`` (the kind value) routes here too.
    "field": "spatial_field",
    "spatial_field": "spatial_field",
}

#: Per-route allow-list of the extra keyword(s) accepted via ``**kind_kw`` — kept
#: off the ``__plot_spec__`` signature because each is valid for one kind only.
#: This table is the one place the per-kind options live; extending a kind's
#: options is a one-line edit here (the validation + ``plot()`` forwarding both
#: read it), so the surface grows without reshaping the signature.
_KIND_KW: dict[str, frozenset[str]] = {
    # ``delay`` is a lag in SAMPLES (the library-wide meaning — what
    # ``optimal_delay`` returns); ``delay_time`` is the same lag in TIME UNITS.
    # ``tau`` is accepted only so that the front door can raise the error that
    # names both, instead of Python's bare "unexpected keyword argument".
    "delay_embedding": frozenset({"delay", "delay_time", "tau"}),
    "time_series": frozenset({"color_by"}),
    "phase_portrait_2d": frozenset({"color_by"}),
    "phase_portrait_3d": frozenset({"color_by"}),
    "spacetime": frozenset({"transpose"}),
    # ``"field"`` takes no per-kind keyword — the spatial grid comes from the
    # system's ``_field_shape`` (via meta), and the field-block selector rides on
    # the main ``components=`` argument.
    "spatial_field": frozenset(),
    # A section is a point cloud of recorded crossings; the in-plane axes come
    # from the section plane in ``meta``, so there is nothing to configure here.
    "poincare_section": frozenset(),
}

#: The keywords ``plot()`` peels off and forwards to :meth:`Trajectory.__plot_spec__`
#: (rather than leaking them to the renderer).  Derived from the routing tables so
#: it can never drift out of sync with the per-kind options above.
_PLOT_SPEC_KEYS: frozenset[str] = frozenset({"kind", "components", "animate", "primitive"}).union(
    *_KIND_KW.values()
)

#: Routing key → the registered plot transform that computes the same geometry
#: (:mod:`tsdynamics.viz.transforms`).  This front door keeps its own spec
#: assembly for the default view — so no picture moved when the transforms
#: landed — and consults the registry only when ``primitive=`` asks for a
#: *different drawing* of the same numbers, which is the one thing the inline
#: builders structurally cannot do.
#:
#: ``poincare_section`` is deliberately absent: its geometry is not a registered
#: transform yet, so asking for a primitive there raises rather than silently
#: ignoring the request.
_TRANSFORM_ROUTE: dict[str, str] = {
    "time_series": "time_series",
    "phase_portrait_2d": "phase_portrait",
    "phase_portrait_3d": "phase_portrait",
    "delay_embedding": "delay_embedding",
    "spacetime": "spacetime",
    "spatial_field": "spatial_field",
}

#: The routing keys :meth:`Trajectory.__plot_spec__` can actually **build** — the
#: auto-dispatch targets plus the recipes in :data:`_KIND_ALIASES`.  Validation
#: keys off *this*, not off :class:`~tsdynamics.viz.spec.PlotKind` membership:
#: the enum is the vocabulary of every plot the *library* can describe, while
#: this front door builds exactly one panel from one trajectory.  Before v6
#: ``kind=`` was resolved straight through ``PlotKind(route)``, so
#: ``__plot_spec__(kind="basins_image")`` returned a spec **labelled**
#: ``basins_image`` whose only layer was a plain ``LINE`` — a mislabelled plot,
#: rendered with a basin-image preset — and ``kind="composite"`` produced a
#: zero-panel composite that silently discarded the trajectory.  A kind this
#: front door cannot build now raises.
_BUILDABLE_ROUTES: frozenset[str] = frozenset(
    {
        "time_series",
        "phase_portrait_2d",
        "phase_portrait_3d",
        "spacetime",
        "poincare_section",
        "spatial_field",
        "delay_embedding",
    }
)


class MinMax(NamedTuple):
    """The per-component extent of a point set — ``lo, hi = traj.minmax()``."""

    lo: np.ndarray
    """Per-component minimum, shape ``(dim,)``."""
    hi: np.ndarray
    """Per-component maximum, shape ``(dim,)``."""


@dataclass(frozen=True)
class Neighbors:
    """The answer :meth:`Trajectory.neighbors` gives: **states**, with their distances.

    A small record rather than the KD-tree's raw ``(distances, indices)`` pair,
    because a caller asking "what is near here?" means the nearby *states*, and
    two unlabelled arrays are not an answer.  It still unpacks the way the pair
    did, so ``d, i = traj.neighbors(q)`` is unchanged.

    Attributes
    ----------
    distance : ndarray, shape (k,) or (m, k)
        Distance from each query point to each neighbour, nearest first.
    index : ndarray, shape (k,) or (m, k)
        Row of :attr:`Trajectory.y` each neighbour came from.
    state : ndarray, shape (k, dim) or (m, k, dim)
        The neighbouring states themselves.  ``NaN`` where fewer than ``k``
        neighbours exist.
    """

    distance: np.ndarray
    index: np.ndarray
    state: np.ndarray

    def __iter__(self) -> Iterator[np.ndarray]:
        """Unpack as ``(distance, index)`` — what the raw pair used to be."""
        return iter((self.distance, self.index))

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """``np.asarray(near)`` is the neighbouring **states**."""
        arr = self.state if dtype is None else np.asarray(self.state, dtype=dtype)
        return arr.copy() if copy is True else arr

    def __repr__(self) -> str:
        """State the answer: how many neighbours, and how far the nearest is."""
        k = int(self.index.shape[-1])
        queries = "" if self.index.ndim == 1 else f" for {self.index.shape[0]} query points"
        nearest = float(np.min(self.distance)) if self.distance.size else float("nan")
        return (
            f"Neighbors  {k} nearest state{'s' if k != 1 else ''}{queries}"
            f"   ·  closest at d = {nearest:.4g}"
            "\n    near.state   near.distance   near.index"
        )


def _validated_variables(names: Sequence[str], dim: int) -> tuple[str, ...]:
    """Return ``names`` as a tuple, or raise naming the mismatch.

    One name per state component, each distinct — the contract a system's
    ``variables`` ClassVar carries, enforced at the *data* constructor so a
    measured-data user gets the library's usual taught refusal instead of a
    silently ignored keyword.
    """
    if isinstance(names, str):
        names = (names,)
    resolved = tuple(str(n) for n in names)
    if len(resolved) != dim:
        raise InvalidInputError(
            f"variables names every state component exactly once: this trajectory is "
            f"{dim}-dimensional and {len(resolved)} name"
            f"{'' if len(resolved) == 1 else 's'} were given ({list(resolved)}). Pass "
            f"{dim} name{'' if dim == 1 else 's'}, e.g. "
            f"variables={tuple(f'v{i}' for i in range(dim))!r}."
        )
    if len(set(resolved)) != len(resolved):
        dupes = sorted({n for n in resolved if resolved.count(n) > 1})
        raise InvalidInputError(
            f"variables must be distinct — {dupes} appears more than once, so "
            f"traj[{dupes[0]!r}] could not say which component it means."
        )
    return resolved


#: The public channels :meth:`Trajectory.__setattr__` validates on the way in.
#: ``system`` is deliberately absent — re-pointing a trajectory at another system
#: is provenance the caller owns, and nothing is derived from it.
_GUARDED_CHANNELS = frozenset({"meta", "t", "y"})


def _validated_meta(value: Any) -> dict[str, Any]:
    """Return *value* as a plain provenance dict, or raise naming what ``meta`` is.

    Every consumer in the library reaches into ``meta`` by key
    (``meta["variables"]``, ``meta["dt"]``, ``meta["field_shape"]``), so a
    non-mapping write breaks reading, plotting and every analysis that asks
    where the data came from — each of them far from the assignment.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise InvalidInputError(
            f"Trajectory.meta is the run's provenance mapping (system, params, "
            f"solver, dt, tolerances, ic), got {type(value).__name__}."
            + remedy('traj.meta = {"system": "measured"}', 'traj.meta["source"] = "rig 2"')
        )
    return dict(value)


def _reject_unbuildable_kind(kind: str, route: str) -> None:
    """Raise unless ``route`` names a view this front door can build from a trajectory.

    Parameters
    ----------
    kind : str
        The spelling the caller passed (quoted back in the message).
    route : str
        ``kind`` after :data:`_KIND_ALIASES` resolution.
    """
    from tsdynamics.errors import InvalidParameterError

    if route in _BUILDABLE_ROUTES:
        return
    accepted = sorted(_BUILDABLE_ROUTES | set(_KIND_ALIASES))
    raise InvalidParameterError(
        f"kind={kind!r} is not a view a Trajectory can be plotted as; "
        f"accepted kinds are {accepted}. "
        "(Other PlotKind values name plots built by an analysis result — "
        "e.g. basins_image by basins_of_attraction, recurrence_plot by "
        "recurrence_matrix — or by tsdynamics.viz.plot for a composite.)"
    )


def _auto_route(n_components: int) -> str:
    """Pick the default routing key from the number of selected components.

    1 → time series, 2 → 2-D portrait, 3 → 3-D portrait, **4+ → spacetime image**
    (a high-dimensional field reads as a spacetime plot, never a bogus 3-D
    portrait of its first three coordinates).
    """
    if n_components > 3:
        return "spacetime"
    if n_components == 3:
        return "phase_portrait_3d"
    if n_components == 2:
        return "phase_portrait_2d"
    return "time_series"


def _axis_dt(t: Any) -> float | None:
    """Return the uniform step of time axis ``t``, or ``None`` if it has none.

    The single reading of "what is one sample worth here", shared by
    :attr:`Trajectory.dt` and by the row-selection meta rebuild, so the number a
    trajectory reports and the number it records cannot drift apart.
    """
    axis = np.asarray(t, dtype=float)
    if axis.size < 2:
        return None
    steps = np.diff(axis)
    first = float(steps[0])
    if first <= 0.0 or not np.all(np.isfinite(steps)):
        return None
    # A *uniformity* check, not a solver tolerance — expressed without the
    # ``rtol=``/``atol=`` keywords so it reads as what it is.
    if bool(np.max(np.abs(steps - first)) > _DT_UNIFORM_RTOL * abs(first)):
        return None
    return first


class Trajectory:
    """
    The result of integrating or iterating a dynamical system.

    A trajectory is a **container of samples**: ``len(traj)`` is the number of
    time samples and iterating it yields that many ``(t_i, y_i)`` pairs.

    .. versionchanged:: 6.0
        ``__iter__`` used to be a tuple-unpacking convenience yielding the two
        *columns* ``(t, y)``, which put it in direct contradiction with
        ``__len__`` (``len(traj)`` said ``n_steps``, ``len(list(traj))`` said 2)
        and broke the container contract for any code that sizes an iterable and
        then walks it.  Iteration now yields samples; the column unpacking moved
        to the explicit :meth:`unpack`::

            t, y = traj.unpack()          # was: t, y = traj

    Attributes
    ----------
    t : ndarray, shape (T,)
        Time points (or step indices for discrete maps).
    y : ndarray, shape (T, dim)
        State at each time point.
    system : SystemBase
        Back-reference to the system that produced this trajectory.
    meta : dict
        Provenance: system name, params snapshot, solver, tolerances, ic.

    Examples
    --------
    >>> traj = lor.run(final_time=100)
    >>> traj.dim
    3
    >>> traj["x"]            # named component (via the class's ``variables``)
    array([...])
    >>> traj.after(20.0)     # drop transient
    Trajectory of Lorenz  (n: ..., dim: 3, var: x, y, z)   t ∈ [20, 100]
        measure it:  ts.analysis.find(traj)   ·   draw it:  ts.plot(traj)
    >>> t, y = traj.unpack()          # the two columns
    >>> for t_i, y_i in traj:         # ... or walk the samples
    ...     pass
    """

    #: Declared as a **mapping** rather than a tuple so the four public slots
    #: carry their own documentation.  ``help(traj.y)`` resolves the *value* and
    #: printed NumPy's 200-line ``ndarray`` reference; ``help(traj.meta)`` printed
    #: ``dict()``'s — neither says what the channel means here, and there was no
    #: other place to ask (``CONTRACT.md`` §11.6 defect 2).  ``help(Trajectory.y)``
    #: now answers, and the dict form is otherwise identical: membership
    #: (``name in Trajectory.__slots__``, used by ``__getattr__``) reads the keys,
    #: and slot enforcement is unchanged.
    __slots__ = {
        "t": "Sample times, shape ``(T,)`` — step indices for a discrete map. "
        "This axis is the source of truth for the sampling interval: "
        ":attr:`dt` derives from it, so a sliced trajectory reports the step it "
        "really has rather than the one the run asked for.",
        "y": "State at each sample, shape ``(T, dim)`` — always 2-D, even for a "
        "one-component signal. Prefer ``traj['x']`` / :meth:`component` to a raw "
        "column index: those read the component *names*.",
        "system": "The system that produced this trajectory, or ``None``. "
        "**Measured data has no system**, and every consumer in the library "
        "tolerates that — it is what lets plain arrays reach the analysis layer.",
        "meta": "Provenance of the run that produced this: system name, "
        "parameter snapshot, solver, ``dt``, tolerances, ``ic``, library version. "
        "Preserved verbatim through slicing and :meth:`after`, which is why it "
        "records what was *asked for* and :attr:`dt` reads the ``t`` axis instead.",
        "_kdtree": None,
        "_unbounded": None,
        "_unbounded_checked": None,
    }

    def __init__(
        self,
        t: np.ndarray,
        y: np.ndarray,
        system: Any = None,
        meta: dict[str, Any] | None = None,
        *,
        variables: Sequence[str] | None = None,
    ) -> None:
        """Build a trajectory from time and state arrays.

        Parameters
        ----------
        t : array_like, shape (T,)
            Time points (or step indices).
        y : array_like, shape (T,) or (T, dim)
            State at each time point.  A 1-D array is read as a single
            component and stored as ``(T, 1)``.
        system : SystemBase, optional
            The system that produced this trajectory.  **Measured data has no
            system**, so it defaults to ``None`` — every consumer in the
            library tolerates that, and it is what lets a user's own arrays
            reach the plotting and analysis layers::

                traj = ts.Trajectory(t, y)
                ts.plot(traj)

        meta : dict, optional
            Provenance (system name, params, solver, ``dt``, tolerances, ic).
        variables : sequence of str, optional
            One name per state component — what :meth:`component`, ``traj["x"]``,
            :meth:`to_frame`'s columns and every plot axis label use.  This is
            the door **measured** data comes in by, which has no system to read
            declared names off::

                traj = ts.Trajectory(t, x, variables=("voltage",))
                traj["voltage"]

            .. versionadded:: 6.0
                It used to be reachable only as the undocumented
                ``meta={"variables": (...)}``, and naming it raised a bare
                ``TypeError`` — on the one constructor the package docstring
                points data users at, so everything a measured signal plotted
                was labelled ``y0``.

        Raises
        ------
        InvalidInputError
            If ``variables`` does not name every component exactly once.
        """
        self.t = np.asarray(t)
        y_arr = np.asarray(y)
        self.y = y_arr[:, None] if y_arr.ndim == 1 else y_arr
        self.system = system
        self.meta = dict(meta) if meta else {}
        if variables is not None:
            self.meta["variables"] = _validated_variables(variables, int(self.y.shape[1]))
        self._kdtree: cKDTree | None = None
        self._unbounded: Unbounded | None = None
        self._unbounded_checked = False

    def __setattr__(self, name: str, value: Any) -> None:
        """Write a public channel, keeping the derived answers honest.

        The data channels stay writable — patching ``traj.y`` in place is a
        legitimate thing to do to measured data, and refusing it would cost
        capability for no gain.  What is *not* legitimate is what used to
        happen afterwards: two caches are computed from ``y`` and kept
        (:attr:`unbounded`'s verdict and the KD-tree behind :meth:`neighbors`),
        and a plain slot write left both in place.  Measured, overwriting a
        bounded orbit's states with ``1e300`` left ``traj.unbounded`` reporting
        ``None`` — "this orbit stayed bounded", about data that no longer
        existed (``CONTRACT.md`` §11.6 defect 1).

        So a write to ``y`` or ``t`` drops those caches, ``y`` is reshaped to
        the ``(T, dim)`` the constructor guarantees, and the two axes are
        required to keep describing the same rows.  ``meta`` must stay a
        mapping, since every consumer in the library indexes it.
        """
        if name.startswith("_") or name not in _GUARDED_CHANNELS:
            object.__setattr__(self, name, value)
            return
        if name == "meta":
            object.__setattr__(self, name, _validated_meta(value))
            return
        arr = np.asarray(value)
        if name == "y":
            arr = arr[:, None] if arr.ndim == 1 else arr
            if arr.ndim != 2:
                raise InvalidInputError(
                    f"Trajectory.y must be the (T, dim) state array — 2-D, or 1-D "
                    f"for a single component — got shape {arr.shape}."
                )
            other = getattr(self, "t", None)
        else:
            if arr.ndim != 1:
                raise InvalidInputError(
                    f"Trajectory.t must be the (T,) sample axis, got shape {arr.shape}."
                )
            other = getattr(self, "y", None)
        if other is not None and len(other) != len(arr):
            held = "t" if name == "y" else "y"
            raise InvalidInputError(
                f"Trajectory.{name} would have {len(arr)} rows but .{held} has "
                f"{len(other)}; the two axes describe the same samples, so writing "
                f"one alone leaves the trajectory unreadable. Build the trajectory "
                f"you mean instead." + remedy("traj = ts.Trajectory(t, y, traj.system, traj.meta)")
            )
        object.__setattr__(self, name, arr)
        # Both caches are functions of ``y``; ``t`` selects the rows they are
        # read over, so either write invalidates them.
        object.__setattr__(self, "_kdtree", None)
        object.__setattr__(self, "_unbounded", None)
        object.__setattr__(self, "_unbounded_checked", False)

    # --- derived facts about the sampling ---

    @property
    def unbounded(self) -> Unbounded | None:
        """What this orbit escaped to, or ``None`` if it stayed bounded.

        A run that blows up *without* reaching the engine's hard ``1e150`` guard
        comes back as an ordinary, finite, entirely meaningless trajectory —
        measured, ``Chua().run(ic=[500, 0, 0])`` returns ``max|y| = 1.6e9``, no
        exception, ``isnan`` all ``False``.  Every downstream answer taken from
        it then looks exactly like an honest one.  This is the flag that tells
        the two apart: it is the line the repr prints, and what
        :class:`~tsdynamics.derived.poincare.PoincareSection` carries through.

        Computed on first read and cached, so a trajectory nobody asks about
        pays nothing.

        Returns
        -------
        Unbounded or None
            ``None`` when the orbit is bounded.  Otherwise the record whose
            ``str`` is the warning line — see
            :class:`tsdynamics._utils.escape.Unbounded`.

        Examples
        --------
        >>> import tsdynamics as ts
        >>> lor = ts.systems.Lorenz()
        >>> lor.run(final_time=5.0, ic=[1.0, 1.0, 1.0]).unbounded is None
        True
        """
        if not self._unbounded_checked:
            self._unbounded = detect_unbounded(self.y)
            self._unbounded_checked = True
        return self._unbounded

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_samples, dim)`` — the shape of :attr:`y`."""
        return (int(self.y.shape[0]), int(self.y.shape[1]))

    @property
    def dt(self) -> float | None:
        """The sampling interval, **derived from the ``t`` axis**.

        **The axis is the one truth**, and ``None`` when there is no single
        answer — a non-uniform axis (a Poincaré section) or an axis too short to
        have a step.  ``meta["dt"]`` used to record what the *run* asked for and
        slicing carried it verbatim, so a decimated trajectory reported the
        undecimated step and every per-unit-time estimator taken from ``meta``
        was off by the decimation factor with no exception.  Since v6 round 8
        selecting rows re-derives ``meta["dt"]`` from the new axis
        (:data:`_ROW_DERIVED_META`), so the two agree — but this property is
        still the reading the library itself uses, because it cannot be absent.

        Examples
        --------
        >>> import tsdynamics as ts
        >>> tr = ts.systems.Lorenz().run(final_time=1.0, dt=0.01, ic=[1.0, 1.0, 1.0])
        >>> round(tr.dt, 10)
        0.01
        >>> round(tr[::5].dt, 10)          # the decimated step, not the run's
        0.05
        """
        return _axis_dt(self.t)

    def _columns(self, names: Sequence[str]) -> Trajectory:
        """Build the sub-trajectory holding ``names``, carrying the names along.

        The one place a column selection is assembled, so the selected block and
        the names it answers to cannot disagree.  ``meta["variables"]`` is what
        :attr:`variables` reads first, which is why the result names *its own*
        columns instead of the producing system's full tuple.
        """
        idx = [self._component_index(n) for n in names]
        return Trajectory(
            self.t,
            self.y[:, idx],
            self.system,
            meta={**self.meta, "variables": tuple(self.variables[i] for i in idx)},
        )

    # --- the teaching door (§5.6) ---

    def __getattr__(self, name: str) -> Any:
        """Answer a miss with the free function that does the job.

        ``traj.rqa`` used to be a bare ``AttributeError`` — the one place in the
        library where a wrong guess taught nothing.  Ruling A2 took the analyses
        off the object, so the error **is** the discovery mechanism.

        Every teaching branch is sealed with
        :func:`~tsdynamics.errors.taught`, because CPython appends its own
        ``Did you mean: …?`` to an ``AttributeError`` escaping a ``__getattr__``
        and it was contradicting us: ``traj.dims`` (a retired namespace of four
        free functions) was answered *"Did you mean: 'dim'?"* — the state-space
        dimension, an integer — and ``traj.to_plot_spec`` was answered with the
        dunder the message had just called *"not a verb you type"*.
        """
        if name.startswith("_") or name in Trajectory.__slots__:
            raise AttributeError(name)
        try:
            raise self._miss(name)
        except AttributeError as err:
            raise _taught(err, name) from None

    def _miss(self, name: str) -> AttributeError:
        """Build the teaching error for a missing attribute.  See :meth:`__getattr__`."""
        if name == "to_plot_spec":
            return _plot_seam_error("Trajectory", "traj")
        if name in _DELETED_TRAJECTORY_METHODS:
            why, lines = _DELETED_TRAJECTORY_METHODS[name]
            return AttributeError(
                f"'Trajectory' object has no attribute {name!r}: {why}.\n"
                + "\n".join(f"    {line}" for line in lines)
            )
        from tsdynamics.analysis import _discovery

        try:
            from tsdynamics import registry

            known = set(registry.analyses.names())
        except Exception:  # pragma: no cover - defensive
            return AttributeError(name)
        has_system = object.__getattribute__(self, "system") is not None
        if name in known:
            return _discovery.attribute_error(name, "Trajectory", "data", has_system=has_system)
        if name in _DELETED_TRAJECTORY_ACCESSORS:
            members = _DELETED_TRAJECTORY_ACCESSORS[name]
            return AttributeError(
                f"'Trajectory' object has no attribute {name!r}: the .lyap / .chaos "
                f"/ .dims / .recurrence namespaces are gone — every member is "
                f"a free function.\n"
                + "\n".join(f"    ts.analysis.{fn}(traj)" for fn in members)
                + "\n    "
                + _discovery.find_line("data")
            )
        near = _discovery.near_miss(name, known)
        tail = f"\n    ts.analysis.{near}(traj)" if near else ""
        return AttributeError(
            f"'Trajectory' object has no attribute {name!r}."
            + (f" Did you mean:{tail}" if near else "")
            + "\n    "
            + _discovery.find_line("data", lead="the")
        )

    # --- compatibility / convenience ---

    def __iter__(self) -> Iterator[tuple[Any, np.ndarray]]:
        """Yield one ``(t_i, y_i)`` sample per time point.

        Consistent with :meth:`__len__` by construction — the two used to
        disagree (``len(traj)`` counted samples while iteration yielded the two
        *columns*), which is a broken container contract.  For the columns use
        :meth:`unpack`.
        """
        return zip(self.t, self.y, strict=True)

    def unpack(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the two column arrays ``(t, y)``.

        The explicit spelling of what ``t, y = traj`` used to do implicitly (see
        :meth:`__iter__`)::

            t, y = system.run(final_time=100).unpack()

        Returns
        -------
        tuple of (ndarray, ndarray)
            ``t`` of shape ``(T,)`` and ``y`` of shape ``(T, dim)`` — the live
            arrays, not copies.
        """
        return self.t, self.y

    def __len__(self) -> int:
        """Return the number of time samples — ``len(traj) == traj.n_steps``."""
        return int(self.t.shape[0])

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """Expose the state array to NumPy — ``np.asarray(traj) is traj.y``.

        Without this hook NumPy fell back to treating a trajectory as an opaque
        scalar object: ``np.asarray(traj)`` returned a **0-d object array**, so
        any downstream ``arr.shape`` / ``arr[:, 0]`` on it failed far from the
        cause.  The array interface returns the ``(T, dim)`` state block ``y``
        (the time vector stays on ``traj.t``), matching what every analysis in
        the library already consumes.
        """
        arr = self.y if dtype is None else np.asarray(self.y, dtype=dtype)
        if copy is True:
            return arr.copy()
        if copy is False and arr is not self.y:
            raise ValueError(
                "np.array(trajectory, copy=False) cannot avoid a copy for "
                f"dtype={dtype!r}; the state array is {self.y.dtype}."
            )
        return arr

    def _rows(self, t: np.ndarray, y: np.ndarray) -> Trajectory:
        """Build the sub-trajectory holding rows ``(t, y)``, with honest ``meta``.

        The one place a row selection is assembled, so a decimated trajectory and
        a tail slice cannot describe themselves differently.  The three keys that
        describe the rows (:data:`_ROW_DERIVED_META`) are re-derived from the new
        arrays; a key the parent did not carry is not invented, and a slice with
        no uniform step drops ``dt`` rather than reporting one that is wrong.
        """
        meta = dict(self.meta)
        for key, rederive in _ROW_DERIVED_META.items():
            if key not in meta:
                continue  # the parent did not carry it; do not invent it
            value = rederive(np.asarray(t), np.asarray(y))
            if value is None:
                meta.pop(key)
            else:
                meta[key] = value
        return Trajectory(t, y, self.system, meta=meta)

    def __getitem__(self, key: Any) -> Any:
        """Select **columns by name**, or **rows by position** — one rule each.

        ============================  ==========================================
        ``traj["x"]``                 one component, as a plain ``(T,)`` array
        ``traj["x", "z"]``            several components, as a **Trajectory**
        ``traj[10:50]`` / ``traj[0]`` / ``traj[mask]` / ``traj[[0, 2]]``
                                      rows, as a **Trajectory**
        ============================  ==========================================

        A **name** selects a column and a **number** selects a row; several
        names give back the same type you started with, so the selection
        composes (``traj["x", "z"][100:]``) and keeps its time axis, its names
        and its provenance.  This is the pandas rule — ``df["a"]`` is the
        series, ``df["a", "b"]`` the frame — and it is the only rule here.

        .. versionchanged:: 6.0
            ``traj[["x", "z"]]`` (the list spelling) was a second way to write
            ``traj["x", "z"]``, and **both returned a bare array** while the
            docs promised a ``Trajectory`` — so the time axis and the component
            names were silently dropped by the spelling the docs taught.  The
            list spelling now raises, naming the one that works, because a list
            of *numbers* selects rows and a list cannot mean both.

        Examples
        --------
        >>> import tsdynamics as ts
        >>> tr = ts.systems.Lorenz().run(final_time=1.0, dt=0.1, ic=[1.0, 1.0, 1.0])
        >>> tr["x"].shape
        (11,)
        >>> xz = tr["x", "z"]
        >>> xz.shape, xz.variables
        ((11, 2), ('x', 'z'))
        """
        if isinstance(key, str):
            return self.y[:, self._component_index(key)]
        if isinstance(key, tuple | list) and key:
            named = [k for k in key if isinstance(k, str)]
            if named and len(named) == len(key) and isinstance(key, tuple):
                return self._columns(named)
            if named or isinstance(key, tuple):
                raise InvalidInputError(self._bad_key_message(key, named))
        if isinstance(key, int | np.integer):
            # Keep the result a well-formed Trajectory (one row), not a
            # corrupted one built from scalars.
            return self._rows(np.atleast_1d(self.t[key]), np.atleast_2d(self.y[key]))
        return self._rows(self.t[key], self.y[key])

    def _bad_key_message(self, key: Any, named: list[str]) -> str:
        """Explain a bracket that is neither a column selection nor a row one."""
        cols = ", ".join(repr(n) for n in (named or list(self.variables[:2])))
        if named and len(named) < len(key):
            why = (
                f"traj[{key!r}] mixes component names with positions. "
                "A name selects a column, a number selects a row — never both at once."
            )
        elif isinstance(key, list):
            why = (
                f"traj[[{cols}]] is the retired list spelling of traj[{cols}]. "
                "A list of numbers already selects ROWS, so a list cannot also mean "
                "columns; there is one spelling now, and it gives back a Trajectory."
            )
        else:
            why = (
                "a trajectory has no joint (row, column) index — columns are named "
                "and rows are numbered. Raw NumPy indexing is traj.y[...]."
            )
        return "\n".join(
            (
                why,
                f"    traj[{cols}]".ljust(40) + "# the columns, as a Trajectory",
                "    traj[10:50]".ljust(40) + "# the rows, as a Trajectory",
            )
        )

    @property
    def variables(self) -> tuple[str, ...]:
        """The name of **every** component — never ``None``.

        Resolution order: the names this trajectory carries in
        ``meta["variables"]`` (written by a column selection, so a sub-trajectory
        names *its own* columns), then the producing system's ``variables``, then
        the generated ``y0 … y{dim-1}``.

        .. versionchanged:: 6.0
            This used to be a pass-through to ``system.variables`` returning
            ``None`` for measured data — so a bare-array trajectory printed
            ``y0``/``y1`` column headings in :meth:`to_frame` and on its plot,
            then refused ``traj["y0"]``; and a column selection reported the
            *inner* system's full tuple, which made ``traj["x", "z"]["y"]``
            silently return ``z``.  The names now come from one place and are
            always as long as the state block is wide.
        """
        names = self.meta.get("variables")
        if names is None and self.system is not None:
            # Instance lookup falls back to the ClassVar for the built-in
            # families, and also honours per-instance names (WrappedSystem).
            names = getattr(self.system, "variables", None)
        if names is not None:
            resolved = tuple(str(n) for n in names)
            if len(resolved) == self.dim:
                return resolved
        return tuple(f"y{i}" for i in range(self.dim))

    def _component_index(self, name: str) -> int:
        names = self.variables
        try:
            return names.index(name)
        except ValueError:
            # KeyError renders its argument with repr(), so a multi-line remedy
            # block would reach the user as literal ``\n`` escapes.  Keep the fix
            # on one line -- and spell it as the expression to type, closest
            # declared name first, rather than only listing what is valid.
            import difflib

            # Case first: difflib scores 'Z' against 'z' at zero, yet a wrong-case
            # component name is the single most common way to miss.
            folded = [n for n in names if n.lower() == name.lower()]
            close = folded or difflib.get_close_matches(name, names, n=1, cutoff=0.4)
            pick = close[0] if close else names[0]
            hint = f" Did you mean {close[0]!r}?" if close else ""
            raise KeyError(
                f"Unknown component {name!r}. Declared variables: {names}."
                f"{hint} Type: traj[{pick!r}]"
            ) from None

    @property
    def dim(self) -> int:
        """State-space dimension."""
        return int(self.y.shape[1])

    @property
    def n_steps(self) -> int:
        """Number of time steps."""
        return len(self.t)

    def component(self, i: int | str) -> np.ndarray:
        """
        Return a single state component.

        Parameters
        ----------
        i : int or str
            Component index, or component name when the system declares
            ``variables``.

        Returns
        -------
        ndarray, shape (T,)
        """
        if isinstance(i, str):
            i = self._component_index(i)
        return self.y[:, i]

    def after(self, t0: float) -> Trajectory:
        """
        Drop the initial transient — keep the samples at or after ``t0``.

        Parameters
        ----------
        t0 : float
            A **cut on the recorded time axis**, read in that axis's own unit:
            time units for a flow, the iteration index for a map, the crossing
            time for a section.  (It is a cut, not a horizon: ``run(t0=...)``
            says where the integration *starts*.)

        Returns
        -------
        Trajectory
            The tail, sharing this trajectory's system and provenance.
        """
        mask = self.t >= t0
        return self._rows(self.t[mask], self.y[mask])

    # --- tabular export ---

    @staticmethod
    def _require_pandas() -> Any:
        """Import :mod:`pandas` lazily, raising a friendly hint if it is absent."""
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "Trajectory.to_frame() needs pandas, which is not a dependency of "
                "tsdynamics. Install it with `pip install pandas`. Without it, "
                "`traj.t` / `traj.y` (or `np.asarray(traj)`) give you the same data "
                "as plain NumPy arrays."
            ) from exc
        return pd

    def to_frame(self) -> Any:
        """Return a :class:`pandas.DataFrame` view of the trajectory.

        One row per time sample: the time vector becomes the index (named
        ``"t"``) and each state component a column, named from the system's
        ``variables`` when it declares them and ``y0…y{dim-1}`` otherwise.  The
        provenance travels along on ``frame.attrs["meta"]``.

        ``pandas`` is a **soft** dependency imported lazily here, so importing
        tsdynamics never pulls it in; a missing install raises an
        :class:`ImportError` pointing at ``pip install pandas`` (and at the
        ``traj.t`` / ``traj.y`` arrays, which need nothing extra).

        Returns
        -------
        pandas.DataFrame
            Shape ``(n_steps, dim)``.

        Raises
        ------
        ImportError
            If :mod:`pandas` is not installed.
        """
        pd = self._require_pandas()
        names = self.variables
        frame = pd.DataFrame(
            self.y,
            index=pd.Index(self.t, name="t"),
            columns=list(names),
        )
        frame.attrs["meta"] = dict(self.meta)
        return frame

    # --- visualization seam ---

    def __plot_spec__(
        self,
        kind: str | None = None,
        *,
        components: int | str | Sequence[int | str] | None = None,
        animate: bool | dict[str, Any] | Animation = False,
        primitive: str | None = None,
        **kind_kw: Any,
    ) -> Plot:
        """
        Describe this trajectory as a backend-agnostic :class:`PlotSpec`.

        **This is the internal seam, not a verb you type.**  ``ts.plot``
        classifies a positional argument as a *subject* by asking whether it
        carries ``__plot_spec__`` — one predicate, so a system, a trajectory and
        all 32 analysis results are recognised the same way.  The two spellings a
        user types are :func:`tsdynamics.plot` (hands back the
        :class:`~tsdynamics.viz.spec.Plot`, drawing nothing) and
        :meth:`plot` (the same thing, styled at the door).

        .. versionchanged:: 6.0
            Was the public ``to_plot_spec``.  ``ts.plot(traj)`` returns the very
            same object without rendering, so a third public spelling for
            "build a plot but do not draw it" bought nothing and cost a
            newcomer a choice.  ``traj.to_plot_spec`` now raises, naming both
            spellings that work.

        Every common view goes through here, so the parameterised
        ``viz.producers`` builders stay an internal detail.

        Auto-dispatch
            With ``kind=None`` the semantic kind follows the number of selected
            components (after applying ``components=``): 1 → ``TIME_SERIES``,
            2 → ``PHASE_PORTRAIT_2D``, 3 → ``PHASE_PORTRAIT_3D``, and **4+ →
            ``SPACETIME``** (a Lorenz-96-style field image, *not* a misleading
            3-D portrait of the first three coordinates).  A discrete-map orbit
            draws with a ``SCATTER`` mark (a point sequence) rather than a line.
            A Poincaré-section trajectory (carrying ``meta["plot_kind"]`` of
            ``"poincare_section"``) is recognised and drawn as its in-plane
            scatter.

        Selecting components
            ``components=`` picks what to draw — a name, an index, or a sequence
            of them: ``components="x"`` (a single time series), ``components=
            ["y0", "y1", "y2"]`` (a 3-D portrait of three chosen channels).  The
            auto-dispatch then keys off how many you selected.

        Overriding the kind
            ``kind=`` forces any member of the closed
            :class:`~tsdynamics.viz.spec.PlotKind` vocabulary — e.g.
            ``kind="time_series"`` to overlay component-vs-time on a 3-D
            trajectory, or ``kind="spacetime"`` to image it.  The recipe
            ``kind="delay"`` builds a delay-coordinate embedding ``x(t)`` vs
            ``x(t - delay)``; pass the lag via ``**kind_kw`` as either
            ``delay=`` (**samples**) or ``delay_time=`` (**time units**).

        Per-kind options (``**kind_kw``)
            Options valid for one kind only are accepted as keywords rather than
            cluttering the signature — ``delay`` / ``delay_time`` (exactly one
            required for ``kind="delay"``: a lag in **samples**, or the same lag
            in **time units**, converted via ``meta["dt"]``),
            ``color_by`` (time series / phase portraits — a named field
            ``"time"``/``"speed"``/``"sagitta"``/``"curvature"``/``"acceleration"``/
            ``"arclength"``/``"index"``, a per-point array, or a callable
            ``f(trajectory) -> array``), ``transpose`` (spacetime).  Passing one to
            the wrong kind raises
            :class:`~tsdynamics.errors.InvalidParameterError`.

        The :mod:`tsdynamics.viz` import is local to this method (lazy), and the
        spec carries no rendering code, so building a spec (or importing
        :mod:`tsdynamics`) never imports matplotlib / Plotly.

        Parameters
        ----------
        kind : str, optional
            Override the auto-dispatched kind.  ``None`` auto-dispatches on the
            number of selected components.  Accepted values are the views a
            trajectory can actually *be* — ``"time_series"``,
            ``"phase_portrait_2d"``, ``"phase_portrait_3d"``, ``"spacetime"``,
            ``"poincare_section"``, ``"spatial_field"`` — plus the ``"delay"`` and
            ``"field"`` recipes.  Any other ``PlotKind`` value raises
            :class:`~tsdynamics.errors.InvalidParameterError`: those kinds are
            built by an analysis result (``basins_image`` by
            ``basins_of_attraction``, ``recurrence_plot`` by
            ``recurrence_matrix``, …) or by ``tsdynamics.viz.plot``, not from a
            trajectory.

            .. versionchanged:: 6.0
                Previously *every* ``PlotKind`` spelling (and even a layer mark
                like ``"line"``) was accepted and produced a spec **labelled**
                with the requested kind whose only layer was a plain line —
                a mislabelled plot; ``kind="composite"`` additionally returned a
                zero-panel composite, silently discarding the trajectory.
        components : int or str or sequence of int/str, optional
            Which state components to draw (names or indices).  ``None`` uses all.
        primitive : str, optional
            **How** to draw the view — the drawing primitive, as opposed to
            ``kind``, which is *what* is drawn.  ``None`` (the default) uses this
            front door's own presentation, unchanged.  Naming one routes the same
            geometry through the registered transform
            (:mod:`tsdynamics.viz.transforms`) and draws it that way::

                traj.plot()                          # a line
                traj.plot(primitive="points")        # the same orbit, unconnected
                traj.plot(primitive="density")       # binned — no overdraw

            Validated against the transform's declared row: an invalid pair
            raises, naming the valid primitives, rather than drawing something
            else.  ``ts.viz.compatibility()`` lists the matrix.

            .. versionadded:: 6.0
        animate : bool or dict or Animation, optional
            Turn the spec into a reveal animation.  ``True`` uses sensible per-kind
            defaults (a moving head on portraits / spacetime, off for a plain time
            series); a dict overrides individual
            :class:`~tsdynamics.viz.spec.Animation` fields; an
            :class:`~tsdynamics.viz.spec.Animation` is used as-is.  Tweak further
            with the chainable ``spec.animate()`` / ``.trail()`` / ``.head()`` /
            ``.camera()`` / ``.clock()`` methods.
        **kind_kw
            Per-kind options (see above).

        Returns
        -------
        PlotSpec
        """
        from tsdynamics.viz.spec import PlotKind

        all_names = self.variables

        # A Poincaré section carries its intent in meta; honour it before the
        # dimensionality dispatch (only for the unmodified default view).
        if (
            kind is None
            and components is None
            and not kind_kw
            and str(self.meta.get("plot_kind", "")) == PlotKind.POINCARE_SECTION
        ):
            self._reject_primitive("poincare_section", primitive)
            return self._with_animation(self._poincare_section_spec(all_names), animate)

        # The ``"field"`` / ``"spatial_field"`` recipe routes before component
        # resolution: here ``components=`` selects a *field block* (e.g. Gray–
        # Scott's "u"/"v"), not a state component, so it must not be resolved
        # against the per-cell labels.
        if kind is not None:
            explicit_route = _KIND_ALIASES.get(kind, kind)
            _reject_unbuildable_kind(kind, explicit_route)
            if explicit_route == "spatial_field":
                self._validate_kind_kw("spatial_field", kind_kw)
                field = self._spatial_field_spec(components, primitive)
                return self._with_animation(field, animate)
            if explicit_route == "poincare_section":
                # A section is a *point cloud of crossings*, not a portrait: route
                # to the section builder (which projects onto the plane recorded in
                # meta, else the first two components) rather than letting the
                # kind fall through to ``_phase_portrait_spec`` — which used to
                # return a LINE-layer portrait merely *labelled* POINCARE_SECTION.
                self._validate_kind_kw("poincare_section", kind_kw)
                self._reject_primitive("poincare_section", primitive)
                section = self._poincare_section_spec(all_names)
                return self._with_animation(section, animate)

        sel = self._resolve_components(components, all_names)
        sel_names = tuple(all_names[i] for i in sel)
        n_sel = len(sel)
        discrete = self._is_discrete()

        # Resolve the routing key: a friendly alias (``"delay"``) first, else the
        # auto kind from the number of selected components.
        route = _KIND_ALIASES.get(kind, kind) if kind is not None else _auto_route(n_sel)
        self._validate_kind_kw(route, kind_kw)

        if route == "delay_embedding":
            delay = self._delay_spec(
                sel, sel_names, kind_kw, explicit=components is not None, primitive=primitive
            )
            return self._with_animation(delay, animate)

        spec_kind = PlotKind(route)  # "delay" never reaches here (aliased above)
        ys = self.y[:, sel]

        if spec_kind == PlotKind.SPACETIME:
            if primitive is not None:
                image = self._via_transform(
                    "spacetime", primitive, transpose=bool(kind_kw.get("transpose"))
                )
            else:
                image = self._spacetime_spec(
                    ys, sel_names, transpose=bool(kind_kw.get("transpose"))
                )
            return self._with_animation(image, animate)

        color_by = kind_kw.get("color_by")
        if spec_kind == PlotKind.TIME_SERIES:
            if primitive is not None:
                series = self._via_transform(
                    "time_series", primitive, components=sel, color_by=color_by
                )
            else:
                series = self._time_series_spec(sel, ys, sel_names, discrete, color_by)
            return self._with_animation(series, animate)

        if primitive is not None:
            need = 3 if spec_kind == PlotKind.PHASE_PORTRAIT_3D else 2
            portrait = self._via_transform(
                "phase_portrait", primitive, components=sel[:need], color_by=color_by
            )
        else:
            portrait = self._phase_portrait_spec(spec_kind, sel, ys, sel_names, discrete, color_by)
        return self._with_animation(portrait, animate)

    def _with_animation(self, spec: Plot, animate: bool | dict[str, Any] | Animation) -> Plot:
        """Stamp an :class:`Animation` onto ``spec`` per the ``animate`` request.

        ``False``/``None`` leaves the spec static.  ``True`` uses per-kind defaults
        (head on except for a plain time series); a dict overrides individual
        :class:`~tsdynamics.viz.spec.Animation` fields; an ``Animation`` is used
        verbatim.  The spec carries ``meta["dt"]``, so time-unit trails / the clock
        resolve at render time.

        A :data:`~tsdynamics.viz.spec.PlotKind.SPATIAL_FIELD` spec always animates
        in the **frames** model — the field *movie*: each frame is the spatial
        state at that instant (a 1-D profile line or a 2-D heatmap), so consecutive
        frames carry genuinely different data.  No comet window, no head marker; the
        field itself is the motion.
        """
        if animate is False or animate is None:
            return spec
        from dataclasses import replace as _dc_replace

        from tsdynamics.viz.spec import Animation as _Animation
        from tsdynamics.viz.spec import PlotKind

        if spec.kind == PlotKind.SPATIAL_FIELD:
            # The spatial-field movie is the frames model — every frame is a fresh
            # spatial snapshot.  Force ``mode="frames"`` and the field defaults (no
            # comet window, no head), letting the caller's own keys still win.
            field_defaults: dict[str, Any] = {
                "mode": "frames",
                "head": False,
                "trail_kind": None,
            }
            if isinstance(animate, _Animation):
                # Use dataclasses.replace — never mutate the caller's Animation.
                spec.animation = _dc_replace(
                    animate, mode="frames", head=False, trail_kind=None, trail_length=None
                )
            elif isinstance(animate, dict):
                spec.animation = _Animation(**{**field_defaults, **animate})
            else:  # truthy (e.g. ``True``)
                spec.animation = _Animation(**field_defaults)
            return spec

        head_default = spec.kind != PlotKind.TIME_SERIES
        # Default to a *windowed* comet (a moving trail + head over the full faint
        # curve) — smooth, small, and rotatable in plotly; a persistent "draw-it-in"
        # is one ``.trail(length=None)`` away.  A tight window (≈ 1/10 of the series,
        # capped) keeps the comet crisp and the exported HTML small.
        defaults: dict[str, Any] = {
            "head": head_default,
            "trail_kind": "steps",
            "trail_length": float(max(2, min(self.n_steps // 10, 200))),
        }
        if isinstance(animate, _Animation):
            # Copy — never store (and later mutate) the caller's Animation by
            # reference, so a subsequent ``.trail()``/``.head()`` on the spec does
            # not reach back and mutate the caller's object (matches the
            # SPATIAL_FIELD branch, which already copies via ``replace``).
            spec.animation = _dc_replace(animate)
        elif isinstance(animate, dict):
            spec.animation = _Animation(**{**defaults, **animate})
        else:  # truthy (e.g. ``True``)
            spec.animation = _Animation(**defaults)
        return spec

    def _resolve_components(
        self, components: int | str | Sequence[int | str] | None, names: tuple[str, ...]
    ) -> list[int]:
        """Resolve a ``components=`` selector to a list of column indices.

        ``None`` selects every component; a lone name/index is wrapped; names
        resolve against ``names`` (the declared or generated ``y0…`` labels).
        """
        from tsdynamics.errors import InvalidParameterError

        if components is None:
            return list(range(self.dim))
        items: Sequence[int | str]
        items = (components,) if isinstance(components, (int, str, np.integer)) else components
        out: list[int] = []
        for c in items:
            if isinstance(c, str):
                if c not in names:
                    raise InvalidParameterError(
                        f"unknown component {c!r}; available components: {list(names)}"
                    )
                out.append(names.index(c))
            else:
                idx = int(c)
                if not -self.dim <= idx < self.dim:
                    raise InvalidParameterError(
                        f"component index {idx} out of range for a {self.dim}-D trajectory"
                    )
                out.append(idx % self.dim)
        if not out:
            raise InvalidParameterError("components= selected no channels; pass at least one.")
        return out

    @staticmethod
    def _validate_kind_kw(route: str | None, kind_kw: dict[str, Any]) -> None:
        """Reject per-kind options passed to the wrong kind.

        The suggestion is searched across **every vocabulary this door speaks**,
        not only the one-element per-kind list: ``colour=`` used to be answered
        with *"allowed here: ['color_by']"* — a different concept (a data
        channel) one letter from the style key ``color=`` the caller meant, and
        ``ttle=`` / ``linewith=`` / ``final_tim=`` all got the same wrong hint.

        ``labels=`` — the word that names the CURVES — is in the pool and in the
        printed listing, because it was in neither.  Measured: ``ts.plot(a, b,
        components="x", label=[...])`` answered *"label= — did you mean
        zlabel=?"*, an **axis** name, while ``labels=`` was on the signature and
        working; and the sibling message for ``names=`` enumerated every accepted
        keyword without it.  Three testers gave up and reached into the
        internals.  The ranking is :func:`~tsdynamics.viz.spec.nearest_keyword`,
        shared with the front door, which prefers a candidate that merely
        *extends* what was typed (``label`` → ``labels``) over an equal-scoring
        one that does not (``label`` → ``zlabel``, 0.909 each).
        """
        from tsdynamics.errors import InvalidParameterError
        from tsdynamics.viz.spec import FIGURE_KEYS, nearest_keyword
        from tsdynamics.viz.style import style_names

        allowed = _KIND_KW.get(route or "", frozenset())
        unknown = set(kind_kw) - allowed
        if not unknown:
            return
        pool = sorted(
            set(allowed)
            | set(FIGURE_KEYS)
            | set(style_names())
            | _PLOT_SPEC_KEYS
            | _CURVE_NAMING_KEYS
        )
        hints = ""
        for bad in sorted(unknown):
            near = nearest_keyword(bad, pool)
            if near:
                hints += f"\n    {bad}= — did you mean {near}=?"
        raise InvalidParameterError(
            f"kind={route!r} does not accept keyword(s) {sorted(unknown)}; "
            f"allowed here: {sorted(allowed) or '(none)'}{hints}"
            "\n(plus any style keyword — color=, linewidth=, alpha=, … — any "
            "figure keyword — title=, xlabel=, xlim=, theme=, … — and labels=, "
            "which names the curves.)"
        )

    def _delay_samples(self, delay_time: float) -> int:
        """Convert a delay in **time units** to an integer sample lag.

        The one conversion, shared by both plotting front doors (this class's
        ``__plot_spec__`` and the ``delay_embedding`` transform behind
        ``ts.plot``), so a delay can never mean two different things depending on
        which door you came in through.
        """
        from tsdynamics.errors import InvalidParameterError

        value = float(delay_time)
        if not np.isfinite(value) or value <= 0:
            raise InvalidParameterError(
                f"delay_time= is a positive, finite time, got {delay_time!r}."
            )
        dt = self.meta.get("dt")
        if dt is None and self.meta.get("time") == "index":
            # Measured data handed in as a bare array: its "time" axis is the
            # sample index, so a delay in TIME UNITS has nothing to convert
            # against.  Falling through to the median-diff fallback below would
            # silently use dt=1 — a second delay-in-two-units bug, walking in
            # through the door built to fix the first.
            raise InvalidParameterError(
                "delay_time= is in time units, but this data came in as a bare array, "
                "so its time axis is the sample index. Say what a sample is worth, or "
                "pass the lag in samples:\n"
                "    ts.plot(data, 'delay_embedding', delay=7)     # 7 SAMPLES\n"
                "    ts.plot(data, dt=0.01)                        # 1 sample = 0.01"
            )
        dt_f = float(dt) if dt is not None else None
        if dt_f is None or not np.isfinite(dt_f) or dt_f <= 0:
            diffs = np.diff(self.t)
            dt_f = float(np.median(diffs)) if diffs.size else None
        if dt_f is None or dt_f <= 0:
            raise InvalidParameterError(
                "cannot convert delay_time to samples: the trajectory carries no "
                "'dt' in meta and its time grid is degenerate. Pass the lag in samples "
                "instead, e.g. delay=7."
            )
        samples = max(1, int(round(value / dt_f)))
        if samples >= self.n_steps:
            raise InvalidParameterError(
                f"delay_time={delay_time} (→ {samples} samples at dt={dt_f:g}) must be "
                f"shorter than the series length {self.n_steps}."
            )
        return samples

    def _via_transform(self, transform: str, primitive: str, /, **options: Any) -> Plot:
        """Build this view through the registered transform, drawn by ``primitive``.

        The ``primitive=`` route.  It is deliberately *not* the default path: the
        inline builders above are what every existing figure was drawn with, and
        the transforms landed picture-preserving precisely so that any later
        difference is attributable.  The two presentations differ only where the
        transform is the better one — a multi-component time series labels its y
        axis ``""`` rather than with the first component's name, and colours its
        curves from the palette.
        """
        from tsdynamics.viz.transforms import build_spec

        return build_spec(self, transform, primitive=primitive, **options)

    @staticmethod
    def _reject_primitive(route: str, primitive: str | None) -> None:
        """Raise when ``primitive=`` is asked of a view that has no transform yet."""
        from tsdynamics.errors import InvalidParameterError

        if primitive is None:
            return
        raise InvalidParameterError(
            f"kind={route!r} has no registered plot transform, so primitive={primitive!r} "
            "cannot be honored. Drop primitive= to get the standard view; "
            "ts.viz.compatibility() lists the transforms that do accept one."
        )

    def _delay_spec(
        self,
        sel: list[int],
        sel_names: tuple[str, ...],
        kind_kw: dict[str, Any],
        *,
        explicit: bool,
        primitive: str | None = None,
    ) -> Plot:
        """Build the ``x(t)`` vs ``x(t - delay)`` delay embedding (a ``PHASE_PORTRAIT_2D``).

        A delay embedding reconstructs **one** scalar observable; with no
        ``components=`` it embeds the first component, but an explicit
        multi-component selection is rejected rather than silently dropped.

        ``delay`` (samples) / ``delay_time`` (time units) are resolved by the very
        helper the ``delay_embedding`` transform uses, so this door and
        ``ts.plot(traj, "delay_embedding", ...)`` cannot disagree about units.
        """
        from tsdynamics.errors import InvalidParameterError
        from tsdynamics.viz.transforms._data import _resolve_delay

        if explicit and len(sel) != 1:
            raise InvalidParameterError(
                "kind='delay' embeds a single component; select exactly one via "
                f"components= (got {len(sel)})."
            )
        samples = _resolve_delay(
            self,
            self.n_steps,
            kind_kw.get("delay"),
            kind_kw.get("delay_time"),
            kind_kw.get("tau"),
        )
        return self._via_transform(
            "delay_embedding",
            primitive or "line",
            delay=samples,
            components=sel[0],
            label=sel_names[0],
        )

    def _time_series_spec(
        self,
        sel: list[int],
        ys: np.ndarray,
        sel_names: tuple[str, ...],
        discrete: bool,
        color_by: Any,
    ) -> Plot:
        """Overlay one component-vs-time layer per selected component."""
        from tsdynamics.viz.spec import Axis, Layer, Legend, PlotKind, PlotSpec

        if color_by is not None:
            from tsdynamics.viz import producers

            return producers.time_series(self, components=sel, color_by=color_by)
        mark = PlotKind.SCATTER if discrete else PlotKind.LINE
        layers = [
            Layer(mark, {"x": self.t, "y": ys[:, k]}, label=sel_names[k]) for k in range(len(sel))
        ]
        return PlotSpec(
            kind=PlotKind.TIME_SERIES,
            ndim=1,
            title=self._title(),
            x=Axis(label="t"),
            y=Axis(label=sel_names[0]),
            layers=layers,
            legend=Legend() if len(sel) > 1 else None,
            meta=dict(self.meta),
        )

    def _phase_portrait_spec(
        self,
        spec_kind: Any,
        sel: list[int],
        ys: np.ndarray,
        sel_names: tuple[str, ...],
        discrete: bool,
        color_by: Any,
    ) -> Plot:
        """Build a 2-D / 3-D phase portrait over the selected components."""
        from tsdynamics.errors import InvalidParameterError
        from tsdynamics.viz.spec import Axis, Layer, PlotKind, PlotSpec

        # Shape (2-D vs 3-D) follows the resolved kind, not the raw selection —
        # forcing ``kind="phase_portrait_2d"`` yields a clean 2-D schema (no z).
        want_3d = spec_kind == PlotKind.PHASE_PORTRAIT_3D
        need = 3 if want_3d else 2
        if len(sel) < need:
            raise InvalidParameterError(
                f"kind={spec_kind.value!r} needs at least {need} components, but "
                f"{len(sel)} were selected; use 'time_series' or select more."
            )
        if color_by is not None:
            from tsdynamics.viz import producers

            return producers.phase_portrait(self, components=sel[:need], color_by=color_by)
        cols: dict[str, np.ndarray] = {"x": ys[:, 0], "y": ys[:, 1]}
        z = Axis(label=sel_names[2]) if want_3d else None
        if want_3d:
            cols["z"] = ys[:, 2]
        flow_mark = PlotKind.LINE3D if want_3d else PlotKind.LINE
        layer_kind = PlotKind.SCATTER if discrete else flow_mark
        return PlotSpec(
            kind=spec_kind,
            ndim=3 if want_3d else 2,
            aspect="equal",
            title=self._title(),
            x=Axis(label=sel_names[0]),
            y=Axis(label=sel_names[1]),
            z=z,
            layers=[Layer(layer_kind, cols)],
            meta=dict(self.meta),
        )

    def _is_discrete(self) -> bool:
        """Whether the producing system is a discrete map (default ``False``).

        Reads the **family word**, not the removed ``is_discrete`` flag: that
        name is alive only through an internal alias table, so this would have
        gone quiet — and answered "flow" for every map — the day the table is
        emptied.  A synthetic trajectory with no system is treated as a flow.
        """
        return getattr(self.system, "family", None) == "map"

    def _spacetime_spec(
        self, ys: np.ndarray, names: tuple[str, ...], *, transpose: bool = False
    ) -> Plot:
        """Build the component-index vs time ``IMAGE`` spec (``SPACETIME``).

        The spatiotemporal field view of a high-dimensional flow (a Lorenz-96
        lattice): the selected columns are drawn as a single color-mapped
        ``IMAGE`` with time along ``x`` and component index along ``y`` (or the
        axes swapped when ``transpose=True``); the colorbar / ``clim`` are
        inferred from the field.
        """
        from tsdynamics.viz.spec import Axis, Colorbar, Layer, PlotKind, PlotSpec

        comp_idx = np.arange(ys.shape[1], dtype=float)
        if transpose:
            img = ys
            x_axis, y_axis = Axis(label="component"), Axis(label="t")
            x_data, y_data = comp_idx, self.t
        else:
            img = ys.T
            x_axis, y_axis = Axis(label="t"), Axis(label="component")
            x_data, y_data = self.t, comp_idx
        layer = Layer(PlotKind.IMAGE, {"x": x_data, "y": y_data, "c": img.ravel(), "z": img})
        spec = PlotSpec(
            kind=PlotKind.SPACETIME,
            ndim=2,
            title=self._title(),
            x=x_axis,
            y=y_axis,
            layers=[layer],
            colorbar=Colorbar(label="state"),
            # ``time_axis`` records which image axis runs in time so an
            # animate={"mode":"frames"} movie grows along time under either
            # orientation (rows when transposed, columns otherwise).
            meta={
                **dict(self.meta),
                "component_names": list(names),
                "time_axis": "row" if transpose else "col",
            },
        )
        return spec.autocolor()

    def _spatial_field_spec(
        self,
        component: int | str | Sequence[int | str] | None,
        primitive: str | None = None,
    ) -> Plot:
        """Build a :data:`SPATIAL_FIELD` spec via the ``spatial_field`` producer.

        The spatial grid is read from ``meta["field_shape"]`` (recorded by a system
        declaring ``_field_shape``); ``component`` selects a field block
        (Gray–Scott's ``"u"`` / ``"v"``).  A bare field with no grid metadata falls
        back to a 1-D profile.
        """
        from tsdynamics.errors import InvalidParameterError
        from tsdynamics.viz import producers

        block: int | str | None
        if component is None:
            block = None
        elif isinstance(component, (int, str, np.integer)):
            block = int(component) if isinstance(component, np.integer) else component
        else:  # a sequence — a field movie plots exactly one block
            items = list(component)
            if len(items) != 1:
                raise InvalidParameterError(
                    "kind='field' plots a single field block; select exactly one via "
                    f"components= (got {len(items)})."
                )
            block = items[0]
        if primitive is not None:
            return self._via_transform("spatial_field", primitive, components=block)
        return producers.spatial_field(self, components=block)

    def _plot_impl(self, *transforms: Any, **kwargs: Any) -> Plot:
        """Build this trajectory's :class:`PlotSpec`, applying inline tweaks first.

        ``plot`` **builds**, ``render`` **draws**, ``save`` **writes** — one word,
        one return type, everywhere::

            traj.plot()                          # a PlotSpec
            traj.plot().save("fig.png")          # write it
            traj.plot().render("plotly")         # a plotly figure
            traj.plot(title="Lorenz")            # tweak, still a PlotSpec

        Sugar over :meth:`__plot_spec__`: the spec-shaping keywords (``kind``,
        ``components``, ``primitive``, ``animate``, and the per-kind options
        ``delay`` / ``delay_time`` / ``color_by`` / ``transpose``) are peeled off
        and passed to :meth:`__plot_spec__`; the **style** vocabulary
        (:data:`~tsdynamics.viz.style.STYLE_KEYS` and its aliases — ``color`` /
        ``lw`` / ``alpha`` / …) plus ``theme`` are applied to the finished spec;
        the rest are inline spec tweaks (``xlabel`` / ``yscale`` / ``title`` /
        …).  A *renderer* option (``ax=``, ``figsize=``, a backend name) belongs
        to :meth:`~tsdynamics.viz.spec.PlotSpec.render` and raises here.

        The style keywords are the same set, with the same spellings, that
        ``ts.plot(traj, color=...)`` accepts — one vocabulary, both doors::

            traj.plot(color="crimson", linewidth=0.6, title="Lorenz", theme="dark")

        The viz package is imported lazily (not at module scope) so plain
        ``import tsdynamics`` never pulls it in.

        .. versionchanged:: 6.0
            Returns the :class:`~tsdynamics.viz.spec.PlotSpec` instead of the
            backend figure.  ``ts.plot(traj)`` already returned a spec, so the two
            spellings of the same word returned two different types and
            ``traj.plot().save(...)`` raised ``'Figure' object has no attribute
            'save'``.  Use ``.render(backend, **backend_kw)`` for a figure.
        """
        from tsdynamics.viz.compose import apply_labels
        from tsdynamics.viz.spec import reject_positional_transform
        from tsdynamics.viz.style import style_names

        reject_positional_transform(transforms, "traj")
        spec_kw = {k: kwargs.pop(k) for k in list(kwargs) if k in _PLOT_SPEC_KEYS}
        names = style_names()
        style = {k: kwargs.pop(k) for k in list(kwargs) if k in names}
        theme = kwargs.pop("theme", None)
        labels = kwargs.pop("labels", None)
        spec = self.__plot_spec__(**spec_kw)
        if theme is not None:
            spec.theme(theme)
        if style:
            spec.style(**style)
        if labels is not None:
            apply_labels([spec], labels)
        return spec.tweak(**kwargs)

    #: ``subject.plot`` is BOTH the verb and the namespace (§6.7): ``plot()``
    #: draws the default view, ``plot.psd()`` / ``plot.nullclines()`` name a
    #: transform, and ``plot.<TAB>`` lists every transform that draws THIS
    #: subject — the discovery route ruling A2 promised when it took the
    #: analyses off the object.
    plot = _plot_namespace(_plot_impl)

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> Any:
        """Notebook display hook — lazily delegated to ``Plottable`` (see :meth:`plot`).

        No-ops (returns ``None``) until a backend is installed, so a trajectory
        still reprs as text in a plain console and ``import`` stays backend-free.
        """
        from tsdynamics.viz.spec import Plottable

        return Plottable._repr_mimebundle_(cast("Plottable", self), include, exclude)

    def _poincare_section_spec(self, names: tuple[str, ...]) -> Plot:
        """Build the 2-D in-plane scatter spec for a Poincaré-section trajectory.

        Projects the recorded crossing states onto the section plane (dropping
        the normal coordinate) and picks the two in-plane axes with the largest
        spread to display — so the section reads as a 2-D point cloud, not a 3-D
        flow.  Falls back to the first two components if the plane is unavailable.
        """
        from tsdynamics.viz.spec import Axis, Layer, PlotKind, PlotSpec

        i, j = self._section_axes()
        layers = [Layer(PlotKind.SCATTER, {"x": self.y[:, i], "y": self.y[:, j]})]
        return PlotSpec(
            kind=PlotKind.POINCARE_SECTION,
            ndim=2,
            aspect="equal",
            title=self._title("Poincaré section"),
            x=Axis(label=names[i]),
            y=Axis(label=names[j]),
            layers=layers,
            meta=dict(self.meta),
        )

    def _section_axes(self) -> tuple[int, int]:
        """Pick the two in-plane display axes for a Poincaré section.

        Drops the coordinate the section plane fixes (read from ``meta["plane"]``
        as ``(index, value)`` when present) and, of the remaining coordinates,
        keeps the two with the largest range so the scatter is maximally
        informative.  Defaults to ``(0, 1)`` when the plane / extra columns are
        unavailable.
        """
        plane = self.meta.get("plane")
        normal_idx: int | None = None
        if isinstance(plane, (tuple, list)) and len(plane) == 2 and np.isscalar(plane[0]):
            normal_idx = int(cast(Any, plane[0]))
        candidates = [c for c in range(self.dim) if c != normal_idx]
        if len(candidates) < 2:
            candidates = list(range(self.dim))[:2]
        if len(candidates) < 2:
            return 0, 0 if self.dim == 1 else 1
        if self.y.shape[0] == 0:
            # An empty section (the plane caught no crossings) has no spread to
            # rank — a reduction over the zero-size axis would raise. Keep the
            # first two in-plane candidates so the section still yields a valid
            # (empty) 2-D scatter spec.
            i, j = candidates[:2]
            return (i, j) if i < j else (j, i)
        spreads = self.y.max(axis=0) - self.y.min(axis=0)
        i, j = sorted(candidates, key=lambda c: spreads[c], reverse=True)[:2]
        return (i, j) if i < j else (j, i)

    def _title(self, prefix: str | None = None) -> str:
        """Compose a title from the originating system name and an optional prefix."""
        system = self.meta.get("system")
        name = str(system) if system else ""
        if prefix and name:
            return f"{prefix} — {name}"
        return prefix or name

    # --- point-set operations ---

    def minmax(self) -> MinMax:
        """Return the per-component extent — ``(lo, hi)``, each of shape ``(dim,)``.

        Unpacks like the bare tuple it used to be (``lo, hi = traj.minmax()``)
        and names its halves (``traj.minmax().hi``), so a reader of the call
        site does not have to remember which one comes first.
        """
        return MinMax(self.y.min(axis=0), self.y.max(axis=0))

    def standardize(self) -> Trajectory:
        """
        Return a copy with zero mean and unit standard deviation per component.

        The applied transform is recorded in ``meta["standardized"]``.
        """
        mean = self.y.mean(axis=0)
        std = self.y.std(axis=0)
        std = np.where(std < np.finfo(float).tiny, 1.0, std)
        return Trajectory(
            self.t,
            (self.y - mean) / std,
            self.system,
            meta={**self.meta, "standardized": {"mean": mean, "std": std}},
        )

    def neighbors(self, q: Any, k: int = 1) -> Neighbors:
        """Return the trajectory **states** nearest to query point(s) ``q``.

        Builds a KD-tree lazily on first call and caches it; subsequent
        queries are O(log T).

        Parameters
        ----------
        q : array-like, shape (dim,) or (m, dim)
            Query point(s).  A single point gives a result without the leading
            query axis; ``m`` points keep it.
        k : int, default 1
            Number of neighbours per query point.  The ``k`` axis is **always**
            present, whatever ``k`` is.

        Returns
        -------
        Neighbors
            ``.state`` — the neighbouring states, shape ``(k, dim)`` (or
            ``(m, k, dim)``); ``.distance`` and ``.index`` — shape ``(k,)`` (or
            ``(m, k)``).  It unpacks as ``(distance, index)``, so
            ``d, i = traj.neighbors(q)`` reads as it always did.

        .. versionchanged:: 6.0
            This handed back ``scipy.spatial.cKDTree.query``'s raw
            ``(distances, indices)`` — two unlabelled arrays whose *rank* changed
            with ``k`` (``k=1`` gave two scalars, ``k=2`` two arrays, so
            ``i[0]`` worked for one and raised for the other), and which never
            contained the thing a caller is usually after: the neighbouring
            states.

        Examples
        --------
        >>> import numpy as np, tsdynamics as ts
        >>> tr = ts.systems.Lorenz().run(final_time=1.0, dt=0.1, ic=[1.0, 1.0, 1.0])
        >>> near = tr.neighbors([1.0, 1.0, 1.0], k=2)
        >>> near.state.shape, near.distance.shape, near.index.shape
        ((2, 3), (2,), (2,))
        """
        from scipy.spatial import cKDTree

        if self._kdtree is None:
            self._kdtree = cKDTree(self.y)
        query = np.asarray(q, dtype=float)
        raw_d, raw_i = self._kdtree.query(query, k=k)
        # scipy squeezes the ``k`` axis when ``k == 1``; restore it so the answer
        # has one shape for every ``k``.  The *query* axis follows the input's
        # rank, which is what a caller who passed one point means.
        distance = np.asarray(raw_d, dtype=float)
        index = np.asarray(raw_i)
        if k == 1:
            distance, index = distance[..., None], index[..., None]
        # A query for more neighbours than there are points: scipy reports the
        # one-past-the-end index with an infinite distance.  Report the state as
        # NaN rather than indexing out of the array.
        missing = index >= self.y.shape[0]
        state = np.asarray(self.y, dtype=float)[np.where(missing, 0, index)]
        if bool(missing.any()):
            state = state.copy()
            state[missing] = np.nan
        return Neighbors(distance=distance, index=index, state=state)

    def set_distance(self, other: Any, *, method: str = "centroid") -> float:
        """
        Distance to another point set (Trajectory or array), as a set.

        ``method`` is ``"centroid"`` (default), ``"hausdorff"``, or
        ``"minimum"`` — see :func:`tsdynamics.data.set_distance`.  The
        matching primitive behind attractor deduplication and continuation.
        """
        from tsdynamics.data import set_distance

        return set_distance(
            self, other, method=cast('Literal["centroid", "hausdorff", "minimum"]', method)
        )

    # --- pickling (``__slots__`` needs an explicit state protocol) ---

    def __getstate__(self) -> dict[str, Any]:
        """Return the picklable state; the lazy KD-tree cache is dropped."""
        return {"t": self.t, "y": self.y, "system": self.system, "meta": self.meta}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore from :meth:`__getstate__` (the KD-tree rebuilds on demand)."""
        self.t = state["t"]
        self.y = state["y"]
        self.system = state["system"]
        self.meta = state["meta"]
        self._kdtree = None
        self._unbounded = None
        self._unbounded_checked = False

    def __repr__(self) -> str:
        """Render what you hold, and the one line that says what to do next.

        The trailing ``measure it:`` line is load-bearing: ruling A2 took the
        analyses off the object, so this is the only route that reaches a user
        who never guesses a wrong name.
        """
        who = ""
        if self.system is not None:
            who = f" of {type(self.system).__name__}"
        names = self.variables
        cols = f", var: {', '.join(names)}" if names and len(names) <= 4 else ""
        escaped = self.unbounded
        warning = f"\n    {escaped}" if escaped is not None else ""
        return (
            f"Trajectory{who}  (n: {len(self)}, dim: {self.dim}{cols})"
            f"   t ∈ [{self.t[0]:.4g}, {self.t[-1]:.4g}]"
            f"{warning}"
            f"\n    measure it:  ts.analysis.find(traj)   ·   draw it:  ts.plot(traj)"
        )


def as_trajectory(obj: Any, *, dt: float | None = None) -> Trajectory:
    """Coerce measured data to a :class:`Trajectory` — the door plain arrays come in by.

    A user holding numbers (a recorded signal, a point cloud, a column of a
    dataframe) should not have to construct a library type to plot or analyse
    them.  This is the one coercion every front door uses, so an array means the
    same thing wherever it is handed in.

    Parameters
    ----------
    obj : Trajectory, ndarray, or array-like
        A :class:`Trajectory` (returned unchanged, so a caller who already has
        one loses nothing), a 1-D series of shape ``(T,)``, or a 2-D block of
        shape ``(T, dim)`` — one row per sample, one column per component.
    dt : float, optional
        Sampling interval.  Without it the time axis is the **sample index**
        (recorded as ``meta["time"] == "index"``), which is honest: a bare array
        carries no time.  With it the time axis is ``dt * arange(T)`` and ``dt``
        is recorded in ``meta``, so anything that converts time units to samples
        (a delay embedding, say) has the number it needs.

    Returns
    -------
    Trajectory

    Raises
    ------
    tsdynamics.errors.InvalidInputError
        If ``obj`` is not a Trajectory and is not an array of shape ``(T,)`` or
        ``(T, dim)``.

    Examples
    --------
    >>> import numpy as np
    >>> as_trajectory(np.sin(np.linspace(0, 10, 64))).shape
    (64, 1)
    >>> as_trajectory(np.zeros((64, 3)), dt=0.01).dt
    0.01
    """
    from tsdynamics.errors import InvalidInputError, InvalidParameterError

    if isinstance(obj, Trajectory):
        return obj

    try:
        arr = np.asarray(obj, dtype=float)
    except (TypeError, ValueError) as err:
        raise InvalidInputError(
            f"cannot read {type(obj).__name__} as data: pass a Trajectory, a 1-D "
            "series of shape (T,), or a 2-D block of shape (T, dim)."
        ) from err
    if arr.ndim == 0 or arr.ndim > 2:
        raise InvalidInputError(
            f"cannot read an array of shape {arr.shape} as data: a series is (T,) "
            "and a multi-component block is (T, dim) — one row per sample."
        )
    y = arr[:, None] if arr.ndim == 1 else arr

    meta: dict[str, Any] = {"time": "index"}
    if dt is None:
        t = np.arange(y.shape[0], dtype=float)
    else:
        step = float(dt)
        if not np.isfinite(step) or step <= 0:
            raise InvalidParameterError(f"dt must be a positive, finite number, got {dt!r}.")
        t = step * np.arange(y.shape[0], dtype=float)
        meta["dt"] = step
    return Trajectory(t, y, None, meta=meta)


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
