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

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

    from scipy.spatial import cKDTree

    from tsdynamics.viz.spec import PlotSpec


class Trajectory:
    """
    The result of integrating or iterating a dynamical system.

    Supports tuple-unpacking for backward compatibility::

        t, y = system.integrate(final_time=100)

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
    >>> traj = lor.integrate(final_time=100)
    >>> traj.dim
    3
    >>> traj["x"]            # named component (via the class's ``variables``)
    array([...])
    >>> traj.after(20.0)     # drop transient
    Trajectory(n_steps=..., dim=3, t=[20.0, 100.0])
    >>> t, y = traj          # tuple-unpack still works
    """

    __slots__ = ("t", "y", "system", "meta", "_kdtree")

    def __init__(
        self,
        t: np.ndarray,
        y: np.ndarray,
        system: Any,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.t = np.asarray(t)
        self.y = np.asarray(y)
        self.system = system
        self.meta = dict(meta) if meta else {}
        self._kdtree: cKDTree | None = None

    # --- compatibility / convenience ---

    def __iter__(self) -> Iterator[np.ndarray]:
        """Allow ``t, y = trajectory``."""
        return iter((self.t, self.y))

    def __getitem__(self, key: Any) -> Any:
        """
        Component access by name, or joint row slicing.

        - ``traj["x"]`` → 1-D component array (requires the system class to
          declare ``variables``).
        - ``traj[["x", "z"]]`` → ``(T, 2)`` array.
        - anything else (int/slice/mask) slices ``t`` and ``y`` together and
          returns a new :class:`Trajectory`.
        """
        if isinstance(key, str):
            return self.y[:, self._component_index(key)]
        if isinstance(key, list | tuple) and key and all(isinstance(k, str) for k in key):
            return self.y[:, [self._component_index(k) for k in key]]
        if isinstance(key, int | np.integer):
            # Keep the result a well-formed Trajectory (one row), not a
            # corrupted one built from scalars.
            return Trajectory(
                np.atleast_1d(self.t[key]),
                np.atleast_2d(self.y[key]),
                self.system,
                meta=self.meta,
            )
        return Trajectory(self.t[key], self.y[key], self.system, meta=self.meta)

    @property
    def variables(self) -> tuple[str, ...] | None:
        """Component names declared by the system (instance attr or class ClassVar)."""
        if self.system is None:
            return None
        # Instance lookup falls back to the ClassVar for the built-in families,
        # and also honours per-instance names (e.g. WrappedSystem).
        return getattr(self.system, "variables", None)

    def _component_index(self, name: str) -> int:
        names = self.variables
        if names is None:
            raise KeyError(
                f"{type(self.system).__name__ if self.system else 'This system'} declares no "
                f"`variables`; use integer indexing or add e.g. variables = ('x', 'y', 'z') "
                f"to the system class."
            )
        try:
            return names.index(name)
        except ValueError:
            raise KeyError(f"Unknown component {name!r}. Declared variables: {names}") from None

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
        Drop the initial transient.

        Parameters
        ----------
        t0 : float
            Keep only time points ``t >= t0``.

        Returns
        -------
        Trajectory
        """
        mask = self.t >= t0
        return Trajectory(self.t[mask], self.y[mask], self.system, meta=self.meta)

    # --- visualization seam ---

    def to_plot_spec(self, kind: str | None = None) -> PlotSpec:
        """
        Describe this trajectory as a backend-agnostic :class:`PlotSpec`.

        The kind is chosen from the trajectory's dimensionality (the dispatch
        that the docs figure tooling hand-rolls today): a 1-D trajectory is a
        :data:`~tsdynamics.viz.spec.PlotKind.TIME_SERIES`, a 2-D one a
        :data:`~tsdynamics.viz.spec.PlotKind.PHASE_PORTRAIT_2D`, and ``dim >= 3``
        a :data:`~tsdynamics.viz.spec.PlotKind.PHASE_PORTRAIT_3D` (the first three
        components).  A Poincaré-section trajectory (one carrying a
        ``meta["plot_kind"]`` of ``"poincare_section"``, set by
        :func:`~tsdynamics.analysis.poincare_section` /
        :meth:`~tsdynamics.derived.PoincareMap.trajectory`) is recognised and
        rendered as the 2-D in-plane scatter instead of a flow line — the section
        intent a renderer would otherwise have to reverse-engineer from ``meta``.

        The :mod:`tsdynamics.viz.spec` import is at module scope but pulls in no
        plotting library, so building a spec (or importing :mod:`tsdynamics`)
        never imports matplotlib / Plotly; the spec itself carries no rendering
        code.

        Parameters
        ----------
        kind : str, optional
            Override the auto-dispatched semantic kind with any member of the
            closed :class:`~tsdynamics.viz.spec.PlotKind` vocabulary (e.g.
            ``"time_series"`` to force component-vs-time on a 3-D trajectory).
            ``None`` (the default) auto-dispatches on dimensionality / section
            intent.

        Returns
        -------
        PlotSpec
        """
        from tsdynamics.errors import InvalidParameterError
        from tsdynamics.viz.spec import Axis, Layer, PlotKind, PlotSpec

        names = self.variables or tuple(f"y{i}" for i in range(self.dim))

        # A Poincaré section carries its intent in meta; honour it before the
        # dimensionality dispatch so a section is never mistaken for a flow.
        if kind is None and str(self.meta.get("plot_kind", "")) == PlotKind.POINCARE_SECTION:
            return self._poincare_section_spec(names)

        if kind is None:
            spec_kind = (
                PlotKind.PHASE_PORTRAIT_3D
                if self.dim >= 3
                else PlotKind.PHASE_PORTRAIT_2D
                if self.dim == 2
                else PlotKind.TIME_SERIES
            )
        else:
            spec_kind = PlotKind(kind)

        meta = dict(self.meta)

        if spec_kind == PlotKind.TIME_SERIES:
            return PlotSpec(
                kind=spec_kind,
                ndim=1,
                title=self._title(),
                x=Axis(label="t"),
                y=Axis(label=names[0]),
                layers=[Layer(PlotKind.LINE, {"x": self.t, "y": self.y[:, 0]}, label=names[0])],
                meta=meta,
            )

        # Take the 2-D-vs-3-D shape from the (possibly overridden) kind, not the
        # raw trajectory dim — so forcing ``kind="phase_portrait_2d"`` on a 3-D
        # trajectory yields a consistent 2-D schema (no stray z channel / LINE3D
        # layer that a renderer dispatching on the 2-D kind would choke on).
        want_3d = spec_kind == PlotKind.PHASE_PORTRAIT_3D
        if want_3d and self.dim < 3:
            raise InvalidParameterError(
                f"kind={spec_kind.value!r} needs a 3-D trajectory, but this one has "
                f"dim={self.dim}; use 'phase_portrait_2d' or 'time_series'."
            )
        cols: dict[str, np.ndarray] = {"x": self.y[:, 0], "y": self.y[:, 1]}
        z = Axis(label=names[2]) if want_3d else None
        if want_3d:
            cols["z"] = self.y[:, 2]
        layer_kind = PlotKind.LINE3D if want_3d else PlotKind.LINE
        return PlotSpec(
            kind=spec_kind,
            ndim=3 if want_3d else 2,
            aspect="equal",
            title=self._title(),
            x=Axis(label=names[0]),
            y=Axis(label=names[1]),
            z=z,
            layers=[Layer(layer_kind, cols)],
            meta=meta,
        )

    def plot(self, backend: str | None = None, **tweaks: Any) -> Any:
        """Render this trajectory via a visualization backend.

        Sugar over :meth:`to_plot_spec` — see
        :meth:`tsdynamics.viz.spec.Plottable.plot` for the tweak keywords.  The
        viz package is imported lazily here (not at module scope) so plain
        ``import tsdynamics`` never pulls it in — and thus never triggers
        renderer-backend discovery / a matplotlib import — honouring the
        no-backend-on-import contract.  Raises
        :class:`~tsdynamics.viz.spec.VisualizationNotInstalled` until a backend
        is registered.
        """
        from tsdynamics.viz.spec import Plottable

        return Plottable.plot(cast("Plottable", self), backend, **tweaks)

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> Any:
        """Notebook display hook — lazily delegated to ``Plottable`` (see :meth:`plot`).

        No-ops (returns ``None``) until a backend is installed, so a trajectory
        still reprs as text in a plain console and ``import`` stays backend-free.
        """
        from tsdynamics.viz.spec import Plottable

        return Plottable._repr_mimebundle_(cast("Plottable", self), include, exclude)

    def _poincare_section_spec(self, names: tuple[str, ...]) -> PlotSpec:
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

    def minmax(self) -> tuple[np.ndarray, np.ndarray]:
        """Return per-component ``(minima, maxima)``, each of shape ``(dim,)``."""
        return self.y.min(axis=0), self.y.max(axis=0)

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

    def neighbors(self, q: Any, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """
        Nearest trajectory points to query point(s) ``q``.

        Builds a KD-tree lazily on first call and caches it; subsequent
        queries are O(log T).

        Parameters
        ----------
        q : array-like, shape (dim,) or (m, dim)
            Query point(s).
        k : int
            Number of neighbours per query point.

        Returns
        -------
        (distances, indices)
            As returned by :meth:`scipy.spatial.cKDTree.query`.
        """
        from scipy.spatial import cKDTree

        if self._kdtree is None:
            self._kdtree = cKDTree(self.y)
        return cast(
            "tuple[np.ndarray, np.ndarray]",
            self._kdtree.query(np.asarray(q, dtype=float), k=k),
        )

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

    def __repr__(self) -> str:
        return (
            f"Trajectory(n_steps={self.n_steps}, dim={self.dim}, "
            f"t=[{self.t[0]:.3g}, {self.t[-1]:.3g}])"
        )
