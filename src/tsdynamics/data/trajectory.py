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

from typing import Any

import numpy as np


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
        meta: dict | None = None,
    ) -> None:
        self.t = np.asarray(t)
        self.y = np.asarray(y)
        self.system = system
        self.meta = dict(meta) if meta else {}
        self._kdtree = None

    # --- compatibility / convenience ---

    def __iter__(self):
        """Allow ``t, y = trajectory``."""
        return iter((self.t, self.y))

    def __getitem__(self, key):
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
        return self.y.shape[1]

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
        return self._kdtree.query(np.asarray(q, dtype=float), k=k)

    def set_distance(self, other: Any, *, method: str = "centroid") -> float:
        """
        Distance to another point set (Trajectory or array), as a set.

        ``method`` is ``"centroid"`` (default), ``"hausdorff"``, or
        ``"minimum"`` — see :func:`tsdynamics.data.set_distance`.  The
        matching primitive behind attractor deduplication and continuation.
        """
        from tsdynamics.data import set_distance

        return set_distance(self, other, method=method)

    def __repr__(self) -> str:
        return (
            f"Trajectory(n_steps={self.n_steps}, dim={self.dim}, "
            f"t=[{self.t[0]:.3g}, {self.t[-1]:.3g}])"
        )
