"""Ensemble of identical systems stepped in lockstep."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tsdynamics.data import Trajectory
    from tsdynamics.viz.spec import PlotSpec

__all__ = ["Ensemble", "EnsembleSystem", "TrajectoryBatch"]


class TrajectoryBatch(list["Trajectory"]):
    """What ``Ensemble.run(...)`` returns — the trajectories, plus the batch view.

    A ``list`` subclass, so it iterates, indexes and ``len()``s like the list of
    trajectories it is.  ``.final`` is the ``(n, dim)`` array of end states —
    exactly what ``system.ensemble(ics, final_time=...)`` used to return before
    ``ensemble`` became a noun.
    """

    __slots__ = ()

    @property
    def final(self) -> np.ndarray:
        """The ``(n, dim)`` final states, one row per member."""
        if not self:
            return np.empty((0, 0))
        return np.array([np.asarray(t.y)[-1] for t in self], dtype=float)

    def to_frame(self) -> Any:
        """Return one long-format ``DataFrame``, with a ``member`` column."""
        import pandas as pd

        frames = []
        for i, traj in enumerate(self):
            frame = traj.to_frame()
            frame.insert(0, "member", i)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def plot(self, *transforms: Any, **kwargs: Any) -> Any:
        """Draw every member on one figure."""
        from tsdynamics.viz import plot as _plot

        return _plot(*self, *transforms, **kwargs)

    def __repr__(self) -> str:
        if not self:
            return "TrajectoryBatch(empty)"
        rows, dim = np.asarray(self[0].y).shape
        return f"TrajectoryBatch({len(self)} trajectories, t: {rows}, var: {dim})"


class Ensemble:
    """
    Many copies of one system, advanced synchronously from different states.

    Used for two-trajectory Lyapunov estimates, basin sampling, and ensemble
    statistics.  Members are independent copies — parameters are shared at
    construction, states are per-member.

    Parameters
    ----------
    system : System
        The template system (copied per member; the original is untouched).
    states : array-like, shape (m, dim)
        One initial state per member.

    Examples
    --------
    >>> band = Lorenz().ensemble([[1, 1, 1], [1.001, 1, 1]])
    >>> band.step(0.01)                       # doctest: +SKIP
    array([[...], [...]])
    >>> band.run(final_time=10.0).final       # doctest: +SKIP
    array([[...], [...]])
    """

    def __init__(self, system: Any, states: Any) -> None:
        states_arr = np.atleast_2d(np.asarray(states, dtype=float))
        if states_arr.shape[1] != system.dim:
            raise ValueError(f"states must have shape (m, {system.dim}), got {states_arr.shape}")
        self.template = system
        self.members = []
        for s in states_arr:
            member = system.copy()
            member.reinit(s)
            self.members.append(member)

    @property
    def size(self) -> int:
        """Number of ensemble members."""
        return len(self.members)

    @property
    def dim(self) -> int:
        """State-space dimension of each member."""
        return cast(int, self.template.dim)

    @property
    def _is_discrete(self) -> bool:
        """Match the template system's time semantics."""
        return cast(bool, self.template._is_discrete)

    @property
    def family(self) -> str:
        """The template system's family word."""
        return cast(str, self.template.family)

    @property
    def params(self) -> Any:
        """The template system's parameters (shared by every member)."""
        return self.template.params

    @property
    def variables(self) -> tuple[str, ...]:
        """The template system's component names."""
        return cast("tuple[str, ...]", self.template.variables)

    def state(self) -> np.ndarray:
        """Return the stacked member states — the ``System``-protocol reading."""
        return self.states()

    def reinit(self, u: Any | None = None, **kwargs: Any) -> None:
        """Restart every member (from ``u``, or from its own construction state)."""
        for member in self.members:
            member.reinit(u, **kwargs)

    def run(self, *args: Any, **kwargs: Any) -> TrajectoryBatch:
        """Run every member and return a :class:`TrajectoryBatch`.

        Every argument is forwarded verbatim to the template family's ``run``,
        with each member's own initial state supplied as ``ic=``::

            band = lor.ensemble(np.random.rand(100, 3))
            batch = band.run(final_time=10.0)
            batch.final          # the (100, 3) end states
        """
        seed = kwargs.pop("seed", None)
        per_member_seed: list[int | None]
        if self.family == "sde" and seed is not None:
            # Member ``i`` draws its noise from ``seed_for(seed, i)`` — depending
            # only on the index — which is the engine's parallel-equals-serial
            # contract.  One shared seed would give every member the SAME path.
            from tsdynamics.families.stochastic import _seed_for as seed_for

            per_member_seed = [seed_for(seed, i) for i in range(len(self.members))]
        else:
            if seed is not None:
                kwargs["seed"] = seed
            per_member_seed = [None] * len(self.members)

        return TrajectoryBatch(self._run_members(args, kwargs, per_member_seed))

    def _run_members(
        self, args: tuple[Any, ...], kwargs: dict[str, Any], seeds: list[int | None]
    ) -> list[Trajectory]:
        """Run every member, ISOLATING a divergence as a NaN row.

        A batch is a population, and one member blowing up is a *result* about
        that member — not a reason to discard the other 999.  The engine's own
        fan-out has always recorded a diverged trajectory as a NaN row; the
        member-by-member Python path used to let the exception escape, so the
        two disagreed about the same contract.
        """
        from tsdynamics.data import Trajectory
        from tsdynamics.errors import ConvergenceError

        runs: list[Trajectory | None] = []
        for member, member_seed in zip(self.members, seeds, strict=True):
            extra = {} if member_seed is None else {"seed": member_seed}
            try:
                runs.append(self.template.run(*args, ic=member.state(), **kwargs, **extra))
            except ConvergenceError:
                runs.append(None)
        template = next((r for r in runs if r is not None), None)
        if template is None:
            raise ConvergenceError(
                f"every one of the {len(runs)} ensemble members diverged, so this "
                f"batch measures nothing. Shorten final_time, or start closer to "
                f"the attractor."
            )
        blank = np.full_like(np.asarray(template.y, dtype=float), np.nan)
        return [
            r
            if r is not None
            else Trajectory(template.t, blank.copy(), self.template, meta=dict(template.meta))
            for r in runs
        ]

    def step(self, n_or_dt: float | int | None = None) -> np.ndarray:
        """Advance every member synchronously and return the stacked states.

        Parameters
        ----------
        n_or_dt : float or int, optional
            The per-member increment (a ``dt`` for a flow, an iteration count for
            a map).  ``None`` uses each member's default.

        Returns
        -------
        numpy.ndarray
            The new states, shape ``(size, dim)`` — one row per member.
        """
        return np.array([m.step(n_or_dt) for m in self.members])

    def states(self) -> np.ndarray:
        """Return the current member states, shape ``(size, dim)``."""
        return np.array([m.state() for m in self.members])

    def set_states(self, states: Any) -> None:
        """Overwrite every member's state.

        Parameters
        ----------
        states : array-like, shape (size, dim)
            One new state per member, in member order.

        Raises
        ------
        ValueError
            If ``states`` is not exactly ``(size, dim)``.
        """
        states_arr = np.atleast_2d(np.asarray(states, dtype=float))
        if states_arr.shape != (self.size, self.dim):
            raise ValueError(f"expected shape {(self.size, self.dim)}, got {states_arr.shape}")
        for member, s in zip(self.members, states_arr, strict=True):
            member.set_state(s)

    def time(self) -> float:
        """Return the common member time."""
        return self.members[0].time() if self.members else 0.0

    # --- ensemble collection ---

    def collect(
        self, steps: int, n_or_dt: float | int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Step every member ``steps`` times and stack the sampled states.

        Advances the whole ensemble synchronously, recording each member's state
        after every step.  This is the trajectory collector the static fan chart
        (:meth:`__plot_spec__`) summarises into a median line + percentile band.

        Parameters
        ----------
        steps : int
            Number of samples to collect (one per step).
        n_or_dt : float or int, optional
            The per-step increment forwarded to each member's ``step`` (a ``dt``
            for a flow, an iteration count for a map).  ``None`` uses the member
            default.

        Returns
        -------
        (times, states)
            ``times`` shape ``(steps,)`` (the common member time after each step);
            ``states`` shape ``(steps, size, dim)`` — sample, member, component.
        """
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        times = np.empty(steps)
        states = np.empty((steps, self.size, self.dim))
        for k in range(steps):
            states[k] = self.step(n_or_dt)
            times[k] = self.time()
        return times, states

    # --- visualization seam ---

    def __plot_spec__(
        self, kind: str | None = None, *, steps: int = 200, components: int = 0, band: float = 90.0
    ) -> PlotSpec:
        """Describe the ensemble as a **static fan chart** (median + percentile band).

        Collects the ensemble's evolution of one component and summarises the
        spread across members at each time as a shaded percentile band (an
        ``AREA`` layer carrying ``"lo"`` / ``"hi"`` band edges, with ``lo <= hi``)
        under the across-member **median** line — the standard, animation-free way
        to read an ensemble's dispersion.  This is **not** an animation: it is one
        :data:`~tsdynamics.viz.spec.PlotKind.ENSEMBLE_FAN` static spec.

        The :mod:`tsdynamics.viz.spec` import is lazy, so building a spec never
        pulls in a plotting backend.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind.  ``None`` (the default) uses
            ``ENSEMBLE_FAN``.
        steps : int, optional
            Number of samples to collect across the ensemble.  Default ``200``.
        components : int, optional
            Which state component to chart.  Default ``0``.
        band : float, optional
            Central percentile mass to shade (``90`` → the 5th–95th percentile
            band).  Default ``90.0``; clamped to ``(0, 100]``.

        Returns
        -------
        PlotSpec
        """
        from tsdynamics.viz.spec import Axis, Layer, PlotKind, PlotSpec

        if not 0 <= components < self.dim:
            raise ValueError(f"components must be in [0, {self.dim}), got {components}")
        if not 0.0 < band <= 100.0:
            raise ValueError(f"band must be in (0, 100], got {band}")

        times, states = self.collect(steps)
        comp = states[:, :, components]  # (steps, size)
        lo_pct = (100.0 - band) / 2.0
        hi_pct = 100.0 - lo_pct
        lo = np.percentile(comp, lo_pct, axis=1)
        hi = np.percentile(comp, hi_pct, axis=1)
        median = np.median(comp, axis=1)
        # The band edges are percentiles of the same sample, so lo <= hi holds by
        # construction; enforce it defensively against any float ordering quirk.
        lo = np.minimum(lo, hi)

        ylabel = self.variables[components]
        spec_kind = PlotKind(kind) if kind is not None else PlotKind.ENSEMBLE_FAN
        return PlotSpec(
            kind=spec_kind,
            ndim=2,
            title=f"Ensemble fan — {type(self.template).__name__} (n={self.size})",
            x=Axis(label="iteration" if self._is_discrete else "time"),
            y=Axis(label=ylabel),
            layers=[
                Layer(
                    PlotKind.AREA,
                    {"x": times, "y": median, "lo": lo, "hi": hi},
                    label=f"{int(round(band))}% band",
                    style={"alpha": 0.3},
                ),
                Layer(
                    PlotKind.LINE,
                    {"x": times, "y": median},
                    label="median",
                ),
            ],
        )

    def plot(self, *transforms: Any, **kwargs: Any) -> PlotSpec:
        """Draw the ensemble — the fan chart by default."""
        from tsdynamics.viz import plot as _plot

        return _plot(self, *transforms, **kwargs)

    def __len__(self) -> int:
        return len(self.members)

    def __repr__(self) -> str:
        return f"Ensemble({type(self.template).__name__}, m={self.size})"


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)


#: The v5 name.  ``Ensemble`` is the noun ``system.ensemble(states)`` returns.
EnsembleSystem = Ensemble
