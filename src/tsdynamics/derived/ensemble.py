"""Ensemble of identical systems stepped in lockstep."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, cast, overload

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tsdynamics.data import Trajectory
    from tsdynamics.viz.spec import PlotSpec

__all__ = ["Ensemble", "EnsembleSystem", "TrajectoryBatch"]


class EnsembleSamples(NamedTuple):
    """What :meth:`Ensemble.collect` records — ``times, states = band.collect(n)``.

    A named pair rather than a bare tuple: ``states`` is a three-axis block and
    "which axis is which" is not something a call site should have to remember.
    """

    times: np.ndarray
    """The common member time after each step, shape ``(steps,)``."""
    states: np.ndarray
    """Every member's state after each step, shape ``(steps, size, dim)``."""


class TrajectoryBatch(Sequence["Trajectory"]):
    """What ``Ensemble.run(...)`` returns — the trajectories, plus the batch view.

    A **read-only sequence**: it iterates, indexes, ``len()``s and unpacks like
    the list of trajectories it is, and ``np.asarray(batch)`` is the
    ``(n, T, dim)`` block.  ``.final`` is the ``(n, dim)`` array of end states —
    exactly what ``system.ensemble(ics, final_time=...)`` used to return before
    ``ensemble`` became a noun.

    .. versionchanged:: 6.0
        It was a ``list`` **subclass**, so ten mutation verbs (``append``,
        ``sort``, ``clear``, ``pop``, …) sat on the tab surface of a *result* and
        could corrupt it: ``batch.append("x")`` then made ``.final`` raise
        ``AttributeError: 'str' object has no attribute 'y'``, and ``batch.sort()``
        raised comparing two trajectories.  A batch is a measurement, not a
        workspace; ``list(batch)`` is still one keystroke away.
    """

    __slots__ = ("_members",)

    def __init__(self, members: Iterable[Trajectory] = ()) -> None:
        self._members: tuple[Trajectory, ...] = tuple(members)

    # --- the sequence protocol ---

    @overload
    def __getitem__(self, index: int) -> Trajectory: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Trajectory, ...]: ...

    def __getitem__(self, index: int | slice) -> Trajectory | tuple[Trajectory, ...]:
        """Member ``i`` (or a tuple of members for a slice)."""
        return self._members[index]

    def __len__(self) -> int:
        """Return the number of members in the batch."""
        return len(self._members)

    def __eq__(self, other: object) -> bool:
        """Compare member-wise against another batch or any sequence of members."""
        if isinstance(other, TrajectoryBatch):
            return self._members == other._members
        if isinstance(other, list | tuple):
            return list(self._members) == list(other)
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]

    # --- the batch view ---

    @property
    def final(self) -> np.ndarray:
        """The ``(n, dim)`` final states, one row per member."""
        if not self._members:
            return np.empty((0, 0))
        return np.array([np.asarray(t.y)[-1] for t in self._members], dtype=float)

    @property
    def t(self) -> np.ndarray:
        """The shared time axis, shape ``(T,)`` — every member runs one window."""
        if not self._members:
            return np.empty((0,))
        return np.asarray(self._members[0].t, dtype=float)

    @property
    def y(self) -> np.ndarray:
        """Every member's states stacked, shape ``(n, T, dim)``."""
        if not self._members:
            return np.empty((0, 0, 0))
        return np.asarray([np.asarray(m.y, dtype=float) for m in self._members], dtype=float)

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """``np.asarray(batch)`` is the ``(n, T, dim)`` state block."""
        arr = self.y if dtype is None else np.asarray(self.y, dtype=dtype)
        return arr.copy() if copy is True else arr

    def to_frame(self) -> Any:
        """Return one long-format ``DataFrame``, with a ``member`` column."""
        import pandas as pd

        frames = []
        for i, traj in enumerate(self._members):
            frame = traj.to_frame()
            frame.insert(0, "member", i)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def plot(self, *transforms: Any, **kwargs: Any) -> Any:
        """Draw every member on one figure."""
        from tsdynamics.viz import plot as _plot

        return _plot(*self, *transforms, **kwargs)

    def __repr__(self) -> str:
        """State what the batch holds, and the one attribute most callers want."""
        if not self._members:
            return "TrajectoryBatch(empty)"
        rows, dim = np.asarray(self._members[0].y).shape
        return (
            f"TrajectoryBatch  {len(self._members)} trajectories"
            f"   ·  {rows} samples  ·  {dim}-D states"
            "\n    batch.final   # the (n, dim) end states   ·   batch[i]   ·   batch.plot()"
        )


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

    def __getattr__(self, name: str) -> Any:
        """Answer a miss with the same teaching a system gives.

        ``band.integrate`` used to be a bare ``AttributeError`` while
        ``lor.integrate`` named the retired verb and the line to type — and an
        ``Ensemble`` is the object the headline ``sys.ensemble(...)`` hands you,
        so it is a likely place to guess.  It does **not** delegate to the
        template: it raises, teaching.
        """
        if name.startswith("_") or name in ("template", "members"):
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        from tsdynamics.families.base import _absent_name_error

        raise _absent_name_error(self, name)

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

    def run(
        self,
        final_time: float | None = None,
        dt: float | None = None,
        *,
        steps: int | None = None,
        transient: float | None = None,
        seed: int | None = None,
        **run_kw: Any,
    ) -> TrajectoryBatch:
        """Run every member and return a :class:`TrajectoryBatch`.

        Takes the **template family's own** ``run`` vocabulary — ``final_time`` /
        ``dt`` for a flow, ``steps`` for a map — and forwards it verbatim, with
        each member's own initial state supplied as ``ic=``::

            band = lor.ensemble(np.random.rand(100, 3))
            batch = band.run(final_time=10.0)
            batch.final          # the (100, 3) end states

        Parameters
        ----------
        final_time : float, optional
            Flow horizon, in time units.
        dt : float, optional
            Output sampling interval, in time units.
        steps : int, optional
            Map horizon, in iterations.  A flow refuses it by name.
        transient : float, optional
            Leading stretch discarded, in the template family's own horizon unit.
        seed : int, optional
            For an **SDE** batch, member ``i`` draws its noise from
            ``seed_for(seed, i)`` — depending only on the index, which is the
            engine's parallel-equals-serial contract.  On every other family it
            is forwarded as the shared initial-condition seed.
        **run_kw
            Everything else the template family's ``run`` accepts (``solver``,
            ``rtol``, ``atol``, ``backend``, ``max_step``, …).  An unknown word
            is refused **by the family that owns it**, with its reason.

        Returns
        -------
        TrajectoryBatch

        Raises
        ------
        InvalidParameterError
            If ``ic=`` is passed: an ensemble's initial states are the ones it
            was built with.
        """
        if "ic" in run_kw:
            from tsdynamics.errors import InvalidParameterError, remedy

            raise InvalidParameterError(
                "an ensemble's initial states are the ones you built it with, so "
                "run() has no ic=. One state per member, given once."
                + remedy(
                    "band = system.ensemble(states)   # the (n, dim) starts",
                    "batch = band.run(final_time=10.0)",
                )
            )
        # Forward by NAME, never positionally: a bare ``(dt,)`` tuple would be
        # read by the template as ``final_time``.  A word the template family
        # does not have (``steps`` on a flow) is refused by that family, by name.
        kwargs: dict[str, Any] = dict(run_kw)
        for word, value in (
            ("final_time", final_time),
            ("dt", dt),
            ("steps", steps),
            ("transient", transient),
        ):
            if value is not None:
                kwargs[word] = value
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

        return TrajectoryBatch(self._run_members((), kwargs, per_member_seed))

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

    def collect(self, steps: int, n_or_dt: float | int | None = None) -> EnsembleSamples:
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
        EnsembleSamples
            A named pair — ``.times`` shape ``(steps,)``, ``.states`` shape
            ``(steps, size, dim)`` (sample, member, component).  It unpacks as
            ``times, states = band.collect(n)`` like the bare tuple it replaced.
        """
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        times = np.empty(steps)
        states = np.empty((steps, self.size, self.dim))
        for k in range(steps):
            states[k] = self.step(n_or_dt)
            times[k] = self.time()
        return EnsembleSamples(times, states)

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
