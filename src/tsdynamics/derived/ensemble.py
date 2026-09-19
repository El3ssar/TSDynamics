"""Ensemble of identical systems stepped in lockstep."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, cast, overload

import numpy as np

from tsdynamics.errors import remedy
from tsdynamics.families._hidden import hide

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tsdynamics.data import Trajectory
    from tsdynamics.viz.spec import PlotSpec

#: ``EnsembleSystem`` is the *same class object* as ``Ensemble`` (the v5 name,
#: kept bound at the bottom of this module).  It is off the listing because two
#: spellings of one class is corollary C3's silent-wrong-answer shape: a reader
#: comparing the two would look for a difference that is not there.
__all__ = ["Ensemble", "TrajectoryBatch"]


@hide("count", "index")
class EnsembleSamples(NamedTuple):
    """What :meth:`Ensemble.collect` records — ``times, states = band.collect(n)``.

    A named pair rather than a bare tuple: ``states`` is a three-axis block and
    "which axis is which" is not something a call site should have to remember.

    ``count`` / ``index`` come free with ``NamedTuple`` and are withheld from
    ``dir()`` (``CONTRACT.md`` §11, T2): they search the *two-element outer
    tuple*, so ``samples.count(x)`` can only ever answer 0, 1 or 2 about which of
    ``times`` / ``states`` equals ``x`` — never a question anyone asks.  Still
    bound, as tuple members must be.
    """

    times: np.ndarray
    """The common member time after each step, shape ``(steps,)``."""
    states: np.ndarray
    """Every member's state after each step, shape ``(steps, size, dim)``."""


@hide("count", "index")
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

    ``count`` / ``index`` are the two mixins ``Sequence`` supplies.  They are
    withheld from ``dir()`` (``CONTRACT.md`` §11, T2) because both search by
    ``==`` and a :class:`~tsdynamics.data.Trajectory` is not something you have a
    second copy of to search for — ``batch.index(traj)`` is answerable only when
    you already hold the element. Still bound, and correct if called.
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
    def diverged(self) -> np.ndarray:
        """Which members left the building, as an ``(n,)`` boolean mask.

        A member has diverged when its **end state** is non-finite or past the
        library's one escape scale (:data:`~tsdynamics._utils.escape.ESCAPE_SCALE`
        — the same number :attr:`~tsdynamics.data.Trajectory.unbounded` and the
        Lyapunov verdict use, so a batch and a single run cannot disagree about
        what "diverged" means).

        Measured before v6 round 8: a twelve-member batch of ``dx/dt = x² - a``
        came back with **five NaN rows**, printed ``TrajectoryBatch  12
        trajectories``, warned nothing, and exposed no ``diverged`` / ``status``
        / ``n_diverged`` of any kind — so ``np.mean(batch.final, axis=0)`` was
        ``nan`` and ``np.nanmean`` was a survivor-biased mean over an unflagged
        58% subsample.  Every sibling result in the library (``basins``,
        ``attractors``, ``basin_fractions``) reports its diverged share; an
        ensemble is precisely where a user aggregates and cannot inspect each
        member by hand.

        Returns
        -------
        ndarray of bool, shape (n,)

        Examples
        --------
        >>> import numpy as np, tsdynamics as ts
        >>> band = ts.systems.Lorenz().ensemble([[1, 1, 1], [1.01, 1, 1]])
        >>> band.run(final_time=1.0, dt=0.1).diverged
        array([False, False])
        """
        from tsdynamics._utils.escape import ESCAPE_SCALE

        end = self.final
        if not end.size:
            return np.zeros(len(self._members), dtype=bool)
        magnitude = np.abs(end)
        return np.asarray(
            ~np.all(np.isfinite(magnitude), axis=1) | np.any(magnitude >= ESCAPE_SCALE, axis=1),
            dtype=bool,
        )

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
        """State what the batch holds, what went wrong in it, and what to read next.

        The divergence clause appears **only when there is one** — a count of
        zero is decoration, and a reader learns to skip decoration.
        """
        if not self._members:
            return "TrajectoryBatch(empty)"
        rows, dim = np.asarray(self._members[0].y).shape
        n = len(self._members)
        lost = int(self.diverged.sum())
        line = f"TrajectoryBatch  {n} trajectories   ·  {rows} samples  ·  {dim}-D states"
        if lost:
            line += (
                f"\n    ⚠ {lost} of {n} diverged — batch.diverged is the mask; "
                f"an average over batch.final is over the {n - lost} that survived"
            )
        return (
            line + "\n    batch.final   # the (n, dim) end states   ·   batch[i]   ·   batch.plot()"
        )


@hide("set_states", "states")
class Ensemble:
    """
    Many copies of one system, advanced synchronously from different states.

    Used for two-trajectory Lyapunov estimates, basin sampling, and ensemble
    statistics.  Members are independent copies — parameters are shared at
    construction, states are per-member.

    Two members are withheld from ``dir()`` (``CONTRACT.md`` §11, T2) and stay
    callable.  Both are second spellings of the ``System`` protocol this class
    already answers in the protocol's own words:

    ``states()``
        Identical to :meth:`state` — the very body of ``state()`` is ``return
        self.states()``.  ``state()`` is the word every other view in the library
        uses, so it is the one that stays listed (corollary C3).
    ``set_states(states)``
        The plural of ``set_state``, which is a *capability* rather than a
        protocol member since v6.  Reach it with ``hasattr`` like the others, or
        build the ensemble you want: ``system.ensemble(new_states)`` is one call
        and cannot desynchronise the member clocks.

    Parameters
    ----------
    system : System
        The system to copy per member (the original is untouched).  Readable
        afterwards as :attr:`system`.
    states : array-like, shape (m, dim)
        One initial state per member.

    Attributes
    ----------
    system : System
        The system every member is a copy of.

        .. versionchanged:: 6.0
            Was ``template``.  Every other wrapper in :mod:`tsdynamics.derived`
            calls the thing it wraps ``system``, so the one exception meant that
            code walking a mixed list of views — the thing wrappers exist to make
            possible — had to special-case this class by name (corollary C3: one
            concept, one spelling).  ``band.template`` names the replacement.
    members : list of System
        The per-member copies, in construction order.

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
        self.system = system
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
        so it is a likely place to guess.  It does **not** delegate to the inner
        system: it raises, teaching.
        """
        if name.startswith("_") or name in ("system", "members"):
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        if name == "template":
            # The v5 spelling of ``system``.  A rename is the one kind of miss a
            # near-miss ranker cannot help with — ``template`` resembles nothing
            # on the object — so it is answered by name.
            raise AttributeError(
                "'Ensemble' object has no attribute 'template': every derived view "
                "names the thing it wraps 'system' — one concept, one spelling."
                + remedy("band.system", "band.system.params")
            )
        from tsdynamics.families.base import _absent_name_error

        raise _absent_name_error(self, name)

    @property
    def size(self) -> int:
        """Number of ensemble members."""
        return len(self.members)

    @property
    def dim(self) -> int:
        """State-space dimension of each member."""
        return cast(int, self.system.dim)

    @property
    def _is_discrete(self) -> bool:
        """Match the template system's time semantics."""
        return cast(bool, self.system._is_discrete)

    @property
    def family(self) -> str:
        """The template system's family word."""
        return cast(str, self.system.family)

    @property
    def params(self) -> Any:
        """The template system's parameters (shared by every member)."""
        return self.system.params

    @property
    def variables(self) -> tuple[str, ...]:
        """The template system's component names."""
        return cast("tuple[str, ...]", self.system.variables)

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
                runs.append(self.system.run(*args, ic=member.state(), **kwargs, **extra))
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
            else Trajectory(template.t, blank.copy(), self.system, meta=dict(template.meta))
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

        **This is a different numerical path from** :meth:`run`, which is why it
        stays on the tab surface rather than being folded into it.  ``run``
        launches each member as its own independent integration, on the inner
        family's own adaptive solver and output grid; ``collect`` drives the
        *live steppers* in lockstep, so at every returned sample the members
        share one clock.  That is the reading an ensemble measurement needs — a
        cross-member percentile at "time ``t``" is only meaningful if every
        member really is at ``t`` — and it is also what makes the returned block
        rectangular in ``(sample, member, component)``.  Use ``run`` when you want
        per-member trajectories; use ``collect`` when you want the ensemble's
        spread through time.

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
            title=f"Ensemble fan — {type(self.system).__name__} (n={self.size})",
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
        return f"Ensemble({type(self.system).__name__}, m={self.size})"


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)


#: The v5 name.  ``Ensemble`` is the noun ``system.ensemble(states)`` returns.
EnsembleSystem = Ensemble
