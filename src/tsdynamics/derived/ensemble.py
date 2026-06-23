"""Ensemble of identical systems stepped in lockstep."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tsdynamics.viz.spec import PlotSpec

__all__ = ["EnsembleSystem"]


class EnsembleSystem:
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
    >>> ens = EnsembleSystem(Lorenz(), [[1, 1, 1], [1.001, 1, 1]])
    >>> ens.step(0.01)
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
    def is_discrete(self) -> bool:
        """Match the template system's time semantics."""
        return cast(bool, self.template.is_discrete)

    def step(self, n_or_dt: float | int | None = None) -> np.ndarray:
        """Advance every member and return the stacked states, shape (m, dim)."""
        return np.array([m.step(n_or_dt) for m in self.members])

    def states(self) -> np.ndarray:
        """Return the current states, shape (m, dim)."""
        return np.array([m.state() for m in self.members])

    def set_states(self, states: Any) -> None:
        """Overwrite every member's state."""
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
        (:meth:`to_plot_spec`) summarises into a median line + percentile band.

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

    def to_plot_spec(
        self, kind: str | None = None, *, steps: int = 200, component: int = 0, band: float = 90.0
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
        component : int, optional
            Which state component to chart.  Default ``0``.
        band : float, optional
            Central percentile mass to shade (``90`` → the 5th–95th percentile
            band).  Default ``90.0``; clamped to ``(0, 100]``.

        Returns
        -------
        PlotSpec
        """
        from tsdynamics.viz.spec import Axis, Layer, PlotKind, PlotSpec

        if not 0 <= component < self.dim:
            raise ValueError(f"component must be in [0, {self.dim}), got {component}")
        if not 0.0 < band <= 100.0:
            raise ValueError(f"band must be in (0, 100], got {band}")

        times, states = self.collect(steps)
        comp = states[:, :, component]  # (steps, size)
        lo_pct = (100.0 - band) / 2.0
        hi_pct = 100.0 - lo_pct
        lo = np.percentile(comp, lo_pct, axis=1)
        hi = np.percentile(comp, hi_pct, axis=1)
        median = np.median(comp, axis=1)
        # The band edges are percentiles of the same sample, so lo <= hi holds by
        # construction; enforce it defensively against any float ordering quirk.
        lo = np.minimum(lo, hi)

        names = getattr(type(self.template), "variables", None)
        ylabel = names[component] if names is not None else f"y{component}"
        spec_kind = PlotKind(kind) if kind is not None else PlotKind.ENSEMBLE_FAN
        return PlotSpec(
            kind=spec_kind,
            ndim=2,
            title=f"Ensemble fan — {type(self.template).__name__} (n={self.size})",
            x=Axis(label="iteration" if self.is_discrete else "time"),
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

    def __len__(self) -> int:
        return len(self.members)

    def __repr__(self) -> str:
        return f"EnsembleSystem({type(self.template).__name__}, size={self.size})"


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
