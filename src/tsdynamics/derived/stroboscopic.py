"""Stroboscopic map: a forced flow sampled once per forcing period."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from tsdynamics.families import Trajectory

from ._base import DerivedSystem

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tsdynamics.viz.spec import PlotSpec

__all__ = ["StroboscopicMap"]


class StroboscopicMap(DerivedSystem):
    """
    Present a forced flow as the discrete map of once-per-period samples.

    One ``step()`` advances the underlying continuous system by exactly one
    forcing period and returns the new state.  Orbit diagrams over a
    ``StroboscopicMap`` are the standard way to study forced oscillators
    (Duffing, forced van der Pol, ...).

    Parameters
    ----------
    system : System
        A continuous-time system.
    period : float
        Sampling period (the forcing period).

    Examples
    --------
    >>> smap = StroboscopicMap(ForcedVanDerPol(), period=2 * np.pi / 0.63)
    >>> samples = smap.trajectory(300, transient=100)
    """

    def __init__(self, system: Any, period: float) -> None:
        super().__init__(system)
        if period <= 0:
            raise ValueError(f"period must be positive, got {period}")
        self.period = float(period)

    def _rebuild(self, inner: Any) -> StroboscopicMap:
        return StroboscopicMap(inner, self.period)

    @property
    def is_discrete(self) -> bool:
        """A stroboscopic map is a discrete view of the flow."""
        return True

    def step(self, n_or_dt: int | None = None) -> np.ndarray:
        """Advance ``n`` periods (default 1) and return the new state."""
        n = int(n_or_dt) if n_or_dt is not None else 1
        return cast(np.ndarray, self.system.step(n * self.period))

    def time(self) -> float:
        """Return the inner flow time."""
        return cast(float, self.system.time())

    def trajectory(self, steps: int = 100, *, transient: int = 0, **kwargs: Any) -> Trajectory:
        """Collect ``steps`` once-per-period samples (after ``transient`` periods)."""
        if kwargs:
            self.reinit(kwargs.pop("ic", None), **kwargs)
        if transient:
            self.system.step(transient * self.period)
        times = np.empty(steps)
        points = np.empty((steps, self.system.dim))
        for k in range(steps):
            points[k] = self.step()
            times[k] = self.system.time()
        meta = {
            "derived": "StroboscopicMap",
            "period": self.period,
            "system": type(self.system).__name__,
            "params": self.params.as_dict(),
        }
        return Trajectory(t=times, y=points, system=self.system, meta=meta)

    def to_plot_spec(self, kind: str | None = None, *, steps: int = 300) -> PlotSpec:
        """Describe the strobe sampling as a **scatter** of sampled states.

        A stroboscopic map is a *discrete* sampling — once per forcing period —
        so the natural picture is a cloud of sampled points (the strobed orbit /
        attractor), **not** a connected flow line.  This collects ``steps``
        samples and builds a 2-D / 3-D ``SCATTER`` spec over the first two / three
        components (a 1-D system is a sample-index time series of dots).

        The :mod:`tsdynamics.viz.spec` import is lazy, so building a spec never
        pulls in a plotting backend.

        Parameters
        ----------
        kind : str, optional
            Override the auto-dispatched semantic kind (e.g.
            ``"phase_portrait_2d"``).  ``None`` (the default) dispatches on the
            sampled dimensionality.
        steps : int, optional
            Number of once-per-period samples to collect.  Default ``300``.

        Returns
        -------
        PlotSpec
        """
        from tsdynamics.viz.spec import Axis, Layer, PlotKind, PlotSpec

        section = self.trajectory(steps)
        names = self.variables or tuple(f"y{i}" for i in range(self.system.dim))
        title = f"Stroboscopic map — {type(self.system).__name__}"

        if self.system.dim == 1:
            spec_kind = PlotKind(kind) if kind is not None else PlotKind.TIME_SERIES
            return PlotSpec(
                kind=spec_kind,
                ndim=1,
                title=title,
                x=Axis(label="sample"),
                y=Axis(label=names[0]),
                layers=[
                    Layer(
                        PlotKind.SCATTER,
                        {"x": np.arange(section.y.shape[0], dtype=float), "y": section.y[:, 0]},
                    )
                ],
            )

        if kind is None:
            want_3d = self.system.dim >= 3
        else:
            want_3d = PlotKind(kind) == PlotKind.PHASE_PORTRAIT_3D
        spec_kind = (
            PlotKind(kind)
            if kind is not None
            else (PlotKind.PHASE_PORTRAIT_3D if want_3d else PlotKind.PHASE_PORTRAIT_2D)
        )
        cols: dict[str, np.ndarray] = {"x": section.y[:, 0], "y": section.y[:, 1]}
        z = None
        if want_3d:
            cols["z"] = section.y[:, 2]
            z = Axis(label=names[2])
        return PlotSpec(
            kind=spec_kind,
            ndim=3 if want_3d else 2,
            aspect="equal",
            title=title,
            x=Axis(label=names[0]),
            y=Axis(label=names[1]),
            z=z,
            layers=[Layer(PlotKind.SCATTER, cols)],
        )


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
