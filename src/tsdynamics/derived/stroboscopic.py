"""Stroboscopic map: a forced flow sampled once per forcing period."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from tsdynamics.families import Trajectory

from ._base import DerivedSystem, _reject_wrapper_keywords

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tsdynamics.viz.spec import PlotSpec

__all__ = ["StroboscopicMap", "infer_forcing_period"]


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
    >>> samples = smap.run(300, transient=100)
    """

    def __init__(self, system: Any, period: float) -> None:
        super().__init__(system)
        if period <= 0:
            raise ValueError(f"period must be positive, got {period}")
        self.period = float(period)

    def _rebuild(self, inner: Any) -> StroboscopicMap:
        return StroboscopicMap(inner, self.period)

    def __repr__(self) -> str:
        """Name the inner system AND the period."""
        return f"StroboscopicMap({type(self.system).__name__}, period={self.period:.6g})"

    @property
    def _is_discrete(self) -> bool:
        """A stroboscopic map is a discrete view of the flow."""
        return True

    @property
    def family(self) -> str:
        """A stroboscopic map is a discrete view of the flow."""
        return "map"

    def step(self, n_or_dt: int | None = None) -> np.ndarray:
        """Advance ``n`` forcing periods (default 1) and return the new state.

        ``n_or_dt`` is a **period count**, not a time increment — the wrapper
        presents a discrete map, so the argument is coerced to an integer with
        ``int()`` (a float is *truncated toward zero*, matching the discrete-view
        convention; pass a whole number to be explicit).

        Parameters
        ----------
        n_or_dt : int, optional
            Number of forcing periods to advance.  ``None`` advances one period.

        Returns
        -------
        numpy.ndarray
            The full-dimensional state after ``n`` periods.
        """
        n = int(n_or_dt) if n_or_dt is not None else 1
        return cast(np.ndarray, self.system.step(n * self.period))

    def time(self) -> float:
        """Return the inner flow time."""
        return cast(float, self.system.time())

    def run(
        self, steps: int = 100, *, ic: Any | None = None, transient: int = 0, **unknown: Any
    ) -> Trajectory:
        """Collect ``steps`` once-per-period samples — **from a fresh start**.

        Sampling restarts the inner flow and advances by exactly one forcing
        :attr:`period` per sample; ``transient`` leading periods are stepped
        through and discarded first.  ``run()`` twice returns the same data;
        ``step()`` is the verb that continues.

        Parameters
        ----------
        steps : int
            Number of once-per-period samples to collect — the horizon word,
            counted in **forcing periods**.
        ic : array-like, optional
            Start state for the inner flow — ``system.dim`` numbers.
        transient : int
            Leading stretch to discard, in this view's own horizon unit:
            **forcing periods**.

        Returns
        -------
        Trajectory
            The strobed samples — continuous sample times in ``t`` (one period
            apart), full-dimensional states in ``y``.
        """
        _reject_wrapper_keywords(self, unknown, accepted=("steps", "ic", "transient"))
        self.reinit(ic)
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

    def __plot_spec__(
        self, kind: str | None = None, *, steps: int = 300, **_unused: Any
    ) -> PlotSpec:
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

        Notes
        -----
        Sampling starts from the inner flow's **live cursor** (this calls
        :meth:`run`, which steps the wrapped system as a side effect with
        no transient discarded), so the picture reflects wherever the system
        currently sits — :meth:`reinit` first for a deterministic start state, or
        burn the transient in beforehand to image the attractor rather than the
        approach to it.
        """
        from tsdynamics.viz.spec import Axis, Layer, PlotKind, PlotSpec

        section = self.run(steps)
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


#: Names read as an *angular* drive frequency (period = 2*pi / value).
_DRIVE_FREQUENCY_PARAMS = ("drive_frequency", "omega")

#: Names read as a forcing period, verbatim.
_FORCING_PERIOD_PARAMS = ("forcing_period", "drive_period")


def infer_forcing_period(system: Any) -> float:
    """Infer a forced flow's forcing period from the system itself.

    The stroboscopic map samples a forced flow once per forcing period; that
    period is a property of the *system's* drive, so a user who has already set
    the drive frequency should not have to re-derive ``2*pi/omega`` by hand
    (Parlitz & Lauterborn 1985 study the forced oscillator precisely through
    this once-per-period section).  This helper resolves the period from the
    system, in priority order:

    1. an explicit **period** hook — a ``forcing_period`` (or ``drive_period``)
       ClassVar / property / parameter — used verbatim;
    2. an explicit **frequency** hook — a ``drive_frequency`` ClassVar / property
       / parameter — taken as the *angular* drive frequency, so the period is
       ``2*pi / drive_frequency``;
    3. the catalogue convention — an ``omega`` parameter — likewise angular, so
       the period is ``2*pi / omega``.

    A system with no such hook cannot have its period inferred; the caller then
    raises directing the user to pass ``period=`` explicitly.

    Parameters
    ----------
    system : System
        The forced continuous system to question.

    Returns
    -------
    float
        The inferred forcing period (strictly positive).

    Raises
    ------
    KeyError
        If the system exposes no recognised drive hook.  (Signalled this way so
        the caller can attach a user-facing message naming the failed
        ``stroboscope`` call.)
    InvalidParameterError
        If a hook is present but its value is non-positive / non-finite.

    References
    ----------
    Parlitz, U. & Lauterborn, W. (1985). "Superstructure in the bifurcation set
    of the Duffing equation." *Physics Letters A*, 107(8), 351-355.
    """
    import math

    from tsdynamics.errors import invalid_value

    def _hook(names: tuple[str, ...]) -> tuple[str, float] | None:
        for name in names:
            value = getattr(system, name, None)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            return name, numeric
        return None

    direct = _hook(_FORCING_PERIOD_PARAMS)
    if direct is not None:
        name, period = direct
        if not math.isfinite(period) or period <= 0:
            raise invalid_value(name, value=period, rule="must be a positive forcing period")
        return period

    freq = _hook(_DRIVE_FREQUENCY_PARAMS)
    if freq is not None:
        name, omega = freq
        if not math.isfinite(omega) or omega <= 0:
            raise invalid_value(name, value=omega, rule="must be a positive drive frequency")
        return 2.0 * math.pi / omega

    raise KeyError(
        "no forcing period or drive frequency could be inferred (looked for "
        f"{[*_FORCING_PERIOD_PARAMS, *_DRIVE_FREQUENCY_PARAMS]})"
    )
