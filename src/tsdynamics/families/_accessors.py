"""Object-side topical accessor layer for systems (xarray-style namespaces).

The library's analysis toolkit lives as **free functions** (the canonical,
composable surface — ``ts.correlation_dimension(data)``,
``ts.gali(system, k=2)``, …).  Holding a system in hand, those functions are
hard to *discover*: pressing ``<TAB>`` on a ``Lorenz()`` reveals only the
system's own verbs, never the ~60 analyses that take it as a first argument.

This module adds a thin, **purely additive** object surface that delegates to
those free functions with the system already bound — grouped into a handful of
cached *topical accessors* (the `xarray accessor pattern
<https://docs.xarray.dev/en/latest/internals/extending-xarray.html>`_) so the
toolkit becomes navigable from the object::

    sys.lyap.spectrum()          sys.lyap.maximal()
    sys.dims.correlation()       sys.recurrence.rqa()
    sys.chaos.gali(k=2)          sys.surrogate.test()
    sys.entropy.permutation()
    sys.fixed_points()           sys.poincare(section="y", at=0.0)
    sys.tangent(k=3)             sys.project("x", "z")     sys.ensemble(states)

Every accessor method forwards to the same free function the user would call by
hand, with the system passed *positionally* (the free-function first-argument
name is not yet unified — that is a later stream — so positional delegation is
the robust choice).  The accessors add **zero behaviour**: a result obtained
through an accessor is identical to the free-function result on the same input.

Accessors that operate on a *measured series* (dimensions, recurrence, entropy,
surrogates) accept the data as an optional first positional argument.  When it
is omitted they first run the system (``system.run(**run_kwargs)``) to produce a
trajectory and then delegate — a convenience that *generates a trajectory
implicitly*; pass ``data=`` (or run the system yourself) for full control over
the integration window.

The accessor namespaces are wired onto :class:`~tsdynamics.families.base.SystemBase`
as cached properties in :mod:`tsdynamics.families.base`, so ``sys.lyap is
sys.lyap`` (one instance per system, holding the system reference).  All
analysis / derived imports are **function-local** to keep
:mod:`tsdynamics.families.base` free of import cycles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tsdynamics.families.base import SystemBase

__all__ = [
    "ChaosAccessor",
    "DimensionsAccessor",
    "EntropyAccessor",
    "LyapunovAccessor",
    "RecurrenceAccessor",
    "SurrogateAccessor",
    "infer_forcing_period",
]


# ---------------------------------------------------------------------------
# accessor base
# ---------------------------------------------------------------------------


class _Accessor:
    """Base for a cached topical accessor that holds its owning system.

    Subclasses expose estimator methods that delegate to the canonical free
    functions, passing ``self._system`` positionally.  The accessor caches on
    the instance (see :class:`~tsdynamics.families.base.SystemBase`), so
    ``sys.lyap is sys.lyap``.
    """

    __slots__ = ("_system",)

    def __init__(self, system: SystemBase) -> None:
        self._system = system

    def __repr__(self) -> str:
        return f"{type(self).__name__}({type(self._system).__name__})"

    # -- shared helper for the data-consuming accessors --

    def _resolve_data(self, data: Any, run_kwargs: dict[str, Any]) -> Any:
        """Return ``data`` if given, else a fresh trajectory from the system.

        The data-consuming analyses (dimensions, recurrence, entropy,
        surrogates) want a measured series.  When the caller passes ``data`` it
        is delegated verbatim; otherwise the system is run once
        (``system.run(**run_kwargs)``) and the resulting trajectory is used.
        Splitting the run kwargs out keeps the delegation byte-identical to the
        free function for a given series.
        """
        if data is not None:
            return data
        return self._system.run(**run_kwargs)


# ---------------------------------------------------------------------------
# lyapunov
# ---------------------------------------------------------------------------


class LyapunovAccessor(_Accessor):
    """Lyapunov-exponent estimators bound to the system (``sys.lyap``).

    Groups the spectrum / maximal-exponent / data-driven estimators so they are
    discoverable from the object.  Each method delegates to the matching free
    function in :mod:`tsdynamics.analysis`.
    """

    def spectrum(self, **kwargs: Any) -> Any:
        """Delegate to :func:`tsdynamics.analysis.lyapunov_spectrum`."""
        from tsdynamics.analysis import lyapunov_spectrum

        return lyapunov_spectrum(self._system, **kwargs)

    def maximal(self, **kwargs: Any) -> Any:
        """Delegate to :func:`tsdynamics.analysis.max_lyapunov`."""
        from tsdynamics.analysis import max_lyapunov

        return max_lyapunov(self._system, **kwargs)

    def from_data(
        self, data: Any = None, *, run_kwargs: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        """Delegate to :func:`tsdynamics.analysis.lyapunov_from_data`.

        Estimates the maximal exponent from a measured series.  When ``data`` is
        omitted the system is run first (see :meth:`_Accessor._resolve_data`);
        the series passed to the free function is then a 1-D component, so a
        multi-component system should usually be given an explicit 1-D ``data``.
        """
        from tsdynamics.analysis import lyapunov_from_data

        series = self._resolve_data(data, run_kwargs or {})
        return lyapunov_from_data(series, **kwargs)


# ---------------------------------------------------------------------------
# chaos indicators
# ---------------------------------------------------------------------------


class ChaosAccessor(_Accessor):
    """Chaos indicators bound to the system (``sys.chaos``).

    ``gali`` and ``expansion_entropy`` take the *system* (they integrate its
    tangent dynamics internally); ``zero_one`` takes a measured scalar series,
    so it runs the system first when no ``data`` is supplied.
    """

    def gali(self, k: int = 2, **kwargs: Any) -> Any:
        """Delegate to :func:`tsdynamics.analysis.gali`."""
        from tsdynamics.analysis import gali

        return gali(self._system, k, **kwargs)

    def expansion_entropy(self, region: Any = None, **kwargs: Any) -> Any:
        """Delegate to :func:`tsdynamics.analysis.expansion_entropy`."""
        from tsdynamics.analysis import expansion_entropy

        return expansion_entropy(self._system, region, **kwargs)

    def zero_one(
        self, data: Any = None, *, run_kwargs: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        """Delegate to :func:`tsdynamics.analysis.zero_one_test`.

        The 0--1 test consumes a scalar observable, not a live system; when
        ``data`` is omitted the system is run first to provide one (pass
        ``component=`` to select a column).
        """
        from tsdynamics.analysis import zero_one_test

        observable = self._resolve_data(data, run_kwargs or {})
        return zero_one_test(observable, **kwargs)


# ---------------------------------------------------------------------------
# fractal dimensions
# ---------------------------------------------------------------------------


class DimensionsAccessor(_Accessor):
    """Fractal-dimension estimators bound to the system (``sys.dims``).

    These consume a point set.  Each method accepts the data as an optional
    first positional argument; omitting it runs the system once to produce a
    trajectory (an implicit integration — pass ``data=`` for full control).
    """

    def correlation(
        self, data: Any = None, *, run_kwargs: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        """Delegate to :func:`tsdynamics.analysis.correlation_dimension`."""
        from tsdynamics.analysis import correlation_dimension

        return correlation_dimension(self._resolve_data(data, run_kwargs or {}), **kwargs)

    def correlation_sum(
        self, data: Any = None, *, run_kwargs: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        """Delegate to :func:`tsdynamics.analysis.correlation_sum`."""
        from tsdynamics.analysis import correlation_sum

        return correlation_sum(self._resolve_data(data, run_kwargs or {}), **kwargs)

    def generalized(
        self,
        data: Any = None,
        q: float = 2.0,
        *,
        run_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Delegate to :func:`tsdynamics.analysis.generalized_dimension`."""
        from tsdynamics.analysis import generalized_dimension

        return generalized_dimension(self._resolve_data(data, run_kwargs or {}), q, **kwargs)

    def box_counting(
        self, data: Any = None, *, run_kwargs: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        """Delegate to :func:`tsdynamics.analysis.box_counting_dimension`."""
        from tsdynamics.analysis import box_counting_dimension

        return box_counting_dimension(self._resolve_data(data, run_kwargs or {}), **kwargs)

    def information(
        self, data: Any = None, *, run_kwargs: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        """Delegate to :func:`tsdynamics.analysis.information_dimension`."""
        from tsdynamics.analysis import information_dimension

        return information_dimension(self._resolve_data(data, run_kwargs or {}), **kwargs)

    def spectrum(
        self, data: Any = None, *, run_kwargs: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        """Delegate to :func:`tsdynamics.analysis.dimension_spectrum`."""
        from tsdynamics.analysis import dimension_spectrum

        return dimension_spectrum(self._resolve_data(data, run_kwargs or {}), **kwargs)

    def fixed_mass(
        self, data: Any = None, *, run_kwargs: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        """Delegate to :func:`tsdynamics.analysis.fixed_mass_dimension`."""
        from tsdynamics.analysis import fixed_mass_dimension

        return fixed_mass_dimension(self._resolve_data(data, run_kwargs or {}), **kwargs)


# ---------------------------------------------------------------------------
# recurrence quantification
# ---------------------------------------------------------------------------


class RecurrenceAccessor(_Accessor):
    """Recurrence-quantification estimators bound to the system (``sys.recurrence``).

    Consumes a point set / series; omitting ``data`` runs the system first.
    """

    def matrix(
        self, data: Any = None, *, run_kwargs: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        """Delegate to :func:`tsdynamics.analysis.recurrence_matrix`."""
        from tsdynamics.analysis import recurrence_matrix

        return recurrence_matrix(self._resolve_data(data, run_kwargs or {}), **kwargs)

    def rqa(
        self, data: Any = None, *, run_kwargs: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        """Delegate to :func:`tsdynamics.analysis.rqa`."""
        from tsdynamics.analysis import rqa

        return rqa(self._resolve_data(data, run_kwargs or {}), **kwargs)

    def windowed(
        self, data: Any = None, *, run_kwargs: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        """Delegate to :func:`tsdynamics.analysis.windowed_rqa`."""
        from tsdynamics.analysis import windowed_rqa

        return windowed_rqa(self._resolve_data(data, run_kwargs or {}), **kwargs)


# ---------------------------------------------------------------------------
# entropy / complexity
# ---------------------------------------------------------------------------


class EntropyAccessor(_Accessor):
    """Entropy / complexity estimators bound to the system (``sys.entropy``).

    Consumes a scalar series; omitting ``data`` runs the system first (pass
    ``component=`` on the estimator to select a column of a multi-component
    trajectory).
    """

    def permutation(
        self, data: Any = None, *args: Any, run_kwargs: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        """Delegate to :func:`tsdynamics.analysis.permutation_entropy`."""
        from tsdynamics.analysis import permutation_entropy

        return permutation_entropy(self._resolve_data(data, run_kwargs or {}), *args, **kwargs)

    def weighted_permutation(
        self, data: Any = None, *args: Any, run_kwargs: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        """Delegate to :func:`tsdynamics.analysis.weighted_permutation_entropy`."""
        from tsdynamics.analysis import weighted_permutation_entropy

        return weighted_permutation_entropy(
            self._resolve_data(data, run_kwargs or {}), *args, **kwargs
        )

    def dispersion(
        self, data: Any = None, *args: Any, run_kwargs: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        """Delegate to :func:`tsdynamics.analysis.dispersion_entropy`."""
        from tsdynamics.analysis import dispersion_entropy

        return dispersion_entropy(self._resolve_data(data, run_kwargs or {}), *args, **kwargs)

    def sample(
        self, data: Any = None, *args: Any, run_kwargs: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        """Delegate to :func:`tsdynamics.analysis.sample_entropy`."""
        from tsdynamics.analysis import sample_entropy

        return sample_entropy(self._resolve_data(data, run_kwargs or {}), *args, **kwargs)

    def approximate(
        self, data: Any = None, *args: Any, run_kwargs: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        """Delegate to :func:`tsdynamics.analysis.approximate_entropy`."""
        from tsdynamics.analysis import approximate_entropy

        return approximate_entropy(self._resolve_data(data, run_kwargs or {}), *args, **kwargs)

    def multiscale(
        self, data: Any = None, *args: Any, run_kwargs: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        """Delegate to :func:`tsdynamics.analysis.multiscale_entropy`."""
        from tsdynamics.analysis import multiscale_entropy

        return multiscale_entropy(self._resolve_data(data, run_kwargs or {}), *args, **kwargs)

    def lz76(
        self, data: Any = None, *args: Any, run_kwargs: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        """Delegate to :func:`tsdynamics.analysis.lz76_complexity`."""
        from tsdynamics.analysis import lz76_complexity

        return lz76_complexity(self._resolve_data(data, run_kwargs or {}), *args, **kwargs)


# ---------------------------------------------------------------------------
# surrogates / nonlinearity tests
# ---------------------------------------------------------------------------


class SurrogateAccessor(_Accessor):
    """Surrogate generators + nonlinearity tests bound to the system (``sys.surrogate``).

    Consumes a measured series; omitting ``data`` runs the system first.
    """

    def test(
        self, data: Any = None, *args: Any, run_kwargs: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        """Delegate to :func:`tsdynamics.analysis.surrogate_test`."""
        from tsdynamics.analysis import surrogate_test

        return surrogate_test(self._resolve_data(data, run_kwargs or {}), *args, **kwargs)

    def generate(
        self, data: Any = None, *args: Any, run_kwargs: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        """Delegate to :func:`tsdynamics.analysis.surrogates`."""
        from tsdynamics.analysis import surrogates

        return surrogates(self._resolve_data(data, run_kwargs or {}), *args, **kwargs)

    def time_reversal_asymmetry(
        self, data: Any = None, *args: Any, run_kwargs: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        """Delegate to :func:`tsdynamics.analysis.time_reversal_asymmetry`."""
        from tsdynamics.analysis import time_reversal_asymmetry

        return time_reversal_asymmetry(self._resolve_data(data, run_kwargs or {}), *args, **kwargs)

    def nonlinear_prediction_error(
        self, data: Any = None, *args: Any, run_kwargs: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        """Delegate to :func:`tsdynamics.analysis.nonlinear_prediction_error`."""
        from tsdynamics.analysis import nonlinear_prediction_error

        return nonlinear_prediction_error(
            self._resolve_data(data, run_kwargs or {}), *args, **kwargs
        )


# ---------------------------------------------------------------------------
# forcing-period inference (the stroboscope builder's period hook)
# ---------------------------------------------------------------------------

#: Conventional parameter names a forced system uses for its drive *frequency*
#: (angular, radians per unit time).  The forcing period is then ``2*pi / value``.
#: ``omega`` is the catalogue convention (e.g. :class:`~tsdynamics.systems.Duffing`,
#: whose autonomising phase variable obeys ``zdot = omega``).
_DRIVE_FREQUENCY_PARAMS = ("drive_frequency", "omega")

#: Conventional parameter names a forced system uses for its drive *period*
#: directly (so no ``2*pi`` conversion is applied).
_FORCING_PERIOD_PARAMS = ("forcing_period", "drive_period")


def infer_forcing_period(system: SystemBase) -> float:
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
    system : SystemBase
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
