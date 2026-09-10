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
    sys.chaos.gali(k=2)          sys.chaos.zero_one()
    sys.fixed_points()           sys.poincare(section="y", at=0.0)
    sys.tangent(k=3)             sys.project("x", "z")     sys.copies(states)
    sys.poincare(("y", 0.0, "up"))                    sys.ensemble(ics)

Every accessor method forwards to the same free function the user would call by
hand, passing it *positionally* whichever subject its own signature asks for —
the **system** for a ``system``-first function (it drives the integration
itself), a measured point set for a ``data``-first one.  That routing is read
off the free function's signature by :func:`subject_kind`, not hand-maintained,
so an accessor cannot drift away from the function it delegates to (see "the
delegation contract" below).  The accessors add **zero behaviour**: a result
obtained through an accessor is identical to the free-function result on the
same input.

Accessors that operate on a *measured series* (dimensions, recurrence,
``lyap.from_data``) accept the data as an optional first positional argument.
When it is omitted they first run the system (``system.run(**run_kwargs)``) to
produce a trajectory and then delegate — a convenience that *generates a
trajectory implicitly*; pass ``data=`` (or run the system yourself) for full
control over the integration window.

Those same three namespaces are **also bound to a**
:class:`~tsdynamics.data.Trajectory`, because a trajectory is what a user
holding measured data actually has, and it is the input those analyses take::

    traj.dims.correlation()      traj.recurrence.rqa()   traj.lyap.from_data()

so the discoverable path is not available only on the object half of these
analyses refuse.  A trajectory-bound accessor has nothing to integrate: a
``system``-first method (``lyap.spectrum`` / ``lyap.maximal``) raises a typed
error naming the system-bound spelling rather than guessing a right-hand side,
and ``run_kwargs=`` is refused rather than silently ignored.

The accessor namespaces are wired onto :class:`~tsdynamics.families.base.SystemBase`
as cached properties in :mod:`tsdynamics.families.base` (and onto
:class:`~tsdynamics.data.Trajectory` in :mod:`tsdynamics.data.trajectory`), so
``sys.lyap is sys.lyap`` and ``traj.dims is traj.dims`` — one instance per
subject, holding the subject reference.  All analysis / derived imports are
**function-local** to keep :mod:`tsdynamics.families.base` free of import cycles.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

    from tsdynamics.families.base import SystemBase

__all__ = [
    "ACCESSOR_DELEGATIONS",
    "ChaosAccessor",
    "DimensionsAccessor",
    "LyapunovAccessor",
    "RecurrenceAccessor",
    "infer_forcing_period",
    "is_drivable",
    "subject_kind",
]


# ---------------------------------------------------------------------------
# the delegation contract
# ---------------------------------------------------------------------------
#
# The free functions in ``tsdynamics.analysis`` come in exactly two shapes, told
# apart by the name of their FIRST parameter:
#
#   * ``system``-first  — the function drives the system itself (it integrates /
#     iterates with its own, method-appropriate defaults).  The accessor must
#     hand it the **system**.
#   * ``data``-first    — the function consumes a measured point set.  The
#     accessor may run the system first to produce one.
#
# Getting that wrong is not a cosmetic slip: ``sys.chaos.zero_one()`` used to
# pre-run the system with the *family's* ``run()`` defaults (dt = 0.01) and pass
# the resulting oversampled trajectory to the system-first ``zero_one_test`` as
# data.  Successive samples were then heavily correlated, and Lorenz —
# unambiguously chaotic — measured K = -0.026 through the accessor against
# K = 0.999 through the free function: a qualitatively wrong answer, silently.
#
# :func:`subject_kind` reads the shape off the free function's signature and
# :meth:`_Accessor._delegate` routes on it, so the accessor cannot disagree with
# the function it delegates to.  ``ACCESSOR_DELEGATIONS`` names the pairing for
# the test suite (``tests/test_accessors.py``), which checks every accessor
# method against its free function programmatically.


def subject_kind(free: Callable[..., Any]) -> str:
    """Return ``"system"`` or ``"data"`` — what ``free`` wants as first argument.

    Read off the free function's signature (its first parameter's name), so the
    accessor layer follows the analysis layer rather than duplicating a hand-kept
    table that can drift.
    """
    try:
        params = list(inspect.signature(free).parameters)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return "data"
    return "system" if params and params[0] == "system" else "data"


#: ``accessor class name → {accessor method name: free-function name}``.  Every
#: public accessor method must appear here; the accessor meta-test iterates it.
ACCESSOR_DELEGATIONS: dict[str, dict[str, str]] = {
    "LyapunovAccessor": {
        "spectrum": "lyapunov_spectrum",
        "maximal": "max_lyapunov",
        "from_data": "lyapunov_from_data",
    },
    "ChaosAccessor": {
        "gali": "gali",
        "expansion_entropy": "expansion_entropy",
        "zero_one": "zero_one_test",
    },
    "DimensionsAccessor": {
        "correlation": "correlation_dimension",
        "correlation_sum": "correlation_sum",
        "generalized": "generalized_dimension",
        "box_counting": "box_counting_dimension",
        "information": "information_dimension",
        "spectrum": "dimension_spectrum",
        "fixed_mass": "fixed_mass_dimension",
    },
    "RecurrenceAccessor": {
        "matrix": "recurrence_matrix",
        "rqa": "rqa",
        "windowed": "windowed_rqa",
    },
}


# ---------------------------------------------------------------------------
# accessor base
# ---------------------------------------------------------------------------


def is_drivable(subject: Any) -> bool:
    """Whether ``subject`` is a *system* the accessor may integrate itself.

    A system can be driven (``run``) to produce data on demand; a measured
    :class:`~tsdynamics.data.Trajectory` cannot — it *is* the data.  The same
    accessor classes are bound to both (``lor.dims`` and ``traj.dims``), so the
    test is made once, structurally, rather than by importing either type.
    """
    return callable(getattr(subject, "run", None)) and callable(getattr(subject, "reinit", None))


class _Accessor:
    """Base for a cached topical accessor that holds its owning subject.

    Subclasses expose estimator methods that delegate to the canonical free
    functions, passing ``self._system`` positionally.  The accessor caches on
    the instance (see :class:`~tsdynamics.families.base.SystemBase`), so
    ``sys.lyap is sys.lyap``.

    The subject is a **system** in the usual case, and the accessor may run it
    to produce data for a ``data``-first analysis.  The *data-consuming*
    accessors are also bound to a :class:`~tsdynamics.data.Trajectory`
    (``traj.dims`` / ``traj.recurrence`` / ``traj.lyap.from_data``), because a
    user holding measured data is the normal case for exactly those analyses
    and pushing them onto the flat functions is the discoverability gap this
    layer exists to close.  A trajectory-bound accessor has nothing to
    integrate, so a ``system``-first method on one raises rather than guessing.
    """

    __slots__ = ("_system",)

    def __init__(self, system: SystemBase) -> None:
        self._system = system

    def __repr__(self) -> str:
        return f"{type(self).__name__}({type(self._system).__name__})"

    # -- shared helper for the data-consuming accessors --

    def _resolve_data(self, data: Any, run_kwargs: dict[str, Any]) -> Any:
        """Return ``data`` if given, else a fresh trajectory from the subject.

        The data-consuming analyses (dimensions, recurrence) want a measured
        series.  When the caller passes ``data`` it is delegated verbatim;
        otherwise the system is run once
        (``system.run(**run_kwargs)``) and the resulting trajectory is used.
        Splitting the run kwargs out keeps the delegation byte-identical to the
        free function for a given series.

        Bound to a trajectory there is nothing to run: the subject *is* the
        series, so it is delegated as-is (and ``run_kwargs`` is refused rather
        than silently ignored).
        """
        if data is not None:
            return data
        if not is_drivable(self._system):
            if run_kwargs:
                from tsdynamics.errors import InvalidParameterError

                raise InvalidParameterError(
                    f"run_kwargs={sorted(run_kwargs)} has nothing to run: this accessor is "
                    f"bound to a {type(self._system).__name__}, which is already the measured "
                    f"series. Drop run_kwargs, or reach the accessor from the system "
                    f"(e.g. system.dims.correlation(run_kwargs={run_kwargs!r}))."
                )
            return self._system
        return self._system.run(**run_kwargs)

    def _delegate(
        self,
        free: Callable[..., Any],
        *args: Any,
        data: Any = None,
        run_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Call ``free`` with the right *first argument* for its signature.

        A ``system``-first free function receives the bound system (or ``data``
        verbatim when the caller supplies one — those functions accept a measured
        series too, and drive their own integration otherwise).  A ``data``-first
        one receives the measured series, running the system when needed.  The
        choice is read off ``free``'s signature by :func:`subject_kind`, so an
        accessor can never quietly feed a pre-run trajectory to a function that
        wanted to drive the system itself (see the module note).
        """
        if subject_kind(free) == "system":
            if run_kwargs:
                from tsdynamics.errors import InvalidParameterError

                raise InvalidParameterError(
                    f"{free.__name__}() drives the system itself, so it takes no "
                    f"run_kwargs; pass its own horizon keywords instead "
                    f"(e.g. {sorted(run_kwargs)} → direct keyword arguments)."
                )
            if data is None and not is_drivable(self._system):
                from tsdynamics.errors import InvalidParameterError

                raise InvalidParameterError(
                    f"{free.__name__}() integrates the system itself, and a "
                    f"{type(self._system).__name__} is measured data — there is no right-hand "
                    f"side to integrate. Reach it from the system that produced the data "
                    f"(system.{_TOPIC_OF.get(type(self).__name__, 'lyap')}."
                    f"{_method_of(type(self).__name__, free.__name__)}()), or use the "
                    f"from-data estimator (ts.analysis.lyapunov_from_data(traj))."
                )
            subject = self._system if data is None else data
        else:
            subject = self._resolve_data(data, run_kwargs or {})
        return free(subject, *args, **kwargs)


#: Accessor class name → the attribute it is reached by on a system/trajectory.
#: Used only to spell the remedy in the "measured data has no dynamics" error.
_TOPIC_OF = {
    "LyapunovAccessor": "lyap",
    "ChaosAccessor": "chaos",
    "DimensionsAccessor": "dims",
    "RecurrenceAccessor": "recurrence",
}


def _method_of(accessor: str, free_name: str) -> str:
    """Return the accessor method that delegates to ``free_name`` (for error text)."""
    for method, free in ACCESSOR_DELEGATIONS.get(accessor, {}).items():
        if free == free_name:
            return method
    return free_name


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

        return self._delegate(lyapunov_spectrum, **kwargs)

    def maximal(self, **kwargs: Any) -> Any:
        """Delegate to :func:`tsdynamics.analysis.max_lyapunov`."""
        from tsdynamics.analysis import max_lyapunov

        return self._delegate(max_lyapunov, **kwargs)

    def from_data(
        self, data: Any = None, *, run_kwargs: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        """Delegate to :func:`tsdynamics.analysis.lyapunov_from_data`.

        Estimates the maximal exponent from a measured series.  Bound to a
        :class:`~tsdynamics.data.Trajectory` the trajectory *is* the series;
        bound to a system and called without ``data`` the system is run first
        (see :meth:`_Accessor._resolve_data`).  Either way the estimator receives
        the whole trajectory — a **multivariate** embedding when the system has
        several components, not one channel — and reads the sampling interval
        off it, so the exponent comes out **per unit time** and compares directly
        with :meth:`spectrum`.  Pass an explicit 1-D ``data`` (e.g. ``traj["x"]``)
        for the single-channel reconstruction, and ``dt=`` to override the
        interval.
        """
        from tsdynamics.analysis import lyapunov_from_data

        return self._delegate(lyapunov_from_data, data=data, run_kwargs=run_kwargs, **kwargs)


# ---------------------------------------------------------------------------
# chaos indicators
# ---------------------------------------------------------------------------


class ChaosAccessor(_Accessor):
    """Chaos indicators bound to the system (``sys.chaos``).

    All three take the *system*: ``gali`` and ``expansion_entropy`` integrate its
    tangent dynamics internally, and ``zero_one`` lets
    :func:`~tsdynamics.analysis.zero_one_test` sample the observable itself (a
    measured series may still be passed as ``data``).
    """

    def gali(self, k: int = 2, **kwargs: Any) -> Any:
        """Delegate to :func:`tsdynamics.analysis.gali`."""
        from tsdynamics.analysis import gali

        return self._delegate(gali, k, **kwargs)

    def expansion_entropy(self, region: Any = None, **kwargs: Any) -> Any:
        """Delegate to :func:`tsdynamics.analysis.expansion_entropy`."""
        from tsdynamics.analysis import expansion_entropy

        return self._delegate(expansion_entropy, region, **kwargs)

    def zero_one(self, data: Any = None, **kwargs: Any) -> Any:
        """Delegate to :func:`tsdynamics.analysis.zero_one_test`.

        The **system itself** is handed to the free function, which samples the
        observable with the coarse, decorrelated grid the 0--1 test needs
        (``final_time`` / ``dt`` / ``n`` / ``transient`` / ``ic`` are its own
        keywords; pass ``component=`` to pick a column).  A measured 1-D series
        may be supplied as ``data`` instead — the free function's data overload.

        .. versionchanged:: 6.0
           This used to pre-run the system with the *family's* ``run()`` defaults
           and pass the resulting trajectory as data.  On a flow that grid is far
           too fine for the test's skew-translation statistic, so the accessor
           reported Lorenz as regular (``K = -0.026``) where the free function
           reported it chaotic (``K = 0.999``).  The ``run_kwargs=`` keyword is
           gone with it — use the free function's own horizon keywords.
        """
        from tsdynamics.analysis import zero_one_test

        return self._delegate(zero_one_test, data=data, **kwargs)


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

        return self._delegate(correlation_dimension, data=data, run_kwargs=run_kwargs, **kwargs)

    def correlation_sum(
        self, data: Any = None, *, run_kwargs: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        """Delegate to :func:`tsdynamics.analysis.correlation_sum`."""
        from tsdynamics.analysis import correlation_sum

        return self._delegate(correlation_sum, data=data, run_kwargs=run_kwargs, **kwargs)

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

        return self._delegate(generalized_dimension, q, data=data, run_kwargs=run_kwargs, **kwargs)

    def box_counting(
        self, data: Any = None, *, run_kwargs: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        """Delegate to :func:`tsdynamics.analysis.box_counting_dimension`."""
        from tsdynamics.analysis import box_counting_dimension

        return self._delegate(box_counting_dimension, data=data, run_kwargs=run_kwargs, **kwargs)

    def information(
        self, data: Any = None, *, run_kwargs: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        """Delegate to :func:`tsdynamics.analysis.information_dimension`."""
        from tsdynamics.analysis import information_dimension

        return self._delegate(information_dimension, data=data, run_kwargs=run_kwargs, **kwargs)

    def spectrum(
        self, data: Any = None, *, run_kwargs: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        """Delegate to :func:`tsdynamics.analysis.dimension_spectrum`."""
        from tsdynamics.analysis import dimension_spectrum

        return self._delegate(dimension_spectrum, data=data, run_kwargs=run_kwargs, **kwargs)

    def fixed_mass(
        self, data: Any = None, *, run_kwargs: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        """Delegate to :func:`tsdynamics.analysis.fixed_mass_dimension`."""
        from tsdynamics.analysis import fixed_mass_dimension

        return self._delegate(fixed_mass_dimension, data=data, run_kwargs=run_kwargs, **kwargs)


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

        return self._delegate(recurrence_matrix, data=data, run_kwargs=run_kwargs, **kwargs)

    def rqa(
        self, data: Any = None, *, run_kwargs: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        """Delegate to :func:`tsdynamics.analysis.rqa`."""
        from tsdynamics.analysis import rqa

        return self._delegate(rqa, data=data, run_kwargs=run_kwargs, **kwargs)

    def windowed(
        self, data: Any = None, *, run_kwargs: dict[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        """Delegate to :func:`tsdynamics.analysis.windowed_rqa`."""
        from tsdynamics.analysis import windowed_rqa

        return self._delegate(windowed_rqa, data=data, run_kwargs=run_kwargs, **kwargs)


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


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
