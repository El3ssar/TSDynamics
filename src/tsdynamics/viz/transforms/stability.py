r"""Stability-spectrum plot transforms — the picture you *read stability off*.

Two registered transforms, one idea: linear stability is a statement about where
a spectrum sits **relative to a boundary**, so the boundary is drawn as
geometry rather than left to the reader's memory.

``eigenvalue_plane``
    The Jacobian spectrum of an equilibrium (flow) or fixed point (map) in the
    complex plane, against the boundary its convention uses — the **imaginary
    axis** for a flow (:math:`\mathrm{Re}\,\lambda < 0` is stable) or the **unit
    circle** for a map (:math:`|\lambda| < 1`).
``floquet_multipliers``
    The monodromy spectrum of a periodic orbit on the unit circle, with the
    **trivial** multiplier :math:`\mu \approx 1` — the one users misread as an
    instability — split into its own marked series.

Both are ``source="model"``: handed a system they must build a Jacobian at a
point that is not in their input (and, for a cycle, integrate the variational
equation), which is exactly the greppable rule the registry uses.  Handed a
result that already carries a spectrum (a
:class:`~tsdynamics.analysis.fixedpoints.FixedPoint`, a
:class:`~tsdynamics.analysis.fixedpoints.FixedPointSet`, a
:class:`~tsdynamics.analysis.fixedpoints.PeriodicOrbit`) they only reshape it,
which is why the gate's examples are cheap.

The boundary is a **part**, not an annotation
---------------------------------------------
The pre-transform ``FixedPointSet.eigenvalue_plane`` drew a flow's imaginary
axis as an :class:`~tsdynamics.viz.spec.Annotation`.  An annotation is honored
by the two drawing backends and dropped by the two data-export ones, so the
JSON / three.js export of that figure lost the very line that makes it
readable.  Here the boundary is real geometry in a pinned ``line`` part, so it
survives every backend and every primitive.

Neither transform owns any numerics: the spectra come from
:func:`tsdynamics.analysis.fixedpoints.fixed_points` /
:func:`tsdynamics.analysis.fixedpoints.periodic_orbit`, and a Jacobian at an
explicit point comes from the family's own
:meth:`~tsdynamics.families.continuous.ContinuousSystem.jacobian`.

References
----------
.. [1] Strogatz, S. H. (1994). *Nonlinear Dynamics and Chaos*. Addison-Wesley.
   Ch. 5-6 (the eigenvalue classification of a planar equilibrium).
.. [2] Kuznetsov, Y. A. (2004). *Elements of Applied Bifurcation Theory*, 3rd
   ed. Springer.  Ch. 1-4 (hyperbolicity; Floquet multipliers and the trivial
   multiplier along the flow direction).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .._frames import FrameSpace, OverlayRole
from ..spec import PlotKind
from ._base import Geometry, Part, Presentation, make_frame
from ._registry import plot_transform

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

__all__ = ["eigenvalue_plane", "floquet_multipliers"]


# ---------------------------------------------------------------------------
# Shared presentation constants
# ---------------------------------------------------------------------------

#: Axis labels for the eigenvalue plane.  They are also the frame's axis
#: *names* (through :func:`~tsdynamics.viz.transforms.make_frame`), so two
#: eigenvalue planes overlay and an eigenvalue plane refuses a multiplier plane
#: — different quantities, different axes.
_LAMBDA_AXES: tuple[str, str] = (r"$\mathrm{Re}\,\lambda$", r"$\mathrm{Im}\,\lambda$")
_MU_AXES: tuple[str, str] = (r"$\mathrm{Re}\,\mu$", r"$\mathrm{Im}\,\mu$")

#: Style of the stability-boundary geometry: present but never the subject.
_BOUNDARY_STYLE: dict[str, Any] = {
    "color": "gray",
    "linewidth": 1.0,
    "linestyle": "dashed",
    "alpha": 0.7,
}

#: How the two half-planes / annuli are drawn.  Filled = the stable side, hollow
#: = the unstable one, mirroring the fixed-point overlay's convention so the two
#: figures read the same way.
_STABLE_STYLE: dict[str, Any] = {"marker": "circle", "markersize": 7.5, "filled": True}
_UNSTABLE_STYLE: dict[str, Any] = {"marker": "circle", "markersize": 7.5, "filled": False}
_TRIVIAL_STYLE: dict[str, Any] = {"marker": "x", "markersize": 9.0, "color": "black"}

#: Points used to draw the unit circle.  200 is what the pre-transform spec used;
#: at typical figure sizes the polygon is indistinguishable from a circle.
_CIRCLE_POINTS = 200

#: Half-height of the imaginary-axis segment when every eigenvalue is real (the
#: Lorenz origin, every node): the data span is zero, so the segment needs a
#: scale of its own, taken from the real parts.
_REAL_ONLY_AXIS_FRACTION = 0.25


# ---------------------------------------------------------------------------
# Coercion
# ---------------------------------------------------------------------------


def _as_complex(values: Any) -> np.ndarray:
    """Return ``values`` as a flat complex array."""
    return np.asarray(values, dtype=complex).ravel()


def _is_system(subject: Any) -> bool:
    """Whether ``subject`` looks like a dynamical system rather than a result."""
    return hasattr(subject, "dim") and (
        hasattr(subject, "_equations") or hasattr(subject, "_step") or hasattr(subject, "jacobian")
    )


def _jacobian_at(system: Any, point: Any) -> tuple[np.ndarray, bool]:
    """Return ``(J, continuous)`` — the system's Jacobian at ``point``.

    Uses the family's own Jacobian (autogenerated and cached for a flow, the
    declared ``_jacobian`` for a map) through the shared
    :mod:`tsdynamics.analysis._tangent` accessors, so no differentiation
    happens here.
    """
    from tsdynamics.analysis._tangent import flow_fns, map_fns

    x = np.asarray(point, dtype=float).ravel()
    if bool(getattr(system, "is_discrete", False)):
        _, jac = map_fns(system)
        return np.asarray(jac(x), dtype=float), False
    _, jac = flow_fns(system)
    return np.asarray(jac(x), dtype=float), True


def _given(**options: Any) -> dict[str, Any]:
    """Keep only the options the caller actually set (drop every ``None``).

    Transform signatures name the estimator options they forward **explicitly**
    rather than taking ``**kwargs``: the composition front door routes a shared
    keyword only to a transform whose signature names it, so a ``**kwargs``
    catch-all would make ``ts.plot(sys, "eigenvalue_plane", n_seeds=400)``
    *silently drop* the argument.  Defaulting each to ``None`` and forwarding
    only what was set keeps the estimator's own defaults authoritative, so the
    two cannot drift apart.
    """
    return {name: value for name, value in options.items() if value is not None}


def _point_label(x: Any) -> str:
    """Short spelling of a state-space point for a title."""
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        return "?"
    return "(" + ", ".join(f"{v:.4g}" for v in arr) + ")"


# ---------------------------------------------------------------------------
# Geometry construction
# ---------------------------------------------------------------------------


def _unit_circle_part(symbol: str) -> Part:
    """Build the ``|z| = 1`` boundary, as a pinned ``line`` part."""
    theta = np.linspace(0.0, 2.0 * np.pi, _CIRCLE_POINTS)
    return Part(
        {"x": np.cos(theta), "y": np.sin(theta)},
        label=rf"$|{symbol}| = 1$",
        style=_BOUNDARY_STYLE,
        primitive="line",
    )


def _imaginary_axis_part(eig: np.ndarray, symbol: str) -> Part:
    """Build the ``Re = 0`` boundary spanning the data, as a pinned ``line`` part.

    Drawn as geometry (not an annotation) so the JSON and three.js exports carry
    it too, and sized from the data so it is always visible: a spectrum that is
    entirely real (a node — the Lorenz origin) has zero imaginary span, so the
    segment takes its half-height from the real parts instead.
    """
    imag = eig.imag if eig.size else np.zeros(1)
    lo, hi = float(np.min(imag)), float(np.max(imag))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        scale = float(np.max(np.abs(eig.real))) if eig.size else 1.0
        half = _REAL_ONLY_AXIS_FRACTION * (scale if scale > 0.0 else 1.0)
        lo, hi = -half, half
    else:
        pad = 0.1 * (hi - lo)
        lo, hi = lo - pad, hi + pad
    return Part(
        {"x": np.zeros(2), "y": np.array([lo, hi])},
        label=rf"$\mathrm{{Re}}\,{symbol} = 0$",
        style=_BOUNDARY_STYLE,
        primitive="line",
    )


def _spectrum_parts(eig: np.ndarray, *, continuous: bool, symbol: str) -> list[Part]:
    """Split a spectrum into its contracting and expanding halves.

    The split is **per eigenvalue**, and the labels say so.  Labelling the two
    series "stable" / "unstable" would be wrong for a saddle, whose point is
    unstable while several of its eigenvalues contract; the criterion the picture
    actually shows is where each eigenvalue sits relative to the boundary.
    """
    if eig.size == 0:
        return []
    if continuous:
        inside = eig.real < 0.0
        in_label = rf"$\mathrm{{Re}}\,{symbol} < 0$ (contracting)"
        out_label = rf"$\mathrm{{Re}}\,{symbol} \geq 0$ (expanding)"
    else:
        inside = np.abs(eig) < 1.0
        in_label = rf"$|{symbol}| < 1$ (contracting)"
        out_label = rf"$|{symbol}| \geq 1$ (expanding)"

    parts: list[Part] = []
    for mask, label, style in (
        (inside, in_label, _STABLE_STYLE),
        (~inside, out_label, _UNSTABLE_STYLE),
    ):
        if not bool(np.any(mask)):
            continue
        sel = eig[mask]
        parts.append(
            Part(
                {"x": sel.real.astype(float), "y": sel.imag.astype(float)}, label=label, style=style
            )
        )
    return parts


def _circle_limits(eig: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    """Symmetric ``(x, y)`` limits that always contain the unit circle.

    Without this a strongly stable cycle (every multiplier inside ``|mu| = 0.1``)
    autoscales to its own cloud and the unit circle — the entire point of the
    figure — falls outside the axes.
    """
    reach = 1.0
    if eig.size:
        finite = eig[np.isfinite(eig)]
        if finite.size:
            reach = max(reach, float(np.max(np.abs(finite))))
    span = 1.15 * reach
    return (-span, span), (-span, span)


def _spectrum_geometry(
    name: str,
    eig: np.ndarray,
    *,
    continuous: bool,
    boundary: str,
    symbol: str,
    axes: tuple[str, str],
    title: str,
    meta: dict[str, Any],
    extra: Sequence[Part] = (),
) -> Geometry:
    """Assemble the shared complex-plane geometry for both transforms."""
    parts: list[Part] = []
    if boundary == "circle":
        parts.append(_unit_circle_part(symbol))
    elif boundary == "axis":
        parts.append(_imaginary_axis_part(eig, symbol))
    parts.extend(extra)
    parts.extend(_spectrum_parts(eig, continuous=continuous, symbol=symbol))

    limits: tuple[tuple[float, float] | None, ...] = ()
    if boundary == "circle":
        lo_hi = _circle_limits(eig)
        limits = (lo_hi[0], lo_hi[1])

    criterion = rf"$\mathrm{{Re}}\,{symbol}<0$" if continuous else rf"$|{symbol}|<1$"
    return Geometry(
        name,
        make_frame(FrameSpace.COMPLEX, 2, axes),
        parts,
        axis_labels=axes,
        axis_limits=limits,
        aspect="equal",
        title=f"{title} — stable if {criterion}",
        meta=meta,
    )


# ---------------------------------------------------------------------------
# eigenvalue_plane
# ---------------------------------------------------------------------------


def _fixed_point_spectrum(
    subject: Any, at: Any, search: dict[str, Any]
) -> tuple[np.ndarray, bool, str, dict[str, Any]]:
    """Resolve ``(eigenvalues, continuous, title, meta)`` for :func:`eigenvalue_plane`."""
    from tsdynamics.errors import InvalidInputError

    # 1. An explicit linearisation point on a system: the cheapest, most direct
    #    route, and the one that lets a user ask about a *known* equilibrium
    #    (the Lorenz origin) without a root search that might not return it.
    if at is not None:
        if not _is_system(subject):
            raise InvalidInputError(
                "at= linearises a *system* at a point, but the subject is a "
                f"{type(subject).__name__}. Pass the system, or drop at= and pass the "
                "fixed-point result you already have."
            )
        jac, continuous = _jacobian_at(subject, at)
        eig = np.linalg.eigvals(jac)
        kind = "equilibrium" if continuous else "fixed point"
        return (
            eig,
            continuous,
            f"{type(subject).__name__} {kind} at {_point_label(at)}",
            {
                "analysis": "jacobian",
                "at": np.asarray(at, dtype=float).tolist(),
                "continuous": continuous,
            },
        )

    # 2. A single fixed point / equilibrium result.
    if hasattr(subject, "eigenvalues"):
        eig = _as_complex(subject.eigenvalues)
        continuous = bool(getattr(subject, "continuous", True))
        kind = "equilibrium" if continuous else "fixed point"
        state = "stable" if bool(getattr(subject, "stable", False)) else "unstable"
        return (
            eig,
            continuous,
            f"{state} {kind} at {_point_label(getattr(subject, 'x', ()))}",
            {"analysis": "fixed_points", "continuous": continuous, "n_points": 1},
        )

    # 3. A set of them (FixedPointSet, or any sequence of FixedPoint).
    members = _as_fixed_point_sequence(subject)
    if members is not None:
        if not members:
            raise InvalidInputError(
                "eigenvalue_plane got an empty fixed-point set: there is no spectrum to "
                "draw. Widen region= / raise n_seeds= on the search that produced it."
            )
        eig = np.concatenate([_as_complex(m.eigenvalues) for m in members])
        continuous = bool(getattr(members[0], "continuous", True))
        n_stable = sum(1 for m in members if bool(getattr(m, "stable", False)))
        word = "equilibria" if continuous else "fixed points"
        return (
            eig,
            continuous,
            f"{len(members)} {word} — {n_stable} stable, {len(members) - n_stable} unstable",
            {
                "analysis": "fixed_points",
                "continuous": continuous,
                "n_points": len(members),
                "points": [np.asarray(m.x, dtype=float).tolist() for m in members],
            },
        )

    # 4. A bare system: run the search.
    if _is_system(subject):
        from tsdynamics.analysis.fixedpoints import fixed_points

        found = fixed_points(subject, **search)
        return _fixed_point_spectrum(found, None, {})

    # 5. Raw eigenvalues.  A Trajectory is caught here explicitly: it *is*
    #    array-coercible, so without the guard a measured orbit would be drawn as
    #    a "spectrum" of its own state values — a plot that draws fine and means
    #    nothing, which is the exact failure this layer exists to prevent.
    if hasattr(subject, "t") and hasattr(subject, "y"):
        raise InvalidInputError(
            "eigenvalue_plane is a model transform: a Trajectory carries samples, not a "
            "Jacobian. Pass the system (optionally with at=), or a FixedPoint / "
            "FixedPointSet computed from it."
        )
    eig = _as_complex(subject)
    if eig.size == 0:
        raise InvalidInputError(
            "eigenvalue_plane needs a spectrum: pass a system (optionally with at=), a "
            "FixedPoint / FixedPointSet, or an array of eigenvalues."
        )
    return eig, True, "spectrum", {"analysis": None, "continuous": True}


def _as_fixed_point_sequence(subject: Any) -> list[Any] | None:
    """Return ``subject`` as a list of fixed-point-like members, or ``None``."""
    if isinstance(subject, (str, bytes, np.ndarray)) or _is_system(subject):
        return None
    try:
        members = list(subject)
    except TypeError:
        return None
    if members and all(hasattr(m, "eigenvalues") for m in members):
        return members
    # An *empty* result collection is still a fixed-point set; distinguish it
    # from an empty array by the class it came from.
    if not members and type(subject).__name__.endswith("Set"):
        return members
    return None


def _example_fixed_point(_primitive: str) -> tuple[Any, dict[str, Any]]:
    """Return the gate's subject: the analytic Lorenz-origin spectrum, no search, no solver.

    Hand-built rather than searched so the compatibility gate stays a *drawing*
    test costing microseconds — and so the numbers in it are the ones this
    module's docstring claims.
    """
    from tsdynamics.analysis.fixedpoints import FixedPoint

    return (
        FixedPoint(
            x=np.zeros(3),
            eigenvalues=np.array([11.82772, -2.66667, -22.82772], dtype=complex),
            stable=False,
            continuous=True,
        ),
        {},
    )


@plot_transform(
    name="eigenvalue_plane",
    source="model",
    kind=PlotKind.EIGENVALUE_PLANE,
    frame=FrameSpace.COMPLEX,
    ndim=2,
    role=OverlayRole.BASE,
    default_primitive="points",
    primitives=("points", "line"),
    presentation=Presentation(aspect="equal"),
    analysis="tsdynamics.analysis.fixedpoints.fixed_points",
    example=_example_fixed_point,
    doc="The Jacobian spectrum of an equilibrium against its stability boundary.",
)
def eigenvalue_plane(
    subject: Any,
    *,
    at: Any | None = None,
    boundary: bool = True,
    region: Any | None = None,
    n_seeds: int | None = None,
    method: str | None = None,
    seed: int | None = None,
) -> Geometry:
    r"""Draw the Jacobian spectrum in the complex plane against its stability boundary.

    Stability of a hyperbolic equilibrium is entirely a statement about which
    side of a boundary its eigenvalues sit on, so the boundary is drawn: the
    **imaginary axis** for a flow (stable iff every
    :math:`\mathrm{Re}\,\lambda < 0`) or the **unit circle** for a map (stable
    iff every :math:`|\lambda| < 1`).  The spectrum is split into the
    contracting and expanding eigenvalues — per *eigenvalue*, not per point, so
    a saddle reads correctly.

    Parameters
    ----------
    subject : System, FixedPoint, FixedPointSet, or array-like
        A system (its equilibria are found with
        :func:`~tsdynamics.analysis.fixedpoints.fixed_points`, unless ``at=`` is
        given), an already-computed fixed point or set of them, or a bare array
        of eigenvalues.
    at : array-like, optional
        Linearise ``subject`` (which must then be a system) at this point and
        draw *that* spectrum.  The direct route to a known equilibrium — the
        Lorenz origin, a symmetric branch — with no root search involved.
    boundary : bool, optional
        Draw the stability boundary.  ``True`` (the default) is the whole point
        of the figure; ``False`` is for overlaying two spectra without drawing
        the same reference twice.
    region, n_seeds, method, seed
        Forwarded to :func:`~tsdynamics.analysis.fixedpoints.fixed_points` when
        ``subject`` is a system and ``at`` is not given.  Each defaults to
        ``None`` — meaning *"leave the estimator's own default alone"* — so the
        two cannot drift apart.  For any option not named here, call
        ``fixed_points`` yourself and plot the result.

    Returns
    -------
    Geometry
        A ``complex``-framed, equal-aspect geometry whose parts are the
        boundary, the contracting eigenvalues and the expanding ones.

    Raises
    ------
    tsdynamics.errors.InvalidInputError
        If ``at=`` is given for a subject that is not a system, or if the
        subject carries no spectrum at all.

    Examples
    --------
    >>> ts.plot(ts.systems.Lorenz(), "eigenvalue_plane", at=[0, 0, 0])  # doctest: +SKIP
    >>> ts.plot(ts.fixed_points(ts.systems.Rossler()), "eigenvalue_plane")  # doctest: +SKIP
    """
    eig, continuous, title, meta = _fixed_point_spectrum(
        subject, at, _given(region=region, n_seeds=n_seeds, method=method, seed=seed)
    )
    return _spectrum_geometry(
        "eigenvalue_plane",
        eig,
        continuous=continuous,
        boundary=("axis" if continuous else "circle") if boundary else "none",
        symbol=r"\lambda",
        axes=_LAMBDA_AXES,
        title=title,
        meta={**meta, "n_eigenvalues": int(eig.size)},
    )


# ---------------------------------------------------------------------------
# floquet_multipliers
# ---------------------------------------------------------------------------


def _trivial_index(mu: np.ndarray, continuous: bool, tol: float) -> int | None:
    """Index of the trivial multiplier ``mu ~ 1``, or ``None``.

    Only a **flow** cycle has one: the monodromy matrix always has the
    eigenvalue 1 along the flow direction (Kuznetsov 2004, §1.5).  A map cycle
    has no such multiplier, so nothing is marked.  The location is
    ``argmin|mu - 1|`` guarded by ``tol`` — a *presentation* heuristic, matching
    the one :meth:`PeriodicOrbit.eigenvalue_plane` uses; the stability flag on
    the result itself is decided by the more robust eigenvector-alignment test
    inside :func:`~tsdynamics.analysis.fixedpoints.periodic_orbit`.
    """
    if not continuous or mu.size == 0:
        return None
    idx = int(np.argmin(np.abs(mu - 1.0)))
    return idx if abs(mu[idx] - 1.0) <= tol else None


def _example_periodic_orbit(_primitive: str) -> tuple[Any, dict[str, Any]]:
    """Return the gate's subject: a hand-built stable cycle (no shooting, no integration)."""
    from tsdynamics.analysis.fixedpoints import PeriodicOrbit

    theta = np.linspace(0.0, 2.0 * np.pi, 32)
    return (
        PeriodicOrbit(
            points=np.column_stack([np.cos(theta), np.sin(theta)]),
            period=6.6,
            multipliers=np.array([1.0, 0.04], dtype=complex),
            stable=True,
            continuous=True,
        ),
        {},
    )


@plot_transform(
    name="floquet_multipliers",
    source="model",
    kind=PlotKind.EIGENVALUE_PLANE,
    frame=FrameSpace.COMPLEX,
    ndim=2,
    role=OverlayRole.BASE,
    default_primitive="points",
    primitives=("points", "line"),
    presentation=Presentation(aspect="equal"),
    analysis="tsdynamics.analysis.fixedpoints.periodic_orbit",
    example=_example_periodic_orbit,
    doc="A periodic orbit's monodromy spectrum on the unit circle, trivial one marked.",
)
def floquet_multipliers(
    subject: Any,
    *,
    trivial_tol: float = 0.1,
    boundary: bool = True,
    ic: Any | None = None,
    period_guess: float | None = None,
    transient: float | None = None,
    n_points: int | None = None,
    seed: int | None = None,
) -> Geometry:
    r"""Draw the Floquet multipliers of a periodic orbit on the unit circle.

    A limit cycle is stable iff every **non-trivial** multiplier satisfies
    :math:`|\mu| < 1`.  A flow's monodromy matrix always carries one multiplier
    :math:`\mu \approx 1` along the flow direction — it is the single most
    misread number in this figure, so it is drawn as its own marked series
    rather than left to sit on the boundary looking marginal.

    Parameters
    ----------
    subject : PeriodicOrbit, System, or array-like
        A computed orbit, a continuous system (shot with
        :func:`~tsdynamics.analysis.fixedpoints.periodic_orbit`), or a bare
        array of multipliers.
    trivial_tol : float, optional
        How close to ``1`` a multiplier must be to be marked trivial.  Default
        ``0.1``.  A converged cycle puts it within ``~1e-6``; the loose default
        still catches a coarsely-shot one, and nothing is marked when no
        multiplier is near ``1``.
    boundary : bool, optional
        Draw the unit circle.  Default ``True``.
    ic, period_guess, transient, n_points, seed
        Forwarded to :func:`~tsdynamics.analysis.fixedpoints.periodic_orbit`
        when ``subject`` is a system.  Each defaults to ``None``, meaning
        *"leave the estimator's own default alone"* (see :func:`_given`).

    Returns
    -------
    Geometry
        A ``complex``-framed, equal-aspect geometry: the unit circle, the
        trivial multiplier (when there is one), and the contracting / expanding
        multipliers.  The axes are ``Re mu`` / ``Im mu``, so a multiplier plane
        never silently overlays an eigenvalue plane.

    Raises
    ------
    tsdynamics.errors.InvalidInputError
        If the subject carries no multipliers.

    Examples
    --------
    >>> vdp = ts.systems.VanDerPol(params={"mu": 1.0})              # doctest: +SKIP
    >>> ts.plot(vdp, "floquet_multipliers", ic=[2.0, 0.0])          # doctest: +SKIP
    """
    from tsdynamics.errors import InvalidInputError

    if hasattr(subject, "multipliers"):
        mu = _as_complex(subject.multipliers)
        continuous = bool(getattr(subject, "continuous", True))
        period = getattr(subject, "period", 0)
        per = f"T = {float(period):.4g}" if continuous else f"p = {int(period)}"
        state = "stable" if bool(getattr(subject, "stable", False)) else "unstable"
        title = f"{state} cycle, {'Floquet multipliers' if continuous else 'multipliers'} ({per})"
        meta: dict[str, Any] = {
            "analysis": "periodic_orbit",
            "continuous": continuous,
            "period": float(period) if continuous else int(period),
            "residual": float(getattr(subject, "residual", 0.0)),
        }
    elif _is_system(subject):
        from tsdynamics.analysis.fixedpoints import periodic_orbit

        shooting = _given(
            ic=ic,
            period_guess=period_guess,
            transient=transient,
            n_points=n_points,
            seed=seed,
        )
        return floquet_multipliers(
            periodic_orbit(subject, **shooting), trivial_tol=trivial_tol, boundary=boundary
        )
    else:
        mu = _as_complex(subject)
        if mu.size == 0:
            raise InvalidInputError(
                "floquet_multipliers needs a spectrum: pass a PeriodicOrbit, a continuous "
                "system to shoot, or an array of multipliers."
            )
        continuous, title = True, "multipliers"
        meta = {"analysis": None, "continuous": True}

    trivial = _trivial_index(mu, continuous, trivial_tol)
    extra: list[Part] = []
    if trivial is not None:
        tv = mu[trivial]
        extra.append(
            Part(
                {"x": np.array([tv.real]), "y": np.array([tv.imag])},
                label=r"trivial $\mu \approx 1$",
                style=_TRIVIAL_STYLE,
            )
        )
        mu = np.delete(mu, trivial)

    return _spectrum_geometry(
        "floquet_multipliers",
        mu,
        # Multipliers live on the unit-circle convention whether the cycle
        # belongs to a map or a flow: |mu| < 1 is the criterion in both cases.
        continuous=False,
        boundary="circle" if boundary else "none",
        symbol=r"\mu",
        axes=_MU_AXES,
        title=title,
        meta={
            **meta,
            "n_multipliers": int(mu.size) + (1 if trivial is not None else 0),
            "trivial_marked": trivial is not None,
        },
        extra=extra,
    )


def __dir__() -> list[str]:
    """Expose only the registered transforms to ``dir()`` / autocomplete."""
    return sorted(__all__)
