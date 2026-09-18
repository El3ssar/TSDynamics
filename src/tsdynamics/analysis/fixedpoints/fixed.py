r"""
Fixed points of maps and equilibria of flows.

:func:`fixed_points` finds the roots of the defining residual by multi-start
root finding and classifies their linear stability from the Jacobian spectrum:

- **maps** (:class:`~tsdynamics.families.DiscreteMap`): solve :math:`f(x) = x`;
  stable iff every multiplier :math:`|\lambda_i| < 1`.
- **flows** (:class:`~tsdynamics.families.ContinuousSystem`): solve the
  equilibrium condition :math:`f(x) = 0` on the right-hand side; stable iff every
  eigenvalue has :math:`\operatorname{Re}\lambda_i < 0`.

The default ``method="newton"`` uses the exact analytic Jacobian.  For maps,
``method="sd"`` / ``"dl"`` additionally engage the Schmelcher--Diakonos (1997) /
Davidchack--Lai (1999) stabilising transformations, which find unstable fixed
points that pure Newton can miss by cycling a set of orthogonal matrices that
turn each instability type into a contracting one.

``method="interval"`` is a *rigorous* alternative (maps **and** flows): the
Krawczyk operator brackets **all** roots inside the (required) search ``region``
by interval branch-and-prune, certifying existence + uniqueness per sub-box — so
it cannot silently miss a root the way a finite multi-start can, and is faster on
the analytic/polynomial systems it applies to.  It needs an interval-extensible
right-hand side (the engine lives in
:mod:`tsdynamics.analysis.fixedpoints._interval`); a system whose kernel uses an
op the interval engine cannot enclose (a comparison, a modulo, a non-integer
power) raises :class:`~tsdynamics.errors.InvalidInputError`, pointing back at
``method="newton"``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

import numpy as np

from tsdynamics.errors import InvalidParameterError, invalid_value, remedy
from tsdynamics.families import ContinuousSystem, DiscreteMap

from .._common import reject_data
from .._result import AnalysisResult, CollectionResult, _ArrayBacked, _build_meta
from .._result_json import _sig, _state
from . import _common as _c

__all__ = ["FixedPoint", "FixedPointSet", "fixed_points"]


@dataclass(frozen=True)
class FixedPoint(_ArrayBacked, AnalysisResult):
    """A fixed point (map) or equilibrium (flow) with its linear stability data.

    **It IS its point** (v6): ``np.asarray(fp)`` is the ``(dim,)`` state,
    ``fp[0]`` is a float, ``fp - other`` and ``np.linalg.norm(fp)`` work, and
    ``len(fp)`` is the state-space dimension — so ``fixed_points(sys)[0]`` hands
    back numbers and the class stays invisible, showing itself only in the repr,
    which states the answer.  The stability data is one dot away and never in the
    way: :attr:`eigenvalues`, :attr:`stable`, :attr:`continuous`.

    It is also an :class:`~tsdynamics.analysis._result.AnalysisResult`, so it
    carries ``.meta`` / the readout ``repr`` / ``.to_dict()`` / ``.to_frame()`` /
    the ``.plot`` seam.

    Attributes
    ----------
    x : ndarray
        The point, shape ``(dim,)``.  ``np.asarray(fp)`` returns it.
    eigenvalues : ndarray
        Eigenvalues of the Jacobian at ``x`` — map multipliers (of :math:`Df`) for
        a discrete map, or eigenvalues of the vector-field Jacobian for a flow.
    stable : bool
        For a map, ``True`` iff every ``|lambda| < 1``; for a flow, ``True`` iff
        every ``Re(lambda) < 0``.
    continuous : bool
        ``True`` for a flow equilibrium, ``False`` for a map fixed point — sets
        which stability convention ``stable`` uses.
    """

    #: The numbers this result *is* — see :class:`_ArrayBacked`.
    _array_field: ClassVar[str] = "x"

    x: np.ndarray = field(default_factory=lambda: np.empty(0), compare=False)
    eigenvalues: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False, compare=False)
    stable: bool = False
    continuous: bool = False

    def __float__(self) -> float:
        """Return the coordinate of a **1-D** system's fixed point.

        A scalar map's fixed point is a number, so ``float(fp)`` is the obvious
        thing to write; on a higher-dimensional system it raises, naming the
        array, rather than silently returning the first coordinate.
        """
        x = np.asarray(self.x, dtype=float).ravel()
        if x.size == 1:
            return float(x[0])
        raise TypeError(
            f"this fixed point has {x.size} coordinates, so it is not one number. "
            "Use np.asarray(fp) for the point, or fp[i] for a coordinate."
        )

    def _gauge(self) -> str:
        """Return the eigenvalue reading that decides the classification."""
        e = np.asarray(self.eigenvalues)
        if not e.size:
            return "no eigenvalues"
        if self.continuous:
            return f"Re(λ)max = {float(e.real.max()):+.4g}"
        return f"|λ|max = {_sig(float(np.abs(e).max()), 4)}"

    def _answer(self) -> str:
        """Return ``x* = [...]`` — the point itself."""
        return f"x* = {_state(self.x)}"

    def _interpretation(self) -> str | None:
        """Name the stability, and the reading it was decided from."""
        return ("stable" if self.stable else "unstable") + f"  {self._gauge()}"

    def _context(self) -> str | None:
        """Return the subject, and which stability convention applies."""
        bits = [b for b in (self._system_label(),) if b]
        bits.append("equilibrium of a flow" if self.continuous else "fixed point of a map")
        return ", ".join(bits)

    def _as_item(self) -> str:
        """Return the compact one-line form used inside a set's item list."""
        return f"x* = {_state(self.x)}  {'stable' if self.stable else 'unstable'}  {self._gauge()}"

    def __plot_spec__(
        self,
        kind: str | None = None,
        *,
        components: Sequence[int | str] | None = None,
        annotate: bool | None = None,
    ) -> Any:
        r"""Describe this fixed point as a backend-agnostic :class:`PlotSpec`.

        Builds a ``FIXED_POINTS_OVERLAY``: a single ``SCATTER`` point in the chosen
        projection plane, styled by stability (a filled marker for a stable point,
        an open marker for an unstable one) and annotated with its **leading
        eigenvalue** — the number that decides the classification the marker is
        drawing.  Designed to be drawn *over* a phase portrait via
        :meth:`AnalysisResult.overlay_on`, which keeps the host layers first.  For
        the eigenvalue/multiplier picture use :meth:`eigenvalue_plane`.  The
        :mod:`tsdynamics.viz.spec` import is lazy, so building a spec never pulls a
        plotting library.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind (a :class:`~tsdynamics.viz.spec.PlotKind`
            value).  ``None`` uses ``FIXED_POINTS_OVERLAY``.
        components : sequence, optional
            The two state-vector coordinates to project onto, as indices
            (``(0, 2)``) or as the names the system declares (``("x", "z")``).
            Defaults to the first two.  Pass the same pair to the host portrait so
            the overlay lands on the plane it was drawn for — since v6 an overlay
            onto a host drawn on a *different* plane raises rather than silently
            putting the equilibria in the wrong place (the frame check in
            :meth:`~tsdynamics.analysis._result.AnalysisResult.overlay_on`).
        annotate : bool, optional
            Label the marker with its leading eigenvalue (largest real part for a
            flow, largest modulus for a map).  ``None`` (default) means *on* for a
            single point.

        Returns
        -------
        PlotSpec
        """
        from .. import _plotbuilder as pb

        x = np.asarray(self.x, dtype=float).ravel()
        label = "stable" if self.stable else "unstable"
        style = _fixed_point_style(self.stable)
        i, j = _resolve_components(self.meta, int(x.size), components)
        if x.size >= 2:
            px, py = float(x[i]), float(x[j])
            layer = pb.scatter(np.array([px]), np.array([py]), label=label, style=style)
            xlabel, ylabel = pb.axis_labels(self.meta, (i, j))
        else:
            px, py = 0.0, float(x[0]) if x.size else 0.0
            layer = pb.scatter(np.array([px]), np.array([py]), label=label, style=style)
            xlabel, ylabel = "", pb.axis_labels(self.meta, (0,))[0]
        annotations, xlim, ylim = _eigenvalue_annotations(
            [(px, py, np.asarray(self.eigenvalues))],
            continuous=self.continuous,
            enabled=True if annotate is None else annotate,
        )
        return pb.spec(
            kind,
            "fixed_points_overlay",
            layers=[layer],
            aspect="equal",
            xlabel=xlabel,
            ylabel=ylabel,
            title=f"{label} fixed point",
            annotations=annotations,
            xlimits=xlim,
            ylimits=ylim,
            meta=self.meta,
        )

    def eigenvalue_plane(self, kind: str | None = None) -> Any:
        r"""Describe the Jacobian spectrum as an :class:`EIGENVALUE_PLANE` spec.

        Plots the eigenvalues / multipliers of :attr:`eigenvalues` in the complex
        plane.  The stability boundary is drawn as a reference geometry: the unit
        circle for a **map** (``|λ| = 1``) or the imaginary axis for a **flow**
        (``Re λ = 0``), per :attr:`continuous`.  The :mod:`tsdynamics.viz.spec`
        import is lazy, so building a spec never pulls a plotting library.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind.  ``None`` uses ``EIGENVALUE_PLANE``.

        Returns
        -------
        PlotSpec
        """
        return _eigenvalue_plane_spec(
            np.asarray(self.eigenvalues),
            continuous=self.continuous,
            title="fixed-point spectrum",
            meta=dict(self.meta) if self.meta else {},
            kind=kind,
        )


@dataclass(frozen=True, eq=False)
class FixedPointSet(CollectionResult):
    """The set of fixed points / equilibria found, indexing to **numbers**.

    A :class:`~tsdynamics.analysis._result.CollectionResult`: iterate it, index
    it, take its ``len``, and read :attr:`stable` / :attr:`unstable` — while it
    carries ``.meta`` / the readout ``repr`` / ``.to_frame()`` / the ``.plot``
    seam.

    ``fps[0]`` is the ``(dim,)`` point as a plain :class:`numpy.ndarray`, equal
    to ``np.asarray(fps)[0]`` — so it has ``.shape``, ``.tolist()`` and a numeric
    ``dtype``, and nothing about the :class:`FixedPoint` class has to be learned
    to use the answer.  That record is :attr:`details`::

        fps = ts.analysis.fixed_points(system)
        fps[0]                 # array([-1.13, -0.34])      the point
        fps.points             # (n, dim)   every point
        fps.eigenvalues        # (n, dim)   every spectrum
        fps.is_stable          # (n,) bool  the mask
        fps.details[0]         # FixedPoint  x* = [-1.13, -0.34]  unstable

    :attr:`points`, :attr:`eigenvalues` and :attr:`is_stable` are the same data
    column-wise, so nothing here needs a loop.
    """

    @property
    def points(self) -> np.ndarray:
        """Every point, as one ``(n, dim)`` float array (same as ``np.asarray(self)``)."""
        return np.asarray(self)

    @property
    def eigenvalues(self) -> np.ndarray:
        """Every member's Jacobian spectrum, as one ``(n, dim)`` array.

        Complex when any eigenvalue is; a ragged set (members of differing
        dimension) falls back to an object array rather than padding.
        """
        rows = [np.asarray(fp.eigenvalues).ravel() for fp in self.items]
        if rows and len({r.size for r in rows}) == 1:
            return np.asarray(rows)
        return np.asarray(rows, dtype=object)

    @property
    def is_stable(self) -> np.ndarray:
        """A boolean **mask** over the members — one ``bool`` each, in order.

        Not a filtered set: this is the ``(n,)`` array you index the *other*
        vectorised columns with, which is what makes a custom selection possible
        without ever touching a record::

            fps.points[fps.is_stable]          # coordinates of the stable ones
            fps.eigenvalues[~fps.is_stable]    # spectra of the unstable ones

        :attr:`stable` is the other half of the pair and returns the filtered
        :class:`FixedPointSet` itself; use that when you want the repr, the
        table and the plot to come with it.

        Returns
        -------
        numpy.ndarray
            Shape ``(n,)``, ``dtype=bool``.
        """
        return np.array([bool(fp.stable) for fp in self.items], dtype=bool)

    @property
    def stable(self) -> FixedPointSet:
        """The stable members, as a **:class:`FixedPointSet` of their own**.

        Not a mask: this is the narrowed *set*, so everything a set does still
        works on it — ``fixed_points(sys).stable`` prints its own readout,
        tabulates with ``to_frame()``, draws with ``.plot()`` and can be
        overlaid.  (Before v6 it handed back a plain ``list`` and all of that was
        lost the moment you narrowed.)

        :attr:`is_stable` is the other half of the pair and gives the boolean
        mask instead; reach for that when you want to index the vectorised
        columns yourself.

        Returns
        -------
        FixedPointSet
        """
        return self._select(stable=True)

    @property
    def unstable(self) -> FixedPointSet:
        """The unstable members, as a :class:`FixedPointSet` (the complement of :attr:`stable`)."""
        return self._select(stable=False)

    def _select(self, *, stable: bool) -> FixedPointSet:
        """Return the sub-set with the given stability, as this same class."""
        return FixedPointSet(
            items=tuple(fp for fp in self.items if bool(fp.stable) is stable), meta=self.meta
        )

    def _noun(self) -> str:
        """Return ``point`` — what one item of this collection is."""
        return "point"

    def _answer(self) -> str:
        """Return the count and the stable/unstable split."""
        if not self.items:
            return "none found"
        n_stable = len(self.stable)
        return (
            f"{len(self.items)} {self._noun()}"
            + ("s" if len(self.items) != 1 else "")
            + f" · {n_stable} stable, {len(self.items) - n_stable} unstable"
        )

    def _derived(self) -> dict[str, Any]:
        """Export the stable/unstable split the repr reports."""
        return {"n_stable": len(self.stable), "n_unstable": len(self.unstable)}

    def __plot_spec__(
        self,
        kind: str | None = None,
        *,
        components: Sequence[int | str] | None = None,
        annotate: bool | None = None,
    ) -> Any:
        r"""Describe the whole set as one ``FIXED_POINTS_OVERLAY`` :class:`PlotSpec`.

        Draws the stable and the unstable fixed points as two separately styled
        ``SCATTER`` layers (filled vs open markers) in the chosen projection plane,
        each annotated with its **leading eigenvalue**, so the set reads at a
        glance — the overlay a phase portrait hosts via
        :meth:`AnalysisResult.overlay_on` (host layers first).  A 1-D set scatters
        against a zero baseline.  An empty set yields a valid layer-less spec.
        For the spectrum picture use :meth:`eigenvalue_plane`.  The
        :mod:`tsdynamics.viz.spec` import is lazy.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind.  ``None`` uses ``FIXED_POINTS_OVERLAY``.
        components : sequence, optional
            The two state-vector coordinates to project onto, as indices
            (``(0, 2)``) or as the names the system declares (``("x", "z")``).
            Defaults to the first two.  Lorenz's two nontrivial equilibria coincide
            under the ``(x, y)`` default only up to a sign and separate cleanly in
            ``("x", "z")`` — the projection is a real choice, not a formality.
        annotate : bool, optional
            Label each marker with its leading eigenvalue (largest real part for a
            flow, largest modulus for a map).  ``None`` (default) annotates while
            the set has at most :data:`_ANNOTATE_AUTO_MAX` members, above which the
            labels would obscure the markers they describe.

        Returns
        -------
        PlotSpec
        """
        from .. import _plotbuilder as pb

        dim = min((np.asarray(fp.x).size for fp in self.items), default=0)
        i, j = _resolve_components(self.meta, int(dim), components) if dim else (0, 0)

        layers = []
        placed: list[tuple[float, float, np.ndarray]] = []
        for stable, label in ((True, "stable"), (False, "unstable")):
            pts = [
                np.asarray(fp.x, dtype=float).ravel() for fp in self.items if fp.stable is stable
            ]
            eigs = [np.asarray(fp.eigenvalues) for fp in self.items if fp.stable is stable]
            if not pts:
                continue
            arr = np.asarray([p[:dim] for p in pts], dtype=float)
            xs = arr[:, i]
            ys = arr[:, j] if dim >= 2 else np.zeros(arr.shape[0])
            layers.append(pb.scatter(xs, ys, label=label, style=_fixed_point_style(stable)))
            placed.extend(zip(xs.tolist(), ys.tolist(), eigs, strict=False))

        continuous = bool(self.items[0].continuous) if self.items else False
        enabled = len(placed) <= _ANNOTATE_AUTO_MAX if annotate is None else annotate
        annotations, xlim, ylim = _eigenvalue_annotations(
            placed, continuous=continuous, enabled=enabled
        )
        labels = pb.axis_labels(self.meta, (i, j))
        return pb.spec(
            kind,
            "fixed_points_overlay",
            layers=layers,
            aspect="equal",
            xlabel=labels[0] if dim >= 2 else "",
            ylabel=labels[1] if dim >= 2 else labels[0],
            title=f"fixed points ({len(self.items)} found)",
            legend=len(layers) > 1,
            annotations=annotations,
            xlimits=xlim,
            ylimits=ylim,
            meta=self.meta,
        )

    def eigenvalue_plane(self, kind: str | None = None) -> Any:
        r"""Describe every member's spectrum in one :class:`EIGENVALUE_PLANE` spec.

        Pools the eigenvalues / multipliers of all fixed points and plots them in
        the complex plane against the stability boundary (the unit circle for
        maps, the imaginary axis for flows).  The :mod:`tsdynamics.viz.spec` import
        is lazy.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind.  ``None`` uses ``EIGENVALUE_PLANE``.

        Returns
        -------
        PlotSpec
        """
        eigs = (
            np.concatenate([np.asarray(fp.eigenvalues).ravel() for fp in self.items])
            if self.items
            else np.empty(0, dtype=complex)
        )
        continuous = bool(self.items[0].continuous) if self.items else False
        return _eigenvalue_plane_spec(
            eigs,
            continuous=continuous,
            title="fixed-point spectra",
            meta=dict(self.meta) if self.meta else {},
            kind=kind,
        )


def fixed_points(
    system: Any,
    region: Any = None,
    *,
    n_seeds: int = 200,
    tol: float = 1e-12,
    max_iter: int = 60,
    dedup_tol: float = 1e-6,
    method: str = "newton",
    lam: float = 0.05,
    beta: float = 1.0,
    max_c: int | None = None,
    seed: int | None = 0,
) -> FixedPointSet:
    r"""
    Find the fixed points of a map, or the equilibria of a flow.

    Seeds are drawn uniformly from ``region`` plus points sampled from a short
    orbit; each runs the chosen root finder, and converged roots are deduplicated
    and classified by the Jacobian spectrum (maps: ``|lambda| < 1``; flows:
    ``Re lambda < 0``).

    Parameters
    ----------
    system : DiscreteMap or ContinuousSystem
        A discrete map (fixed points) or a continuous flow (equilibria).  Delay
        and stochastic systems are not supported.
    region : Box, Grid, (lo, hi) tuple, optional
        Search region.  An explicit region is a hard search **domain**: seeds are
        drawn in it and converged roots outside it are discarded.  ``None`` (the
        default) is a pure *seeding* heuristic instead — nothing is clipped — and
        seeds two boxes derived from a 20-time-unit burn-in orbit: its bare
        bounding box (``n_seeds`` draws, where equilibria cluster) *and* the same
        box padded by 50 % (a further ``n_seeds / 2``, which reaches the ones
        outside the attractor), plus 20 on-orbit points.  The padded box is what
        finds equilibria the orbit never approaches: Rossler's second one is at
        ``(5.69, -28.47, 28.47)`` while its attractor never leaves ``|y| < 12``,
        and before v6 (a 2.0-time-unit hull) ``fixed_points(Rossler())`` returned
        1 of 2 for **every** seed, with no warning.  If the burn-in orbit escapes
        instead of settling, both boxes fall back to ``[-2, 2]^dim``.
    n_seeds : int
        Random seeds in the hull box; the padded box takes half as many again,
        and 20 orbit points are added on top.  This is a
        completeness knob, and the cheap one: multi-start Newton is a heuristic
        that can always miss a root, so raise it when the count matters.  For a
        *guaranteed* complete set use ``method="interval"`` (which recovers all
        27 of Thomas's equilibria with a per-box existence certificate).
    tol : float
        Residual tolerance (``‖f(x) − x‖`` for maps, ``‖f(x)‖`` for flows).
    max_iter : int
        Root-finding iterations per seed.
    dedup_tol : float
        Distance below which two roots are merged.
    method : {"newton", "sd", "dl", "interval"}
        ``"newton"`` (default) — Newton on the exact Jacobian.  ``"sd"`` /
        ``"dl"`` — Schmelcher--Diakonos / Davidchack--Lai stabilising
        transformations (maps only) for systematically reaching unstable points.
        ``"interval"`` — the rigorous Krawczyk branch-and-prune (maps **and**
        flows): it brackets *all* roots in ``region`` with an existence +
        uniqueness certificate per sub-box, so it cannot silently miss a root.
        It requires a bounded ``region`` (the box it certifies over) and an
        interval-extensible right-hand side; a system whose kernel uses an op the
        interval engine cannot enclose raises
        :class:`~tsdynamics.errors.InvalidInputError` (use ``"newton"`` there).
        The ``n_seeds`` / ``lam`` / ``beta`` / ``max_c`` knobs do not apply.
    lam : float
        Step size of the Schmelcher--Diakonos iteration (``method="sd"``).
    beta : float
        Regularisation strength of the Davidchack--Lai iteration
        (``method="dl"``); ``beta=0`` is plain Newton, larger ``beta`` enlarges
        the basin at the cost of more iterations.
    max_c : int, optional
        Cap on the number of stabilising matrices tried (``sd``/``dl``).  The full
        set has ``2^dim · dim!`` members; if capped, a warning is emitted.
    seed : int, default 0
        RNG seed for the multi-start sampling.  All randomness (the box seeds and
        the burn-in orbit's starting state) is drawn from a *local*
        :class:`numpy.random.Generator` seeded with this value, so a given ``seed``
        is fully reproducible regardless of the global ``numpy.random`` state.
        Pass ``seed=None`` for an explicitly non-deterministic search.

        .. versionchanged:: 6.0
            Was ``None``.  Multi-start is a *sampling* method, so an unseeded
            default made the shipped answer vary run to run: measured, three
            identical ``fixed_points(Thomas(), region=[(-5, 5)] * 3,
            n_seeds=60)`` calls found 16, 19 and 17 equilibria.

    Returns
    -------
    FixedPointSet
        A list-like ``CollectionResult`` of :class:`FixedPoint`, sorted by
        coordinate.

    Raises
    ------
    NotImplementedError
        If ``system`` is neither a discrete map nor a continuous flow.
    ValueError
        If ``method`` is not ``"newton"``/``"sd"``/``"dl"``/``"interval"``, or
        ``"sd"``/``"dl"`` is requested for a flow (use ``"newton"`` on
        ``f(x)=0``), or ``"interval"`` is requested without a bounded ``region``.
    InvalidInputError
        If ``method="interval"`` and the system's right-hand side uses an
        operation the interval engine cannot enclose.

    Examples
    --------
    >>> fixed_points(Henon())              # two saddles of the Hénon map
    >>> fixed_points(Lorenz())             # the origin and the two C± equilibria
    >>> fixed_points(Henon(), region=[(-3, 3), (-3, 3)], method="interval")  # rigorous

    References
    ----------
    Schmelcher & Diakonos (1997), *Phys. Rev. Lett.* 78, 4733.
    Davidchack & Lai (1999), *Phys. Rev. E* 60, 6172.
    Krawczyk (1969), *Computing* 4, 187.
    Neumaier (1990), *Interval Methods for Systems of Equations*, CUP.
    """
    reject_data(system, analysis="fixed_points")
    if isinstance(system, DiscreteMap):
        continuous = False
    elif isinstance(system, ContinuousSystem):
        continuous = True
    else:
        raise NotImplementedError(
            f"fixed_points solves f(x)=x for a map and f(x)=0 for a flow; "
            f"{type(system).__name__} is neither. A delay system has no "
            f"finite-dimensional root to solve for, and a stochastic one has no "
            f"fixed point at all — but the *drift* of an SDE, and the flow a delay "
            f"system reduces to at zero delay, do."
            + remedy(
                "ts.analysis.fixed_points(ts.systems.Lorenz())",
                lead="Pass a map or a flow:",
            )
        )

    method = method.lower()
    if method not in ("newton", "sd", "dl", "interval"):
        raise invalid_value(
            "method",
            method,
            options=["newton", "sd", "dl", "interval"],
            hint=(
                "'newton' is multi-start Newton; 'sd'/'dl' add stabilising "
                "transformations that reach unstable orbits of maps; 'interval' "
                "is the rigorous Krawczyk search, which cannot miss a root."
                + remedy(
                    "ts.analysis.fixed_points(system, method='newton')",
                    lead="The default is the one to start from:",
                )
            ),
        )
    if continuous and method in ("sd", "dl"):
        raise ValueError(
            "the 'sd'/'dl' stabilising transformations target unstable orbits of "
            "maps; flow equilibria are found with method='newton' on f(x)=0."
        )

    dim = int(system.dim)
    rng = np.random.default_rng(seed)

    if method == "interval":
        return _interval_fixed_points(system, continuous, region, dim, tol)

    if continuous:
        rhs, jac = _c.flow_fns(system)

        def residual(x: np.ndarray) -> np.ndarray:
            return rhs(x, 0.0)

        def jac_resid(x: np.ndarray) -> np.ndarray:
            return jac(x, 0.0)

        def classify(r: np.ndarray) -> FixedPoint:
            eig = np.linalg.eigvals(jac(r, 0.0))
            return FixedPoint(
                x=r, eigenvalues=eig, stable=bool(np.all(eig.real < 0.0)), continuous=True
            )
    else:
        step, jac = _c.map_fns(system)
        eye = np.eye(dim)

        def residual(x: np.ndarray) -> np.ndarray:
            return cast("np.ndarray", step(x) - x)

        def jac_resid(x: np.ndarray) -> np.ndarray:
            return jac(x) - eye

        def classify(r: np.ndarray) -> FixedPoint:
            eig = np.linalg.eigvals(jac(r))
            return FixedPoint(
                x=r, eigenvalues=eig, stable=bool(np.all(np.abs(eig) < 1.0)), continuous=False
            )

    # One burn-in orbit serves both the automatic box and the on-orbit seeds (it
    # is 2500 RK4 steps of pure Python — sampling it twice was pure waste).
    orbit = _c.sample_orbit_box(system, dim, rng=rng) if region is None else None
    lo, hi = (
        _c.hull_box(orbit, dim, _c.HULL_PAD)
        if orbit is not None
        else _c.resolve_box(system, region, dim, rng)
    )
    seeds = _build_seeds(dim, lo, hi, n_seeds, rng, orbit=orbit)
    c_mats = _stabilising_matrices(method, dim, max_c)

    # The box only *seeds* the search.  An explicit ``region`` is also a hard
    # search domain, so converged roots outside it are clipped; but when the box
    # is the auto burn-in bounding box (``region is None``) it must not filter
    # results — a flow's equilibria are saddles the on-attractor orbit never
    # visits (e.g. the Lorenz origin and the C± centres sit outside the chaotic
    # attractor's hull), so clipping to that box would silently drop genuine
    # equilibria (the FIX-FPFLOW defect).
    bounds = (lo, hi) if region is not None else None

    roots = _c.solve_roots(
        residual,
        jac_resid,
        dim,
        seeds,
        method=method,
        c_mats=c_mats,
        lam=lam,
        beta=beta,
        tol=tol,
        max_iter=max_iter,
        dedup_tol=dedup_tol,
        bounds=bounds,
    )
    out = [classify(r) for r in roots]
    out.sort(key=lambda fp: tuple(fp.x))
    return FixedPointSet(
        items=tuple(out),
        meta=_build_meta(
            system,
            analysis="fixed_points",
            method=method,
            variables=_variable_names(system),
        ),
    )


def _interval_fixed_points(
    system: Any,
    continuous: bool,
    region: Any,
    dim: int,
    tol: float,
) -> FixedPointSet:
    r"""Rigorous Krawczyk branch-and-prune over the search box (``method="interval"``).

    Brackets every root of the residual inside ``region`` with an existence +
    uniqueness certificate per sub-box, then classifies each by the analytic
    Jacobian spectrum (the same stability convention as ``method="newton"``).

    A bounded ``region`` is required — the interval method certifies completeness
    *within a box*, so an unbounded search has no meaning (and the auto burn-in
    box would not enclose off-attractor equilibria).  An interval-extensible RHS
    is required too; a kernel the interval engine cannot enclose raises
    :class:`~tsdynamics.errors.InvalidInputError` at build time.
    """
    from . import _interval as _iv

    if region is None:
        raise InvalidParameterError(
            "method='interval' needs a bounded 'region' (the box it certifies "
            "completeness over); pass one (lo, hi) bound per state component, "
            "e.g. region=[(-3, 3), (-3, 3)] — or a Box/Ball/Grid."
        )
    lo, hi = _c.resolve_box(system, region, dim, rng=np.random.default_rng(0))

    if continuous:
        resid_jac = _iv.flow_interval_fn(system)
        rhs, jac = _c.flow_fns(system)

        def residual_float(x: np.ndarray) -> np.ndarray:
            return np.asarray(rhs(x, 0.0), dtype=float)

        def jac_float(x: np.ndarray) -> np.ndarray:
            return jac(x, 0.0)

        def classify(r: np.ndarray) -> FixedPoint:
            eig = np.linalg.eigvals(jac(r, 0.0))
            return FixedPoint(
                x=r, eigenvalues=eig, stable=bool(np.all(eig.real < 0.0)), continuous=True
            )
    else:
        resid_jac = _iv.map_interval_fn(system)
        step, jac = _c.map_fns(system)
        eye = np.eye(dim)

        def residual_float(x: np.ndarray) -> np.ndarray:
            return cast("np.ndarray", step(x) - x)

        def jac_float(x: np.ndarray) -> np.ndarray:
            return jac(x) - eye

        def classify(r: np.ndarray) -> FixedPoint:
            eig = np.linalg.eigvals(jac(r))
            return FixedPoint(
                x=r, eigenvalues=eig, stable=bool(np.all(np.abs(eig) < 1.0)), continuous=False
            )

    roots = _iv.krawczyk_roots(lo, hi, resid_jac, residual_float, jac_float, tol=tol)
    # Deduplicate (a root straddling a bisection cut may be certified twice).
    kept = _c.dedup_points(roots, max(1e-6, 100.0 * tol))
    out = [classify(r) for r in kept]
    out.sort(key=lambda fp: tuple(fp.x))
    return FixedPointSet(
        items=tuple(out),
        meta=_build_meta(
            system,
            analysis="fixed_points",
            method="interval",
            variables=_variable_names(system),
        ),
    )


def _build_seeds(
    dim: int,
    lo: np.ndarray,
    hi: np.ndarray,
    n_seeds: int,
    rng: np.random.Generator,
    *,
    orbit: np.ndarray | None,
) -> np.ndarray:
    r"""Random box seeds augmented with a subsample of an on-orbit burn-in.

    ``orbit is None`` means the caller supplied an explicit ``region``: that box
    is the search domain, so it is seeded uniformly and nothing is added outside
    it.

    ``orbit`` given means the box was derived automatically from that burn-in
    orbit, and it is only a *seeding* aid (roots outside it are kept — see
    :func:`fixed_points`).  Two scales are then drawn, because a flow's
    equilibria sit in two different places:

    * the **hull** — the orbit's bare bounding box, seeded with the full
      ``n_seeds``.  This is where equilibria cluster (Thomas has 27 inside its
      attractor's hull, Lorenz 3, Chua 3), and density is the binding constraint
      there, so it gets the larger share.
    * the **padded** box — the same hull grown by :data:`~_common.HULL_PAD`
      spans, seeded with :data:`~_common.HULL_PAD_FRACTION` of ``n_seeds`` —
      reaches the saddles that sit outside the attractor.  Rossler's second
      equilibrium is at ``(5.69, -28.47, 28.47)`` while its attractor never
      leaves ``|y| < 12``; against the pre-v6 2.0-time-unit hull
      ``fixed_points(Rossler())`` returned 1 of 2 equilibria for **every** seed,
      with no warning.

    Padding costs volume as ``(1 + 2 * pad) ** dim``, so it is bought sparingly:
    seeding the *bare* hull rather than a heavily padded third box is what takes
    Thomas from 19/23/19 recovered equilibria to 27/27/27 (see
    :data:`~_common.HULL_PAD`).
    """
    seeds = rng.uniform(lo, hi, size=(int(n_seeds), dim))
    if orbit is None or not orbit.size:
        return seeds
    tight_lo, tight_hi = _c.hull_box(orbit, dim, 0.0)
    n_pad = max(1, int(int(n_seeds) * _c.HULL_PAD_FRACTION))
    return np.vstack(
        [
            rng.uniform(tight_lo, tight_hi, size=(int(n_seeds), dim)),
            seeds[:n_pad],
            orbit[:: max(1, len(orbit) // 20)],
        ]
    )


def _stabilising_matrices(method: str, dim: int, max_c: int | None) -> list[np.ndarray]:
    """Return the ``C`` set for SD/DL (empty for Newton), with a truncation warning."""
    if method == "newton":
        return []
    full = _c._signed_permutation_count(dim)
    mats = _c.signed_permutation_matrices(dim, max_c)
    if max_c is not None and full > max_c:
        import warnings

        warnings.warn(
            f"using {max_c} of {full} stabilising matrices for dim={dim}; "
            f"some unstable orbits may be missed (raise max_c to search more).",
            stacklevel=3,
        )
    return mats


# ── visualization helpers (shared with periodic.py) ──────────────────────────


def _variable_names(system: Any) -> tuple[str, ...] | None:
    """Return a system's declared component names, or ``None``.

    Recorded into the result's provenance so the plot can label its axes ``x`` /
    ``y`` rather than ``$x_0$`` / ``$x_1$`` — the difference between a figure you
    can publish and one you must relabel by hand.
    """
    names = getattr(system, "variables", None)
    if isinstance(names, (list, tuple)) and names:
        return tuple(str(n) for n in names)
    return None


#: Marker *diameter* (pt) for a fixed-point / equilibrium marker.
#:
#: These emitters used to speak matplotlib's ``s`` (a marker **area**, pt²) —
#: ``{"s": 40.0}``.  ``s`` is no longer a canonical style key, and the canonical
#: ``markersize`` is a **diameter**, so the value must be ``sqrt(40) ~= 6.3``, not
#: ``40``.  Emitting 40 made the renderer square it to ``s=1600``: markers 40x too
#: large linearly, 1600x by area — the "three enormous discs" a Lorenz equilibrium
#: plot used to be.
_FIXED_POINT_MARKERSIZE: float = 6.3

#: Colour for a **stable** fixed point / equilibrium (Okabe-Ito blue).
_STABLE_COLOR = "#0173b2"

#: Colour for an **unstable** one (Okabe-Ito vermilion).
_UNSTABLE_COLOR = "#d55e00"


#: Above this many points a set's plot annotates nothing unless asked.
#:
#: Annotation is worth its ink only while each label is readable.  Thomas' attractor
#: has 27 equilibria; 27 eigenvalue labels over a 5-inch axes is a grey smear that
#: hides the very markers it describes.  Lorenz (3) and Rössler (2) are comfortably
#: under, which is why ``annotate=None`` means *auto* rather than *on*.
_ANNOTATE_AUTO_MAX = 8


def _resolve_components(
    meta: Mapping[str, Any] | None,
    dim: int,
    components: Sequence[int | str] | None,
) -> tuple[int, int]:
    """Resolve a requested projection plane to a pair of state-vector indices.

    A fixed-point plot of a 3-D flow has to choose two of the three coordinates, and
    silently always choosing ``(0, 1)`` is a real loss: Lorenz's two nontrivial
    equilibria are symmetric under ``(x, y) -> (-x, -y)`` and differ visibly in
    ``z``, so the ``(x, z)`` projection says something the default one does not.

    Accepts indices (``(0, 2)``) or the component *names* a system declares
    (``("x", "z")``), resolved against ``meta["variables"]`` — the same provenance
    :func:`~tsdynamics.analysis._plotbuilder.axis_labels` reads, so a named
    projection and its axis labels can never disagree.

    Parameters
    ----------
    meta : mapping, optional
        The result's provenance.
    dim : int
        Length of the state vectors being plotted.
    components : sequence, optional
        Two indices or names.  ``None`` gives ``(0, 1)`` (or ``(0, 0)`` in 1-D,
        which the callers draw against a zero baseline).

    Returns
    -------
    tuple of int

    Raises
    ------
    InvalidParameterError
        If the request is not a pair, names a component the system does not
        declare, or indexes past the state vector — all of which would otherwise
        surface as a silently wrong projection.
    """
    if components is None:
        return (0, 1 if dim >= 2 else 0)
    items = list(components)
    if len(items) != 2:
        raise InvalidParameterError(
            f"components must name exactly two coordinates for a 2-D projection, "
            f"got {len(items)}: {components!r}."
        )
    names_raw = meta.get("variables") if meta else None
    names = list(names_raw) if isinstance(names_raw, (list, tuple)) else []
    out: list[int] = []
    for item in items:
        if isinstance(item, str):
            if item not in names:
                raise InvalidParameterError(
                    f"unknown component {item!r}; this result declares "
                    f"{names or 'no component names'}. Use an integer index instead."
                )
            out.append(names.index(item))
        else:
            out.append(int(item))
    for idx in out:
        if not 0 <= idx < dim:
            raise InvalidParameterError(
                f"component index {idx} is out of range for a {dim}-D state vector."
            )
    return (out[0], out[1])


def _leading_eigenvalue(eigenvalues: np.ndarray, *, continuous: bool) -> complex | None:
    """Return the eigenvalue that decides stability, or ``None`` for an empty spectrum.

    For a **flow** that is the largest real part (the slowest-decaying / fastest-
    growing direction); for a **map** the largest modulus.  The two conventions are
    the two stability criteria (``Re λ < 0`` versus ``|λ| < 1``), so reading the
    wrong one off a plot inverts its meaning.
    """
    eig = np.asarray(eigenvalues).ravel()
    if eig.size == 0:
        return None
    eig = eig.astype(complex)
    order = eig.real if continuous else np.abs(eig)
    return complex(eig[int(np.argmax(order))])


def _eigenvalue_label(value: complex) -> str:
    r"""Format a leading eigenvalue compactly enough to sit next to a marker.

    A complex pair collapses to ``λ=a±bi`` rather than printing the conjugate
    twice, and a real eigenvalue drops the imaginary part entirely — a label with a
    dead ``+0.0000i`` on it is a label people stop reading.
    """
    re = float(value.real)
    im = float(value.imag)
    if abs(im) <= 1e-12 * max(1.0, abs(re)):
        return rf"$\lambda={re:+.3g}$"
    return rf"$\lambda={re:+.3g}\pm{abs(im):.3g}i$"


def _eigenvalue_annotations(
    placed: Sequence[tuple[float, float, np.ndarray]],
    *,
    continuous: bool,
    enabled: bool,
) -> tuple[list[Any], tuple[float, float] | None, tuple[float, float] | None]:
    """Build one leading-eigenvalue text label per plotted marker.

    The offset is the part worth explaining.  A label written *at* the marker's
    coordinates sits on top of the marker, so it is nudged along ``+x`` by a
    fraction of the plotted spread — a data-coordinate offset, because it must
    survive every backend, and plotly drops an annotation's style entirely (so an
    ``ha``/``va`` nudge would work in matplotlib and silently not in plotly).  When
    the spread is zero — a single point, the common case for
    :meth:`FixedPoint.__plot_spec__` — the offset falls back to a fraction of the
    point's own magnitude, and finally to a bare constant at the origin.

    Labels are pushed **outward**, away from the centre of the point cloud, and the
    returned x limits are widened to hold them.  Both halves are needed: pointing
    every label the same way runs the right-most one off the canvas (Lorenz's
    ``C+`` equilibrium did exactly that), while pointing them all inward makes the
    two symmetric equilibria of Lorenz's ``(x, z)`` projection — same height,
    opposite sides — write their labels straight through each other.

    Widening the limits is safe for the overlay path:
    :meth:`~tsdynamics.analysis._result.AnalysisResult.overlay_on` merges only
    layers and annotations onto the host spec and discards the overlay's own axes,
    so these limits apply exactly when this spec is drawn on its own.

    Returns
    -------
    (list, tuple or None, tuple or None)
        The annotations, and the x / y limits that keep them on the canvas.
    """
    if not enabled or not placed:
        return [], None, None
    from .. import _plotbuilder as pb

    xs = [p[0] for p in placed]
    ys = [p[1] for p in placed]
    spread = max(max(xs) - min(xs), max(ys) - min(ys))
    if spread <= 0.0:
        spread = max(abs(xs[0]), abs(ys[0]), 1.0)
    dx = 0.045 * spread
    midx = 0.5 * (min(xs) + max(xs))

    out = []
    for px, py, eig in placed:
        value = _leading_eigenvalue(eig, continuous=continuous)
        if value is None:
            continue
        left = px < midx
        out.append(
            pb.text(
                px - dx if left else px + dx,
                py,
                _eigenvalue_label(value),
                style={"fontsize": 8, "ha": "right" if left else "left", "va": "center"},
            )
        )
    if not out:
        return [], None, None
    # Room for the widest label (~22 glyphs at 8 pt) on either side, plus a little
    # air above and below so a marker never sits on the frame.
    xlim = (min(xs) - 0.45 * spread, max(xs) + 0.45 * spread)
    ylim = (min(ys) - 0.12 * spread, max(ys) + 0.12 * spread)
    return out, xlim, ylim


def _fixed_point_style(stable: bool) -> dict[str, Any]:
    """Backend-neutral marker style distinguishing a stable from an unstable point.

    Stability is the *meaning* of a fixed-point plot, so it is encoded twice —
    redundantly and deliberately:

    - ``filled``: a stable point is a filled disc, an unstable one an open circle
      (the textbook convention).
    - ``color``: a colourblind-safe blue / vermilion pair, so the distinction
      survives a backend that does not honor ``filled`` (three.js) and a
      greyscale print.

    Both keys are canonical members of
    :data:`~tsdynamics.viz.style.STYLE_KEYS`, so they survive
    :func:`~tsdynamics.viz.style.normalize_style` and the two styles are
    genuinely distinguishable after normalization — they were not: the old style
    dicts differed *only* in ``filled``, which was not a style key and was
    dropped, making ``normalize_style(stable) == normalize_style(unstable)``.
    """
    return {
        "marker": "circle",
        "filled": bool(stable),
        "color": _STABLE_COLOR if stable else _UNSTABLE_COLOR,
        "markersize": _FIXED_POINT_MARKERSIZE,
    }


def _eigenvalue_plane_spec(
    eigenvalues: np.ndarray,
    *,
    continuous: bool,
    title: str,
    meta: dict[str, Any],
    kind: str | None = None,
    trivial_index: int | None = None,
) -> Any:
    r"""Build an ``EIGENVALUE_PLANE`` :class:`PlotSpec` for a spectrum.

    Scatters ``eigenvalues`` in the complex plane (real part on ``x``, imaginary
    part on ``y``) against the stability boundary: the unit circle ``|λ| = 1`` for
    a map (``continuous=False``), or the imaginary axis ``Re λ = 0`` for a flow
    (``continuous=True``).  When ``trivial_index`` is given (a flow's trivial
    Floquet multiplier ``≈ 1``), that eigenvalue is split into its own
    distinctly-styled layer.  The :mod:`tsdynamics.viz.spec` import is lazy.
    """
    from .. import _plotbuilder as pb

    eig = np.asarray(eigenvalues).ravel().astype(complex)
    n = eig.size

    layers = []
    annotations = []

    # The stability-boundary reference geometry.
    if continuous:
        # Deliberately *untitled*.  A ``vline``'s text is drawn rotated against the
        # top of the axes, which on this plot is exactly where a complex conjugate
        # pair near the imaginary axis sits — on Lorenz the label ran straight
        # through the eigenvalue at ``+0.094 + 10.2i``.  The criterion goes in the
        # title instead, where nothing can collide with it, and the axis is already
        # labelled ``Re λ``, so the line needs no gloss.
        annotations.append(pb.vline(0.0, style={"color": "gray", "alpha": 0.6}))
    else:
        theta = np.linspace(0.0, 2.0 * np.pi, 200)
        layers.append(
            pb.line(
                np.cos(theta),
                np.sin(theta),
                label=r"$|\lambda| = 1$",
                style={"color": "gray", "lw": 1.0, "alpha": 0.6},
            )
        )

    if n:
        keep = np.ones(n, dtype=bool)
        if trivial_index is not None and 0 <= trivial_index < n:
            keep[trivial_index] = False
            tv = eig[trivial_index]
            layers.append(
                pb.scatter(
                    np.array([tv.real]),
                    np.array([tv.imag]),
                    label=r"trivial $\mu \approx 1$",
                    style={"marker": "x", "markersize": 7.75},
                )
            )
        rest = eig[keep]
        if rest.size:
            layers.append(
                pb.scatter(
                    rest.real.astype(float),
                    rest.imag.astype(float),
                    label="multipliers" if not continuous else "eigenvalues",
                    style={"marker": "circle", "markersize": _FIXED_POINT_MARKERSIZE},
                )
            )

    return pb.spec(
        kind,
        "eigenvalue_plane",
        layers=layers,
        aspect="equal",
        xlabel=r"$\mathrm{Re}\,\lambda$",
        ylabel=r"$\mathrm{Im}\,\lambda$",
        # The stability criterion belongs in the title: it is what the reference
        # geometry *means*, and the title is the one place on this plot where no
        # eigenvalue can land on top of it.
        title=(
            f"{title} — stable if "
            + (r"$\mathrm{Re}\,\lambda<0$" if continuous else r"$|\lambda|<1$")
        ),
        legend=len(layers) > 1,
        annotations=annotations,
        meta=meta,
    )


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
