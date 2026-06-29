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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from tsdynamics.families import ContinuousSystem, DiscreteMap

from .._result import AnalysisResult, CollectionResult
from . import _common as _c

__all__ = ["FixedPoint", "FixedPointSet", "fixed_points"]


@dataclass(frozen=True)
class FixedPoint(AnalysisResult):
    """A fixed point (map) or equilibrium (flow) with its linear stability data.

    An :class:`~tsdynamics.analysis._result.AnalysisResult`, so it carries
    ``.meta`` / ``.summary()`` / ``.to_dict()`` / the ``.plot`` seam alongside its
    point and stability data.

    Attributes
    ----------
    x : ndarray
        The point, shape ``(dim,)``.
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

    x: np.ndarray = field(default_factory=lambda: np.empty(0), compare=False)
    eigenvalues: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False, compare=False)
    stable: bool = False
    continuous: bool = False

    def __repr__(self) -> str:  # noqa: D105
        kind = "stable" if self.stable else "unstable"
        if self.continuous:
            gauge = f"Re(λ)max={self.eigenvalues.real.max():+.4f}"
        else:
            gauge = f"|λ|max={np.abs(self.eigenvalues).max():.4f}"
        return f"FixedPoint({np.round(self.x, 6)}, {kind}, {gauge})"

    def to_plot_spec(self, kind: str | None = None) -> Any:
        r"""Describe this fixed point as a backend-agnostic :class:`PlotSpec`.

        Builds a ``FIXED_POINTS_OVERLAY``: a single ``SCATTER`` point at the first
        two coordinates of :attr:`x`, styled by stability (a filled marker for a
        stable point, an open marker for an unstable one).  Designed to be drawn
        *over* a phase portrait via :meth:`AnalysisResult.overlay_on`, which keeps
        the host layers first.  For the eigenvalue/multiplier picture use
        :meth:`eigenvalue_plane`.  The :mod:`tsdynamics.viz.spec` import is lazy,
        so building a spec never pulls a plotting library.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind (a :class:`~tsdynamics.viz.spec.PlotKind`
            value).  ``None`` uses ``FIXED_POINTS_OVERLAY``.

        Returns
        -------
        PlotSpec
        """
        from .. import _plotbuilder as pb

        x = np.asarray(self.x, dtype=float).ravel()
        label = "stable" if self.stable else "unstable"
        style = _fixed_point_style(self.stable)
        if x.size >= 2:
            layer = pb.scatter(np.array([x[0]]), np.array([x[1]]), label=label, style=style)
            ylabel = "$x_1$"
        else:
            layer = pb.scatter(
                np.array([0.0]), np.array([x[0] if x.size else 0.0]), label=label, style=style
            )
            ylabel = "$x$"
        return pb.spec(
            kind,
            "fixed_points_overlay",
            layers=[layer],
            aspect="equal",
            xlabel="$x_0$",
            ylabel=ylabel,
            title=f"{label} fixed point",
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
    """The set of fixed points / equilibria found, behaving like a ``list``.

    A :class:`~tsdynamics.analysis._result.CollectionResult`: iterate it, index it
    (``fps[0]`` is a :class:`FixedPoint`), take its ``len``, and read
    :attr:`stable` / :attr:`unstable` sublists — while it carries ``.meta`` /
    ``.summary()`` / ``.to_frame()`` / the ``.plot`` seam.
    """

    @property
    def stable(self) -> list[FixedPoint]:
        """The stable fixed points / equilibria in the set."""
        return [fp for fp in self.items if fp.stable]

    @property
    def unstable(self) -> list[FixedPoint]:
        """The unstable fixed points / equilibria in the set."""
        return [fp for fp in self.items if not fp.stable]

    def to_plot_spec(self, kind: str | None = None) -> Any:
        r"""Describe the whole set as one ``FIXED_POINTS_OVERLAY`` :class:`PlotSpec`.

        Draws the stable and the unstable fixed points as two separately styled
        ``SCATTER`` layers (filled vs open markers) at the first two coordinates,
        so the set reads at a glance — the overlay a phase portrait hosts via
        :meth:`AnalysisResult.overlay_on` (host layers first).  A 1-D set scatters
        against a zero baseline.  An empty set yields a valid layer-less spec.
        For the spectrum picture use :meth:`eigenvalue_plane`.  The
        :mod:`tsdynamics.viz.spec` import is lazy.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind.  ``None`` uses ``FIXED_POINTS_OVERLAY``.

        Returns
        -------
        PlotSpec
        """
        from .. import _plotbuilder as pb

        layers = []
        for stable, label in ((True, "stable"), (False, "unstable")):
            pts = [
                np.asarray(fp.x, dtype=float).ravel() for fp in self.items if fp.stable is stable
            ]
            if not pts:
                continue
            dim = min(p.size for p in pts)
            arr = np.asarray([p[:dim] for p in pts], dtype=float)
            xs = arr[:, 0]
            ys = arr[:, 1] if dim >= 2 else np.zeros(arr.shape[0])
            layers.append(pb.scatter(xs, ys, label=label, style=_fixed_point_style(stable)))
        return pb.spec(
            kind,
            "fixed_points_overlay",
            layers=layers,
            aspect="equal",
            xlabel="$x_0$",
            ylabel="$x_1$",
            title=f"fixed points ({len(self.items)} found)",
            legend=len(layers) > 1,
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
    *,
    region: Any = None,
    n_seeds: int = 200,
    tol: float = 1e-12,
    max_iter: int = 60,
    dedup_tol: float = 1e-6,
    method: str = "newton",
    lam: float = 0.05,
    beta: float = 1.0,
    max_c: int | None = None,
    seed: int | None = None,
) -> FixedPointSet:
    r"""
    Find fixed points of a map (``f(x) = x``) or equilibria of a flow (``f(x) = 0``).

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
        Search region; defaults to a burn-in orbit's bounding box padded by 50 %,
        or ``[-2, 2]^dim`` if the orbit diverges.
    n_seeds : int
        Random seeds (orbit points are added on top).
    tol : float
        Residual tolerance (``‖f(x) − x‖`` for maps, ``‖f(x)‖`` for flows).
    max_iter : int
        Root-finding iterations per seed.
    dedup_tol : float
        Distance below which two roots are merged.
    method : {"newton", "sd", "dl"}
        ``"newton"`` (default) — Newton on the exact Jacobian.  ``"sd"`` /
        ``"dl"`` — Schmelcher--Diakonos / Davidchack--Lai stabilising
        transformations (maps only) for systematically reaching unstable points.
    lam : float
        Step size of the Schmelcher--Diakonos iteration (``method="sd"``).
    beta : float
        Regularisation strength of the Davidchack--Lai iteration
        (``method="dl"``); ``beta=0`` is plain Newton, larger ``beta`` enlarges
        the basin at the cost of more iterations.
    max_c : int, optional
        Cap on the number of stabilising matrices tried (``sd``/``dl``).  The full
        set has ``2^dim · dim!`` members; if capped, a warning is emitted.
    seed : int, optional
        RNG seed for the multi-start sampling.  All randomness (the box seeds and
        the burn-in orbit's starting state) is drawn from a *local*
        :class:`numpy.random.Generator` seeded with this value, so a given ``seed``
        is fully reproducible regardless of the global ``numpy.random`` state.
        ``seed=None`` (the default) is non-deterministic — the sampling varies from
        call to call.

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
        If ``method`` is not ``"newton"``/``"sd"``/``"dl"``, or ``"sd"``/``"dl"``
        is requested for a flow (use ``"newton"`` on ``f(x)=0``).

    Examples
    --------
    >>> fixed_points(Henon())              # two saddles of the Hénon map
    >>> fixed_points(Lorenz())             # the origin and the two C± equilibria

    References
    ----------
    Schmelcher & Diakonos (1997), *Phys. Rev. Lett.* 78, 4733.
    Davidchack & Lai (1999), *Phys. Rev. E* 60, 6172.
    """
    if isinstance(system, DiscreteMap):
        continuous = False
    elif isinstance(system, ContinuousSystem):
        continuous = True
    else:
        raise NotImplementedError(
            f"fixed_points supports discrete maps and continuous flows, not "
            f"{type(system).__name__}."
        )

    method = method.lower()
    if method not in ("newton", "sd", "dl"):
        raise ValueError(f"method must be 'newton', 'sd', or 'dl', got {method!r}.")
    if continuous and method != "newton":
        raise ValueError(
            "the 'sd'/'dl' stabilising transformations target unstable orbits of "
            "maps; flow equilibria are found with method='newton' on f(x)=0."
        )

    dim = int(system.dim)
    rng = np.random.default_rng(seed)

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

    lo, hi = _c.resolve_box(system, region, dim, rng)
    seeds = _build_seeds(system, dim, lo, hi, n_seeds, rng)
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
        meta=AnalysisResult.build_meta(system, analysis="fixed_points", method=method),
    )


def _build_seeds(
    system: Any,
    dim: int,
    lo: np.ndarray,
    hi: np.ndarray,
    n_seeds: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Random box seeds augmented with a subsample of an on-orbit burn-in."""
    seeds = rng.uniform(lo, hi, size=(int(n_seeds), dim))
    orbit = _c.sample_orbit_box(system, dim, rng=rng)
    if orbit.size:
        seeds = np.vstack([seeds, orbit[:: max(1, len(orbit) // 20)]])
    return seeds


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


def _fixed_point_style(stable: bool) -> dict[str, Any]:
    """Backend-neutral marker style distinguishing a stable from an unstable point.

    A stable point gets a filled disc; an unstable one an open circle — the
    convention a renderer maps to its own marker idioms (unknown keys are
    ignored, per the :class:`~tsdynamics.viz.spec.Layer` contract).
    """
    if stable:
        return {"marker": "o", "filled": True, "s": 40.0}
    return {"marker": "o", "filled": False, "s": 40.0}


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
        annotations.append(pb.vline(0.0, text=r"$\mathrm{Re}\,\lambda = 0$"))
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
                    style={"marker": "x", "s": 60.0},
                )
            )
        rest = eig[keep]
        if rest.size:
            layers.append(
                pb.scatter(
                    rest.real.astype(float),
                    rest.imag.astype(float),
                    label="multipliers" if not continuous else "eigenvalues",
                    style={"marker": "o", "s": 40.0},
                )
            )

    return pb.spec(
        kind,
        "eigenvalue_plane",
        layers=layers,
        aspect="equal",
        xlabel=r"$\mathrm{Re}\,\lambda$",
        ylabel=r"$\mathrm{Im}\,\lambda$",
        title=title,
        legend=len(layers) > 1,
        annotations=annotations,
        meta=meta,
    )


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
