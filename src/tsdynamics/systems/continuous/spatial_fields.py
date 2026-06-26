"""Spatially-extended 2-D field systems (method-of-lines PDEs).

These are reaction-diffusion / pattern-formation PDEs discretised by the method
of lines on a **periodic** ``N x N`` grid: each spatial cell becomes one ODE, the
state vector is the flattened field(s), and spatial derivatives are compact
finite-difference stencils with modular (wrap-around) index arithmetic.  Each
system declares :attr:`~tsdynamics.families.base.SystemBase._field_shape` (the
grid) so the visualization layer's ``kind="field"`` recipe (stream
VIZ-SPATIAL-FIELD) reshapes every per-time state vector back to the grid and
plays it as an evolving **2-D heatmap movie**.

The catalogue
-------------
- :class:`GrayScott` — the Gray–Scott two-species reaction-diffusion model
  (Pearson 1993): a fast-diffusing substrate ``u`` and a slow activator ``v``
  forming spots / worms / labyrinths.  The plotted field is the activator ``v``.
- :class:`SwiftHohenberg` — the Swift–Hohenberg equation (Swift & Hohenberg
  1977), the canonical pattern-forming PDE: a single field relaxing onto a
  stripe / labyrinth pattern of a preferred wavelength.

Why Swift–Hohenberg rather than a 2-D Kuramoto–Sivashinsky?
-----------------------------------------------------------
A 2-D KS ``u_t = -∇²u - ∇⁴u - ½|∇u|²`` is **not** mean-conserving — unlike its
1-D form ``-½(u²)_x``, the 2-D advective term ``½|∇u|² ≥ 0`` injects a positive
spatial-mean source, so the field's DC offset drifts without bound and a
fixed-colormap movie washes out.  Projecting that drift out (subtracting the
spatial mean of the nonlinear term) restores boundedness but couples **every**
grid cell to a global sum, which makes the symbolic right-hand side a dense
``O(N⁴)`` tape that the engine cannot lower at any useful grid size.  Swift–
Hohenberg is local (compact stencils only), unconditionally bounded by its cubic
saturation, and forms equally vivid evolving 2-D structure — so it is the robust
2-D field example here.  (The shipped 1-D
:class:`~tsdynamics.systems.continuous.chaotic_attractors.KuramotoSivashinsky`
remains the spatiotemporally-chaotic *1-D* field example — its profile plays as a
travelling-wave line movie.)

References
----------
.. [1] Pearson, J. E. (1993). "Complex Patterns in a Simple System." *Science*,
   261(5118), 189-192.
.. [2] Swift, J. & Hohenberg, P. C. (1977). "Hydrodynamic fluctuations at the
   convective instability." *Physical Review A*, 15(1), 319-328.
"""

from __future__ import annotations

import numpy as np

from tsdynamics.errors import InvalidParameterError
from tsdynamics.families import ContinuousSystem

__all__ = ["GrayScott", "SwiftHohenberg"]


class GrayScott(ContinuousSystem):
    r"""Gray–Scott reaction-diffusion on a periodic ``N x N`` grid (method of lines).

    Two scalar fields — a substrate ``u`` and an activator ``v`` — react and
    diffuse on a doubly-periodic square lattice [1]_:

    .. code-block:: text

        u_t = Du ∇²u - u v² + F (1 - u)
        v_t = Dv ∇²v + u v² - (F + k) v

    The reaction ``u v²`` consumes ``u`` to make ``v``; the feed ``F`` replenishes
    ``u`` and the kill ``F + k`` removes ``v``.  Depending on ``(F, k)`` the field
    settles into spots, stripes, worms or self-replicating "mitosis" patterns.

    State layout
        The state vector is the two fields flattened and concatenated:
        ``[u.ravel(), v.ravel()]`` (length ``2 N²``).  :attr:`field_labels`
        ``("u", "v")`` name the two blocks and :attr:`_field_shape` is the grid
        ``(N, N)``; the ``kind="field"`` plot recipe defaults to the **activator
        ``v``** (select ``u`` with ``components="u"``).

    Discretisation
        The Laplacian is the 5-point periodic stencil with **unit grid spacing**
        (``h = 1``, the Pearson convention): ``∇²u_{ij} ≈ u_{i+1,j} + u_{i-1,j} +
        u_{i,j+1} + u_{i,j-1} - 4 u_{ij}``.  The effective domain is therefore
        ``L = N`` cells wide.

    Parameters
    ----------
    N : int, optional
        Grid points per side.  Structural — changing it recompiles.  Default 48
        (state dimension ``2 · 48² = 4608``).
    Du, Dv : float, optional
        Substrate / activator diffusion coefficients.  Default ``0.16`` / ``0.08``.
    F : float, optional
        Feed rate.  Default ``0.06``.
    k : float, optional
        Kill rate.  Default ``0.062`` (spots / worms regime).

    References
    ----------
    .. [1] Pearson, J. E. (1993). "Complex Patterns in a Simple System."
       *Science*, 261(5118), 189-192.
    """

    # N drives the symbolic loop length (one equation per cell per field), so it
    # is baked into the tape; the rate / diffusion coefficients are runtime knobs.
    _structural_params = frozenset({"N"})

    params = {"N": 48, "Du": 0.16, "Dv": 0.08, "F": 0.06, "k": 0.062}

    #: The grid one field block lives on (so a bare Trajectory plays as a field
    #: movie of one ``N x N`` block).
    _field_shape = (48, 48)
    #: The two field blocks packed into the state vector, in state order.
    field_labels = ("u", "v")

    def __init__(
        self,
        N: int | None = None,
        Du: float | None = None,
        Dv: float | None = None,
        F: float | None = None,
        k: float | None = None,
        *,
        params: dict[str, float] | None = None,
        ic=None,
    ):
        p = dict(type(self).params)
        if params:
            unknown = set(params) - set(p)
            if unknown:
                raise InvalidParameterError(
                    f"GrayScott: unknown parameter(s) {sorted(unknown)}. Declared: {sorted(p)}"
                )
            p.update(params)
        if N is not None:
            p["N"] = int(N)
        if Du is not None:
            p["Du"] = float(Du)
        if Dv is not None:
            p["Dv"] = float(Dv)
        if F is not None:
            p["F"] = float(F)
        if k is not None:
            p["k"] = float(k)
        n_val = int(p["N"])
        if n_val < 3:
            raise ValueError("GrayScott requires N >= 3.")
        if ic is None:
            ic = self._nucleation_ic(n_val)
        # ``field_shape=(N, N)`` so a custom-N instance is self-describing (the
        # base resolves it onto ``traj.meta["field_shape"]``).
        super().__init__(dim=2 * n_val * n_val, params=p, ic=ic, field_shape=(n_val, n_val))

    @staticmethod
    def _nucleation_ic(N: int, *, seed: int = 0) -> np.ndarray:
        """Build the seeded nucleation IC: ``u ≈ 1``, ``v ≈ 0``, a central square seed.

        The whole grid starts at the trivial fixed point ``(u, v) = (1, 0)`` with a
        small central square perturbed to ``(0.5, 0.25)`` (plus light noise) to
        nucleate a growing pattern, the standard Gray–Scott seeding (Pearson 1993).
        """
        rng = np.random.default_rng(seed)
        u = np.ones((N, N), dtype=float)
        v = np.zeros((N, N), dtype=float)
        c = N // 2
        r = max(3, N // 8)
        u[c - r : c + r, c - r : c + r] = 0.5
        v[c - r : c + r, c - r : c + r] = 0.25
        u += 0.02 * rng.standard_normal((N, N))
        v += 0.02 * rng.standard_normal((N, N))
        return np.concatenate([u.ravel(), v.ravel()])

    @staticmethod
    def _equations(Y, t, *, N, Du, Dv, F, k):
        # State layout: [u (N*N), v (N*N)]; unit grid spacing (h = 1).
        def uidx(r, c):
            return (r % N) * N + (c % N)

        def vidx(r, c):
            return N * N + (r % N) * N + (c % N)

        rhs = []
        # u-block.
        for r in range(N):
            for c in range(N):
                u0 = Y(uidx(r, c))
                v0 = Y(vidx(r, c))
                lap_u = (
                    Y(uidx(r + 1, c))
                    + Y(uidx(r - 1, c))
                    + Y(uidx(r, c + 1))
                    + Y(uidx(r, c - 1))
                    - 4 * u0
                )
                rhs.append(Du * lap_u - u0 * v0 * v0 + F * (1 - u0))
        # v-block.
        for r in range(N):
            for c in range(N):
                u0 = Y(uidx(r, c))
                v0 = Y(vidx(r, c))
                lap_v = (
                    Y(vidx(r + 1, c))
                    + Y(vidx(r - 1, c))
                    + Y(vidx(r, c + 1))
                    + Y(vidx(r, c - 1))
                    - 4 * v0
                )
                rhs.append(Dv * lap_v + u0 * v0 * v0 - (F + k) * v0)
        return rhs


class SwiftHohenberg(ContinuousSystem):
    r"""Swift–Hohenberg pattern formation on a periodic ``N x N`` grid (method of lines).

    A single scalar field relaxing onto a striped / labyrinthine pattern of a
    preferred wavelength [1]_:

    .. code-block:: text

        u_t = r u - (1 + ∇²)² u - u³

    The linear operator ``-(1 + ∇²)²`` is maximally destabilising at the
    wavenumber ``|k| = 1`` (so the pattern wavelength is ``2π``), the control
    ``r > 0`` sets the unstable bandwidth, and the cubic ``-u³`` saturates the
    growth — the field is bounded for all time.  Expanding,
    ``(1 + ∇²)² u = u + 2 ∇²u + ∇⁴u``.

    State layout
        The state vector is the single field flattened (``u.ravel()``, length
        ``N²``).  :attr:`_field_shape` is the grid ``(N, N)``, so the
        ``kind="field"`` recipe plays it directly as a 2-D heatmap movie.

    Discretisation
        ``∇²`` is the 5-point periodic Laplacian and ``∇⁴`` the compact 13-point
        biharmonic stencil, both on a uniform periodic grid of spacing
        ``h = L / N``.

    Parameters
    ----------
    N : int, optional
        Grid points per side.  Structural — changing it recompiles.  Default 32
        (state dimension ``32² = 1024``).
    L : float, optional
        Domain length per side.  Default ``40.0`` (several pattern wavelengths).
    r : float, optional
        Linear growth control.  Default ``0.3`` (a robust stripe / labyrinth
        former).

    Notes
    -----
    The cubic saturation keeps the field bounded, so a non-stiff explicit method
    (the default ``"rk45"``) integrates it stably at a modest step; a fixed-order
    implicit solver is unnecessary (and its dense Jacobian on the flattened field
    is needlessly expensive).

    References
    ----------
    .. [1] Swift, J. & Hohenberg, P. C. (1977). "Hydrodynamic fluctuations at the
       convective instability." *Physical Review A*, 15(1), 319-328.
    """

    # N drives the symbolic loop length, so it is baked into the tape; L and r are
    # runtime control parameters (changing them only changes the coefficients).
    _structural_params = frozenset({"N"})

    params = {"N": 32, "L": 40.0, "r": 0.3}

    #: The grid the single field lives on.
    _field_shape = (32, 32)

    #: Seed for the small-random IC builder.
    _ic_seed = 0

    def __init__(
        self,
        N: int | None = None,
        L: float | None = None,
        r: float | None = None,
        *,
        params: dict[str, float] | None = None,
        ic=None,
    ):
        p = dict(type(self).params)
        if params:
            unknown = set(params) - set(p)
            if unknown:
                raise InvalidParameterError(
                    f"SwiftHohenberg: unknown parameter(s) {sorted(unknown)}. Declared: {sorted(p)}"
                )
            p.update(params)
        if N is not None:
            p["N"] = int(N)
        if L is not None:
            p["L"] = float(L)
        if r is not None:
            p["r"] = float(r)
        n_val = int(p["N"])
        if n_val < 3:
            raise ValueError("SwiftHohenberg requires N >= 3.")
        if ic is None:
            rng = np.random.default_rng(type(self)._ic_seed)
            ic = 0.1 * rng.standard_normal(n_val * n_val)
        super().__init__(dim=n_val * n_val, params=p, ic=ic, field_shape=(n_val, n_val))

    @staticmethod
    def _equations(Y, t, *, N, L, r):
        h = L / N
        inv_h2 = 1.0 / (h * h)
        inv_h4 = inv_h2 * inv_h2

        def idx(rr, cc):
            return (rr % N) * N + (cc % N)

        rhs = []
        for rr in range(N):
            for cc in range(N):

                def u(dr, dc, rr=rr, cc=cc):
                    return Y(idx(rr + dr, cc + dc))

                u0 = u(0, 0)
                lap = (u(1, 0) + u(-1, 0) + u(0, 1) + u(0, -1) - 4 * u0) * inv_h2
                # Compact 13-point biharmonic (∇⁴) stencil on a periodic grid.
                bih = (
                    20 * u0
                    - 8 * (u(1, 0) + u(-1, 0) + u(0, 1) + u(0, -1))
                    + 2 * (u(1, 1) + u(1, -1) + u(-1, 1) + u(-1, -1))
                    + (u(2, 0) + u(-2, 0) + u(0, 2) + u(0, -2))
                ) * inv_h4
                # u_t = r u - (1 + ∇²)² u - u³ = r u - (u + 2 ∇²u + ∇⁴u) - u³
                rhs.append(r * u0 - (u0 + 2.0 * lap + bih) - u0 * u0 * u0)
        return rhs
