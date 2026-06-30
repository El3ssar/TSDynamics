from typing import ClassVar

import numpy as np
from symengine import cos, exp, sign, sin

from tsdynamics.errors import InvalidParameterError
from tsdynamics.families import ContinuousSystem


class Lorenz(ContinuousSystem):
    """
    Lorenz (1963) strange attractor.

    Parameters
    ----------
    sigma, rho, beta : float
        Classic Lorenz parameters.  Default values produce the well-known
        chaotic attractor with Lyapunov spectrum ≈ [0.91, 0, −14.57].
    """

    params = {"sigma": 10.0, "rho": 28.0, "beta": 8 / 3}
    dim = 3
    variables = ("x", "y", "z")
    reference = "Lorenz (1963), J. Atmos. Sci. 20, 130-141"
    doi = "10.1175/1520-0469(1963)020<0130:dnf>2.0.co;2"
    known_lyapunov = {
        "spectrum": (0.906, 0.0, -14.57),
        "atol": (0.45, 0.2, 4.6),
        "ic": (1.0, 1.0, 1.0),
        "kwargs": {
            "dt": 0.1,
            "burn_in": 50.0,
            "final_time": 200.0,
            "method": "dop853",
            "rtol": 1e-7,
            "atol": 1e-10,
        },
        "source": "Sprott (2003), Chaos and Time-Series Analysis",
    }

    @staticmethod
    def _equations(y, t, *, sigma, rho, beta):
        x, yv, z = y(0), y(1), y(2)
        return (
            sigma * (yv - x),
            rho * x - x * z - yv,
            x * yv - beta * z,
        )


class LorenzBounded(ContinuousSystem):
    """
    Bounded variant of the Lorenz (1963) attractor.

    A reformulation of the classic Lorenz equations in which the dynamics are
    confined to a bounded region of phase space by polynomial correction terms
    scaled by ``1/r**2``; as ``r -> infinity`` it reduces to the standard
    Lorenz system.

    Parameters
    ----------
    sigma, rho, beta : float
        The usual Lorenz parameters (Prandtl number, Rayleigh ratio, geometric
        factor). Defaults give the canonical chaotic regime.
    r : float
        Bounding radius. Larger ``r`` recovers the unbounded Lorenz flow.
    """

    params = {"beta": 2.667, "r": 64, "rho": 28, "sigma": 10}
    dim = 3
    reference = "Sprott & Xiong (2015), Chaos 25, 083101"
    doi = "10.1063/1.4927643"

    @staticmethod
    def _equations(Y, t, *, beta, r, rho, sigma):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = (
            sigma * y
            - sigma * x
            - sigma / r**2 * y * x**2
            - sigma / r**2 * y**3
            - sigma / r**2 * y * z**2
            + sigma / r**2 * x**3
            + sigma / r**2 * x * y**2
            + sigma / r**2 * x * z**2
        )
        ydot = (
            rho * x
            - x * z
            - y
            - rho / r**2 * x**3
            - rho / r**2 * x * y**2
            - rho / r**2 * x * z**2
            + 1 / r**2 * z * x**3
            + 1 / r**2 * x * z * y**2
            + 1 / r**2 * x * z**3
            + 1 / r**2 * y * x**2
            + 1 / r**2 * y**3
            + 1 / r**2 * y * z**2
        )
        zdot = (
            x * y
            - beta * z
            - 1 / r**2 * y * x**3
            - 1 / r**2 * x * y**3
            - 1 / r**2 * x * y * z**2
            + beta / r**2 * z * x**2
            + beta / r**2 * z * y**2
            + beta / r**2 * z**3
        )
        return xdot, ydot, zdot


class LorenzCoupled(ContinuousSystem):
    """
    Two diffusively coupled Lorenz (1963) systems.

    Two identical Lorenz subsystems whose x-variables are linearly coupled with
    strength ``kappa``; a canonical testbed for chaos synchronization and the
    onset of (anti-)synchronized and desynchronized states. The state is laid
    out as ``[x1, y1, z1, x2, y2, z2]`` (``dim = 6``).

    Parameters
    ----------
    sigma, rho, beta : float
        Standard Lorenz parameters, shared by both subsystems.
    kappa : float
        Diffusive coupling strength between the two x-variables.
    """

    params = {"beta": 8 / 3, "kappa": 2.85, "rho": 28, "sigma": 10}
    dim = 6
    reference = "Lorenz (1963), J. Atmos. Sci. 20, 130-141"
    doi = "10.1175/1520-0469(1963)020<0130:dnf>2.0.co;2"

    @staticmethod
    def _equations(Y, t, *, beta, kappa, rho, sigma):
        x1, y1, z1, x2, y2, z2 = Y(0), Y(1), Y(2), Y(3), Y(4), Y(5)
        x1dot = sigma * (y1 - x1) + kappa * (x2 - x1)
        y1dot = rho * x1 - x1 * z1 - y1
        z1dot = x1 * y1 - beta * z1
        x2dot = sigma * (y2 - x2) + kappa * (x1 - x2)
        y2dot = rho * x2 - x2 * z2 - y2
        z2dot = x2 * y2 - beta * z2
        return x1dot, y1dot, z1dot, x2dot, y2dot, z2dot


class Lorenz96(ContinuousSystem):
    """
    Lorenz-96 model on a 1D ring of ``N`` weakly coupled scalar variables.

    Forced by a single scalar ``f``; chaotic for ``f >= 8`` and ``N``
    large enough (the canonical attractor sets in around ``N = 5``).

    Parameters
    ----------
    N : int
        Number of sites on the ring. Structural — changing it recompiles.
    f : float
        Forcing strength. Runtime-tunable (no recompile).
    """

    reference = "Lorenz (1996), Proc. ECMWF Seminar on Predictability 1, 1-18"
    doi = "10.1017/cbo9780511617652.004"
    params = {"f": 8.0, "N": 20}
    # N affects the symbolic structure (loop length), so it must be baked in.
    _structural_params = frozenset({"N"})

    def __init__(
        self,
        N: int | None = None,
        f: float | None = None,
        *,
        params: dict[str, float] | None = None,
        ic=None,
    ):
        p = dict(type(self).params)
        if params:
            # super().__init__ validates unknown parameters (InvalidParameterError).
            p.update(params)
        if N is not None:
            p["N"] = int(N)
        if f is not None:
            p["f"] = float(f)
        super().__init__(dim=int(p["N"]), params=p, ic=ic)

    @staticmethod
    def _equations(y_sym, t_sym, *, f, N):
        return [
            (y_sym((i + 1) % N) - y_sym((i - 2) % N)) * y_sym((i - 1) % N) - y_sym(i) + f
            for i in range(N)
        ]


class Lorenz84(ContinuousSystem):
    """
    Lorenz (1984) low-order general-circulation model.

    A three-variable model of the large-scale mid-latitude atmospheric
    circulation: ``x`` is the intensity of the westerly wind current (and the
    poleward temperature gradient), while ``y`` and ``z`` are the cosine and
    sine phases of a chain of superposed large-scale eddies. For different
    thermal forcings the system has stable steady, periodic, or irregular
    (chaotic) solutions; the default parameters give the chaotic attractor.

    Parameters
    ----------
    a, b : float
        Damping coefficients of the wind current and the eddies.
    f, g : float
        Thermal (symmetric) and asymmetric forcing amplitudes.
    """

    params = {"a": 1.32, "b": 7.91, "f": 4.83, "g": 4.194}
    dim = 3
    reference = "Lorenz (1984), Tellus 36A, 98-110"
    doi = "10.3402/tellusa.v36i2.11473"

    @staticmethod
    def _equations(Y, t, *, a, b, f, g):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = -a * x - y**2 - z**2 + a * f
        ydot = -y + x * y - b * x * z + g
        zdot = -z + b * x * y + x * z
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a, b, f, g):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [-a, -2 * y, -2 * z]
        row2 = [y - b * z, x - 1, -b * x]
        row3 = [b * y + z, b * x, -1 + x]
        return row1, row2, row3


class Rossler(ContinuousSystem):
    """
    Rössler (1976) strange attractor.

    A three-dimensional flow with a single quadratic nonlinearity, designed as a
    minimal prototype of chaos simpler than Lorenz. The dynamics spiral outward
    in the ``x``–``y`` plane and are reinjected by an occasional fold in ``z``,
    producing a band-like attractor that arises through a period-doubling
    cascade. The defaults give the canonical chaotic regime.

    Parameters
    ----------
    a, b : float
        Coefficients controlling the spiral growth and the reinjection.
    c : float
        Parameter governing the fold; the standard route to chaos.
    """

    params = {"a": 0.2, "b": 0.2, "c": 5.7}
    dim = 3
    variables = ("x", "y", "z")
    reference = "Rössler (1976), Phys. Lett. A 57, 397-398"
    doi = "10.1016/0375-9601(76)90101-8"
    known_lyapunov = {
        "spectrum": (0.0714, 0.0, -5.39),
        "atol": (0.06, 0.06, 1.5),
        "ic": (1.0, 0.0, 0.0),
        "kwargs": {"dt": 0.1, "burn_in": 100.0, "final_time": 500.0},
        "source": "Sprott (2003), Chaos and Time-Series Analysis",
    }

    @staticmethod
    def _equations(Y, t, *, a, b, c):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = -y - z
        ydot = x + a * y
        zdot = b + z * x - c * z
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a, b, c):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [0, -1, -1]
        row2 = [1, a, 0]
        row3 = [z, 0, x - c]
        return row1, row2, row3


class Thomas(ContinuousSystem):
    """
    Thomas' cyclically symmetric attractor.

    A three-dimensional flow that is cyclically symmetric in ``x``, ``y`` and
    ``z`` and can be read as a frictionally damped particle moving in a 3-D
    lattice of sinusoidal forces. As the damping ``a`` is lowered the system
    passes from a fixed point through limit cycles to chaos, and for weak
    damping exhibits "labyrinth chaos" — random-walk-like motion across the
    lattice of unstable equilibria.

    Parameters
    ----------
    a : float
        Dissipation (friction) coefficient; smaller ``a`` is more chaotic.
    b : float
        Amplitude of the sinusoidal forcing.
    """

    params = {"a": 1.85, "b": 10}
    dim = 3
    reference = "Thomas (1999), Int. J. Bifurc. Chaos 9, 1889-1905"
    doi = "10.1142/s0218127499001383"

    @staticmethod
    def _equations(Y, t, *, a, b):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = -a * x + b * sin(y)
        ydot = -a * y + b * sin(z)
        zdot = -a * z + b * sin(x)
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a, b):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [-a, b * cos(y), 0]
        row2 = [0, -a, b * cos(z)]
        row3 = [b * cos(x), 0, -a]
        return row1, row2, row3


class KuramotoSivashinsky(ContinuousSystem):
    """
    1D Kuramoto–Sivashinsky PDE on a periodic domain, discretized with N grid points.

    PDE (common sign convention):

    .. code-block:: text

        u_t = - u u_x - u_xx - u_xxxx

    The ``-u_xx`` term is linearly *destabilising* (long-wavelength growth), the
    ``-u_xxxx`` term is dissipative (short-wavelength damping), and the nonlinear
    term ``-u u_x = -0.5*(u^2)_x`` provides saturation.  For ``L ≈ 22`` the
    attractor is spatio-temporally chaotic with ~2–3 positive Lyapunov exponents.

    Spatial discretisation uses central finite differences on a uniform periodic
    grid (7-point stencils, Fornberg weights).  The 7-point stencils give 6th-order
    accuracy for ``u_x`` and ``u_xx``; the 7-point ``u_xxxx`` stencil is 4th-order
    accurate (the same 7 points cannot reach 6th order for the 4th derivative):

    .. code-block:: text

        (u^2)_x  ≈ (1/dx) Σ w1_k u_{j+k}^2    (6th-order, k = ±1,±2,±3)
        u_xx     ≈ (1/dx²) Σ w2_k u_{j+k}       (6th-order, k = 0,±1,±2,±3)
        u_xxxx   ≈ (1/dx⁴) Σ w4_k u_{j+k}       (4th-order, k = 0,±1,±2,±3)

    Notes
    -----
    - The spatial mean of u is conserved by all three terms; initial conditions
      should therefore be zero-mean to obtain the canonical chaotic attractor.
      The default IC is a small sinusoidal perturbation with zero mean.
    - Requires N >= 7 (minimum span needed by the ±3 stencil).
    - KS is stiff: the ``u_xxxx`` term gives eigenvalues ∝ (π/dx)^4.  Explicit
      RK methods require dt ≪ (dx/π)^4 and are impractical for fine grids.
      The default integrator is the variable-order ``"bdf"``; override with
      ``method=`` if needed.
    """

    # The u_xxxx term makes KS stiff — explicit RK will blow up for any useful grid.
    _default_method = "bdf"
    # N drives the symbolic loop length, so it must be baked in at compile time.
    # L is kept as a runtime control parameter (changing it does not change the
    # number of equations, only the coefficients).
    _structural_params = frozenset({"N"})

    reference = (
        "Kuramoto & Tsuzuki (1976), Prog. Theor. Phys. 55, 356-369; "
        "Sivashinsky (1977), Acta Astronaut. 4, 1177-1206"
    )
    params = {"N": 32, "L": 22.0}

    #: Seed for the deterministic broadband IC builder.  Override per-instance
    #: by passing a custom ``ic=`` array, or globally by subclassing.
    _ic_seed: ClassVar[int] = 0

    def __init__(
        self,
        N: int | None = None,
        L: float | None = None,
        *,
        params: dict[str, float] | None = None,
        ic=None,
    ):
        p = dict(type(self).params)
        if params:
            # Validate before the N>=7 / IC-build steps below so an unknown
            # parameter surfaces first (super().__init__ would also reject it,
            # but only after those steps). Raise the project error type to match
            # SystemBase.__init__ (InvalidParameterError, a ValueError subclass).
            unknown = set(params) - set(p)
            if unknown:
                raise InvalidParameterError(
                    f"KuramotoSivashinsky: unknown parameter(s) {sorted(unknown)}. "
                    f"Declared: {sorted(p)}"
                )
            p.update(params)
        if N is not None:
            p["N"] = int(N)
        if L is not None:
            p["L"] = float(L)
        N_val, L_val = int(p["N"]), float(p["L"])
        if N_val < 7:
            raise ValueError("KuramotoSivashinsky requires N >= 7 (uses ±3 stencil).")
        if ic is None:
            ic = self._broadband_ic(N_val, L_val, seed=type(self)._ic_seed)
        super().__init__(dim=N_val, params=p, ic=ic)

    @staticmethod
    def _broadband_ic(N: int, L: float, *, seed: int = 0, amplitude: float = 0.5) -> np.ndarray:
        """
        Build a zero-mean broadband initial condition for KS.

        The previous default ``0.01·cos(2π x/L)`` excites only the lowest
        wavenumber ``k = 2π/L``, which lies in a marginally unstable band for
        ``L ≳ 30`` and grows so slowly that the trajectory stays near the
        zero solution for the integration window — visible as horizontal
        stripes in a space-time plot.

        Instead we excite every linearly unstable Fourier mode
        (``|k_j| < 1``, i.e. wavenumber indices ``j < L / (2π)``) with
        seeded random amplitudes and phases, normalised to a target RMS so
        the trajectory enters the nonlinear regime in O(10) time units
        across the full L range.

        Parameters
        ----------
        N : int
            Number of grid points.
        L : float
            Domain length.
        seed : int, optional
            Seed for ``numpy.random.default_rng``.  Default 0 — keep this
            fixed to get reproducible trajectories.
        amplitude : float, optional
            Target RMS amplitude of the IC.  Default 0.5.

        Returns
        -------
        ic : ndarray, shape (N,)
            Zero-mean initial condition.
        """
        rng = np.random.default_rng(seed)
        x = np.linspace(0.0, L, N, endpoint=False)
        # Highest wavenumber index in the linearly unstable band ``|k_j| < 1``.
        # Always include at least mode 1 so even small-L domains evolve.
        k_max = max(2, int(L / (2.0 * np.pi)) + 1)
        # Cap at Nyquist so we never alias on small grids.
        k_max = min(k_max, N // 2)
        ic = np.zeros(N, dtype=float)
        for k in range(1, k_max + 1):
            a = rng.standard_normal()
            b = rng.standard_normal()
            ic += a * np.cos(2.0 * np.pi * k * x / L) + b * np.sin(2.0 * np.pi * k * x / L)
        ic -= ic.mean()
        rms = float(np.sqrt(np.mean(ic**2)))
        if rms > 0.0:
            ic *= amplitude / rms
        return ic

    @staticmethod
    def _equations(Y, t, *, N, L):
        # 7-point central weights (Trefethen-style) for periodic, equispaced grid.
        # First derivative (6th-order): D1 * f / dx
        w1 = (
            -1.0 / 60.0,
            3.0 / 20.0,
            -3.0 / 4.0,
            0.0,
            3.0 / 4.0,
            -3.0 / 20.0,
            1.0 / 60.0,
        )
        # Second derivative (6th-order): D2 * f / dx^2
        w2 = (
            1.0 / 90.0,
            -3.0 / 20.0,
            3.0 / 2.0,
            -49.0 / 18.0,
            3.0 / 2.0,
            -3.0 / 20.0,
            1.0 / 90.0,
        )
        # Fourth derivative (7-point central, 4th-order accurate): D4 * f / dx^4
        w4 = (-1.0 / 6.0, 2.0, -6.5, 28.0 / 3.0, -6.5, 2.0, -1.0 / 6.0)
        offsets = (-3, -2, -1, 0, 1, 2, 3)

        dx = L / N
        inv_dx = 1.0 / dx
        inv_dx2 = inv_dx * inv_dx
        inv_dx4 = inv_dx2 * inv_dx2

        rhs = []
        for j in range(N):
            # Nonlinear term: -u * u_x (structure-preserving)
            ux = 0.0
            for r, c in zip(offsets, w1, strict=True):
                idx = (j + r) % N
                ux += c * Y(idx)
            ux *= inv_dx

            nonlinear = -Y(j) * ux

            # u_xx: 6th-order central
            uxx = 0.0
            for r, c in zip(offsets, w2, strict=True):
                idx = (j + r) % N
                uxx += c * Y(idx)
            uxx *= inv_dx2

            # u_xxxx: 7-point central
            uxxxx = 0.0
            for r, c in zip(offsets, w4, strict=True):
                idx = (j + r) % N
                uxxxx += c * Y(idx)
            uxxxx *= inv_dx4

            rhs.append(nonlinear - uxx - uxxxx)
        return rhs


class Halvorsen(ContinuousSystem):
    """
    Halvorsen's cyclically symmetric attractor.

    A three-dimensional flow proposed by Arne Dehli Halvorsen that is symmetric
    under cyclic interchange of ``x``, ``y`` and ``z``, combining linear damping
    and cross-coupling with a single quadratic self-term per equation. It
    produces a visually intricate cyclically symmetric strange attractor.

    Parameters
    ----------
    a : float
        Linear damping coefficient (default 1.4 gives the chaotic attractor).
    b : float
        Coupling coefficient between the three variables.
    """

    # Originally circulated by A. D. Halvorsen on the sci.fractals newsgroup and
    # documented (without a formal primary paper) by Sprott (2003).
    params = {"a": 1.4, "b": 4}
    dim = 3
    reference = "Sprott (2010), Elegant Chaos, World Scientific"
    doi = "10.1142/9789812838827"

    @staticmethod
    def _equations(Y, t, *, a, b):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = -a * x - b * y - b * z - y**2
        ydot = -a * y - b * z - b * x - z**2
        zdot = -a * z - b * x - b * y - x**2
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a, b):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [-a, -b - 2 * y, -b]
        row2 = [-b, -a, -b - 2 * z]
        row3 = [-b - 2 * x, -b, -a]
        return row1, row2, row3


class Chua(ContinuousSystem):
    """
    Chua's circuit (double-scroll attractor).

    The canonical autonomous electronic oscillator whose only nonlinearity is a
    piecewise-linear resistor (the "Chua diode"), here represented by the
    odd ramp ``m1 x + 0.5 (m0 - m1)(|x+1| - |x-1|)``. For the classic parameters
    it produces the double-scroll strange attractor.

    Parameters
    ----------
    alpha, beta : float
        Dimensionless circuit parameters (capacitor/inductor ratios).
    m0, m1 : float
        Inner and outer slopes of the piecewise-linear Chua diode
        characteristic.
    """

    params = {"alpha": 15.6, "beta": 28.0, "m0": -1.142857, "m1": -0.71429}
    dim = 3
    reference = "Chua (1969), Introduction to Nonlinear Network Theory, McGraw-Hill"
    # Classic double-scroll Chua circuit (α=15.6, β=28, m0=-8/7, m1=-5/7). The
    # piecewise-linear nonlinearity makes the *exact* leading exponent sensitive
    # to the breakpoint handling, so only the robust sign structure is asserted:
    # one positive, one ~zero and one strongly negative exponent (largest LE
    # ≈0.4-0.5 for this canonical double-scroll set, matching the engine estimate).
    # A small off-origin IC reliably lands on the double-scroll attractor.
    known_lyapunov = {
        "n_positive": 1,
        "ic": (0.1, 0.0, 0.0),
        "kwargs": {
            "dt": 0.02,
            "burn_in": 100.0,
            "final_time": 400.0,
            "method": "dop853",
            "rtol": 1e-8,
            "atol": 1e-11,
        },
        "source": "Matsumoto, Chua & Komuro (1985), IEEE Trans. Circuits Syst. 32, 798-818",
    }

    @staticmethod
    def _equations(Y, t, *, alpha, beta, m0, m1):
        x, y, z = Y(0), Y(1), Y(2)
        ramp_x = m1 * x + 0.5 * (m0 - m1) * (abs(x + 1) - abs(x - 1))
        xdot = alpha * (y - x - ramp_x)
        ydot = x - y + z
        zdot = -beta * y
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, alpha, beta, m0, m1):
        x, y, z = Y(0), Y(1), Y(2)
        dramp_xdx = m1 + 0.5 * (m0 - m1) * (sign(x + 1) - sign(x - 1))
        row1 = [-alpha - alpha * dramp_xdx, alpha, 0]
        row2 = [1, -1, 1]
        row3 = [0, -beta, 0]
        return row1, row2, row3


class MultiChua(ContinuousSystem):
    """
    Ring of ``n_circuits`` Chua circuits coupled through their x-variables.

    The state is laid out as ``[x1, y1, z1, x2, y2, z2, ...]`` and the
    dimension is therefore ``3 * n_circuits``.

    Parameters
    ----------
    n_circuits : int
        Number of circuits in the ring. Structural — changing it recompiles.
    """

    params = {
        "alpha": 15.6,
        "beta": 28.0,
        "m0": -1.143,
        "m1": -0.714,
        "kappa": 0.1,
        "n_circuits": 3,
    }
    reference = (
        "Yalçın, Suykens & Vandewalle (2005), Cellular Neural Networks, "
        "Multi-Scroll Chaos and Synchronization, World Scientific"
    )
    doi = "10.1142/9789812567741"
    # n_circuits drives the loop length in _equations, so bake it in.
    _structural_params = frozenset({"n_circuits"})

    def __init__(
        self,
        n_circuits: int | None = None,
        *,
        params: dict[str, float] | None = None,
        ic=None,
    ):
        p = dict(type(self).params)
        if params:
            # super().__init__ validates unknown parameters (InvalidParameterError).
            p.update(params)
        if n_circuits is not None:
            p["n_circuits"] = int(n_circuits)
        super().__init__(dim=3 * int(p["n_circuits"]), params=p, ic=ic)

    @staticmethod
    def _equations(Y, t, *, alpha, beta, m0, m1, kappa, n_circuits):
        """
        Right-hand side of the MultiChua model.

        X: State vector [x1, y1, z1, x2, y2, z2, ..., xn, yn, zn]
        """
        dim = 3 * n_circuits
        dXdt = [None] * dim

        for i in range(n_circuits):
            # Extract indices for the current circuit
            x_idx = 3 * i
            y_idx = x_idx + 1
            z_idx = x_idx + 2

            # State variables for this circuit
            x = Y(x_idx)
            y = Y(y_idx)
            z = Y(z_idx)

            # Coupled neighbor indices (periodic boundary conditions)
            x_prev = Y((x_idx - 3) % dim)  # Previous x (cyclic indexing)
            x_next = Y((x_idx + 3) % dim)  # Next x (cyclic indexing)

            # Nonlinear Chua diode function
            ramp_x = m1 * x + 0.5 * (m0 - m1) * (abs(x + 1) - abs(x - 1))

            # Chua equations with coupling
            xdot = alpha * (y - x - ramp_x) + kappa * (x_next - x_prev)
            ydot = x - y + z
            zdot = -beta * y

            # Assign derivatives
            dXdt[x_idx] = xdot
            dXdt[y_idx] = ydot
            dXdt[z_idx] = zdot

        return dXdt


class Duffing(ContinuousSystem):
    """
    Forced Duffing oscillator (double-well, periodically driven).

    The damped, harmonically driven oscillator with a cubic restoring force,
    written as an autonomous 3-D system by carrying the drive phase ``z`` with
    ``z' = omega``. The cubic nonlinearity makes the potential a double well
    (for ``beta < 0``), and for suitable forcing the response is chaotic with a
    strange attractor. The defaults are a standard chaotic regime.

    Parameters
    ----------
    alpha : float
        Linear stiffness coefficient.
    beta : float
        Cubic stiffness coefficient (negative gives a double well).
    delta : float
        Linear damping coefficient.
    gamma : float
        Amplitude of the periodic forcing.
    omega : float
        Angular frequency of the periodic forcing.
    """

    params = {"alpha": 1.0, "beta": -1.0, "delta": 0.1, "gamma": 0.35, "omega": 1.4}
    dim = 3
    reference = (
        "Duffing (1918), Erzwungene Schwingungen bei veränderlicher "
        "Eigenfrequenz, Vieweg, Braunschweig"
    )
    # The explicit default (rk45) fails to integrate this system; an implicit
    # solver handles it robustly, so make that the default.
    _default_method = "bdf"

    @staticmethod
    def _equations(Y, t, *, alpha, beta, delta, gamma, omega):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = y
        ydot = -delta * y - alpha * x - beta * x**3 + gamma * cos(z)
        zdot = omega
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, alpha, beta, delta, gamma, omega):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [0, 1, 0]
        row2 = [-alpha - 3 * beta * x**2, -delta, -gamma * sin(z)]
        row3 = [0, 0, 0]
        return row1, row2, row3


class RabinovichFabrikant(ContinuousSystem):
    """
    Rabinovich–Fabrikant system.

    A three-dimensional model derived for the stochastic self-modulation of
    waves in a non-equilibrium dissipative medium. With cubic nonlinearities it
    exhibits an unusually rich phase space, including chaotic attractors and
    hidden/transient chaos that is highly sensitive to the parameters and
    initial conditions.

    Parameters
    ----------
    a : float
        Parameter controlling the modulation/dissipation balance.
    g : float
        Parameter controlling the linear growth/damping.
    """

    params = {"a": 1.1, "g": 0.87}
    dim = 3
    reference = "Rabinovich & Fabrikant (1979), Sov. Phys. JETP 50, 311-317"
    doi = "10.1007/bf01034469"
    default_ic = [-1.0, 0.0, 0.5]  # random U[0,1)^3 escapes the basin

    @staticmethod
    def _equations(Y, t, *, a, g):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = y * (z - 1 + x**2) + g * x
        ydot = x * (3 * z + 1 - x**2) + g * y
        zdot = -2 * z * (a + x * y)
        return (xdot, ydot, zdot)

    @staticmethod
    def _jacobian(Y, t, a, g):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [2 * x * y + g, z - 1 + x**2, y]
        row2 = [3 * z + 1 - 3 * x**2, g, 3 * x]
        row3 = [-2 * y * z, -2 * x * z, -2 * (a + x * y)]
        return row1, row2, row3


class Dadras(ContinuousSystem):
    """
    Dadras–Momeni system.

    An eight-term three-dimensional autonomous system with quadratic
    nonlinearities that can generate two-, three- and four-scroll chaotic
    attractors as a single parameter is varied. The default parameters give a
    multi-scroll chaotic attractor.

    Parameters
    ----------
    c, e, o, p, r : float
        Coefficients of the polynomial terms controlling the scroll structure
        and the dissipation of the flow.
    """

    params = {"c": 2.0, "e": 9.0, "o": 2.7, "p": 3.0, "r": 1.7}
    dim = 3
    reference = "Dadras & Momeni (2009), Phys. Lett. A 373, 3637-3642"
    doi = "10.1016/j.physleta.2009.07.088"

    @staticmethod
    def _equations(Y, t, *, c, e, o, p, r):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = y - p * x + o * y * z
        ydot = r * y - x * z + z
        zdot = c * x * y - e * z
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, c, e, o, p, r):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [-p, 1 + o * z, o * y]
        row2 = [-z, r, 1 - x]
        row3 = [c * y, c * x, -e]
        return row1, row2, row3


class PehlivanWei(ContinuousSystem):
    """
    Pehlivan–Wei chaotic system.

    A parameter-free three-dimensional autonomous system with two quadratic
    nonlinearities, notable for its simple algebraic form and its equilibrium
    structure. It produces a chaotic attractor from the fixed initial
    conditions.
    """

    params = {}
    dim = 3
    reference = "Pehlivan & Wei (2012), Turk. J. Electr. Eng. Comput. Sci. 20, 1229-1239"
    doi = "10.3906/elk-1103-14"

    @staticmethod
    def _equations(Y, t):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = y - y * z
        ydot = y + y * z - 2 * x
        zdot = 2 - x * y - y**2
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [0, 1 - z, -y]
        row2 = [-2, 1 + z, y]
        row3 = [-y, -x - 2 * y, 0]
        return row1, row2, row3


# region Sprott Attractors


class SprottTorus(ContinuousSystem):
    """
    Sprott's torus system (strange attractor with coexisting invariant tori).

    A simple time-reversible three-dimensional flow with quadratic
    nonlinearities and no equilibria. It is conservative (quasi-periodic motion
    on nested invariant tori) for some initial conditions and dissipative
    (a hidden strange attractor with a symmetric twin repellor) for others.
    Parameter-free.
    """

    params = {}
    dim = 3
    reference = "Sprott (2014), Phys. Lett. A 378, 1361-1363"
    doi = "10.1016/j.physleta.2013.11.004"

    @staticmethod
    def _equations(Y, t):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = y + 2 * x * y + x * z
        ydot = 1 - 2 * x**2 + y * z
        zdot = x - x**2 - y**2
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [2 * y + z, 2 * x + 1, x]
        row2 = [-4 * x, z, y]
        row3 = [1 - 2 * x, -2 * y, 0]
        return row1, row2, row3


class SprottA(ContinuousSystem):
    """
    Sprott case A — one of the algebraically simplest chaotic flows.

    Case A from Sprott's 1994 computer search for minimal three-dimensional
    autonomous chaotic systems (five terms, two quadratic nonlinearities). It is
    conservative and identical to the Nosé–Hoover thermostatted oscillator at
    its standard parameter, exhibiting a chaotic sea interspersed with invariant
    tori. Parameter-free.
    """

    params = {}
    dim = 3
    reference = "Sprott (1994), Phys. Rev. E 50, R647-R650"
    doi = "10.1103/physreve.50.r647"

    @staticmethod
    def _equations(Y, t):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = y
        ydot = -x + y * z
        zdot = 1 - y**2
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [0, 1, 0]
        row2 = [-1, z, y]
        row3 = [0, -2 * y, 0]
        return row1, row2, row3


class SprottB(ContinuousSystem):
    """
    Sprott case B — an algebraically simple chaotic flow.

    Case B from Sprott's 1994 search for minimal three-dimensional chaotic
    systems (five terms, two quadratic nonlinearities). Parameter-free.
    """

    params = {}
    dim = 3
    reference = "Sprott (1994), Phys. Rev. E 50, R647-R650"
    doi = "10.1103/physreve.50.r647"

    @staticmethod
    def _equations(Y, t):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = y * z
        ydot = x - y
        zdot = 1 - x * y
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [0, z, y]
        row2 = [1, -1, 0]
        row3 = [-y, -x, 0]
        return row1, row2, row3


class SprottC(ContinuousSystem):
    """
    Sprott case C — an algebraically simple chaotic flow.

    Case C from Sprott's 1994 search for minimal three-dimensional chaotic
    systems (five terms, two quadratic nonlinearities). Parameter-free.
    """

    params = {}
    dim = 3
    reference = "Sprott (1994), Phys. Rev. E 50, R647-R650"
    doi = "10.1103/physreve.50.r647"

    @staticmethod
    def _equations(Y, t):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = y * z
        ydot = x - y
        zdot = 1 - x**2
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [0, z, y]
        row2 = [1, -1, 0]
        row3 = [-2 * x, 0, 0]
        return row1, row2, row3


class SprottD(ContinuousSystem):
    """
    Sprott case D — an algebraically simple chaotic flow.

    Case D from Sprott's 1994 search for minimal three-dimensional chaotic
    systems (five terms, two quadratic nonlinearities). Parameter-free.
    """

    params = {}
    dim = 3
    reference = "Sprott (1994), Phys. Rev. E 50, R647-R650"
    doi = "10.1103/physreve.50.r647"
    default_ic = [0.1, 0.05, 0.05]  # random U[0,1)^3 escapes the basin

    @staticmethod
    def _equations(Y, t):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = -y
        ydot = x + z
        zdot = x * z + 3 * y**2
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [0, -1, 0]
        row2 = [1, 0, 1]
        row3 = [z, 6 * y, x]
        return row1, row2, row3


class SprottE(ContinuousSystem):
    """
    Sprott case E — an algebraically simple chaotic flow.

    Case E from Sprott's 1994 search for minimal three-dimensional chaotic
    systems (five terms, two quadratic nonlinearities). Parameter-free.
    """

    params = {}
    dim = 3
    reference = "Sprott (1994), Phys. Rev. E 50, R647-R650"
    doi = "10.1103/physreve.50.r647"

    @staticmethod
    def _equations(Y, t):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = y * z
        ydot = x**2 - y
        zdot = 1 - 4 * x
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [0, z, y]
        row2 = [2 * x, -1, 0]
        row3 = [-4, 0, 0]
        return row1, row2, row3


class SprottF(ContinuousSystem):
    """
    Sprott case F — an algebraically simple chaotic flow.

    Case F from Sprott's 1994 search (six terms, one quadratic nonlinearity).
    Chaotic at the default ``a = 0.5``.

    Parameters
    ----------
    a : float
        Single adjustable coefficient of the flow.
    """

    params = {"a": 0.5}
    dim = 3
    reference = "Sprott (1994), Phys. Rev. E 50, R647-R650"
    doi = "10.1103/physreve.50.r647"

    @staticmethod
    def _equations(Y, t, *, a):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = y + z
        ydot = -x + a * y
        zdot = x**2 - z
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [0, 1, 1]
        row2 = [-1, a, 0]
        row3 = [2 * x, 0, -1]
        return row1, row2, row3


class SprottG(ContinuousSystem):
    """
    Sprott case G — an algebraically simple chaotic flow.

    Case G from Sprott's 1994 search (six terms, one quadratic nonlinearity).
    Chaotic at the default ``a = 0.4``.

    Parameters
    ----------
    a : float
        Single adjustable coefficient of the flow.
    """

    params = {"a": 0.4}
    dim = 3
    reference = "Sprott (1994), Phys. Rev. E 50, R647-R650"
    doi = "10.1103/physreve.50.r647"

    @staticmethod
    def _equations(Y, t, *, a):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = a * x + z
        ydot = x * z - y
        zdot = -x + y
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [a, 0, 1]
        row2 = [z, -1, x]
        row3 = [-1, 1, 0]
        return row1, row2, row3


class SprottH(ContinuousSystem):
    """
    Sprott case H — an algebraically simple chaotic flow.

    Case H from Sprott's 1994 search (six terms, one quadratic nonlinearity).
    Chaotic at the default ``a = 0.5``.

    Parameters
    ----------
    a : float
        Single adjustable coefficient of the flow.
    """

    params = {"a": 0.5}
    dim = 3
    reference = "Sprott (1994), Phys. Rev. E 50, R647-R650"
    doi = "10.1103/physreve.50.r647"

    @staticmethod
    def _equations(Y, t, *, a):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = -y + z**2
        ydot = x + a * y
        zdot = x - z
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [0, -1, 2 * z]
        row2 = [1, a, 0]
        row3 = [1, 0, -1]
        return row1, row2, row3


class SprottI(ContinuousSystem):
    """
    Sprott case I — an algebraically simple chaotic flow.

    Case I from Sprott's 1994 search (six terms, one quadratic nonlinearity).
    Chaotic at the default ``a = 0.2``.

    Parameters
    ----------
    a : float
        Single adjustable coefficient of the flow.
    """

    params = {"a": 0.2}
    dim = 3
    reference = "Sprott (1994), Phys. Rev. E 50, R647-R650"
    doi = "10.1103/physreve.50.r647"
    default_ic = [0.1, 0.05, 0.05]  # random U[0,1)^3 escapes the basin

    @staticmethod
    def _equations(Y, t, *, a):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = -a * y
        ydot = x + z
        zdot = x + y**2 - z
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [0, -a, 0]
        row2 = [1, 0, 1]
        row3 = [1, 2 * y, -1]
        return row1, row2, row3


class SprottJ(ContinuousSystem):
    """
    Sprott case J — an algebraically simple chaotic flow.

    Case J from Sprott's 1994 search (six terms, one quadratic nonlinearity).
    Parameter-free.
    """

    params = {}
    dim = 3
    reference = "Sprott (1994), Phys. Rev. E 50, R647-R650"
    doi = "10.1103/physreve.50.r647"

    @staticmethod
    def _equations(Y, t):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = 2 * z
        ydot = -2 * y + z
        zdot = -x + y + y**2
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [0, 0, 2]
        row2 = [0, -2, 1]
        row3 = [-1, 1 + 2 * y, 0]
        return row1, row2, row3


class SprottK(ContinuousSystem):
    """
    Sprott case K — an algebraically simple chaotic flow.

    Case K from Sprott's 1994 search (six terms, one quadratic nonlinearity).
    Chaotic at the default ``a = 0.3``.

    Parameters
    ----------
    a : float
        Single adjustable coefficient of the flow.
    """

    params = {"a": 0.3}
    dim = 3
    reference = "Sprott (1994), Phys. Rev. E 50, R647-R650"
    doi = "10.1103/physreve.50.r647"

    @staticmethod
    def _equations(Y, t, *, a):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = x * y - z
        ydot = x - y
        zdot = x + a * z
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [y, x, -1]
        row2 = [1, -1, 0]
        row3 = [1, 0, a]
        return row1, row2, row3


class SprottL(ContinuousSystem):
    """
    Sprott case L — an algebraically simple chaotic flow.

    Case L from Sprott's 1994 search (six terms, one quadratic nonlinearity).
    Chaotic at the default parameters.

    Parameters
    ----------
    a, b : float
        Adjustable coefficients of the flow.
    """

    params = {"a": 0.9, "b": 3.9}
    dim = 3
    reference = "Sprott (1994), Phys. Rev. E 50, R647-R650"
    doi = "10.1103/physreve.50.r647"
    _default_method = "bdf"  # explicit default solver fails; use an implicit one

    @staticmethod
    def _equations(Y, t, *, a, b):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = y + b * z
        ydot = a * x**2 - y
        zdot = 1 - x
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a, b):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [0, 1, b]
        row2 = [2 * a * x, -1, 0]
        row3 = [-1, 0, 0]
        return row1, row2, row3


class SprottM(ContinuousSystem):
    """
    Sprott case M — an algebraically simple chaotic flow.

    Case M from Sprott's 1994 search (six terms, one quadratic nonlinearity).
    Chaotic at the default ``a = 1.7``.

    Parameters
    ----------
    a : float
        Single adjustable coefficient of the flow.
    """

    params = {"a": 1.7}
    dim = 3
    reference = "Sprott (1994), Phys. Rev. E 50, R647-R650"
    doi = "10.1103/physreve.50.r647"
    default_ic = [0.1, 0.05, 0.05]  # random U[0,1)^3 escapes the basin

    @staticmethod
    def _equations(Y, t, *, a):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = -z
        ydot = -(x**2) - y
        zdot = a + a * x + y
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [0, 0, -1]
        row2 = [-2 * x, -1, 0]
        row3 = [a, 1, 0]
        return row1, row2, row3


class SprottN(ContinuousSystem):
    """
    Sprott case N — an algebraically simple chaotic flow.

    Case N from Sprott's 1994 search (six terms, one quadratic nonlinearity).
    Parameter-free.
    """

    params = {}
    dim = 3
    reference = "Sprott (1994), Phys. Rev. E 50, R647-R650"
    doi = "10.1103/physreve.50.r647"

    @staticmethod
    def _equations(Y, t):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = -2 * y
        ydot = x + z**2
        zdot = 1 + y - 2 * z
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [0, -2, 0]
        row2 = [1, 0, 2 * z]
        row3 = [0, 1, -2]
        return row1, row2, row3


class SprottO(ContinuousSystem):
    """
    Sprott case O — an algebraically simple chaotic flow.

    Case O from Sprott's 1994 search (six terms, one quadratic nonlinearity).
    Chaotic at the default ``a = 2.7``.

    Parameters
    ----------
    a : float
        Single adjustable coefficient of the flow.
    """

    params = {"a": 2.7}
    dim = 3
    reference = "Sprott (1994), Phys. Rev. E 50, R647-R650"
    doi = "10.1103/physreve.50.r647"
    default_ic = [0.1, 0.05, 0.05]  # random U[0,1)^3 escapes the basin

    @staticmethod
    def _equations(Y, t, *, a):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = y
        ydot = x - z
        zdot = x + x * z + a * y
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [0, 1, 0]
        row2 = [1, 0, -1]
        row3 = [1 + z, a, x]
        return row1, row2, row3


class SprottP(ContinuousSystem):
    """
    Sprott case P — an algebraically simple chaotic flow.

    Case P from Sprott's 1994 search (six terms, one quadratic nonlinearity).
    Chaotic at the default ``a = 2.7``.

    Parameters
    ----------
    a : float
        Single adjustable coefficient of the flow.
    """

    params = {"a": 2.7}
    dim = 3
    reference = "Sprott (1994), Phys. Rev. E 50, R647-R650"
    doi = "10.1103/physreve.50.r647"
    _default_method = "bdf"  # explicit default solver fails; use an implicit one

    @staticmethod
    def _equations(Y, t, *, a):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = a * y + z
        ydot = -x + y**2
        zdot = x + y
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [0, a, 1]
        row2 = [-1, 2 * y, 0]
        row3 = [1, 1, 0]
        return row1, row2, row3


class SprottQ(ContinuousSystem):
    """
    Sprott case Q — an algebraically simple chaotic flow.

    Case Q from Sprott's 1994 search (six terms, one quadratic nonlinearity).
    Chaotic at the default parameters.

    Parameters
    ----------
    a, b : float
        Adjustable coefficients of the flow.
    """

    params = {"a": 3.1, "b": 0.5}
    dim = 3
    reference = "Sprott (1994), Phys. Rev. E 50, R647-R650"
    doi = "10.1103/physreve.50.r647"

    @staticmethod
    def _equations(Y, t, *, a, b):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = -z
        ydot = x - y
        zdot = a * x + y**2 + b * z
        return (xdot, ydot, zdot)

    @staticmethod
    def _jacobian(Y, t, a, b):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [0, 0, -1]
        row2 = [1, -1, 0]
        row3 = [a, 2 * y, b]
        return row1, row2, row3


class SprottR(ContinuousSystem):
    """
    Sprott case R — an algebraically simple chaotic flow.

    Case R from Sprott's 1994 search (six terms, one quadratic nonlinearity).
    Chaotic at the default parameters.

    Parameters
    ----------
    a, b : float
        Adjustable coefficients of the flow.
    """

    params = {"a": 0.9, "b": 0.4}
    dim = 3
    reference = "Sprott (1994), Phys. Rev. E 50, R647-R650"
    doi = "10.1103/physreve.50.r647"

    @staticmethod
    def _equations(Y, t, *, a, b):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = a - y
        ydot = b + z
        zdot = x * y - z
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a, b):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [0, -1, 0]
        row2 = [0, 0, 1]
        row3 = [y, x, -1]
        return row1, row2, row3


class SprottS(ContinuousSystem):
    """
    Sprott case S — an algebraically simple chaotic flow.

    Case S from Sprott's 1994 search (six terms, one quadratic nonlinearity).
    Parameter-free.
    """

    params = {}
    dim = 3
    reference = "Sprott (1994), Phys. Rev. E 50, R647-R650"
    doi = "10.1103/physreve.50.r647"

    @staticmethod
    def _equations(Y, t):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = -x - 4 * y
        ydot = x + z**2
        zdot = 1 + x
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [-1, -4, 0]
        row2 = [1, 0, 2 * z]
        row3 = [1, 0, 0]
        return row1, row2, row3


class SprottMore(ContinuousSystem):
    """
    Sprott-style jerk flow with a Gaussian nonlinearity.

    A three-dimensional jerk-type flow combining a sign-function (Coulomb-like)
    damping term with a Gaussian ``exp(-x**2)`` nonlinearity. An example of a
    minimal chaotic flow built from non-polynomial elementary functions.
    Parameter-free.
    """

    params = {}
    dim = 3
    reference = "Sprott (2020), Chaos Theory Appl. 2, 1-3"
    doi = "10.1016/j.chaos.2020.109990"

    @staticmethod
    def _equations(Y, t):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = y
        ydot = -x - sign(z) * y
        zdot = y**2 - exp(-(x**2))
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [0, 1, 0]
        row2 = [-1, -sign(z), 0]
        row3 = [2 * x * exp(-(x**2)), 2 * y, 0]
        return row1, row2, row3


class SprottJerk(ContinuousSystem):
    """
    Sprott's simplest dissipative chaotic flow (jerk system).

    The algebraically simplest dissipative chaotic flow with a single quadratic
    nonlinearity, written in jerk form ``x''' + mu x'' - x'^2 + x = 0`` (state
    space: ``x' = y``, ``y' = z``, ``z' = -x + y^2 - mu z``). It has been proven
    that no simpler chaotic system with a quadratic nonlinearity exists; the
    attractor resembles a Möbius strip and reaches chaos via period doubling
    around ``mu ≈ 2.017``.

    Parameters
    ----------
    mu : float
        Damping coefficient; chaos occurs for ``2.017 < mu < 2.082``.
    """

    params = {"mu": 2.017}
    dim = 3
    reference = "Sprott (1997), Phys. Lett. A 228, 271-274"
    doi = "10.1016/s0375-9601(97)00088-1"
    _default_method = "bdf"  # explicit default solver fails; use an implicit one

    @staticmethod
    def _equations(Y, t, *, mu):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = y
        ydot = z
        zdot = -x + y**2 - mu * z
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, mu):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [0, 1, 0]
        row2 = [0, 0, 1]
        row3 = [-1, 2 * y, -mu]
        return row1, row2, row3


# endregion


class Arneodo(ContinuousSystem):
    """
    Arneodo–Coullet–Tresser system.

    A third-order jerk-type system with a cubic nonlinearity that exhibits a
    spiral-type ("Shilnikov") strange attractor arising from a homoclinic
    bifurcation. Written as ``x' = y``, ``y' = z``, ``z' = -a x - b y - c z +
    d x^3``.

    Parameters
    ----------
    a, b, c : float
        Linear feedback coefficients.
    d : float
        Cubic nonlinearity coefficient.
    """

    params = {"a": -5.5, "b": 4.5, "c": 1.0, "d": -1.0}
    dim = 3
    reference = "Arneodo, Coullet & Tresser (1980), Phys. Lett. A 79, 259-263"
    doi = "10.1016/0375-9601(80)90342-4"

    @staticmethod
    def _equations(Y, t, *, a, b, c, d):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = y
        ydot = z
        zdot = -a * x - b * y - c * z + d * x**3
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a, b, c, d):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [0, 1, 0]
        row2 = [0, 0, 1]
        row3 = [-a + 3 * d * x**2, -b, -c]
        return row1, row2, row3


class Rucklidge(ContinuousSystem):
    """
    Rucklidge model of double convection.

    A third-order model that is an asymptotically exact description of weakly
    nonlinear two-dimensional convection in a fluid layer subject to a lateral
    constraint (double / double-diffusive convection). It exhibits chaotic
    behaviour for suitable parameters.

    Parameters
    ----------
    a : float
        Dissipation parameter.
    b : float
        Forcing (Rayleigh-like) parameter.
    """

    params = {"a": 2.0, "b": 6.7}
    dim = 3
    reference = "Rucklidge (1992), J. Fluid Mech. 237, 209-229"
    doi = "10.1017/s0022112092003392"

    @staticmethod
    def _equations(Y, t, *, a, b):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = -a * x + b * y - y * z
        ydot = x
        zdot = -z + y**2
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a, b):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [-a, b - z, -y]
        row2 = [1, 0, 0]
        row3 = [0, 2 * y, -1]
        return row1, row2, row3


class HyperRossler(ContinuousSystem):
    """
    Rössler hyperchaotic system.

    The first four-dimensional flow shown to be hyperchaotic — i.e. to possess
    two positive Lyapunov exponents (two directions of instability on the
    attractor) — using a single quadratic nonlinearity. The defaults are
    Rössler's original hyperchaotic parameter set.

    Parameters
    ----------
    a, b, c, d : float
        Coefficients of the four-variable flow (defaults give hyperchaos).
    """

    params = {"a": 0.25, "b": 3.0, "c": 0.5, "d": 0.05}
    dim = 4
    reference = "Rössler (1979), Phys. Lett. A 71, 155-157"
    doi = "10.1016/0375-9601(79)90150-6"
    default_ic = [-10.0, -6.0, 0.0, 10.0]  # random U[0,1)^4 escapes the basin

    @staticmethod
    def _equations(Y, t, *, a, b, c, d):
        x, y, z, w = Y(0), Y(1), Y(2), Y(3)
        xdot = -y - z
        ydot = x + a * y + w
        zdot = b + x * z
        wdot = -c * z + d * w
        return xdot, ydot, zdot, wdot

    @staticmethod
    def _jacobian(Y, t, a, b, c, d):
        x, y, z, w = Y(0), Y(1), Y(2), Y(3)
        row1 = [0, -1, -1, 0]
        row2 = [1, a, 0, 1]
        row3 = [z, 0, x, 0]
        row4 = [0, 0, -c, d]
        return row1, row2, row3, row4


class HyperLorenz(ContinuousSystem):
    """
    Hyperchaotic Lorenz-type system.

    A four-dimensional extension of the Lorenz system obtained by adding a
    fourth state ``w`` that feeds back into the dynamics, producing hyperchaos
    (two positive Lyapunov exponents).

    Parameters
    ----------
    a, b, c : float
        Lorenz-like coefficients of the underlying three-variable core.
    d : float
        Feedback coefficient of the added fourth variable.
    """

    params = {"a": 10, "b": 2.667, "c": 28, "d": 1.1}
    dim = 4
    reference = "Meier (2003), Presentation of Attractors with Cinema"
    doi = "10.1007/978-3-540-24699-2_13"

    @staticmethod
    def _equations(Y, t, *, a, b, c, d):
        x, y, z, w = Y(0), Y(1), Y(2), Y(3)
        xdot = a * y - a * x + w
        ydot = -x * z + c * x - y
        zdot = -b * z + x * y
        wdot = d * w - x * z
        return xdot, ydot, zdot, wdot


class HyperYangChen(ContinuousSystem):
    """
    Hyperchaotic Yang–Chen-type system.

    A four-dimensional Chen-like flow with an added linear feedback variable
    ``w`` that yields hyperchaotic dynamics (two positive Lyapunov exponents).

    Parameters
    ----------
    a, b, c, d : float
        Coefficients of the four-variable flow.
    """

    params = {"a": 30, "b": 3, "c": 35, "d": 8}
    dim = 4
    reference = "Meier (2003), Presentation of Attractors with Cinema"
    doi = "10.1007/978-3-540-24699-2_13"

    @staticmethod
    def _equations(Y, t, *, a, b, c, d):
        x, y, z, w = Y(0), Y(1), Y(2), Y(3)
        xdot = a * y - a * x
        ydot = c * x - x * z + w
        zdot = -b * z + x * y
        wdot = -d * x
        return xdot, ydot, zdot, wdot


class HyperYan(ContinuousSystem):
    """
    Hyperchaotic Yan-type system.

    A four-dimensional flow with multiple cross-product nonlinearities and a
    feedback variable ``w`` producing hyperchaos (two positive Lyapunov
    exponents).

    Parameters
    ----------
    a, b, c, d : float
        Coefficients of the four-variable flow.
    """

    params = {"a": 37, "b": 3, "c": 26, "d": 38}
    dim = 4
    reference = "Meier (2003), Presentation of Attractors with Cinema"
    doi = "10.1007/978-3-540-24699-2_13"

    @staticmethod
    def _equations(Y, t, *, a, b, c, d):
        x, y, z, w = Y(0), Y(1), Y(2), Y(3)
        xdot = a * y - a * x
        ydot = (c - a) * x - x * z + c * y
        zdot = -b * z + x * y - y * z + x * z - w
        wdot = -d * w + y * z - x * z
        return xdot, ydot, zdot, wdot


class GuckenheimerHolmes(ContinuousSystem):
    """
    Guckenheimer–Holmes structurally stable heteroclinic cycle.

    A symmetric three-dimensional flow exhibiting a robust (structurally stable)
    heteroclinic cycle connecting saddle equilibria, the prototypical model of
    "cycling chaos". The dynamics slow down near each saddle in turn, giving
    long, intermittent excursions.

    Parameters
    ----------
    a, b, c, d, e, f : float
        Coefficients setting the linear growth/rotation and the nonlinear
        coupling that close the heteroclinic cycle.
    """

    params = {"a": 0.4, "b": 20.25, "c": 3, "d": 1.6, "e": 1.7, "f": 0.44}
    dim = 3
    reference = "Guckenheimer & Holmes (1983), Nonlinear Oscillations, Springer"
    doi = "10.1007/978-1-4612-1140-2"

    @staticmethod
    def _equations(Y, t, *, a, b, c, d, e, f):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = a * x - b * y + c * z * x + d * z * x**2 + d * z * y**2
        ydot = a * y + b * x + c * z * y
        zdot = e - z**2 - f * x**2 - f * y**2 - a * z**3
        return xdot, ydot, zdot


class HenonHeiles(ContinuousSystem):
    """
    Hénon–Heiles system.

    A two-degree-of-freedom Hamiltonian model of the planar motion of a star in
    an axisymmetric galactic potential, written as a four-dimensional flow in
    ``(x, y, px, py)``. As the energy increases the phase space transitions from
    mostly regular (invariant tori) to predominantly chaotic motion — a classic
    illustration of the breakdown of a third integral of motion.

    Parameters
    ----------
    lam : float
        Strength of the cubic anharmonic coupling in the potential.
    """

    params = {"lam": 1}
    dim = 4
    reference = "Hénon & Heiles (1964), Astron. J. 69, 73-79"
    doi = "10.1086/109234"
    default_ic = [0.1, 0.1, 0.1, 0.1]  # low-energy bounded orbit; random U[0,1)^4 can be unbound

    @staticmethod
    def _equations(Y, t, *, lam):
        x, y, px, py = Y(0), Y(1), Y(2), Y(3)
        xdot = px
        ydot = py
        pxdot = -x - 2 * lam * x * y
        pydot = -y - lam * x**2 + lam * y**2
        return xdot, ydot, pxdot, pydot

    @staticmethod
    def _jacobian(Y, t, lam):
        x, y, px, py = Y(0), Y(1), Y(2), Y(3)
        row1 = [0, 0, 1, 0]
        row2 = [0, 0, 0, 1]
        row3 = [-1 - 2 * lam * y, -2 * lam * x, 0, 0]
        row4 = [-2 * lam * x, 2 * lam * y - 1, 0, 0]
        return row1, row2, row3, row4


class NoseHoover(ContinuousSystem):
    """
    Nosé–Hoover thermostatted oscillator.

    A one-dimensional harmonic oscillator coupled to a single deterministic
    Nosé–Hoover thermostat variable. This time-reversible, conservative
    three-dimensional flow exhibits a coexistence of regular (toroidal) and
    chaotic trajectories depending on the initial conditions, and is a canonical
    example of deterministic thermostatted dynamics.

    Parameters
    ----------
    a : float
        Target-temperature parameter of the thermostat.
    """

    params = {"a": 1.5}
    dim = 3
    reference = "Nosé (1984), J. Chem. Phys. 81, 511-519; Hoover (1985), Phys. Rev. A 31, 1695-1697"

    @staticmethod
    def _equations(Y, t, *, a):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = y
        ydot = -x + y * z
        zdot = a - y**2
        return xdot, ydot, zdot


class RikitakeDynamo(ContinuousSystem):
    """
    Rikitake two-disk dynamo.

    A coupled two-disk homopolar dynamo in which the current from each disk
    excites the coil of the other. It produces chaotic reversals of the current
    direction that mimic the irregular polarity reversals of the geomagnetic
    field — an early dynamical model of geomagnetic reversal.

    Parameters
    ----------
    a : float
        Asymmetry / coupling coefficient between the two disks.
    mu : float
        Common damping (dissipation) coefficient.
    """

    params = {"a": 1.0, "mu": 1.0}
    dim = 3
    reference = "Rikitake (1958), Proc. Cambridge Philos. Soc. 54, 89-105"
    doi = "10.1017/s0305004100033223"

    @staticmethod
    def _equations(Y, t, *, a, mu):
        x, y, z = Y(0), Y(1), Y(2)
        xdot = -mu * x + y * z
        ydot = -mu * y - a * x + x * z
        zdot = 1 - x * y
        return xdot, ydot, zdot

    @staticmethod
    def _jacobian(Y, t, a, mu):
        x, y, z = Y(0), Y(1), Y(2)
        row1 = [-mu, z, y]
        row2 = [-a + z, -mu, x]
        row3 = [-y, -x, 0]
        return row1, row2, row3
