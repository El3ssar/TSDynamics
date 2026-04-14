"""ContinuousSystem — ODE base class via JiTCODE."""

from __future__ import annotations

import os
import pathlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np

from .base import SystemBase, Trajectory

# ---------------------------------------------------------------------------
# Cache directory
# ---------------------------------------------------------------------------

_CACHE_DIR = pathlib.Path(
    os.environ.get("TSDYNAMICS_CACHE", pathlib.Path.home() / ".cache" / "tsdynamics")
)

# ---------------------------------------------------------------------------
# Integrator name map (SciPy-style aliases → JiTCODE names)
# ---------------------------------------------------------------------------

_INTEGRATOR_MAP: dict[str, str] = {
    "RK45":   "dopri5",
    "dopri5": "dopri5",
    "DOP853": "dop853",
    "dop853": "dop853",
    "LSODA":  "lsoda",
    "lsoda":  "lsoda",
    "VODE":   "vode",
    "vode":   "vode",
}
_EXPLICIT_METHODS = frozenset({"dopri5", "dop853"})


def _make_t_eval(t0: float, tf: float, dt: float) -> np.ndarray:
    """Build a uniform output grid from t0 to tf (inclusive)."""
    t_arr = np.arange(t0, tf, dt)
    if t_arr.size == 0 or t_arr[-1] < tf - 1e-12:
        t_arr = np.append(t_arr, tf)
    return t_arr


# ---------------------------------------------------------------------------
# ContinuousSystem
# ---------------------------------------------------------------------------

class ContinuousSystem(SystemBase, ABC):
    """
    Base class for ODE-based dynamical systems, compiled via JiTCODE.

    Subclass contract
    -----------------
    1. Declare ``params = {...}`` and ``dim = N`` at class level.
    2. Implement ``_equations`` as a ``@staticmethod`` returning a
       length-``dim`` sequence of JiTCODE / SymEngine symbolic expressions.
    3. Optionally mark integer or loop-structural parameters in
       ``_structural_params`` — these are baked into the compiled C code
       rather than exposed as runtime control parameters.

    Compilation
    -----------
    The first call to ``integrate`` or ``lyapunov_spectrum`` triggers JiTCODE
    compilation. Non-structural parameters become JiTCODE ``control_pars``,
    meaning the resulting ``.so`` module is compiled **once per class** (or
    once per structural-param combination) and reused for all subsequent runs
    and parameter changes, even across process restarts.

    Class-level attributes
    ----------------------
    _structural_params : frozenset[str]
        Parameter names that appear as integer loop bounds or affect the
        symbolic structure of ``_equations``.  These are baked in at compile
        time.  For most systems this is empty (the default).

        Example — Lorenz96 uses ``N`` to build the list comprehension::

            _structural_params = frozenset({"N"})

    _default_method : str
        Default integrator name (default ``"RK45"``).

    Examples
    --------
    >>> lor = Lorenz()
    >>> traj = lor.integrate(final_time=100, dt=0.01)
    >>> t, y = traj          # tuple-unpack
    >>> lor.sigma = 15.0     # change param — zero recompile cost
    >>> traj2 = lor.integrate(final_time=100)
    """

    _default_method: ClassVar[str] = "RK45"

    #: Parameters whose values affect the symbolic *structure* of _equations
    #: (e.g. integer loop bounds). These are baked in at compile time.
    _structural_params: ClassVar[frozenset[str]] = frozenset()

    # Per-class in-process cache: module_path (str) → jitcode object.
    # Keyed by a path string derived from (class, dim, structural-param hash).
    _compiled_odes: ClassVar[dict[str, Any]] = {}
    _compiled_lyap:  ClassVar[dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # Subclass interface
    # ------------------------------------------------------------------ #

    @staticmethod
    @abstractmethod
    def _equations(y, t, **params) -> Sequence:
        """
        Build the symbolic RHS.

        Parameters
        ----------
        y : JiTCODE ``y``-accessor — call ``y(i)`` for state component ``i``.
        t : JiTCODE time symbol.
        **params
            Current parameter values.  For non-structural params these are
            SymEngine symbols during compilation and float values during any
            Python-fallback evaluation.

        Returns
        -------
        Sequence of ``dim`` SymEngine expressions.

        Notes
        -----
        Use only symbolic / arithmetic operations.  No NumPy, no ``math``,
        no Python ``if``.  Periodic index: ``y((i+1) % N)``.
        """
        ...

    # ------------------------------------------------------------------ #
    # Internal compilation helpers
    # ------------------------------------------------------------------ #

    def _structural_vals(self) -> dict[str, Any]:
        """Return the structural parameter key→value pairs (baked in)."""
        return {k: self.params[k] for k in type(self)._structural_params}

    def _control_params(self) -> dict[str, Any]:
        """Return the non-structural parameters (become control_pars)."""
        structural = type(self)._structural_params
        return {k: v for k, v in self.params.items() if k not in structural}

    def _module_path(self) -> pathlib.Path:
        """
        Stable filesystem path for the compiled module.

        Based on: class name + dim + hash of structural params.
        Changing non-structural params does NOT produce a new path because
        those become runtime control parameters.
        """
        import hashlib
        import json

        struct_vals = self._structural_vals()
        if struct_vals:
            h = hashlib.md5(
                json.dumps(sorted(struct_vals.items()), default=str).encode()
            ).hexdigest()[:8]
            name = f"tsdyn_{type(self).__name__}_{self.dim}_{h}"
        else:
            name = f"tsdyn_{type(self).__name__}_{self.dim}"
        return _CACHE_DIR / name

    def _ensure_compiled(self, for_lyap: bool = False, n_lyap: int = 0) -> Any:
        """
        Return a compiled JiTCODE object, compiling (and caching) if needed.

        For regular integration uses ``jitcode``; for Lyapunov uses
        ``jitcode_lyap``. The compiled shared library is saved to
        ``_CACHE_DIR`` so subsequent runs load from disk without recompiling.

        Parameters
        ----------
        for_lyap : bool
            If True, return a ``jitcode_lyap`` instance.
        n_lyap : int
            Number of Lyapunov exponents (only used when ``for_lyap=True``).

        Returns
        -------
        jitcode or jitcode_lyap

        Notes
        -----
        ``jitcode_lyap`` objects are **not** cached in-process.  After
        integration the object carries internal state (tangent vectors); if
        the same object is then handed back to a second ``lyapunov_spectrum``
        call, ``jitcode.set_integrator`` tries to restore that stale state by
        calling ``jitcode_lyap.set_initial_value`` with the full augmented
        vector, which appends extra random Lyapunov directions and produces a
        dimension mismatch.  Creating a fresh wrapper each call is negligible
        cost; the disk cache already prevents C recompilation.
        """
        import symengine
        from jitcode import jitcode as _jitcode
        from jitcode import jitcode_lyap as _jitcode_lyap
        from jitcode import t as t_sym
        from jitcode import y

        module_path = self._module_path()

        # Regular integration: safe to cache in-process (jitcode objects are
        # stateless between set_initial_value calls once the .so is loaded).
        # Lyapunov: always create a fresh wrapper — see docstring.
        if not for_lyap:
            cache = type(self)._compiled_odes
            cache_key = str(module_path)
            if cache_key in cache:
                return cache[cache_key]

        control_keys = list(self._control_params())
        control_syms = {k: symengine.Symbol(k) for k in control_keys}
        control_par_list = list(control_syms.values())

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Check disk cache
        so_suffix = f"_lyap{n_lyap}" if for_lyap else ""
        saved_path = pathlib.Path(str(module_path) + so_suffix)
        # JiTCODE appends a platform-specific extension; check for any match
        existing = list(_CACHE_DIR.glob(f"{saved_path.name}.*"))

        cls_jitc   = _jitcode_lyap if for_lyap else _jitcode
        lyap_kwargs = {"n_lyap": n_lyap} if for_lyap else {}

        if existing:
            # Load from disk — no recompilation needed
            ode = cls_jitc(
                module_location=str(existing[0]),
                n=self.dim,
                control_pars=control_par_list,
                verbose=False,
                **lyap_kwargs,
            )
        else:
            # Build symbolic RHS and compile fresh
            struct_vals = self._structural_vals()
            all_params  = {**struct_vals, **control_syms}
            f_sym = list(type(self)._equations(y, t_sym, **all_params))
            if len(f_sym) != self.dim:
                raise ValueError(
                    f"_equations must return {self.dim} expressions, got {len(f_sym)}"
                )
            ode = cls_jitc(
                f_sym,
                n=self.dim,
                control_pars=control_par_list,
                verbose=False,
                **lyap_kwargs,
            )
            ode.generate_f_C()
            ode.save_compiled(destination=str(saved_path), overwrite=True)

        if not for_lyap:
            cache[cache_key] = ode
        return ode

    # ------------------------------------------------------------------ #
    # Integration
    # ------------------------------------------------------------------ #

    def integrate(
        self,
        final_time: float = 100.0,
        dt: float = 0.02,
        *,
        t0: float = 0.0,
        ic: Any | None = None,
        method: str | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        **integrator_kwargs,
    ) -> Trajectory:
        """
        Integrate the ODE and return a :class:`~tsdynamics.base.Trajectory`.

        Parameters
        ----------
        final_time : float
            End of integration window. Default 100.0.
        dt : float
            Output sampling interval. The internal stepper is adaptive.
        t0 : float
            Start time. Default 0.0. Allows warm restarts from a non-zero
            time (the IC is interpreted as the state at ``t0``).
        ic : array-like, optional
            Initial state at ``t0``. Falls back to ``self.ic``, then
            ``U[0, 1)^dim``.
        method : str, optional
            Integrator: ``"RK45"`` / ``"dopri5"`` (default), ``"DOP853"``,
            ``"LSODA"``, ``"VODE"``.
        rtol, atol : float
            Solver tolerances (default 1e-6 / 1e-9).
        **integrator_kwargs
            Forwarded to ``jitcode.set_integrator`` (e.g. ``max_step``).

        Returns
        -------
        Trajectory
            Supports tuple-unpacking: ``t, y = sys.integrate(...)``.
        """
        method     = method or self._default_method
        integ_name = _INTEGRATOR_MAP.get(method, method)
        ic_arr     = self.resolve_ic(ic)
        t_eval     = _make_t_eval(t0, final_time, dt)

        ode = self._ensure_compiled(for_lyap=False)
        ode.set_integrator(integ_name, rtol=rtol, atol=atol, **integrator_kwargs)
        ode.set_parameters(*self._control_params().values())
        ode.set_initial_value(ic_arr, t0)

        y_out = np.empty((t_eval.size, self.dim), dtype=float)
        for k, tk in enumerate(t_eval):
            y_out[k] = ode.integrate(float(tk))

        return Trajectory(t=t_eval, y=y_out, system=self)

    # ------------------------------------------------------------------ #
    # Lyapunov spectrum
    # ------------------------------------------------------------------ #

    def lyapunov_spectrum(
        self,
        final_time: float = 200.0,
        dt: float = 0.1,
        *,
        ic: Any | None = None,
        n_exp: int | None = None,
        burn_in: float = 50.0,
        method: str | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        **integrator_kwargs,
    ) -> np.ndarray:
        """
        Estimate the Lyapunov spectrum using :func:`jitcode.jitcode_lyap`.

        Results are stored in ``self.meta['lyapunov_spectrum']``.

        Parameters
        ----------
        final_time : float
            Averaging window length after burn-in. Default 200.0.
        dt : float
            Sampling interval for local exponent accumulation. Default 0.1.
        ic : array-like, optional
            Initial state. Falls back to ``self.ic``, then random.
        n_exp : int, optional
            Number of exponents. Defaults to ``dim``.
        burn_in : float
            Discard this much time before averaging. Default 50.0.
        method : str, optional
            Integrator (default ``"RK45"``).
        rtol, atol : float
            Tolerances.

        Returns
        -------
        ndarray, shape (n_exp,)
            Lyapunov exponents ordered from largest to smallest.
        """
        n_exp  = n_exp or self.dim
        method = method or self._default_method
        integ_name = _INTEGRATOR_MAP.get(method, method)
        ic_arr = self.resolve_ic(ic)

        ode = self._ensure_compiled(for_lyap=True, n_lyap=n_exp)
        ode.set_integrator(integ_name, rtol=rtol, atol=atol, **integrator_kwargs)
        ode.set_parameters(*self._control_params().values())
        ode.set_initial_value(ic_arr, 0.0)

        # Burn-in
        T = 0.0
        while burn_in > T:
            Tn = min(burn_in, T + dt)
            ode.integrate(Tn)
            T = Tn

        # Production: time-weighted average of local exponents
        T_end   = float(ode.t) + final_time
        weights = []
        ly_steps = []
        T = float(ode.t)

        while T_end > T:
            Tn  = min(T_end, T + dt)
            ret = ode.integrate(Tn)

            local_lyaps = ret[1] if isinstance(ret, tuple) and len(ret) >= 2 else ret
            v = np.asarray(local_lyaps, float).ravel()
            if v.size != n_exp:
                raise ValueError(f"Expected {n_exp} local LEs, got shape {v.shape}")

            ly_steps.append(v)
            weights.append(Tn - T)
            T = Tn

        W = np.asarray(weights, float)
        L = np.vstack(ly_steps) if ly_steps else np.empty((0, n_exp), float)
        exponents = (
            (W[:, None] * L).sum(axis=0) / W.sum() if L.size else np.zeros(n_exp)
        )

        self.meta["lyapunov_spectrum"] = exponents
        return exponents
