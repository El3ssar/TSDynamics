"""ContinuousSystem — ODE base via Pure-Rust IR steppers + JiTCODE fallback."""

from __future__ import annotations

import os
import pathlib
import sysconfig
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np

from ._ir import CompiledOde, NotLowerableError
from ._ode_lowering import lower_ode_to_ir
from .base import SystemBase, Trajectory

# Platform-specific compiled extension suffix (e.g. ".cpython-312-x86_64-linux-gnu.so").
# Used to filter glob results to actual shared libraries only.
_EXT_SUFFIX: str = sysconfig.get_config_var("EXT_SUFFIX") or ".so"


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
    "RK45": "dopri5",
    "dopri5": "dopri5",
    "DOP853": "dop853",
    "dop853": "dop853",
    "LSODA": "lsoda",
    "lsoda": "lsoda",
    "VODE": "vode",
    "vode": "vode",
}
_EXPLICIT_METHODS = frozenset({"dopri5", "dop853"})

# Rust-native methods (N2.b ERK + N2.c Rosenbrock); optional Jacobian for stiff family.
_STIFF_RUST_METHODS = frozenset({"ROSENBROCK23", "ROSENBROCK34", "RODAS4"})
# LSODA/VODE may route to Rust Rosenbrock only for modest state dimensions (dense ROW solves).
_LSODA_AUTO_ROSS_DIM_CAP = 24
_RUST_NATIVE_METHODS = frozenset(
    {"DP5", "DP8", "TSIT5", "VERN9", "BS3", "RK4", *_STIFF_RUST_METHODS}
)


def _rust_integrator_name(method: str | None, default: str) -> str:
    m = (method or default).upper()
    if m in ("RK45", "DOPRI5"):
        return "DP5"
    if m in ("DOP853",):
        return "DP8"
    return m


def _ode_ir_cache_key(inst: Any) -> tuple[type, int, int]:
    sv = inst._structural_vals()
    hsh = hash(tuple(sorted(sv.items()))) if sv else 0
    return (type(inst), inst.dim, hsh)


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
    Base class for finite-dimensional ODE dynamical systems.

    Subclasses supply a symbolic :meth:`_equations` RHS; integration uses a **pure-Rust**
    adaptive IR stepper when SymEngine lowering to bytecode succeeds (**N2**) and ``method``
    names a catalogue entry. :meth:`integrate` still falls back silently to JiTCODE when lowering
    fails (unsupported expression nodes).

    Variational Lyapunov (:meth:`lyapunov_spectrum`) stays on JiTCODE until milestone **N3**.

    Subclass contract
    -----------------
    1. Declare ``params = {...}`` and ``dim = N`` at class level.
    2. Implement ``_equations`` as a ``@staticmethod`` returning a length-``dim``
       sequence of symbolic expressions lowering can walk (SymEngine + JiTCODE rules).
    3. Optionally declare ``_structural_params`` for loop bounds baked into bytecode.

    Native ``method`` catalogue (Rust, case-insensitive)
    -----------------------------------------------------

    Explicit adaptive Runge–Kutta (I step-size control):

    ``"DP5"``, alias ``"RK45"`` / ``"dopri5"``; ``"DP8"``, alias ``"DOP853"`` / ``"dop853"``;
    ``"TSIT5"``; ``"VERN9"``; ``"BS3"``; fixed-step ``"RK4"``.

    Stiff Rosenbrock–Wanner (PI controller; Jacobian from IR bytecode):

    ``"ROSENBROCK23"``, ``"ROSENBROCK34"``, ``"RODAS4"``.
    Requires a Jacobian in the bytecode; lowering without one emits a warning and JiTCODE is used instead.

    Legacy stiff switchers (**deprecated**):

    ``"LSODA"``, ``"VODE"`` issue :class:`~warnings.DeprecationWarning`. When the Jacobian is available
    and ``dim ≤ 24``, integrate may automatically remap onto ``"ROSENBROCK23"`` instead of JiTCODE;
    larger sparse-friendly problems stay on JiTCODE unless you pick an explicit catalogue method.

    Default integrator behaviour
    ----------------------------

    Class-level :attr:`_default_method` is ``"RK45"`` — it denotes the SciPy-era name while the Rust hot
    path runs **DP5** (same order family). Smooth non-stiff problems often benefit from ``"DP8"`` /
    ``"TSIT5"`` / ``"VERN9"`` at similar tolerances; switching the library default remains a deliberate **N2.e+**
    release-note change.

    Compilation and caching
    -----------------------
    First :meth:`integrate` on a lowering-capable path decodes bytecode once into an in-memory cache keyed
    by ``(class, dim, hash(structural_values))``. First JiTCODE use still emits a compiled ``.so`` under
    ``~/.cache/tsdynamics/`` (environment ``TSDYNAMICS_CACHE``); changing only non-structural parameters
    reuses bytecode / shared objects across calls.

    Class-level attributes
    ----------------------
    _structural_params : frozenset[str]
        Symbols baked into the IR / JIT like Lorenz96's loop length ``N``.

    _default_method : str
        Default ``integrate(method=…)`` stem; aliases above still apply.

    Examples
    --------
    >>> lor = Lorenz()
    >>> traj = lor.integrate(final_time=100, dt=0.01)
    >>> t, y = traj          # tuple-unpack
    >>> lor.sigma = 15.0     # change param — bytecode stays valid
    >>> traj2 = lor.integrate(final_time=100)
    """

    _default_method: ClassVar[str] = "RK45"

    #: Parameters whose values affect the symbolic *structure* of _equations
    #: (e.g. integer loop bounds). These are baked in at compile time.
    _structural_params: ClassVar[frozenset[str]] = frozenset()

    # Per-class in-process caches.
    # _compiled_odes  : cache_key (str) → jitcode object (stateless between
    #                   set_initial_value calls; safe to reuse).
    # _compiled_lyap  : lyap_cache_key (str) → absolute path of the saved .so.
    #                   jitcode_lyap objects are NOT cached (they carry tangent-
    #                   vector state); we cache the path so each call can create
    #                   a fresh wrapper without recompiling.
    _compiled_odes: ClassVar[dict[str, Any]] = {}
    _compiled_lyap: ClassVar[dict[str, str]] = {}
    #: SymEngine → IR bytecode for the Rust stepper (N2); keyed like JiTCODE.
    _compiled_ode_ir: ClassVar[dict[tuple[type, int, int], CompiledOde]] = {}

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
            # 64-bit slice of the MD5 digest matches ParamSet.param_hash so all
            # cache keys in the project share a single collision budget.
            h = hashlib.md5(
                json.dumps(sorted(struct_vals.items()), default=str).encode()
            ).hexdigest()[:16]
            name = f"tsdyn_{type(self).__name__}_{self.dim}_{h}"
        else:
            name = f"tsdyn_{type(self).__name__}_{self.dim}"
        return _CACHE_DIR / name

    def _ensure_compiled(self, for_lyap: bool = False, n_lyap: int = 0) -> Any:
        """
        Return a compiled JiTCODE object, compiling (and caching) if needed.

        ``jitcode`` objects (regular integration) are cached in-process because
        they are stateless between ``set_initial_value`` calls.
        ``jitcode_lyap`` objects are **not** cached — they carry tangent-vector
        state that must start fresh each call.  Instead, the compiled ``.so``
        path is cached so each call can construct a new wrapper without
        recompiling.

        Lookup order
        ------------
        1. In-process object cache (regular) / path cache (lyap).
        2. Disk cache (``_CACHE_DIR``).
        3. Check ``sys.modules``: if a prior interrupted compilation registered
           the module name but never saved the ``.so`` to disk, recover from the
           still-live temp dir or fall back to a unique per-process module name
           so the session keeps working without a kernel restart.
        4. Fresh compilation.
        """
        import shutil
        import sys

        import symengine
        from jitcode import jitcode as _jitcode
        from jitcode import jitcode_lyap as _jitcode_lyap
        from jitcode import t as t_sym
        from jitcode import y
        from jitcxde_common.modules import modulename_from_path

        cls_jitc = _jitcode_lyap if for_lyap else _jitcode
        lyap_kwargs = {"n_lyap": n_lyap} if for_lyap else {}

        so_suffix = f"_lyap{n_lyap}" if for_lyap else ""
        saved_path = pathlib.Path(str(self._module_path()) + so_suffix)
        cache_key = str(saved_path)

        control_syms = {k: symengine.Symbol(k) for k in self._control_params()}
        control_par_list = list(control_syms.values())

        def _load(path: str | pathlib.Path) -> Any:
            return cls_jitc(
                module_location=str(path),
                n=self.dim,
                control_pars=control_par_list,
                verbose=False,
                **lyap_kwargs,
            )

        def _find_so(base: pathlib.Path) -> pathlib.Path | None:
            hits = [
                f
                for f in _CACHE_DIR.glob(f"{base.name}.*")
                if f.name.endswith(_EXT_SUFFIX) and f.stat().st_size > 0
            ]
            return hits[0] if hits else None

        # 1. In-process cache
        if not for_lyap and cache_key in type(self)._compiled_odes:
            return type(self)._compiled_odes[cache_key]
        if for_lyap:
            cached = type(self)._compiled_lyap.get(cache_key)
            if cached and pathlib.Path(cached).exists():
                return _load(cached)
            if cached:
                del type(self)._compiled_lyap[cache_key]  # stale — clear it

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # 2. Disk cache
        so = _find_so(saved_path)
        if so:
            ode = _load(so)
            if for_lyap:
                type(self)._compiled_lyap[cache_key] = str(so)
            else:
                type(self)._compiled_odes[cache_key] = ode
            return ode

        # 3. Determine compile destination.
        # JiTCODE registers the module name in sys.modules inside compile_C,
        # *before* save_compiled copies the .so to its permanent location.
        # If a prior call was interrupted in that window, the name is already
        # in sys.modules and a fresh compile_C would raise NameError.
        # Check upfront and either recover from the still-live temp dir or
        # redirect to a unique per-process name.
        dest = saved_path
        mname = modulename_from_path(str(saved_path))
        if mname in sys.modules:
            live_so = getattr(sys.modules[mname], "__file__", None)
            if live_so and pathlib.Path(live_so).exists():
                # The interrupted call's temp dir is still alive — copy the
                # .so to the permanent location and load from there.
                so = pathlib.Path(str(saved_path) + _EXT_SUFFIX)
                try:
                    shutil.copy(live_so, so)
                except OSError:
                    so = pathlib.Path(live_so)
                ode = _load(so)
                if for_lyap:
                    type(self)._compiled_lyap[cache_key] = str(so)
                else:
                    type(self)._compiled_odes[cache_key] = ode
                return ode
            # Temp dir already cleaned — compile under a unique name so the
            # session works without a restart.
            dest = pathlib.Path(f"{saved_path}_{os.getpid()}")

        # 4. Fresh compilation
        struct_vals = self._structural_vals()
        f_sym = list(type(self)._equations(y, t_sym, **{**struct_vals, **control_syms}))
        if len(f_sym) != self.dim:
            raise ValueError(f"_equations must return {self.dim} expressions, got {len(f_sym)}")

        ode = cls_jitc(
            f_sym, n=self.dim, control_pars=control_par_list, verbose=False, **lyap_kwargs
        )
        ode.generate_f_C()
        so = pathlib.Path(ode.save_compiled(destination=str(dest), overwrite=True))

        if for_lyap:
            type(self)._compiled_lyap[cache_key] = str(so)
        else:
            type(self)._compiled_odes[cache_key] = ode
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

        Sampling uses a uniform grid from ``t0`` through ``final_time`` (endpoint included),
        analogous to wrapping ``numpy.arange(t0, final_time, dt)`` and appending ``final_time``.

        Parameters
        ----------
        final_time : float
            End of integration window. Default ``100.0``.
        dt : float
            Uniform output spacing. Internal steps are adaptive and unrelated to ``dt``.
            Default ``0.02``.
        t0 : float
            Start time (``ic`` holds the state there).
        ic : array-like, optional
            Initial state resolved via :meth:`~tsdynamics.base.SystemBase.resolve_ic`.
        method : str, optional
            Catalogue name routed to Pure-Rust when IR lowering succeeds.

            Explicit: ``"DP5"`` (**alias** ``"RK45"``, ``"dopri5"``), ``"DP8"`` (**alias**
            ``"DOP853"``, ``"dop853"``), ``"TSIT5"``, ``"VERN9"``, ``"BS3"``, ``"RK4"``.
            Stiff: ``"ROSENBROCK23"``, ``"ROSENBROCK34"``, ``"RODAS4"``.
            Deprecated: ``"LSODA"``, ``"VODE"``.
            Default :attr:`_default_method` is ``"RK45"`` (**DP5** on Rust); smoother problems
            often merit ``method="DP8"`` without changing library defaults globally.
        rtol, atol : float
            Solver tolerances. Default ``1e-6`` / ``1e-9``.
        **integrator_kwargs
            JiTCODE-only keyword arguments forwarded from ``jitcode.set_integrator``.

        Notes
        -----
        If lowering emits :exc:`NotLowerableError`, integrate silently selects JiTCODE with the usual
        SciPy name map. Selecting a Rosenbrock method without a Jacobian issues ``UserWarning`` and
        continues on dopri5.

        Returns
        -------
        Trajectory
            Supports tuple-unpacking: ``t, y = sys.integrate(...)``.
        """
        method = method or self._default_method
        ic_arr = self.resolve_ic(ic)
        t_eval = _make_t_eval(t0, final_time, dt)

        lsoda_vode = str(method).strip().upper() in ("LSODA", "VODE")
        if lsoda_vode:
            warnings.warn(
                "method='LSODA' and method='VODE' are deprecated; for stiff ODEs prefer "
                "method='Rosenbrock23' (or 'Rosenbrock34' / 'Rodas4') when the lowered IR "
                "includes a Jacobian.",
                DeprecationWarning,
                stacklevel=2,
            )

        rust_m = _rust_integrator_name(method, self._default_method)

        try:
            key = _ode_ir_cache_key(self)
            cache = type(self)._compiled_ode_ir
            if key not in cache:
                co = lower_ode_to_ir(
                    type(self),
                    dim=self.dim,
                    params=dict(self.params),
                    structural_params=type(self)._structural_params,
                )
                cache[key] = co
            co = cache[key]
        except NotLowerableError:
            pass
        else:
            eff_rust = rust_m
            if lsoda_vode and co.has_jacobian and co.dim <= _LSODA_AUTO_ROSS_DIM_CAP:
                eff_rust = "ROSENBROCK23"

            use_rust = eff_rust in _RUST_NATIVE_METHODS and (
                eff_rust not in _STIFF_RUST_METHODS or co.has_jacobian
            )

            if eff_rust in _STIFF_RUST_METHODS and not co.has_jacobian:
                warnings.warn(
                    f"{eff_rust.lower()} requires a symbolic Jacobian in the IR; "
                    "using an explicit JiTCODE integrator instead.",
                    UserWarning,
                    stacklevel=2,
                )

            if use_rust:
                from .._native import integrate_ode as _rust_integrate_ode

                nonstruct = [k for k in self.params if k not in type(self)._structural_params]
                params_np = np.asarray([self.params[k] for k in nonstruct], dtype=float)
                t_out, y_out = _rust_integrate_ode(
                    co.bytecode,
                    float(t0),
                    float(final_time),
                    ic_arr,
                    params_np,
                    eff_rust,
                    float(dt),
                    float(rtol),
                    float(atol),
                )
                y_stack = np.asarray(y_out, dtype=float).reshape(t_out.shape[0], self.dim)
                return Trajectory(t=t_out.copy(), y=y_stack, system=self)

        integ_name = _INTEGRATOR_MAP.get(method, method)
        jit_stiff_name = _rust_integrator_name(method, self._default_method)
        if jit_stiff_name in _STIFF_RUST_METHODS:
            integ_name = "dopri5"

        ode = self._ensure_compiled(for_lyap=False)
        ode.set_integrator(integ_name, rtol=rtol, atol=atol, **integrator_kwargs)
        ode.set_parameters(*self._control_params().values())
        ode.set_initial_value(ic_arr, t0)

        y_out = np.empty((t_eval.size, self.dim), dtype=float)
        for k, tk in enumerate(t_eval):
            state = ode.integrate(float(tk))
            if not np.isfinite(state).all():
                raise RuntimeError(
                    f"{type(self).__name__}: ODE diverged at t={tk:.6g} — "
                    f"state contains non-finite values: {state}"
                )
            y_out[k] = state

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
        if n_exp is not None and n_exp <= 0:
            raise ValueError(f"n_exp must be a positive integer, got {n_exp!r}")
        n_exp = n_exp if n_exp is not None else self.dim
        method = method or self._default_method
        integ_name = _INTEGRATOR_MAP.get(method, method)
        ic_arr = self.resolve_ic(ic)

        ode = self._ensure_compiled(for_lyap=True, n_lyap=n_exp)
        ode.set_integrator(integ_name, rtol=rtol, atol=atol, **integrator_kwargs)
        ode.set_parameters(*self._control_params().values())
        ode.set_initial_value(ic_arr, 0.0)

        # Burn-in (discard transient; no exponent accumulation)
        T = 0.0
        while burn_in > T:
            Tn = min(burn_in, T + dt)
            ode.integrate(float(Tn))
            T = Tn

        # Production: time-weighted average of local exponents
        T_end = float(ode.t) + final_time
        weights = []
        ly_steps = []
        T = float(ode.t)

        while T_end > T:
            Tn = min(T_end, T + dt)
            ret = ode.integrate(float(Tn))

            # jitcode_lyap.integrate returns (state, lyapunov_exponents).
            # The fallback "else ret" would silently use the state vector as
            # exponents — always enforce the tuple contract instead.
            if not (isinstance(ret, tuple) and len(ret) >= 2):
                raise RuntimeError(
                    f"{type(self).__name__}: jitcode_lyap.integrate returned "
                    f"unexpected type {type(ret)!r}; expected (state, lyap_exps) tuple."
                )
            local_lyaps = ret[1]
            v = np.asarray(local_lyaps, float).ravel()
            if v.size != n_exp:
                raise ValueError(f"Expected {n_exp} local LEs, got shape {v.shape}")

            ly_steps.append(v)
            weights.append(Tn - T)
            T = Tn

        W = np.asarray(weights, float)
        L = np.vstack(ly_steps) if ly_steps else np.empty((0, n_exp), float)
        exponents = (W[:, None] * L).sum(axis=0) / W.sum() if L.size else np.zeros(n_exp)

        self.meta["lyapunov_spectrum"] = exponents
        return exponents
