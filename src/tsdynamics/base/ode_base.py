"""ContinuousSystem — ODE base class via JiTCODE."""

from __future__ import annotations

import os
import pathlib
import sysconfig
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np

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


def _make_t_eval(t0: float, tf: float, dt: float) -> np.ndarray:
    """Build a uniform output grid from t0 to tf (inclusive)."""
    t_arr = np.arange(t0, tf, dt)
    if t_arr.size == 0 or t_arr[-1] < tf - 1e-12:
        t_arr = np.append(t_arr, tf)
    return t_arr


def _resolve_derivative_nodes(expr):
    """
    Replace unevaluated SymEngine ``Derivative`` nodes with a.e. derivatives.

    SymEngine leaves ``d|u|/du`` and ``d sign(u)/du`` unevaluated, which
    ``Lambdify`` cannot compile.  Almost everywhere, ``d sign(u)/d· = 0``
    and ``d|u|/du = sign(u)`` — the measure-zero kink is irrelevant for
    numeric Jacobian evaluation along an orbit.

    Works directly on the SymEngine tree: these nodes may differentiate with
    respect to *expressions* (chain-rule dummies), which cannot round-trip
    through SymPy at all.
    """
    import symengine

    if "Derivative" not in str(expr):
        return expr

    def walk(e):
        name = type(e).__name__
        if name == "Derivative":
            target = e.args[0]
            wrt = e.args[1]
            tname = type(target).__name__
            if tname == "sign":
                return symengine.Integer(0)
            if tname == "Abs":
                g = target.args[0]
                if g == wrt:
                    return symengine.sign(g)
                try:  # wrt may be a symbol or a y(i) application — diff handles both
                    return symengine.sign(g) * walk(g.diff(wrt))
                except RuntimeError:
                    return e
            return e  # unknown derivative — leave it; Lambdify will fail loudly
        args = e.args
        if not args:
            return e
        new_args = [walk(a) for a in args]
        if all(na is a for na, a in zip(new_args, args, strict=True)):
            return e
        try:
            return e.func(*new_args)
        except (TypeError, RuntimeError):
            # Some node types (e.g. Piecewise) flatten their args and cannot
            # be reconstructed via func(*args) — keep the original node.
            return e

    return walk(symengine.sympify(expr))


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

    #: Whether JiTCODE should run SymEngine's ``simplify(ratio=1)`` on each RHS
    #: expression before emitting C.  ``None`` keeps JiTCODE's own default
    #: (enabled for ``dim <= 10``).  Set ``False`` on systems whose RHS is a
    #: large rational expression: the simplify pass is super-linear and can
    #: effectively hang at compile time, while the C compiler optimises the
    #: unsimplified code just as well (see ``BlinkingRotlet``).
    _compile_simplify: ClassVar[bool | None] = None

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
    # _lambdified     : cache_key (str) → (rhs_fn, jac_fn, control_names) —
    #                   SymEngine-Lambdified numeric RHS/Jacobian evaluators,
    #                   keyed like the compile cache (class + dim + structural).
    _compiled_odes: ClassVar[dict[str, Any]] = {}
    _compiled_lyap: ClassVar[dict[str, str]] = {}
    _compiled_so: ClassVar[dict[str, str]] = {}
    _lambdified: ClassVar[dict[str, tuple]] = {}

    # Protocol stepping state (instances shadow these class defaults on first
    # ``reinit``).  Each instance owns a private jitcode wrapper loaded from
    # a unique copy of the compiled .so, so two live steppers never clobber
    # each other and module re-initialisation UB cannot occur.
    _stepper: Any = None
    _state_now: np.ndarray | None = None
    _t_now: float = 0.0
    _default_step_dt: ClassVar[float] = 0.01
    _stepper_counter: ClassVar[int] = 0

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

    def _equations_hash(self) -> str:
        """
        Content hash of the RHS definition, part of every compile-cache key.

        Without it, two same-named classes (user shadowing a builtin, or a
        notebook cell redefining a class with edited equations) would silently
        reuse each other's compiled dynamics.
        """
        import hashlib
        import inspect

        fn = type(self)._equations
        fn = getattr(fn, "__func__", fn)
        try:
            src = inspect.getsource(fn)
        except (OSError, TypeError):  # dynamically defined without source
            src = repr(getattr(fn, "__code__", fn).co_code)
        return hashlib.md5(src.encode()).hexdigest()[:8]

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

        eq = self._equations_hash()
        struct_vals = self._structural_vals()
        if struct_vals:
            # 64-bit slice of the MD5 digest matches ParamSet.param_hash so all
            # cache keys in the project share a single collision budget.
            h = hashlib.md5(
                json.dumps(sorted(struct_vals.items()), default=str).encode()
            ).hexdigest()[:16]
            name = f"tsdyn_{type(self).__name__}_{self.dim}_{h}_{eq}"
        else:
            name = f"tsdyn_{type(self).__name__}_{self.dim}_{eq}"
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
            # save_compiled writes plain "<name>.so"; interrupted-compile
            # recovery writes the full EXT_SUFFIX form — accept both.
            hits = [
                f
                for f in _CACHE_DIR.glob(f"{base.name}.*")
                if (f.name.endswith(_EXT_SUFFIX) or f.name.endswith(".so")) and f.stat().st_size > 0
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
                type(self)._compiled_so[cache_key] = str(so)
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
                    type(self)._compiled_so[cache_key] = str(so)
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
        ode.generate_f_C(simplify=type(self)._compile_simplify)
        so = pathlib.Path(ode.save_compiled(destination=str(dest), overwrite=True))

        if for_lyap:
            type(self)._compiled_lyap[cache_key] = str(so)
        else:
            type(self)._compiled_odes[cache_key] = ode
            type(self)._compiled_so[cache_key] = str(so)
        return ode

    # ------------------------------------------------------------------ #
    # System protocol — incremental stepping
    # ------------------------------------------------------------------ #

    @property
    def is_discrete(self) -> bool:
        """ODEs are continuous-time systems."""
        return False

    def _so_path(self) -> str:
        """Ensure the non-lyap module is compiled and return its .so path."""
        cache_key = str(self._module_path())
        so = type(self)._compiled_so.get(cache_key)
        if so is not None and not pathlib.Path(so).exists():
            # Cache directory was wiped while we were running — recompile.
            so = None
            type(self)._compiled_so.pop(cache_key, None)
            type(self)._compiled_odes.pop(cache_key, None)
        self._ensure_compiled(for_lyap=False)
        so = type(self)._compiled_so.get(cache_key)
        if so is None:  # object-cache hit recorded before path tracking
            hits = [
                f
                for f in _CACHE_DIR.glob(f"{self._module_path().name}.*")
                if (f.name.endswith(_EXT_SUFFIX) or f.name.endswith(".so")) and f.stat().st_size > 0
            ]
            if not hits:
                raise RuntimeError(f"{type(self).__name__}: compiled module not found on disk.")
            so = str(hits[0])
            type(self)._compiled_so[cache_key] = so
        return so

    def _fresh_stepper(self) -> Any:
        """
        Build a private jitcode wrapper with genuinely isolated module state.

        Loading the same compiled ``.so`` path twice re-runs its single-phase
        init on shared static state — undefined behaviour that can segfault
        once many extension modules are loaded.  Each stepper therefore loads
        a uniquely named *copy* of the module, giving it its own dlopen
        handle and namespace.
        """
        import shutil
        import tempfile
        import weakref

        import symengine
        from jitcode import jitcode as _jitcode

        so = pathlib.Path(self._so_path())
        type(self)._stepper_counter += 1

        # Reclaim the previous stepper's copy before making a new one —
        # parameter sweeps reinit thousands of times and must not accumulate
        # one temp dir per reinit.
        old = getattr(self, "_stepper_tmpdir", None)
        if old:
            shutil.rmtree(old, ignore_errors=True)

        # Same filename (PyInit_<stem> is baked into the binary), unique
        # directory → distinct dlopen image per stepper.
        tmpdir = tempfile.mkdtemp(
            prefix=f"tsdyn_stepper_{os.getpid()}_{type(self)._stepper_counter}_"
        )
        unique = pathlib.Path(tmpdir) / so.name
        shutil.copy(so, unique)
        # Keep the temp dir referenced for the stepper's lifetime, and make
        # sure instance death / interpreter exit removes it.
        self._stepper_tmpdir = tmpdir
        weakref.finalize(self, shutil.rmtree, tmpdir, ignore_errors=True)

        control_syms = [symengine.Symbol(k) for k in self._control_params()]
        return _jitcode(
            module_location=str(unique),
            n=self.dim,
            control_pars=control_syms,
            verbose=False,
        )

    def reinit(
        self,
        u: Any | None = None,
        *,
        t: float | None = None,
        params: dict | None = None,
        method: str | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        **integrator_kwargs,
    ) -> None:
        """
        (Re)start the incremental stepper from state ``u`` at time ``t``.

        Parameters
        ----------
        u : array-like, optional
            Initial state (falls back to ``self.ic``, then random).
        t : float, optional
            Start time (default 0.0).
        params : dict, optional
            Parameter overrides applied (in place) before restarting.
        method, rtol, atol, **integrator_kwargs
            Stepper configuration, as in :meth:`integrate`.
        """
        if params:
            for k, v in params.items():
                self.params[k] = v
        t0 = float(t) if t is not None else 0.0
        ic_arr = self.resolve_ic(u)

        stepper = self._fresh_stepper()
        m = method or self._default_method
        stepper.set_integrator(_INTEGRATOR_MAP.get(m, m), rtol=rtol, atol=atol, **integrator_kwargs)
        stepper.set_parameters(*self._control_params().values())
        stepper.set_initial_value(ic_arr, t0)

        self._stepper = stepper
        self._state_now = ic_arr.copy()
        self._t_now = t0

    def step(self, n_or_dt: float | None = None) -> np.ndarray:
        """
        Advance the system by ``dt`` (default 0.01) and return the new state.

        The first call performs an implicit :meth:`reinit`.  Parameter changes
        made after ``reinit`` take effect on the next ``reinit``, not on a
        live stepper.
        """
        if self._stepper is None:
            self.reinit()
        dt = float(n_or_dt) if n_or_dt is not None else self._default_step_dt
        self._t_now = self._t_now + dt
        state = np.asarray(self._stepper.integrate(self._t_now), dtype=float)
        if not np.isfinite(state).all():
            raise RuntimeError(
                f"{type(self).__name__}: ODE diverged at t={self._t_now:.6g} during step()."
            )
        self._state_now = state.copy()
        return state

    def state(self) -> np.ndarray:
        """Return a copy of the current state (implicit ``reinit`` if cold)."""
        if self._state_now is None:
            self.reinit()
        return self._state_now.copy()

    def set_state(self, u: Any) -> None:
        """Overwrite the current state without changing the current time."""
        u_arr = np.asarray(u, dtype=float).reshape(self.dim)
        if self._stepper is None:
            self.reinit(u_arr)
        else:
            self._stepper.set_initial_value(u_arr, self._t_now)
            self._state_now = u_arr.copy()

    def time(self) -> float:
        """Return the current stepper time."""
        return self._t_now

    def trajectory(
        self,
        final_time: float = 100.0,
        *,
        dt: float = 0.02,
        transient: float = 0.0,
        **kwargs,
    ) -> Trajectory:
        """Protocol-uniform trajectory: ``integrate`` plus optional transient drop."""
        traj = self.integrate(final_time=transient + final_time, dt=dt, **kwargs)
        return traj.after(transient) if transient > 0 else traj

    # ------------------------------------------------------------------ #
    # Symbolic Jacobian autogeneration + numeric RHS
    # ------------------------------------------------------------------ #

    def jacobian_sym(self) -> list[list[Any]]:
        """
        Return the symbolic Jacobian of ``_equations``, differentiated by SymEngine.

        Rows are ``d f_i / d y(j)`` for the *current* structural parameters;
        non-structural parameters appear as symbols.  Hand-written
        ``_jacobian`` methods on system classes are never used at runtime —
        this autogenerated form is the single source of truth (the test suite
        cross-checks hand-written ones against it).

        Returns
        -------
        list of ``dim`` rows, each a list of ``dim`` SymEngine expressions.
        """
        import symengine
        from jitcode import t as t_sym
        from jitcode import y

        struct_vals = self._structural_vals()
        control_syms = {k: symengine.Symbol(k) for k in self._control_params()}
        f_sym = list(type(self)._equations(y, t_sym, **{**struct_vals, **control_syms}))
        if len(f_sym) != self.dim:
            raise ValueError(f"_equations must return {self.dim} expressions, got {len(f_sym)}")
        return [
            [_resolve_derivative_nodes(symengine.sympify(fi).diff(y(j))) for j in range(self.dim)]
            for fi in f_sym
        ]

    def _build_lambdified(self) -> tuple[Any, Any, list[str]]:
        """
        Build (and cache) SymEngine-Lambdified numeric RHS and Jacobian.

        Both take a flat argument vector ``[y_0..y_{dim-1}, t, *control_params]``.
        Cached per (class, dim, structural-hash) — parameter value changes
        need no rebuild because control params are call-time arguments.
        """
        key = str(self._module_path())
        cached = type(self)._lambdified.get(key)
        if cached is not None:
            return cached

        import symengine
        from jitcode import t as t_sym
        from jitcode import y

        struct_vals = self._structural_vals()
        control_names = list(self._control_params())
        control_syms = {k: symengine.Symbol(k) for k in control_names}
        f_sym = [
            symengine.sympify(e)
            for e in type(self)._equations(y, t_sym, **{**struct_vals, **control_syms})
        ]
        jac_rows = [
            [_resolve_derivative_nodes(fi.diff(y(j))) for j in range(self.dim)] for fi in f_sym
        ]

        # Lambdify needs plain symbols — swap the y(i) function applications out.
        y_syms = [symengine.Symbol(f"y_{i}") for i in range(self.dim)]
        subs = {y(i): y_syms[i] for i in range(self.dim)}
        args = [*y_syms, t_sym, *(control_syms[k] for k in control_names)]
        rhs_fn = symengine.Lambdify(args, [e.subs(subs) for e in f_sym])
        jac_fn = symengine.Lambdify(args, [e.subs(subs) for row in jac_rows for e in row])

        entry = (rhs_fn, jac_fn, control_names)
        type(self)._lambdified[key] = entry
        return entry

    def jacobian(self, u: Any, t: float = 0.0) -> np.ndarray:
        """
        Evaluate the (autogenerated) Jacobian numerically at state ``u``.

        Parameters
        ----------
        u : array-like, shape (dim,)
            State at which to evaluate.
        t : float
            Time (matters only for non-autonomous systems).

        Returns
        -------
        ndarray, shape (dim, dim)
        """
        _, jac_fn, control_names = self._build_lambdified()
        vals = [float(self.params[k]) for k in control_names]
        arg = np.concatenate([np.asarray(u, dtype=float).ravel(), [t], vals])
        return np.asarray(jac_fn(arg), dtype=float).reshape(self.dim, self.dim)

    def _rhs_numeric(self):
        """
        Return a fast numeric RHS callable ``f(u, t) -> ndarray``.

        Parameter values are captured at call time of this method; build a
        fresh callable after changing parameters.  Used by figure tooling,
        Poincaré crossing refinement, and backend cross-validation — the
        compiled JiTCODE path remains the integrator of record.
        """
        rhs_fn, _, control_names = self._build_lambdified()
        vals = np.array([float(self.params[k]) for k in control_names])

        def rhs(u: Any, t: float = 0.0) -> np.ndarray:
            arg = np.concatenate([np.asarray(u, dtype=float).ravel(), [t], vals])
            return np.asarray(rhs_fn(arg), dtype=float).ravel()

        return rhs

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
        backend: str = "jitcode",
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
            ``"LSODA"``, ``"VODE"``.  The diffsol backend maps these onto
            ``tsit45`` / ``bdf`` / ``tr_bdf2`` / ``esdirk34``.
        rtol, atol : float
            Solver tolerances (default 1e-6 / 1e-9).
        backend : {"jitcode", "diffsol", "auto"}
            ``"jitcode"`` (default) compiles the RHS to C; ``"diffsol"`` uses
            the Rust solver suite via LLVM JIT — no C compiler, prebuilt
            wheels (``pip install tsdynamics[diffsol]``), ~10× faster on small
            chaotic systems, and validated against JiTCODE across the whole
            ODE catalogue (see :mod:`tsdynamics.backends.diffsol`).  ``"auto"``
            picks ``"diffsol"`` when it is installed, else ``"jitcode"`` —
            the recommended zero-compiler fast path.
        **integrator_kwargs
            Forwarded to ``jitcode.set_integrator`` (e.g. ``max_step``).

        Returns
        -------
        Trajectory
            Supports tuple-unpacking: ``t, y = sys.integrate(...)``.
        """
        if backend == "auto":
            # Prefer the zero-compiler Rust path when its optional dependency
            # is installed; otherwise fall back to the always-available
            # JiTCODE path. Lets `tsdynamics[diffsol]` users get the fast
            # backend without naming it, with no surprise for everyone else.
            from tsdynamics.backends import diffsol as _diffsol

            backend = "diffsol" if _diffsol.available() else "jitcode"

        if backend == "diffsol":
            from tsdynamics.backends import diffsol as _diffsol

            ic_arr = self.resolve_ic(ic)
            t_eval, y_out = _diffsol.integrate(
                self, final_time, dt, t0=t0, ic=ic_arr, method=method, rtol=rtol, atol=atol
            )
            return Trajectory(
                t=t_eval,
                y=y_out,
                system=self,
                meta=self._provenance(
                    family="ode",
                    backend="diffsol",
                    method=method or "tsit45",
                    dt=dt,
                    t0=t0,
                    rtol=rtol,
                    atol=atol,
                    ic=ic_arr.copy(),
                ),
            )
        if backend != "jitcode":
            raise ValueError(f"Unknown backend {backend!r}; use 'jitcode', 'diffsol', or 'auto'.")

        method = method or self._default_method
        integ_name = _INTEGRATOR_MAP.get(method, method)
        ic_arr = self.resolve_ic(ic)
        t_eval = _make_t_eval(t0, final_time, dt)

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

        return Trajectory(
            t=t_eval,
            y=y_out,
            system=self,
            meta=self._provenance(
                family="ode",
                method=integ_name,
                dt=dt,
                t0=t0,
                rtol=rtol,
                atol=atol,
                ic=ic_arr.copy(),
            ),
        )

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

        self.meta.record(
            "lyapunov_spectrum",
            exponents,
            dt=dt,
            final_time=final_time,
            burn_in=burn_in,
            n_exp=n_exp,
            method=integ_name,
        )
        return exponents
