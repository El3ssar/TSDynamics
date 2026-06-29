"""Tangent dynamics: state + deviation vectors, the one Lyapunov engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np

from tsdynamics.families import ContinuousSystem, DelaySystem, DiscreteMap

from ._base import DerivedSystem
from ._variational import build_variational_tape, embed_extended, split_extended

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tsdynamics.viz.spec import PlotSpec

__all__ = ["TangentSystem"]

#: ODE backends that integrate the *extended* variational system on the engine
#: (interpreter / Cranelift JIT / pure-Python reference).
_ENGINE_BACKENDS = frozenset({"interp", "jit", "reference"})


def _qr_growths(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reorthonormalise deviation vectors; return ``(Q, log|diag R|)``.

    The logarithmic stretch factors ``log|diag R|`` are the per-step Lyapunov
    contributions; their sign-independence (``|·|``) absorbs the arbitrary
    column-sign convention of ``numpy.linalg.qr``.
    """
    q, r = np.linalg.qr(w)
    diag = np.abs(np.diag(r))
    tiny = np.finfo(float).tiny
    diag = np.where(diag < tiny, tiny, diag)
    return q, np.log(diag)


class TangentSystem(DerivedSystem):
    """
    Evolve a system together with ``k`` deviation (tangent) vectors.

    Each ``step()`` advances the state and the deviation vectors, then
    QR-reorthonormalises; :meth:`growths` exposes the per-step logarithmic
    stretch factors ``log |diag R|`` and :meth:`exponents` their running
    time-average — the Lyapunov spectrum estimate.  :meth:`lyapunov_spectrum`
    wraps that into the standard burn-in + time-averaged estimate, and is the
    single implementation every family's ``lyapunov_spectrum`` delegates to.

    Implementation per family
    -------------------------
    - **Maps**: pure NumPy — ``W ← J(x)·W`` with ``_jacobian`` evaluated at the
      pre-image (the correct tangent-map convention), then QR.
    - **ODEs**: the *extended* ODE (state ⊕ ``k`` tangent vectors, see
      :mod:`tsdynamics.derived._variational`) is lowered to an engine tape and
      integrated per step on the Rust engine (or the pure-Python reference
      oracle), then QR-reorthonormalised here.  Select the variational backend
      with ``backend=``: ``"interp"`` (default), ``"jit"``, or ``"reference"``.
    - **DDEs**: not supported — tangent dynamics of a DDE lives in an
      infinite-dimensional history space; use
      ``DelaySystem.lyapunov_spectrum`` (the engine DDE Lyapunov estimator).

    Parameters
    ----------
    system : DiscreteMap or ContinuousSystem
        The base system whose tangent dynamics to evolve.
    k : int, optional
        Number of deviation vectors (``1 ≤ k ≤ system.dim``).  Defaults to the
        full state dimension.
    backend : str, optional
        ODE variational backend (ignored for maps): ``"interp"`` (default),
        ``"jit"``, or ``"reference"``.

    Examples
    --------
    >>> tang = TangentSystem(Henon(), k=2)
    >>> tang.reinit([0.1, 0.1])
    >>> for _ in range(5000):
    ...     tang.step()
    >>> tang.exponents()        # ≈ [0.42, -1.62]
    """

    _ODE_DEFAULT_DT: ClassVar[float] = 0.1

    def __init__(self, system: Any, k: int | None = None, *, backend: str | None = None) -> None:
        if isinstance(system, DelaySystem):
            raise NotImplementedError(
                "TangentSystem does not support delay systems — their tangent space is the "
                "infinite-dimensional history space. Use DelaySystem.lyapunov_spectrum."
            )
        if isinstance(system, DiscreteMap):
            self._mode = "map"
        elif isinstance(system, ContinuousSystem):
            self._mode = "ode"
        else:
            raise TypeError(
                f"TangentSystem needs a DiscreteMap or ContinuousSystem, "
                f"got {type(system).__name__}."
            )
        super().__init__(system)
        self.k = int(k) if k is not None else system.dim
        if not 1 <= self.k <= system.dim:
            raise ValueError(f"k must be in [1, {system.dim}], got {self.k}")

        # Both maps and ODEs select among the same three backends: the compiled
        # engine (``interp``/``jit``) or the pure-Python ``reference`` oracle.  For
        # maps ``interp``/``jit`` run the Rust QR tangent-map kernel (stream
        # perf/map-lyapunov-kernel) and ``reference`` the pure-NumPy QR loop (also
        # the transparent fallback when the engine declines — a non-lowering
        # ``_step`` or an absent wheel); for ODEs they select the variational
        # integration backend as before.
        self._backend = (backend or "interp").lower()
        if self._backend not in _ENGINE_BACKENDS:
            raise ValueError(
                f"unknown {'map' if self._mode == 'map' else 'ODE'} tangent backend "
                f"{backend!r}; choose from {sorted(_ENGINE_BACKENDS)}."
            )

        self._W: np.ndarray | None = None  # (dim, k) deviation vectors (map mode)
        self._ext_tape: Any = None  # extended variational tape (ode engine mode)
        self._ext_tape_key: Any = None  # structural-param key the tape was built for
        self._ext_tape_arrays: Any = None  # the extended tape's engine wire arrays (cached)
        self._z: np.ndarray | None = None  # extended state (ode engine mode)
        self._t = 0.0
        # ODE integration options (engine mode), captured at reinit.
        self._method: str | None = None
        self._rtol = 1e-6
        self._atol = 1e-9
        self._integrator_kwargs: dict[str, Any] = {}
        # Resumable engine stepper amortisation (stream perf/lyapunov-stepper): when
        # the resolved kernel is *explicit* and the backend is the compiled engine
        # (``interp``/``jit``), the per-dt-chunk loop reuses one durable
        # ``OdeStepper`` handle (built once over the extended variational tape,
        # re-seeded after each QR via ``set_state``) instead of constructing a fresh
        # ``ODEProblem`` + calling ``run.integrate`` every chunk.  ``advance(dt)`` is
        # bit-for-bit identical to the per-dt ``integrate_dense`` it supersedes, so
        # the spectrum is unchanged.  ``make_ode_stepper`` rejects an implicit kernel
        # at construction, so a stiff base flow (an implicit ``_default_method``) and
        # the ``reference`` oracle keep the per-chunk ``run.integrate`` path.
        self._ode_stepper: Any = None  # the durable OdeStepper handle (explicit only)
        self._step_kernel: str | None = None  # canonical kernel name (e.g. "rk45")
        self._step_explicit_engine = False  # eligible for the stepper fast path?
        self._last_growths = np.zeros(self.k)
        self._sum_growths = np.zeros(self.k)
        self._elapsed = 0.0
        # The batch engine map kernel (:meth:`_map_spectrum_engine`) returns only
        # the time-averaged spectrum and the interval count, never the final
        # orthonormal frame nor the last interval's per-step stretch.  When that
        # path runs, the streaming accessors ``deviations()`` / ``growths()`` cannot
        # be reproduced and are marked stale (set True there, cleared by ``reinit``)
        # so they raise an honest error instead of returning the long-run average.
        self._map_engine_stale = False

    def _rebuild(self, inner: Any) -> TangentSystem:
        return TangentSystem(inner, self.k, backend=self._backend)

    # --- lifecycle ---

    def _reset_accumulators(self) -> None:
        """Zero the accumulated and last growth records (start a fresh average)."""
        self._last_growths = np.zeros(self.k)
        self._sum_growths = np.zeros(self.k)
        self._elapsed = 0.0
        self._map_engine_stale = False

    def reinit(
        self,
        u: Any | None = None,
        *,
        t: float | None = None,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Restart state, deviation vectors, and accumulated growth sums."""
        self._reset_accumulators()

        if self._mode == "map":
            self.system.reinit(u, t=t, params=params)
            self._W = np.eye(self.system.dim)[:, : self.k]
            return

        # ode mode — apply any parameter overrides to the inner system first.
        if params:
            for key, value in params.items():
                self.params[key] = value
        ic_arr = self.system.resolve_ic(u)
        t0 = float(t) if t is not None else 0.0
        self._t = t0
        self._method = kwargs.pop("method", None)
        self._rtol = kwargs.pop("rtol", 1e-6)
        self._atol = kwargs.pop("atol", 1e-9)
        self._integrator_kwargs = dict(kwargs)

        self._reinit_ode_engine(ic_arr)

    def _reinit_ode_engine(self, ic_arr: np.ndarray) -> None:
        """Lower the extended variational tape (rebuilding on a structural change).

        The lowered tape bakes in the base system's *structural* parameters
        (``_structural_params``: a variable-dimension count, a delay, …), so a
        live structural change must re-lower it.  The tape is keyed on the
        current structural values and rebuilt when they differ; an ordinary
        *control*-parameter change reads live from the system and needs no
        rebuild.
        """
        key = tuple(sorted(self.system._structural_vals().items()))
        if self._ext_tape is None or key != self._ext_tape_key:
            self._ext_tape = build_variational_tape(self.system, self.k)
            self._ext_tape_key = key
            self._ext_tape_arrays = None  # re-marshal the wire arrays for the new tape
        w0 = np.eye(self.system.dim)[:, : self.k]
        self._z = embed_extended(ic_arr, w0)
        self._setup_ode_stepper()

    def _setup_ode_stepper(self) -> None:
        """Decide the per-chunk integration path and build the resumable stepper.

        Resolves the requested method to a canonical engine kernel.  When that
        kernel is **explicit** and the backend is the compiled engine
        (``interp``/``jit``), build one durable
        :class:`tsdynamics._rust.OdeStepper` over the extended variational tape —
        the per-chunk loop then re-seeds it with ``set_state`` after each QR and
        :func:`~tsdynamics.engine.run.step_advance` advances it one ``dt``,
        marshalling the tape into the engine exactly once instead of per chunk
        (the ``TODO(stepper-amortize)``).  ``advance(dt)`` reproduces the per-dt
        ``integrate_dense`` bit-for-bit, so the spectrum is unchanged.

        The fast path is declined — leaving the per-chunk ``run.integrate`` path in
        :meth:`_step_ode_engine` untouched — for an **implicit / stiff** kernel
        (``make_ode_stepper`` rejects it at construction), for ``method="auto"``
        (whose per-IC stiffness probe is owned by ``run.integrate``), and for the
        ``reference`` backend (no engine stepper).  Any failure to build the
        handle (e.g. the engine wheel absent) silently falls back to the
        per-chunk path.
        """
        self._ode_stepper = None
        self._step_kernel = None
        self._step_explicit_engine = False

        if self._backend == "reference":
            return

        from tsdynamics import solvers
        from tsdynamics.engine import run

        requested = self._method or self.system._default_method
        # ``method="auto"`` is owned by run.integrate (it probes the start-state
        # Jacobian spectrum per IC); leave that path on run.integrate.
        if solvers.normalize(requested) == "auto":
            return
        try:
            resolution = solvers.resolve(requested, family="ode")
        except ValueError:
            return  # an unknown name surfaces loudly on the run.integrate path
        if resolution.is_implicit:
            return  # make_ode_stepper rejects implicit kernels — keep run.integrate

        assert self._z is not None  # caller seeds the extended state first
        if self._ext_tape_arrays is None:
            self._ext_tape_arrays = self._ext_tape.to_arrays()
        try:
            self._ode_stepper = run.make_ode_stepper(
                self._ext_tape_arrays,
                self._z,
                self._t,
                method=resolution.name,
                rtol=self._rtol,
                atol=self._atol,
                jit=self._backend == "jit",
            )
        except Exception:
            # The engine wheel may be absent / decline; fall back transparently to
            # the per-chunk run.integrate path (it raises the canonical errors).
            self._ode_stepper = None
            return
        self._step_kernel = resolution.name
        self._step_explicit_engine = True

    # --- protocol ---

    @property
    def is_discrete(self) -> bool:
        """Match the wrapped system's time semantics."""
        return cast(bool, self.system.is_discrete)

    def step(self, n_or_dt: float | None = None) -> np.ndarray:
        """
        Advance state + deviation vectors and reorthonormalise.

        For maps ``n_or_dt`` is the number of iterations (QR after each);
        for ODEs it is the time increment (default 0.1).
        Returns the new state.
        """
        if self._mode == "map":
            return self._step_map(int(n_or_dt) if n_or_dt is not None else 1)
        dt = float(n_or_dt) if n_or_dt is not None else self._ODE_DEFAULT_DT
        return self._step_ode_engine(dt)

    def _step_map(self, n: int) -> np.ndarray:
        if self._W is None:
            self.reinit()
        sys = self.system
        jac = type(sys)._jacobian
        params = sys.params.as_tuple()
        for _ in range(n):
            x = sys.state()
            j = np.atleast_2d(np.asarray(jac(x, *params), dtype=float))
            sys.step()
            self._W = j @ self._W
            self._W, growths = _qr_growths(self._W)
            self._last_growths = growths
            self._sum_growths += growths
            self._elapsed += 1.0
        return cast(np.ndarray, sys.state())

    def _step_ode_engine(self, dt: float) -> np.ndarray:
        # Advance the extended variational state one ``dt`` chunk, then QR-reorthonormalise.
        #
        # Fast path (stream perf/lyapunov-stepper): for an *explicit* engine kernel
        # the loop reuses one durable ``OdeStepper`` handle built in
        # :meth:`_setup_ode_stepper` — re-seed it with the post-QR extended state
        # (``set_state``) and advance one ``dt`` (``run.step_advance``).  The tape is
        # marshalled into the engine exactly once (at reinit), not per chunk, and
        # ``advance(dt)`` is byte-for-byte identical to the per-dt ``integrate_dense``
        # the slow path called, so the spectrum is unchanged.  The slow per-chunk
        # ``run.integrate`` path below is kept verbatim for the implicit/stiff kernel
        # (``make_ode_stepper`` rejects it), the ``reference`` oracle, ``method="auto"``,
        # and a non-positive ``dt`` (so the canonical ``InvalidParameterError`` still
        # fires) — exactly the cases :meth:`_setup_ode_stepper` declines.
        if self._z is None:
            self.reinit()
        assert self._z is not None  # reinit() seeds the extended state in ODE mode
        from tsdynamics.engine import run

        dim = self.system.dim
        t0 = self._t

        if self._step_explicit_engine and self._ode_stepper is not None and dt > 0:
            assert self._step_kernel is not None
            params_vec = self._engine_params_vec()
            self._ode_stepper.set_state(self._z, t0)
            z_next = run.step_advance(
                self._ode_stepper, dt, params_vec, name=type(self.system).__name__
            )
        else:
            from tsdynamics.engine.problem import ODEProblem

            method = self._method or self.system._default_method
            problem = ODEProblem(tape=self._ext_tape, ic=self._z, t0=t0, system=self.system)
            traj = run.integrate(
                problem,
                final_time=t0 + dt,
                dt=dt,
                t0=t0,
                method=method,
                rtol=self._rtol,
                atol=self._atol,
                backend=self._backend,
            )
            z_next = traj.y[-1]

        x, w = split_extended(z_next, dim, self.k)
        q, growths = _qr_growths(w)
        self._z = embed_extended(x, q)
        self._t = t0 + dt
        self._last_growths = growths
        self._sum_growths += growths
        self._elapsed += dt
        return x

    def _engine_params_vec(self) -> np.ndarray:
        """Return the extended tape's live control-parameter vector.

        The extended variational tape carries the base system's control parameters
        (in ``system._control_params()`` order — see
        :func:`~tsdynamics.derived._variational.build_variational_tape`), so the
        live parameter vector is read off a thin :class:`ODEProblem` wrapping that
        tape and the base system; the values come straight from
        ``system.params``, so a control-parameter change still takes effect on the
        next ``step`` exactly as the per-chunk ``run.integrate`` path did.
        """
        from tsdynamics.engine.problem import ODEProblem

        assert self._z is not None  # only called on the live (seeded) fast path
        prob = ODEProblem(tape=self._ext_tape, ic=self._z, t0=self._t, system=self.system)
        return prob.params_vec()

    def state(self) -> np.ndarray:
        """Return a copy of the current base-system state."""
        if self._mode == "map":
            return cast(np.ndarray, self.system.state())
        # ODE mode: the constructor guarantees an engine backend, so the extended
        # state ``self._z`` always carries the base state in its leading slots.
        if self._z is None:
            self.reinit()
        assert self._z is not None  # reinit() seeds the extended state in ODE mode
        return self._z[: self.system.dim].copy()

    def set_state(self, u: Any) -> None:
        """Overwrite the base state (map mode only — ODE tangent vectors would desync)."""
        if self._mode == "map":
            self.system.set_state(u)
        else:
            raise NotImplementedError(
                "TangentSystem(ode).set_state is not supported — the tangent vectors "
                "would desynchronise. Use reinit(u)."
            )

    def time(self) -> float:
        """Return the current time / iteration count."""
        return self.system.time() if self._mode == "map" else self._t

    # --- Lyapunov accessors ---

    def deviations(self) -> np.ndarray:
        """Return the current orthonormal deviation vectors, shape ``(dim, k)``.

        Available on every supported backend — maps and the ODE engine
        backends both carry the deviation matrix explicitly.
        """
        if self._mode == "map":
            if self._map_engine_stale:
                raise RuntimeError(
                    "deviations() is unavailable after a batch engine map "
                    "lyapunov_spectrum: the Rust kernel returns only the averaged "
                    "spectrum, not the final orthonormal frame. Call reinit() and "
                    "step() to drive the streaming tangent frame explicitly, or use "
                    "backend='reference' for the streaming pure-Python QR loop."
                )
            if self._W is None:
                raise RuntimeError("deviations() is available after reinit()")
            return self._W.copy()
        # ODE engine backend (interp/jit/reference): the constructor rejects any
        # other backend, so the deviation vectors are always carried explicitly.
        if self._z is None:
            raise RuntimeError("deviations() is available after reinit()")
        return split_extended(self._z, self.system.dim, self.k)[1]

    def growths(self) -> np.ndarray:
        """Return the log stretch factors ``log|diag R|`` from the most recent step.

        Raises
        ------
        RuntimeError
            In map mode, after a *batch* engine ``lyapunov_spectrum`` call: the
            Rust kernel returns only the time-averaged spectrum, not the last
            interval's per-step stretch, so the most-recent-step value cannot be
            reproduced. Drive the frame with ``reinit()`` + ``step()`` (or use
            ``backend='reference'``) for streaming per-step growths.
        """
        if self._mode == "map" and self._map_engine_stale:
            raise RuntimeError(
                "growths() is unavailable after a batch engine map "
                "lyapunov_spectrum: the Rust kernel returns only the averaged "
                "spectrum, not the most recent step's log|diag R|. Call reinit() and "
                "step() to drive the streaming tangent frame explicitly, or use "
                "backend='reference' for the streaming pure-Python QR loop."
            )
        return self._last_growths.copy()

    def exponents(self) -> np.ndarray:
        """Return the running Lyapunov-spectrum estimate (accumulated growths / elapsed)."""
        if self._elapsed == 0.0:
            return np.zeros(self.k)
        return self._sum_growths / self._elapsed

    # --- convergence history (the Lyapunov estimates settling over time) ---

    def convergence(
        self,
        steps: int = 2000,
        n_or_dt: float | None = None,
        *,
        ic: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Record the running Lyapunov estimates as they converge.

        Reinitialises the tangent frame and steps it ``steps`` times, capturing
        the running :meth:`exponents` estimate after every step.  The estimates
        settle as the time-average accumulates — the curve a user inspects to
        judge whether a Lyapunov run has converged.

        Parameters
        ----------
        steps : int, optional
            Number of tangent steps to record.  Default ``2000``.
        n_or_dt : float, optional
            Per-step increment (iterations for a map, ``dt`` for an ODE).
            ``None`` uses the family default.
        ic : array-like, optional
            Initial condition; ``None`` resolves the system's default.

        Returns
        -------
        (times, estimates)
            ``times`` shape ``(steps,)``; ``estimates`` shape ``(steps, k)`` —
            the running estimate of each of the ``k`` exponents after each step.
        """
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        # Random-IC retry on divergence, mirroring the map Lyapunov path
        # (:meth:`_map_spectrum`): when called without an explicit ``ic`` the
        # system resolves a random one, which for a small-basin map (Hénon) can
        # land off-attractor and diverge.  Re-draw a fresh IC and retry; an
        # explicitly supplied ``ic`` that diverges is the caller's, so re-raise.
        max_retries = 10
        for attempt in range(max_retries):
            use_ic = ic if attempt == 0 else None
            times = np.empty(steps)
            estimates = np.empty((steps, self.k))
            try:
                self.reinit(use_ic)
                for s in range(steps):
                    self.step(n_or_dt)
                    times[s] = self.time()
                    estimates[s] = self.exponents()
            except RuntimeError:
                if ic is not None or attempt == max_retries - 1:
                    raise
                # Force a fresh random IC on the next attempt.
                object.__setattr__(self.system, "ic", None)
                continue
            return times, estimates
        # Unreachable: the loop returns on success or re-raises on the last
        # attempt; this satisfies the type checker that all paths are covered.
        raise RuntimeError(  # pragma: no cover
            f"{type(self.system).__name__}: tangent convergence diverged from every IC."
        )

    # --- visualization seam ---

    def to_plot_spec(
        self,
        kind: str | None = None,
        *,
        steps: int = 2000,
        n_or_dt: float | None = None,
        ic: Any | None = None,
    ) -> PlotSpec:
        """Describe the Lyapunov-estimate convergence as a :class:`PlotSpec`.

        Builds a :data:`~tsdynamics.viz.spec.PlotKind.DIAGNOSTIC_CURVE` of each
        exponent's running estimate against time — a labelled family of lines
        (one ``LINE`` layer per exponent, legended), the standard read-out for
        "has the Lyapunov spectrum settled?".  The estimates are collected via
        :meth:`convergence`.

        The :mod:`tsdynamics.viz.spec` import is lazy, so building a spec never
        pulls in a plotting backend.

        Parameters
        ----------
        kind : str, optional
            Override the semantic kind.  ``None`` (the default) uses
            ``DIAGNOSTIC_CURVE``.
        steps : int, optional
            Number of tangent steps to record.  Default ``2000``.
        n_or_dt : float, optional
            Per-step increment (iterations for a map, ``dt`` for an ODE).
        ic : array-like, optional
            Initial condition; ``None`` resolves the system's default.

        Returns
        -------
        PlotSpec
        """
        from tsdynamics.viz.spec import Axis, Layer, Legend, PlotKind, PlotSpec

        times, estimates = self.convergence(steps, n_or_dt, ic=ic)
        spec_kind = PlotKind(kind) if kind is not None else PlotKind.DIAGNOSTIC_CURVE
        layers = [
            Layer(
                PlotKind.LINE,
                {"x": times, "y": estimates[:, i]},
                label=f"$\\lambda_{{{i + 1}}}$",
            )
            for i in range(self.k)
        ]
        return PlotSpec(
            kind=spec_kind,
            ndim=2,
            title=f"Lyapunov convergence — {type(self.system).__name__}",
            x=Axis(label="iteration" if self.is_discrete else "time"),
            y=Axis(label="Lyapunov estimate"),
            layers=layers,
            legend=Legend(),
        )

    # --- the one Lyapunov engine: burn-in + time-averaged spectrum ---

    def lyapunov_spectrum(self, **kwargs: Any) -> np.ndarray:
        """
        Estimate the Lyapunov spectrum — the unified engine for every family.

        Map and ODE families both delegate their ``lyapunov_spectrum`` here, so
        the QR/variational machinery lives in exactly one place.  Mode-specific
        keywords:

        - **maps**: ``steps`` (default 5000), ``ic``, ``reortho_interval`` (1).
        - **ODEs**: ``final_time`` (200.0), ``dt`` (0.1), ``ic``, ``burn_in``
          (50.0), ``method``, ``rtol`` (1e-6), ``atol`` (1e-9), and any extra
          integrator keywords.

        The estimate is recorded in ``self.meta['lyapunov_spectrum']`` (the inner
        system's :class:`~tsdynamics.families.base.MetaStore`).

        Returns
        -------
        ndarray, shape (k,)
            Lyapunov exponents, largest first (QR order).
        """
        if self._mode == "map":
            return self._lyapunov_spectrum_map(**kwargs)
        return self._lyapunov_spectrum_ode(**kwargs)

    def _lyapunov_spectrum_map(
        self,
        steps: int = 5000,
        ic: Any | None = None,
        reortho_interval: int = 1,
    ) -> np.ndarray:
        """QR tangent-map spectrum for a map, with random-IC retry on divergence.

        For the compiled-engine backends (``interp``/``jit``, the default) the whole
        QR tangent-map iteration runs in one Rust kernel call
        (:func:`~tsdynamics.engine.run.map_lyapunov`) — no per-step Python→FFI
        round-trip, ~thousands of times faster than the NumPy loop it supersedes.
        The ``reference`` backend, and any map whose ``_step`` will not lower to the
        engine IR (piecewise/ufunc) or a missing engine wheel, transparently use the
        pure-Python QR loop (:meth:`_accumulate_map`) — the oracle the engine is
        validated against.  The engine drives the same algorithm (Jacobian at the
        pre-image ``x_n``, propagate the frame, QR every ``reortho_interval``), so
        the spectrum matches to tolerance (it differs only by the lowered-IR vs
        pure-Python float order — the WS-MAPITER caveat).
        """
        max_retries = 10

        # Fast path: the compiled engine kernel (interp/jit).  Reference, a
        # non-lowering _step, or an absent wheel fall through to the NumPy loop.
        if self._backend in ("interp", "jit"):
            engine_result = self._map_spectrum_engine(steps, ic, reortho_interval, max_retries)
            if engine_result is not None:
                return engine_result

        for attempt in range(max_retries):
            use_ic = ic if attempt == 0 else None
            if self._accumulate_map(steps, use_ic, reortho_interval):
                exponents = self.exponents()
                self.meta.record(
                    "lyapunov_spectrum",
                    exponents,
                    steps=steps,
                    n_exp=self.k,
                    reortho_interval=reortho_interval,
                    backend=self._backend,
                )
                return exponents
            if attempt < max_retries - 1:
                # Force a fresh random IC: clear any stored one on the inner map.
                object.__setattr__(self.system, "ic", None)

        raise ValueError(
            f"{type(self.system).__name__}.lyapunov_spectrum: failed after "
            f"{max_retries} retries — iterates diverge from every tried IC. "
            f"Try a larger `steps` budget or pass an `ic` from a known basin point."
        )

    def _map_spectrum_engine(
        self,
        steps: int,
        ic: Any | None,
        reortho_interval: int,
        max_retries: int,
    ) -> np.ndarray | None:
        """Run the map QR spectrum on the Rust engine kernel, or decline.

        Lowers the map ``_step`` (with its Jacobian) once and runs the entire QR
        tangent-map iteration in one :func:`~tsdynamics.engine.run.map_lyapunov`
        call — the per-step Python QR loop replaced by a single FFI round-trip.

        Returns the spectrum (also seeding ``self`` so ``exponents()`` reports it)
        on success, or ``None`` to decline so the caller falls back to the pure-NumPy
        loop — when the map will not lower to the engine IR
        (:class:`~tsdynamics.engine.compile.TapeCompileError`) or the compiled
        wheel is absent
        (:class:`~tsdynamics.engine.run.EngineNotAvailableError`).  Mirrors the NumPy
        path's random-IC retry on divergence (a random draw can land off-basin); an
        explicit ``ic`` that diverges re-raises rather than retrying silently.
        """
        from tsdynamics.engine import run
        from tsdynamics.engine.compile import (
            TapeCompileError,
            lower_map_cached,
            tape_jacobian_is_smooth,
        )
        from tsdynamics.errors import ConvergenceError

        sys = self.system
        name = type(sys).__name__
        try:
            tape = lower_map_cached(sys, with_jacobian=True)
        except TapeCompileError:
            return None  # a non-lowering _step (piecewise/ufunc) → NumPy fallback
        # A piecewise map (abs/sign/floor/min/max/comparison in _step) has a lowered
        # Jacobian that is a.e.-correct but collapses to the a.e. value (0) at a kink —
        # and a map orbit can land exactly on it (the full-height Tent's dyadic orbit
        # hits x=0.5), poisoning the QR growth.  Decline so the NumPy loop, which reads
        # the one-sided hand-written `_jacobian`, runs instead.
        if not tape_jacobian_is_smooth(tape):
            return None

        try:
            tape_arrays = tape.to_arrays()
        except run.EngineNotAvailableError:  # pragma: no cover - defensive
            return None

        ic_explicit = ic is not None
        for attempt in range(max_retries):
            use_ic = sys.resolve_ic(ic if attempt == 0 else None)
            try:
                exponents, intervals = run.map_lyapunov(
                    tape_arrays,
                    use_ic,
                    steps=steps,
                    k=self.k,
                    reortho_interval=reortho_interval,
                    jit=self._backend == "jit",
                    name=name,
                )
            except run.EngineNotAvailableError:  # pragma: no cover - wheel-free env
                return None  # no engine → NumPy fallback (the wheel-free oracle)
            except ConvergenceError:
                if ic_explicit or attempt == max_retries - 1:
                    raise
                # Off-basin random draw diverged; clear the stored IC and retry.
                object.__setattr__(sys, "ic", None)
                continue

            # Seed ``exponents()`` so it reports the just-computed spectrum (a
            # downstream consumer may read it after this call).  The map kernel
            # returns only the averaged spectrum and the interval count — never the
            # final orthonormal frame nor the last interval's per-step stretch — so
            # ``deviations()`` / ``growths()`` cannot be reproduced: mark them stale
            # (they raise an honest error) rather than fabricating the long-run
            # average as if it were the most recent step (the ODE engine path, by
            # contrast, re-seats ``self._z`` and the kernel returns ``last_growths``,
            # so its streaming accessors stay genuinely coherent).
            self._W = None
            self._sum_growths = exponents * float(intervals * reortho_interval)
            self._last_growths = np.full(self.k, np.nan)
            self._elapsed = float(intervals * reortho_interval)
            self._map_engine_stale = True
            self.meta.record(
                "lyapunov_spectrum",
                exponents,
                steps=steps,
                n_exp=self.k,
                reortho_interval=reortho_interval,
                backend=self._backend,
            )
            return exponents

        # Unreachable: the loop returns on success or re-raises on the last attempt.
        raise ConvergenceError(  # pragma: no cover
            f"{name}.lyapunov_spectrum: map iterates diverge from every tried IC."
        )

    def _accumulate_map(self, steps: int, ic: Any | None, reortho_interval: int) -> bool:
        """Run the map QR loop; return ``False`` (soft failure) if it diverges.

        Evaluates the Jacobian at the pre-image ``x_n`` (so the accumulated
        product is ``J(x_{N-1})···J(x_0)`` — the exact tangent map), QR-ing every
        ``reortho_interval`` steps.  Float overflow on transient huge iterates is
        silenced locally and tracked explicitly as a soft failure.
        """
        self.reinit(ic)
        sys = self.system
        jac = type(sys)._jacobian
        params = sys.params.as_tuple()
        w = self._W
        sums = np.zeros(self.k)
        last = np.zeros(self.k)
        intervals = 0
        tiny = np.finfo(float).tiny

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for i in range(steps):
                x = sys.state()
                j = np.atleast_2d(np.asarray(jac(x, *params), dtype=float))
                if not np.all(np.isfinite(j)):
                    return False
                try:
                    sys.step()  # raises RuntimeError on a non-finite iterate
                except RuntimeError:
                    return False
                w = j @ w
                if not np.all(np.isfinite(w)):
                    return False
                if (i + 1) % reortho_interval == 0:
                    q, r = np.linalg.qr(w)
                    if not np.all(np.isfinite(q)):
                        return False
                    w = q
                    diag = np.abs(np.diag(r))
                    diag = np.where(diag < tiny, tiny, diag)
                    last = np.log(diag)
                    sums += last
                    intervals += 1

        if intervals == 0:
            return False
        self._W = w
        self._last_growths = last  # most recent reorthonormalisation's stretch
        self._sum_growths = sums
        self._elapsed = float(intervals * reortho_interval)
        return True

    def _lyapunov_spectrum_ode(
        self,
        final_time: float = 200.0,
        dt: float = 0.1,
        *,
        ic: Any | None = None,
        burn_in: float = 50.0,
        method: str | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        **integrator_kwargs: Any,
    ) -> np.ndarray:
        """Burn-in then time-averaged spectrum for a flow (either ODE backend).

        Fast path (stream ``perf/ode-lyapunov-engine``): when the resolved kernel
        is *explicit* and the backend is the compiled engine (``interp``/``jit``)
        — exactly the cases :meth:`_setup_ode_stepper` builds a stepper for — the
        **whole** burn-in + averaging Benettin loop (per-``dt`` extended-variational
        integration + QR + ``Σ log|diag R|`` accumulation) runs in one
        :func:`~tsdynamics.engine.run.lyapunov_spectrum_ode` engine call instead of
        the per-``dt`` Python loop (one Python→FFI round-trip, no per-chunk NumPy
        ``qr``).  The per-``dt`` numerics are byte-for-byte the per-chunk path
        (``interp == jit``), so the spectrum is unchanged to floating-point
        tolerance.  The slow per-chunk loop below is kept verbatim for the
        implicit/stiff kernel, the ``reference`` oracle, and ``method="auto"`` —
        the cases :meth:`_setup_ode_stepper` declines.
        """
        if final_time <= 0:
            raise ValueError(f"final_time must be positive, got {final_time!r}")
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt!r}")

        self.reinit(ic, method=method, rtol=rtol, atol=atol, **integrator_kwargs)

        if self._step_explicit_engine and self._ode_stepper is not None:
            exponents = self._engine_lyapunov_spectrum(dt, burn_in, final_time)
        else:
            # Burn-in: advance the trajectory + tangent frame without accumulating.
            t_burn = self._t + max(0.0, burn_in)
            while self._t < t_burn - 1e-12:
                self.step(min(dt, t_burn - self._t))
            self._reset_accumulators()

            # Production: uniform-dt steps, time-weighted growth average.
            t_end = self._t + final_time
            while self._t < t_end - 1e-12:
                self.step(min(dt, t_end - self._t))
            exponents = self.exponents()

        self.meta.record(
            "lyapunov_spectrum",
            exponents,
            dt=dt,
            final_time=final_time,
            burn_in=burn_in,
            n_exp=self.k,
            method=method or self.system._default_method,
            backend=self._backend,
        )
        return exponents

    def _engine_lyapunov_spectrum(self, dt: float, burn_in: float, final_time: float) -> np.ndarray:
        """Run the entire Benettin loop in one Rust engine call (the fast path).

        Eligibility is decided by :meth:`_setup_ode_stepper` (an explicit engine
        kernel): ``self._step_kernel`` is the registry-canonical method and
        ``self._ext_tape_arrays`` the marshalled extended variational tape.  The
        engine integrates the burn-in + averaging windows chunk-by-chunk and
        returns the spectrum + the final extended state, which is re-seated into
        ``self`` so :meth:`state` / :meth:`deviations` / :meth:`exponents` read
        exactly as the per-chunk loop would have left them.
        """
        from tsdynamics.engine import run

        assert self._z is not None  # reinit() seeds the extended state
        assert self._step_kernel is not None  # explicit-engine path resolved it
        if self._ext_tape_arrays is None:
            self._ext_tape_arrays = self._ext_tape.to_arrays()
        params_vec = self._engine_params_vec()
        out = run.lyapunov_spectrum_ode(
            self._ext_tape_arrays,
            params_vec,
            self._z,
            dim=self.system.dim,
            k=self.k,
            t0=self._t,
            dt=dt,
            burn_in=max(0.0, burn_in),
            final_time=final_time,
            method=self._step_kernel,
            rtol=self._rtol,
            atol=self._atol,
            jit=self._backend == "jit",
        )
        # Re-seat the live state from the engine's final extended state, and the
        # accumulators so exponents()/growths() reflect this run.
        self._z = np.asarray(out["final_state"], dtype=np.float64)
        self._t = self._t + max(0.0, burn_in) + final_time
        self._last_growths = np.asarray(out["last_growths"], dtype=np.float64)
        exponents = np.asarray(out["spectrum"], dtype=np.float64)
        # Mirror the Python accumulator state so a subsequent exponents() call
        # returns the same estimate (sum_growths / elapsed == spectrum).
        self._elapsed = float(final_time)
        self._sum_growths = exponents * self._elapsed
        return exponents


def __dir__() -> list[str]:
    """Expose only the curated public API (``__all__``) to ``dir()`` / autocomplete."""
    return sorted(__all__)
