"""Rust-vs-v2 trajectory cross-validation harness (ROADMAP stream F4).

The migration plan (ROADMAP §9, decision D1) makes the Rust engine the sole
integration backend and *deletes* the v2 backends (JiTCODE/JiTCDDE/Numba/diffsol)
once parity is proven. Before any v2 backend is removed (milestone M3, stream
I-XVAL) every system's Rust trajectory must be shown to match its v2 trajectory
within tolerance, and its Rust RHS to match the symbolic RHS to ~1e-15.

This module is the *scaffold* that gate is built on: a small, backend-agnostic
comparison engine plus a couple of concrete backends. It deliberately ships in
``tests/`` (not in ``src/``) — it is validation tooling, not public API. Two
downstream streams extend it:

* **I-XVAL** sweeps :func:`crossvalidate` / :func:`crossvalidate_rhs` over the
  whole catalogue and turns the aggregate into the removal gate.
* the **engine streams (E1–E7)** add the new engine as one more
  :class:`IntegrationBackend`, so the same harness compares it to the v2
  reference with no new plumbing.

Design
------
``IntegrationBackend`` is a duck-typed seam: anything with a ``name`` and an
``integrate_dense(system, ic, t_eval)`` method qualifies, and reports
``available()`` so a sweep *skips* (never fails) when an optional backend is
absent. Three backends ship today:

* :class:`ScipyReference` — integrates the system's symbolic RHS
  (``_rhs_numeric``, the v2 numeric truth) with SciPy at tight tolerance. Always
  available; needs no C compiler. The trustworthy reference the migration
  candidate is measured against.
* :class:`RustEngine` — **the migration candidate**: the shipping engine
  (:mod:`tsdynamics._rust`) through its public seam ``engine.run.integrate`` /
  ``engine.run.eval_rhs`` (``backend="interp"`` or ``"jit"``). Available only
  when the compiled extension is built; exposes ``eval_rhs`` for
  :func:`crossvalidate_rhs`. The M3 removal gate (stream I-XVAL) sweeps this over
  the registry catalogue.
* :class:`RustCore` — the v2-seed accelerator
  (:mod:`tsdynamics.engine.rustcore`), retired at M3. Available only when
  ``tsdynamics-core`` is installed; also exposes ``eval_rhs``. Kept for the
  legacy Lorenz scaffold (``test_xval.py``) until the v2 seed is removed.

The comparison functions return an :class:`XValReport` (max abs/rel error,
tolerance, pass/fail) instead of asserting, so callers can aggregate across a
sweep. Two correct integrators of a *chaotic* flow diverge exponentially, so
trajectory comparison is only meaningful on an early ``window=`` where Lyapunov
amplification is small — the same convention the rustcore numeric tests use. The
RHS check has no such limit: it compares pointwise evaluations and holds to
~1e-9 everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "IntegrationBackend",
    "RustCore",
    "RustEngine",
    "ScipyReference",
    "XValReport",
    "crossvalidate",
    "crossvalidate_rhs",
]


@runtime_checkable
class IntegrationBackend(Protocol):
    """The seam every backend implements (duck-typed; this Protocol documents it)."""

    name: str

    def available(self) -> bool:
        """Whether this backend can run in the current environment."""
        ...

    def integrate_dense(self, system: Any, ic: Any, t_eval: np.ndarray) -> np.ndarray:
        """Integrate ``system`` from ``ic`` and sample at ``t_eval`` → ``(len(t_eval), dim)``."""
        ...


class ScipyReference:
    """Reference backend: SciPy on the symbolic RHS (the v2 numeric truth).

    Parameters
    ----------
    method, rtol, atol
        Forwarded to :func:`scipy.integrate.solve_ivp`. Defaults are a tight,
        high-accuracy reference (``DOP853`` at ``rtol=1e-12``).
    """

    def __init__(self, *, method: str = "DOP853", rtol: float = 1e-12, atol: float = 1e-14) -> None:
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.name = f"scipy:{method}"

    def available(self) -> bool:
        """SciPy is a hard project dependency, so this backend is always available."""
        try:
            import scipy  # noqa: F401
        except ImportError:  # pragma: no cover - scipy is a required dependency
            return False
        return True

    def integrate_dense(self, system: Any, ic: Any, t_eval: np.ndarray) -> np.ndarray:
        """Integrate the symbolic RHS and sample at ``t_eval``."""
        from scipy.integrate import solve_ivp

        t_eval = np.ascontiguousarray(t_eval, dtype=np.float64)
        ic = np.asarray(system.resolve_ic(ic), dtype=np.float64).ravel()
        if t_eval.size == 0:
            return np.empty((0, ic.size), dtype=np.float64)
        f = system._rhs_numeric()
        sol = solve_ivp(
            lambda t, u: f(u, t),
            (float(t_eval[0]), float(t_eval[-1])),
            ic,
            t_eval=t_eval,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
        )
        if not sol.success:
            raise RuntimeError(f"{self.name}: solve_ivp failed: {sol.message}")
        return np.ascontiguousarray(sol.y.T)


class RustCore:
    """Candidate backend: the experimental ``tsdynamics-core`` accelerator.

    Wraps :mod:`tsdynamics.engine.rustcore`. ``method`` selects the kernel
    (``RK45`` adaptive default, ``RK4`` fixed-step, ``stiff``); see that module
    for the full list. Exposes :meth:`eval_rhs` so :func:`crossvalidate_rhs` can
    check the tape lowering pointwise.
    """

    def __init__(
        self,
        *,
        method: str = "RK45",
        rtol: float = 1e-10,
        atol: float = 1e-12,
        h: float | None = None,
    ) -> None:
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.h = h
        self.name = f"rustcore:{method}"

    def available(self) -> bool:
        """Whether the optional ``tsdynamics-core`` wheel is installed."""
        from tsdynamics.engine import rustcore as rc

        return rc.available()

    def integrate_dense(self, system: Any, ic: Any, t_eval: np.ndarray) -> np.ndarray:
        """Integrate via the Rust kernel and sample at ``t_eval``."""
        from tsdynamics.engine import rustcore as rc

        ic = np.asarray(system.resolve_ic(ic), dtype=np.float64).ravel()
        return rc.integrate_dense(
            system, ic, t_eval, method=self.method, rtol=self.rtol, atol=self.atol, h=self.h
        )

    def eval_rhs(self, system: Any, u: Any, t: float = 0.0) -> np.ndarray:
        """Evaluate ``du/dt`` once in Rust (used by :func:`crossvalidate_rhs`)."""
        from tsdynamics.engine import rustcore as rc

        return rc.eval_rhs(system, u, t)


class RustEngine:
    """The migration candidate: the shipping Rust engine (:mod:`tsdynamics._rust`).

    Unlike :class:`RustCore` (the v2-seed ``tsdynamics-core`` accelerator that the
    F4 scaffold wrapped), this backend drives the *new* engine through its public
    Python seam — :func:`tsdynamics.engine.run.integrate` and
    :func:`tsdynamics.engine.run.eval_rhs` — exactly the path the family base
    classes use.  It is the backend the M3 removal gate (stream I-XVAL) sweeps
    over the whole registry catalogue to authorise deleting the v2 backends.

    Parameters
    ----------
    backend
        ``"interp"`` (the SSA-tape interpreter, default) or ``"jit"`` (the
        Cranelift JIT).  The two are numerically identical by contract, so a
        sweep can run the same comparison through either and an ``interp``-vs-
        ``jit`` cross-check pins that contract.
    method, rtol, atol
        Forwarded to :func:`~tsdynamics.engine.run.integrate`.  Defaults are a
        tight adaptive ``RK45``.

    Notes
    -----
    :meth:`integrate_dense` rides ``run.integrate``, which samples on the uniform
    grid :func:`tsdynamics.utils.grids.make_output_grid` builds from
    ``(t0, final_time, dt)``.  That grid reproduces a uniform ``t_eval`` exactly,
    so the harness contract ("sample at ``t_eval``") is met for uniform grids
    (the only kind the sweeps use); a non-uniform ``t_eval`` raises rather than
    silently resampling.
    """

    def __init__(
        self,
        *,
        backend: str = "interp",
        method: str = "RK45",
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ) -> None:
        self.backend = backend
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.name = f"rust:{backend}:{method}"

    def available(self) -> bool:
        """Whether the compiled engine extension (:mod:`tsdynamics._rust`) is built."""
        try:
            import tsdynamics._rust  # noqa: F401
        except ImportError:
            return False
        return True

    def integrate_dense(self, system: Any, ic: Any, t_eval: np.ndarray) -> np.ndarray:
        """Integrate via the Rust engine and sample at a uniform ``t_eval``."""
        from tsdynamics.engine import run

        t_eval = np.ascontiguousarray(t_eval, dtype=np.float64)
        ic = np.asarray(system.resolve_ic(ic), dtype=np.float64).ravel()
        if t_eval.size == 0:
            return np.empty((0, ic.size), dtype=np.float64)
        if t_eval.size == 1:
            # A single sample is the state at t_eval[0] — the initial condition.
            return ic.reshape(1, -1)
        t0 = float(t_eval[0])
        tf = float(t_eval[-1])
        dt = float(t_eval[1] - t_eval[0])
        if not np.allclose(np.diff(t_eval), dt, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"{self.name}: integrate_dense needs a uniform t_eval grid "
                "(run.integrate samples on a uniform output grid)"
            )
        traj = run.integrate(
            system,
            final_time=tf,
            dt=dt,
            t0=t0,
            ic=ic,
            backend=self.backend,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
        )
        y = np.ascontiguousarray(traj.y, dtype=np.float64)
        if y.shape[0] != t_eval.size or not np.allclose(traj.t, t_eval, rtol=0.0, atol=1e-9):
            raise RuntimeError(
                f"{self.name}: engine output grid (n={y.shape[0]}) did not align with "
                f"the requested t_eval (n={t_eval.size})"
            )
        return y

    def eval_rhs(self, system: Any, u: Any, t: float = 0.0) -> np.ndarray:
        """Evaluate ``du/dt`` once on the engine (used by :func:`crossvalidate_rhs`)."""
        from tsdynamics.engine import run

        return np.asarray(run.eval_rhs(system, u, t, backend=self.backend), dtype=np.float64)


@dataclass(frozen=True)
class XValReport:
    """Outcome of one cross-validation: error magnitudes + a pass/fail verdict.

    ``passed`` uses the same criterion as :func:`numpy.allclose`
    (``|candidate - reference| <= atol + rtol * |reference|`` everywhere, and no
    non-finite mismatch). ``max_rel_err`` is reported for diagnostics but does
    not by itself decide the verdict.
    """

    system: str
    reference: str
    candidate: str
    metric: str  # "trajectory" | "rhs"
    n_samples: int
    max_abs_err: float
    max_rel_err: float
    atol: float
    rtol: float
    passed: bool
    window: tuple[float, float] | None = None
    detail: str = ""

    def summary(self) -> str:
        """One-line human-readable summary (used in assertion messages and logs)."""
        status = "PASS" if self.passed else "FAIL"
        win = "" if self.window is None else f" window={self.window}"
        tail = f" — {self.detail}" if self.detail else ""
        return (
            f"[{status}] {self.system}: {self.candidate} vs {self.reference} "
            f"({self.metric}{win}) max|Δ|={self.max_abs_err:.3e} relΔ={self.max_rel_err:.3e} "
            f"(atol={self.atol:.1e} rtol={self.rtol:.1e} n={self.n_samples}){tail}"
        )

    def __str__(self) -> str:
        return self.summary()


def _compare(
    candidate: np.ndarray, reference: np.ndarray, *, atol: float, rtol: float
) -> tuple[float, float, bool, str]:
    """Return ``(max_abs_err, max_rel_err, passed, detail)`` for two arrays.

    A shape mismatch is a programming error and raises.  A non-finite *mismatch*
    — one side NaN/Inf where the other is finite, or both non-finite but *not the
    same* value (``+inf`` vs ``-inf``, ``inf`` vs ``NaN``) — is a hard validation
    failure carried in the verdict, not an exception.  Two sides that *agree* on
    the same non-finite value (both ``+inf``, both ``-inf``, both ``NaN``) are a
    genuine agreement and count as a pass: an overflow that both the engine and
    the reference reach identically is not a discrepancy.  Errors are measured
    only on the jointly-finite positions, so the ``inf - inf`` subtraction (which
    would raise a RuntimeWarning, escalated to an error under the suite's
    ``filterwarnings=["error"]``) never runs.
    """
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if candidate.shape != reference.shape:
        raise ValueError(
            f"cross-validation shape mismatch: candidate {candidate.shape} "
            f"vs reference {reference.shape}"
        )
    finite = np.isfinite(candidate) & np.isfinite(reference)
    # Non-finite positions where the two sides agree on the same value (both ±inf
    # by `==`, both NaN) are not mismatches; everything else non-finite is.
    both_nonfinite = ~np.isfinite(candidate) & ~np.isfinite(reference)
    both_nan = np.isnan(candidate) & np.isnan(reference)
    agree_nonfinite = both_nonfinite & (both_nan | np.equal(candidate, reference))
    mismatch = ~finite & ~agree_nonfinite
    n_mismatch = int(mismatch.sum())

    if finite.any():
        ref_mag = np.abs(reference[finite])
        abs_err = np.abs(candidate[finite] - reference[finite])
        max_abs = float(abs_err.max())
        max_rel = float((abs_err / (ref_mag + 1e-300)).max())
        within = bool(np.all(abs_err <= atol + rtol * ref_mag))
    elif candidate.size == 0:
        # Nothing to compare (e.g. an empty t_eval) — a vacuous pass.
        max_abs = max_rel = 0.0
        within = True
    else:
        # No jointly-finite position: a pass only if every entry agrees non-finite.
        max_abs = max_rel = 0.0 if n_mismatch == 0 else float("inf")
        within = True
    detail = "" if n_mismatch == 0 else f"{n_mismatch}/{candidate.size} non-finite mismatches"
    return max_abs, max_rel, (within and n_mismatch == 0), detail


def crossvalidate(
    system: Any,
    *,
    reference: IntegrationBackend,
    candidate: IntegrationBackend,
    t_eval: np.ndarray,
    ic: Any = None,
    window: tuple[float, float] | None = None,
    atol: float = 1e-3,
    rtol: float = 0.0,
) -> XValReport:
    """Integrate ``system`` with two backends on ``t_eval`` and compare.

    Both backends start from the same initial condition (``ic`` resolved via the
    system's own ``resolve_ic`` — pass it explicitly for determinism). When
    ``window=(lo, hi)`` is given the comparison is restricted to that time span,
    which is how chaotic flows are validated (agreement is only expected before
    Lyapunov amplification dominates).

    Returns
    -------
    XValReport
        With ``metric="trajectory"``.
    """
    name = type(system).__name__
    t_eval = np.ascontiguousarray(t_eval, dtype=np.float64)
    ic_vec = np.asarray(system.resolve_ic(ic), dtype=np.float64).ravel()
    y_ref = reference.integrate_dense(system, ic_vec, t_eval)
    y_cand = candidate.integrate_dense(system, ic_vec, t_eval)
    if window is None:
        sel = np.ones(t_eval.shape, dtype=bool)
    else:
        lo, hi = window
        sel = (t_eval >= lo) & (t_eval <= hi)
    max_abs, max_rel, passed, detail = _compare(y_cand[sel], y_ref[sel], atol=atol, rtol=rtol)
    return XValReport(
        system=name,
        reference=reference.name,
        candidate=candidate.name,
        metric="trajectory",
        n_samples=int(sel.sum()),
        max_abs_err=max_abs,
        max_rel_err=max_rel,
        atol=atol,
        rtol=rtol,
        passed=passed,
        window=window,
        detail=detail,
    )


def crossvalidate_rhs(
    system: Any,
    *,
    candidate: Any,
    n: int = 16,
    seed: int = 0,
    rtol: float = 1e-9,
    atol: float = 1e-9,
    scale: float = 1.0,
    t_max: float = 5.0,
) -> XValReport:
    """Compare a backend's pointwise RHS to the symbolic RHS at random states.

    This is the strongest, tolerance-tight signal that a system *lowers*
    correctly — it has no chaotic-divergence caveat. ``candidate`` must expose an
    ``eval_rhs(system, u, t)`` method (only :class:`RustCore` does today).

    Returns
    -------
    XValReport
        With ``metric="rhs"`` (``n_samples`` random states checked).
    """
    name = type(system).__name__
    eval_rhs = getattr(candidate, "eval_rhs", None)
    if eval_rhs is None:
        raise TypeError(f"backend {getattr(candidate, 'name', candidate)!r} has no eval_rhs")
    f = system._rhs_numeric()
    rng = np.random.default_rng(seed)
    max_abs = 0.0
    max_rel = 0.0
    passed = True
    detail = ""
    for _ in range(n):
        u = rng.standard_normal(system.dim) * scale
        t = float(rng.uniform(0.0, t_max))
        got = np.asarray(eval_rhs(system, u, t), dtype=np.float64)
        ref = np.asarray(f(u, t), dtype=np.float64)
        a, r, ok, d = _compare(got, ref, atol=atol, rtol=rtol)
        max_abs = max(max_abs, a)
        max_rel = max(max_rel, r)
        passed = passed and ok
        detail = detail or d
    return XValReport(
        system=name,
        reference="symbolic:_rhs_numeric",
        candidate=getattr(candidate, "name", "candidate"),
        metric="rhs",
        n_samples=n,
        max_abs_err=max_abs,
        max_rel_err=max_rel,
        atol=atol,
        rtol=rtol,
        passed=passed,
        detail=detail,
    )
