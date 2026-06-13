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
absent. Two backends ship today:

* :class:`ScipyReference` — integrates the system's symbolic RHS
  (``_rhs_numeric``, the v2 numeric truth) with SciPy at tight tolerance. Always
  available; needs no C compiler. The stand-in "reference" until the new engine
  and the v2 compiled backends are both wired in as backends.
* :class:`RustCore` — the v2-seed accelerator
  (:mod:`tsdynamics.backends.rustcore`). Available only when ``tsdynamics-core``
  is installed; also exposes a pointwise ``eval_rhs`` for :func:`crossvalidate_rhs`.

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

    Wraps :mod:`tsdynamics.backends.rustcore`. ``method`` selects the kernel
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
        from tsdynamics.backends import rustcore as rc

        return rc.available()

    def integrate_dense(self, system: Any, ic: Any, t_eval: np.ndarray) -> np.ndarray:
        """Integrate via the Rust kernel and sample at ``t_eval``."""
        from tsdynamics.backends import rustcore as rc

        ic = np.asarray(system.resolve_ic(ic), dtype=np.float64).ravel()
        return rc.integrate_dense(
            system, ic, t_eval, method=self.method, rtol=self.rtol, atol=self.atol, h=self.h
        )

    def eval_rhs(self, system: Any, u: Any, t: float = 0.0) -> np.ndarray:
        """Evaluate ``du/dt`` once in Rust (used by :func:`crossvalidate_rhs`)."""
        from tsdynamics.backends import rustcore as rc

        return rc.eval_rhs(system, u, t)


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

    A shape mismatch is a programming error and raises; a non-finite *mismatch*
    (one side NaN/Inf where the other is finite) is a hard validation failure,
    not an exception — the report carries the verdict.
    """
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if candidate.shape != reference.shape:
        raise ValueError(
            f"cross-validation shape mismatch: candidate {candidate.shape} "
            f"vs reference {reference.shape}"
        )
    finite = np.isfinite(candidate) & np.isfinite(reference)
    all_finite = bool(finite.all())  # vacuously True for empty arrays
    abs_err = np.abs(candidate - reference)
    if finite.any():
        ref_mag = np.abs(reference[finite])
        max_abs = float(abs_err[finite].max())
        max_rel = float((abs_err[finite] / (ref_mag + 1e-300)).max())
        within = bool(np.all(abs_err[finite] <= atol + rtol * ref_mag))
    elif candidate.size == 0:
        # Nothing to compare (e.g. an empty t_eval) — a vacuous pass.
        max_abs = max_rel = 0.0
        within = True
    else:
        # Non-empty but entirely non-finite on both sides.
        max_abs = max_rel = float("inf")
        within = False
    detail = "" if all_finite else f"{int((~finite).sum())}/{finite.size} non-finite mismatches"
    return max_abs, max_rel, (within and all_finite), detail


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
