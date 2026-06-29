"""Independent method-of-steps oracle for the Rust DDE engine.

Audit gap (the reason this file exists)
----------------------------------------
JiTCDDE was removed at M3 and ``backend="reference"`` is *rejected* for delay
systems (``DelaySystem`` has no pure-Python integrator).  So the Rust
method-of-steps engine (``backend="interp"``) is currently the *only* DDE
integrator in the library — a bug in the method of steps that still produces a
finite, plausible-looking trajectory has nothing to check it against.

This module supplies that missing check: a small, self-contained, clearly-correct
**fixed-step RK4 method-of-steps integrator written here in pure NumPy**, using a
hand-coded RHS for each test system (transcribed from its published equations,
not from the library), and a constant past.  It then compares the engine's
``integrate(...)`` to that oracle over a short, pre-chaotic horizon.

Three layers of bar:

1. ``test_linear_dde_*`` — an *analytic* linear DDE, ``x'(t) = -a*x(t-tau)`` with
   constant past, where the method-of-steps solution is a known closed-form
   polynomial on the first window(s).  Tight tolerance — no oracle, no
   chaotic-divergence caveat.
2. ``test_mackey_glass_matches_oracle`` / ``test_sprott_delay_matches_oracle`` —
   the engine vs. the NumPy oracle for built-in DDEs, loose tolerance over a few
   delay windows (different step schemes diverge on a chaotic DDE, so the horizon
   is short / pre-chaotic).
3. ``test_oracle_self_consistency`` — the oracle reproduces the *analytic* linear
   DDE too, so a calibration bug in the oracle itself is caught (it is not a
   silent rubber stamp).

References
----------
Mackey & Glass (1977), Science 197, 287-289 — ``x' = beta*x(t-tau)/(1+x(t-tau)^n)
- gamma*x``.
Sprott (2007), Phys. Lett. A 366, 397-402 — ``x' = sin(x(t-tau))``.
Bellen & Zennaro (2003), *Numerical Methods for Delay Differential Equations*,
Oxford — method of steps + RK on a stored history (the textbook scheme this
oracle implements).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

# The Rust engine is the only DDE backend, so this whole module needs it.
_rust = pytest.importorskip("tsdynamics._rust")

import tsdynamics as ts  # noqa: E402

# ---------------------------------------------------------------------------
# The independent oracle: fixed-step RK4 method of steps, pure NumPy.
# ---------------------------------------------------------------------------


def _mos_rk4(
    rhs: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    *,
    tau: float,
    history: Callable[[float], np.ndarray],
    final_time: float,
    h: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a single-delay DDE ``x'(t) = rhs(t, x(t), x(t-tau))``.

    A textbook fixed-step RK4 *method of steps* (Bellen & Zennaro 2003): the
    solution is built forward on a uniform grid of step ``h``; the delayed value
    ``x(t-tau)`` is read from the already-computed past by **linear
    interpolation** of the stored grid (and from ``history`` for ``t <= 0``).

    Deliberately simple and independent of the library: it never touches
    ``tsdynamics`` internals, only the system's published RHS handed in as a
    plain Python closure.

    Parameters
    ----------
    rhs : callable ``(t, x, x_delayed) -> dx/dt``
        The DDE right-hand side, all arguments length-``dim`` arrays.
    tau : float
        The (constant, positive) delay.
    history : callable ``s -> x`` for ``s <= 0``
        The past, a length-``dim`` array per scalar time ``s``.
    final_time, h : float
        Integration end time and (fixed) step.  ``h`` should divide ``tau`` so
        the delayed lookup lands on grid points where possible.

    Returns
    -------
    (t_grid, x_grid) : the uniform time grid and the solution on it.
    """
    n_steps = int(round(final_time / h))
    t_grid = np.arange(n_steps + 1) * h
    dim = np.atleast_1d(np.asarray(history(0.0), dtype=float)).size
    x_grid = np.empty((n_steps + 1, dim), dtype=float)
    x_grid[0] = np.atleast_1d(np.asarray(history(0.0), dtype=float))

    def delayed(t_query: float) -> np.ndarray:
        """``x(t_query)`` via the history (past) or linear interp of the grid."""
        if t_query <= 0.0:
            return np.atleast_1d(np.asarray(history(t_query), dtype=float))
        # Linear interpolation between the two bracketing grid points already
        # computed.  t_query is always <= the current frontier (delay > 0).
        pos = t_query / h
        lo = int(np.floor(pos))
        frac = pos - lo
        if frac == 0.0:
            return x_grid[lo]
        return (1.0 - frac) * x_grid[lo] + frac * x_grid[lo + 1]

    for k in range(n_steps):
        t = t_grid[k]
        xk = x_grid[k]
        # RK4 over [t, t+h]; the delayed argument is sampled at the matching
        # sub-stage time (t + c*h - tau), read from the stored past.
        k1 = rhs(t, xk, delayed(t - tau))
        k2 = rhs(t + 0.5 * h, xk + 0.5 * h * k1, delayed(t + 0.5 * h - tau))
        k3 = rhs(t + 0.5 * h, xk + 0.5 * h * k2, delayed(t + 0.5 * h - tau))
        k4 = rhs(t + h, xk + h * k3, delayed(t + h - tau))
        x_grid[k + 1] = xk + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return t_grid, x_grid


def _at(traj_t: np.ndarray, traj_y: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Sample a (t, y) curve at ``times`` by nearest-grid lookup."""
    idx = np.searchsorted(traj_t, times)
    idx = np.clip(idx, 0, len(traj_t) - 1)
    # snap to the nearer of idx-1 / idx
    left = np.clip(idx - 1, 0, len(traj_t) - 1)
    take_left = np.abs(traj_t[left] - times) < np.abs(traj_t[idx] - times)
    idx = np.where(take_left, left, idx)
    return traj_y[idx]


# ---------------------------------------------------------------------------
# Hand-coded RHS closures (transcribed from the published equations).
# ---------------------------------------------------------------------------


def _mackey_glass_rhs(beta: float, gamma: float, n: float):
    def rhs(t: float, x: np.ndarray, xd: np.ndarray) -> np.ndarray:
        return np.array([beta * xd[0] / (1.0 + xd[0] ** n) - gamma * x[0]])

    return rhs


def _sprott_rhs():
    def rhs(t: float, x: np.ndarray, xd: np.ndarray) -> np.ndarray:
        return np.array([np.sin(xd[0])])

    return rhs


def _linear_rhs(a: float):
    def rhs(t: float, x: np.ndarray, xd: np.ndarray) -> np.ndarray:
        return np.array([-a * xd[0]])

    return rhs


# ---------------------------------------------------------------------------
# A library DDE for the analytic linear test (engine path).
# ---------------------------------------------------------------------------


class _LinearDDE(ts.DelaySystem):
    """``x'(t) = -x(t - 1)``; constant past 1 ⇒ exact piecewise polynomial."""

    params = {"tau": 1.0}
    dim = 1

    @staticmethod
    def _equations(y, t, *, tau):
        return [-y(0, t - tau)]


# ---------------------------------------------------------------------------
# Layer 1 — analytic linear DDE: closed-form method of steps.
# ---------------------------------------------------------------------------
#
# x'(t) = -x(t-1), x(s)=1 for s<=0.  Method of steps:
#   [0,1]: x' = -1            => x(t) = 1 - t            => x(1) = 0
#   [1,2]: x' = -(1-(t-1))    = -(2-t) => x(t) = 1 - t + (t-1)^2/2
#                                          => x(2) = 1 - 2 + 1/2 = -0.5
# (Bellen & Zennaro 2003, the standard textbook example.)


def test_linear_dde_engine_matches_closed_form():
    traj = _LinearDDE().integrate(
        backend="interp", final_time=2.0, dt=0.05, ic=[1.0], rtol=1e-10, atol=1e-12
    )
    t, y = traj.t, traj.y[:, 0]
    x1 = _at(t, y[:, None], np.array([1.0]))[0, 0]
    x2 = _at(t, y[:, None], np.array([2.0]))[0, 0]
    assert abs(x1 - 0.0) < 1e-7, f"x(1) = {x1} (expected 0)"
    assert abs(x2 - (-0.5)) < 1e-7, f"x(2) = {x2} (expected -0.5)"


def test_linear_dde_full_window_polynomial():
    """The whole first window must equal the exact polynomial x(t)=1-t."""
    traj = _LinearDDE().integrate(
        backend="interp", final_time=1.0, dt=0.05, ic=[1.0], rtol=1e-10, atol=1e-12
    )
    t, y = traj.t, traj.y[:, 0]
    mask = t <= 1.0
    exact = 1.0 - t[mask]
    assert np.max(np.abs(y[mask] - exact)) < 1e-7


# ---------------------------------------------------------------------------
# Layer 3 — the oracle reproduces the analytic linear DDE (calibration guard).
# ---------------------------------------------------------------------------


def test_oracle_self_consistency_linear():
    """The NumPy oracle itself hits the closed form, so it is not a rubber stamp."""
    history = lambda s: np.array([1.0])  # noqa: E731
    t, x = _mos_rk4(_linear_rhs(1.0), tau=1.0, history=history, final_time=2.0, h=0.01)
    x1 = x[np.argmin(np.abs(t - 1.0)), 0]
    x2 = x[np.argmin(np.abs(t - 2.0)), 0]
    assert abs(x1 - 0.0) < 1e-4, f"oracle x(1) = {x1}"
    assert abs(x2 - (-0.5)) < 1e-4, f"oracle x(2) = {x2}"


# ---------------------------------------------------------------------------
# Layer 2 — engine vs. independent oracle for built-in DDEs.
# ---------------------------------------------------------------------------


def test_mackey_glass_matches_oracle():
    """Mackey-Glass: engine vs. NumPy method-of-steps oracle, pre-chaotic horizon.

    Constant past ``x(s)=0.5`` (off the ``x=1`` equilibrium so the dynamics is
    non-trivial), default params (beta=0.2, gamma=0.1, tau=17, n=10).  Over
    ~1.5 delay windows the trajectory is a smooth relaxation toward the fixed
    point — non-chaotic, so two distinct step schemes must agree to ~1%.
    """
    mg = ts.MackeyGlass()
    beta, gamma, tau, n = (
        mg.params["beta"],
        mg.params["gamma"],
        mg.params["tau"],
        mg.params["n"],
    )
    ic = [0.5]
    final_time = 25.0
    dt = 0.05

    traj = mg.integrate(
        backend="interp", final_time=final_time, dt=dt, ic=ic, rtol=1e-8, atol=1e-10
    )
    t_o, x_o = _mos_rk4(
        _mackey_glass_rhs(beta, gamma, n),
        tau=tau,
        history=lambda s: np.array(ic, dtype=float),
        final_time=final_time,
        h=0.005,
    )

    sample = np.linspace(2.0, final_time, 24)
    eng = _at(traj.t, traj.y, sample)[:, 0]
    orc = _at(t_o, x_o, sample)[:, 0]

    rel = np.abs(eng - orc) / (np.abs(orc) + 1e-3)
    assert np.max(rel) < 1e-2, (
        f"MackeyGlass engine vs oracle: max rel err {np.max(rel):.3e}\n"
        f"  sample t={sample[np.argmax(rel)]:.2f}  eng={eng[np.argmax(rel)]:.6f}  "
        f"orc={orc[np.argmax(rel)]:.6f}"
    )


def test_sprott_delay_matches_oracle():
    """Sprott's x'=sin(x(t-tau)): engine vs oracle over a short pre-chaotic span.

    Chaos sets in around ``tau ~ 5`` (the default is 5.1), so two step schemes
    diverge eventually; we compare only over the first ~2 delay windows where
    they still track, at 2% tolerance.
    """
    sd = ts.SprottDelay()
    tau = sd.params["tau"]
    ic = [0.8]
    final_time = 10.0
    dt = 0.05

    traj = sd.integrate(
        backend="interp", final_time=final_time, dt=dt, ic=ic, rtol=1e-8, atol=1e-10
    )
    t_o, x_o = _mos_rk4(
        _sprott_rhs(),
        tau=tau,
        history=lambda s: np.array(ic, dtype=float),
        final_time=final_time,
        h=0.005,
    )

    sample = np.linspace(1.0, final_time, 20)
    eng = _at(traj.t, traj.y, sample)[:, 0]
    orc = _at(t_o, x_o, sample)[:, 0]

    rel = np.abs(eng - orc) / (np.abs(orc) + 0.1)
    assert np.max(rel) < 2e-2, (
        f"SprottDelay engine vs oracle: max rel err {np.max(rel):.3e}\n"
        f"  sample t={sample[np.argmax(rel)]:.2f}  eng={eng[np.argmax(rel)]:.6f}  "
        f"orc={orc[np.argmax(rel)]:.6f}"
    )
