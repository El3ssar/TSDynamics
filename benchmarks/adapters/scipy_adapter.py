"""SciPy adapter — the general-purpose numerical baseline.

SciPy is not a dynamical-systems library, but ``scipy.integrate.solve_ivp`` is
the canonical Python ODE-integration baseline and ``scipy.optimize.fsolve`` /
event detection cover fixed points and Poincaré sections. It does not provide
Lyapunov exponents, fractal dimensions, bifurcation diagrams or basins, so those
rows stay blank — exactly the "leave it blank" case from the brief.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from adapters._base import BaseAdapter


def _lorenz_rhs(params: dict[str, float]) -> Callable[[float, np.ndarray], list[float]]:
    sigma, rho, beta = params["sigma"], params["rho"], params["beta"]

    def rhs(_t: float, u: np.ndarray) -> list[float]:
        x, y, z = u
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    return rhs


def _rossler_rhs(params: dict[str, float]) -> Callable[[float, np.ndarray], list[float]]:
    a, b, c = params["a"], params["b"], params["c"]

    def rhs(_t: float, u: np.ndarray) -> list[float]:
        x, y, z = u
        return [-y - z, x + a * y, b + z * (x - c)]

    return rhs


class SciPyAdapter(BaseAdapter):
    name = "SciPy"
    language = "python"

    def _probe(self) -> tuple[bool, str, str]:
        import scipy

        return True, scipy.__version__, ""

    def _grid(self, T: float) -> np.ndarray:
        dt = self._intg["dt"]
        return np.linspace(0.0, T, int(round(T / dt)) + 1)

    # -- integration -------------------------------------------------------- #

    def task_integrate_short(self, quick: bool) -> Callable[[], Any]:
        from scipy.integrate import solve_ivp

        rhs = _lorenz_rhs(self.cfg["lorenz"]["params"])
        ic = self.cfg["lorenz"]["ic"]
        T = 50.0 if quick else self._intg["t_short"]
        grid = self._grid(T)

        def run() -> None:
            solve_ivp(rhs, (0.0, T), ic, method="DOP853", t_eval=grid, rtol=1e-9, atol=1e-9)

        return run

    def task_integrate_long(self, quick: bool) -> Callable[[], Any]:
        from scipy.integrate import solve_ivp

        rhs = _lorenz_rhs(self.cfg["lorenz"]["params"])
        ic = self.cfg["lorenz"]["ic"]
        T = 1000.0 if quick else self._intg["t_long"]
        grid = self._grid(T)

        def run() -> None:
            solve_ivp(rhs, (0.0, T), ic, method="DOP853", t_eval=grid, rtol=1e-9, atol=1e-9)

        return run

    def task_integrate_accuracy(self, quick: bool) -> Callable[[], float]:
        from scipy.integrate import solve_ivp

        rhs = _lorenz_rhs(self.cfg["lorenz"]["params"])
        ic = self.cfg["lorenz"]["ic"]
        T = self._intg["t_acc"]
        ref = np.asarray(self._ref["lorenz_acc_final"], dtype=float)

        def run() -> float:
            sol = solve_ivp(
                rhs,
                (0.0, T),
                ic,
                method="DOP853",
                rtol=self._intg["acc_rtol"],
                atol=self._intg["acc_atol"],
            )
            return float(np.max(np.abs(sol.y[:, -1] - ref)))

        return run

    # -- fixed points ------------------------------------------------------- #

    def task_fixed_points(self, quick: bool) -> Callable[[], float]:
        from scipy.optimize import fsolve

        a, b = self.cfg["henon"]["params"]["a"], self.cfg["henon"]["params"]["b"]

        def residual(p: np.ndarray) -> list[float]:
            x, y = p
            return [1.0 - a * x * x + y - x, b * x - y]

        seeds = [np.array([0.6, 0.2]), np.array([-1.1, -0.3])]

        def run() -> float:
            roots = []
            for s in seeds:
                sol, info, ier, _msg = fsolve(residual, s, full_output=True)
                if ier == 1:
                    roots.append(float(sol[0]))
            return max(roots) if roots else float("nan")

        return run

    # -- Poincaré section --------------------------------------------------- #

    def task_poincare_section(self, quick: bool) -> Callable[[], Any]:
        from scipy.integrate import solve_ivp

        rhs = _rossler_rhs(self.cfg["rossler"]["params"])
        ic = self.cfg["rossler"]["ic"]
        n = 200 if quick else 1000

        def event_y(_t: float, u: np.ndarray) -> float:
            return float(u[1])

        event_y.direction = 1.0  # type: ignore[attr-defined]  # upward y=0 crossings
        # ~6 time-units per Rössler revolution → budget enough time for n crossings.
        T = n * 6.5

        def run() -> int:
            sol = solve_ivp(
                rhs,
                (0.0, T),
                ic,
                method="DOP853",
                events=event_y,
                rtol=1e-9,
                atol=1e-9,
                dense_output=False,
            )
            return int(sol.t_events[0].size)

        return run
