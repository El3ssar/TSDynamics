"""dysts adapter — the GilpinLab chaotic-systems catalogue + integrator.

``dysts`` ships ~130 systems with a SciPy ``solve_ivp`` integrator plus analysis
helpers (``gp_dim`` correlation dimension, ``dfa``). It contributes a second real
dynamics library to the integration rows and to the from-data dimension/DFA rows.

Deliberately **not** wired to the Lyapunov rows: dysts rescales each system's time
axis per characteristic period (for ML benchmarking), so its Lyapunov exponents
are in rescaled-time units (Lorenz ≈ 0.44, not the physical 0.906) — incomparable
to this benchmark's physical-time literature reference. Integration is in physical
time when ``m.dt = None`` is set (so the ``dt=`` kwarg governs), so the integration
rows are a clean comparison.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import series

from adapters._base import BaseAdapter


class DystsAdapter(BaseAdapter):
    name = "dysts"
    language = "python"

    def _probe(self) -> tuple[bool, str, str]:
        import importlib.metadata as md

        import dysts  # noqa: F401

        try:
            ver = md.version("dysts")
        except Exception:  # pragma: no cover
            ver = "?"
        return True, ver, ""

    def _lorenz(self) -> Any:
        from dysts.flows import Lorenz

        m = Lorenz()
        m.dt = None  # let the dt= kwarg govern the step (else dysts' tiny built-in dt wins)
        return m

    # -- integration (physical time; SciPy RK45 under the hood) ------------- #

    def _integrate(self, T: float, rtol: float, atol: float, ic: Any | None = None) -> np.ndarray:
        m = self._lorenz()
        if ic is not None:
            m.ic = np.asarray(ic, dtype=float)
        dt = self._intg["dt"]
        n = int(round(T / dt))
        return np.asarray(
            m.make_trajectory(n, dt=dt, resample=False, method="RK45", rtol=rtol, atol=atol),
            dtype=float,
        )

    def task_integrate_short(self, quick: bool) -> Callable[[], Any]:
        T = 50.0 if quick else self._intg["t_short"]

        def run() -> None:
            self._integrate(T, 1e-9, 1e-9)

        return run

    def task_integrate_long(self, quick: bool) -> Callable[[], Any]:
        T = 1000.0 if quick else self._intg["t_long"]

        def run() -> None:
            self._integrate(T, 1e-9, 1e-9)

        return run

    # NOTE: no integrate_accuracy — dysts' IC / parameter / per-period time-rescaling
    # conventions do not match this benchmark's fixed [1,1,1] reference trajectory, so
    # a final-state deviation would reflect a convention mismatch, not solver error.
    # dysts stays on the (IC-independent) short/long integration-SPEED rows.

    # -- from-data dimension / scaling -------------------------------------- #

    def task_correlation_dimension(self, quick: bool) -> Callable[[], float]:
        from dysts.analysis import gp_dim

        s = self.cfg["series"]
        x = series.lorenz_series()[: s["corr_n"]]
        emb = series.delay_embed(x, dim=s["embed_dim"], delay=s["embed_delay"])

        def run() -> float:
            return float(gp_dim(emb))

        return run

    def task_dfa(self, quick: bool) -> Callable[[], float]:
        from dysts.analysis import dfa

        x = np.ascontiguousarray(series.white_noise_series()[: self.cfg["series"]["dfa_n"]])

        def run() -> float:
            return float(dfa(x))

        return run
