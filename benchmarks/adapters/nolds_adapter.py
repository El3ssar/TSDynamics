"""nolds adapter — from-data nonlinear measures (Rosenstein, Grassberger).

``nolds`` estimates quantities from a scalar time series only: it has no notion
of equations, so it covers the correlation dimension and the maximal Lyapunov
exponent (the Rosenstein ``lyap_r`` estimator) but not integration, bifurcation
diagrams, basins, Poincaré sections or fixed points (all left blank). Every
data-driven task reads the *shared* series from :mod:`series`, identical to what
every other library sees.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import series

from adapters._base import BaseAdapter


class NoldsAdapter(BaseAdapter):
    name = "nolds"
    language = "python"

    def _probe(self) -> tuple[bool, str, str]:
        import nolds

        return True, getattr(nolds, "__version__", "?"), ""

    def task_correlation_dimension(self, quick: bool) -> Callable[[], float]:
        import nolds

        s = self.cfg["series"]
        x = np.ascontiguousarray(series.lorenz_series()[: s["corr_n"]])

        def run() -> float:
            return float(nolds.corr_dim(x, s["embed_dim"], lag=s["embed_delay"]))

        return run

    def task_sample_entropy(self, quick: bool) -> Callable[[], float]:
        import nolds

        s = self.cfg["series"]
        x = np.ascontiguousarray(series.lorenz_series()[: s["entropy_n"]])

        def run() -> float:
            return float(nolds.sampen(x, emb_dim=s["entropy_m"]))

        return run

    def task_dfa(self, quick: bool) -> Callable[[], float]:
        import nolds

        x = np.ascontiguousarray(series.white_noise_series()[: self.cfg["series"]["dfa_n"]])

        def run() -> float:
            return float(nolds.dfa(x))

        return run

    def task_hurst(self, quick: bool) -> Callable[[], float]:
        import nolds

        x = np.ascontiguousarray(series.white_noise_series()[: self.cfg["series"]["dfa_n"]])

        def run() -> float:
            return float(nolds.hurst_rs(x))

        return run

    def task_lyapunov_from_data(self, quick: bool) -> Callable[[], float]:
        import nolds

        s = self.cfg["series"]
        x = series.lorenz_series()
        xs = np.ascontiguousarray(x[:: s["lyap_stride"]][: s["lyap_n"]])
        dt_eff = s["dt"] * s["lyap_stride"]

        def run() -> float:
            # tau=dt_eff scales the slope into per-time units (matches λ ref).
            return float(
                nolds.lyap_r(
                    xs,
                    emb_dim=s["embed_dim"],
                    lag=s["lyap_delay"],
                    min_tsep=s["lyap_theiler"],
                    tau=dt_eff,
                )
            )

        return run

    def task_max_lyapunov(self, quick: bool) -> Callable[[], float]:
        import nolds

        # Estimate the Hénon maximal exponent from a generated orbit (per
        # iteration ⇒ tau=1), the data-side counterpart of ts.max_lyapunov.
        h = series.henon_series(4000 if quick else 8000)

        def run() -> float:
            return float(nolds.lyap_r(h, emb_dim=2, lag=1, min_tsep=10, tau=1))

        return run
