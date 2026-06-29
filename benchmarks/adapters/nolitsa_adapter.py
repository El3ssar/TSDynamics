"""nolitsa adapter — from-data nonlinear time-series analysis.

``nolitsa`` returns the raw scaling *curves* (the correlation sum C(r), the
average divergence S(t)); the dimension / exponent is the slope of the scaling
region, which this adapter fits. Like nolds it is a from-data tool only, so
integration / bifurcation / basins / fixed-point rows stay blank.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import series

from adapters._base import BaseAdapter


class NolitsaAdapter(BaseAdapter):
    name = "nolitsa"
    language = "python"

    def _probe(self) -> tuple[bool, str, str]:
        import nolitsa

        return True, getattr(nolitsa, "__version__", "?"), ""

    def task_correlation_dimension(self, quick: bool) -> Callable[[], float]:
        from nolitsa import d2

        s = self.cfg["series"]
        x = np.ascontiguousarray(series.lorenz_series()[: s["corr_n"]])
        r = np.logspace(-1.5, 1.0, 40)

        def run() -> float:
            (rr, cc) = d2.c2_embed(
                x, dim=[s["embed_dim"]], tau=s["embed_delay"], r=r, window=s["theiler"]
            )[0]
            slopes = d2.d2(rr, cc)  # local d log C / d log r
            mid = slopes[len(slopes) // 3 : 2 * len(slopes) // 3]
            return float(np.median(mid))

        return run

    def task_lyapunov_from_data(self, quick: bool) -> Callable[[], float]:
        from nolitsa import lyapunov

        s = self.cfg["series"]
        x = series.lorenz_series()
        xs = np.ascontiguousarray(x[:: s["lyap_stride"]][: s["lyap_n"]])
        dt_eff = s["dt"] * s["lyap_stride"]

        def run() -> float:
            d = np.asarray(
                lyapunov.mle_embed(
                    xs,
                    dim=[s["embed_dim"]],
                    tau=s["lyap_delay"],
                    window=s["lyap_theiler"],
                    maxt=30,
                )[0]
            )
            t = np.arange(d.size) * dt_eff
            return float(np.polyfit(t[1:10], d[1:10], 1)[0])  # slope = λ (per time)

        return run

    def task_max_lyapunov(self, quick: bool) -> Callable[[], float]:
        from nolitsa import lyapunov

        h = series.henon_series(4000 if quick else 8000)

        def run() -> float:
            d = np.asarray(lyapunov.mle_embed(h, dim=[2], tau=1, window=10, maxt=20)[0])
            t = np.arange(d.size)  # per iteration
            return float(np.polyfit(t[1:8], d[1:8], 1)[0])

        return run

    def task_embedding_dimension(self, quick: bool) -> Callable[[], float]:
        from nolitsa import dimension

        s = self.cfg["series"]
        x = np.ascontiguousarray(series.lorenz_series()[: s["entropy_n"]])
        dims = np.arange(1, s["embed_max_dim"] + 1)

        def run() -> float:
            # fnn → (test1, test2, both); the embedding dim is the first where the
            # fraction of false nearest neighbours drops ~to zero.
            _f1, _f2, both = dimension.fnn(x, dim=dims, tau=s["embed_target_delay"], parallel=False)
            below = np.where(np.asarray(both) < 0.01)[0]
            return float(dims[below[0]]) if below.size else float(dims[-1])

        return run

    def task_surrogate_generation(self, quick: bool) -> Callable[[], None]:
        from nolitsa import surrogates

        x = np.ascontiguousarray(series.lorenz_series()[: self.cfg["series"]["entropy_n"]])

        def run() -> None:  # speed-only: generate (timed), no estimate
            surrogates.iaaft(x)

        return run
