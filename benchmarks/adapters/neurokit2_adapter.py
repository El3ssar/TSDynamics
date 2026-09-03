"""neurokit2 adapter — from-data phase-space measures.

``neurokit2`` contributes the correlation-dimension, RQA and embedding-dimension
rows. It has no ODE integration / from-system Lyapunov, so those rows stay blank.
Most functions return a ``(value, info)`` tuple — the scalar is element ``[0]``;
``complexity_rqa`` returns ``(DataFrame, info)``.

Its entropy / DFA / Hurst / surrogate tasks were dropped with the v6 scope
narrowing: TSDynamics no longer ships those estimators, so the rows had no
TSDynamics column left to compare against.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import series

from adapters._base import BaseAdapter


class NeuroKit2Adapter(BaseAdapter):
    name = "neurokit2"
    language = "python"

    def _probe(self) -> tuple[bool, str, str]:
        import neurokit2 as nk

        return True, getattr(nk, "__version__", "?"), ""

    def _lorenz(self, n_key: str = "entropy_n") -> np.ndarray:
        return np.ascontiguousarray(series.lorenz_series()[: self.cfg["series"][n_key]])

    def task_correlation_dimension(self, quick: bool) -> Callable[[], float]:
        import neurokit2 as nk

        s = self.cfg["series"]
        x = np.ascontiguousarray(series.lorenz_series()[: s["corr_n"]])

        def run() -> float:
            return float(
                nk.fractal_correlation(
                    x, delay=s["embed_delay"], dimension=s["embed_dim"], radius=64
                )[0]
            )

        return run

    def task_rqa_determinism(self, quick: bool) -> Callable[[], float]:
        import neurokit2 as nk

        s = self.cfg["series"]
        x = self._lorenz("rqa_n")

        def run() -> float:
            df, _info = nk.complexity_rqa(
                x,
                dimension=s["rqa_embed_dim"],
                delay=s["rqa_embed_delay"],
                tolerance="sd",
                min_linelength=2,
                show=False,
            )
            return float(df["Determinism"].iloc[0])

        return run

    def task_embedding_dimension(self, quick: bool) -> Callable[[], float]:
        import neurokit2 as nk

        s = self.cfg["series"]
        x = self._lorenz()

        def run() -> float:
            val, _info = nk.complexity_dimension(
                x,
                delay=s["embed_target_delay"],
                dimension_max=s["embed_max_dim"],
                method="afnn",
                show=False,
            )
            return float(val)

        return run
