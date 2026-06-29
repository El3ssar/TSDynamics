"""neurokit2 adapter — broad from-data complexity suite.

``neurokit2`` covers the widest from-data surface here: entropy (sample /
permutation / multiscale), DFA, Hurst, correlation dimension, RQA, embedding
dimension, Lyapunov-from-data and surrogate generation. It has no ODE
integration / from-system Lyapunov, so those rows stay blank. Most functions
return a ``(value, info)`` tuple — the scalar is element ``[0]``; ``complexity_rqa``
returns ``(DataFrame, info)``.
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

    def task_sample_entropy(self, quick: bool) -> Callable[[], float]:
        import neurokit2 as nk

        x, m = self._lorenz(), self.cfg["series"]["entropy_m"]

        def run() -> float:
            return float(nk.entropy_sample(x, delay=1, dimension=m, tolerance="sd")[0])

        return run

    def task_permutation_entropy(self, quick: bool) -> Callable[[], float]:
        import neurokit2 as nk

        x, m = self._lorenz(), self.cfg["series"]["entropy_m"]

        def run() -> float:
            return float(nk.entropy_permutation(x, delay=1, dimension=m + 1, corrected=True)[0])

        return run

    def task_multiscale_entropy(self, quick: bool) -> Callable[[], float]:
        import neurokit2 as nk

        x, m = self._lorenz(), self.cfg["series"]["entropy_m"]

        def run() -> float:
            return float(
                nk.entropy_multiscale(x, dimension=m + 1, tolerance="sd", method="MSEn")[0]
            )

        return run

    def task_dfa(self, quick: bool) -> Callable[[], float]:
        import neurokit2 as nk

        x = np.ascontiguousarray(series.white_noise_series()[: self.cfg["series"]["dfa_n"]])

        def run() -> float:
            return float(nk.fractal_dfa(x, order=1)[0])

        return run

    def task_hurst(self, quick: bool) -> Callable[[], float]:
        import neurokit2 as nk

        x = np.ascontiguousarray(series.white_noise_series()[: self.cfg["series"]["dfa_n"]])

        def run() -> float:
            return float(nk.fractal_hurst(x, corrected=True)[0])

        return run

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

    def task_surrogate_generation(self, quick: bool) -> Callable[[], None]:
        import neurokit2 as nk

        x = self._lorenz()

        def run() -> None:  # speed-only: generate (timed), no estimate
            nk.signal_surrogate(x, method="IAAFT", random_state=0)

        return run
