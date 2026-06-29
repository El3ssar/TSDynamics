"""antropy adapter — entropy + DFA from a scalar series.

``antropy`` is a feature/complexity-extraction library (sample/permutation entropy,
DFA, fractal-dimension features). It has no Hurst exponent, correlation dimension,
RQA, embedding, surrogate or dynamics tooling, so those rows stay blank. Every task
reads the shared series so the estimate is comparable to the other libraries.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import series

from adapters._base import BaseAdapter


class AntropyAdapter(BaseAdapter):
    name = "antropy"
    language = "python"

    def _probe(self) -> tuple[bool, str, str]:
        import antropy

        return True, getattr(antropy, "__version__", "?"), ""

    def task_sample_entropy(self, quick: bool) -> Callable[[], float]:
        import antropy

        s = self.cfg["series"]
        x = np.ascontiguousarray(series.lorenz_series()[: s["entropy_n"]])

        def run() -> float:
            return float(antropy.sample_entropy(x, order=s["entropy_m"]))

        return run

    def task_permutation_entropy(self, quick: bool) -> Callable[[], float]:
        import antropy

        s = self.cfg["series"]
        x = np.ascontiguousarray(series.lorenz_series()[: s["entropy_n"]])

        def run() -> float:
            return float(antropy.perm_entropy(x, order=s["entropy_m"] + 1, delay=1, normalize=True))

        return run

    def task_dfa(self, quick: bool) -> Callable[[], float]:
        import antropy

        x = np.ascontiguousarray(series.white_noise_series()[: self.cfg["series"]["dfa_n"]])

        def run() -> float:
            return float(antropy.detrended_fluctuation(x))

        return run
