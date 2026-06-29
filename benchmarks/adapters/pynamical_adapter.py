"""pynamical adapter — discrete-map bifurcation diagrams.

``pynamical`` specialises in 1-D/2-D maps (logistic, cubic, Hénon): its headline
capability is the logistic-map bifurcation diagram via ``simulate``. It offers no
ODE integration, Lyapunov spectra, fractal dimensions, basins or Poincaré tools,
so those rows stay blank. ``simulate`` is numba-jitted; the harness's warm-up
call pays the one-time compile outside the timed runs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from adapters._base import BaseAdapter


class PynamicalAdapter(BaseAdapter):
    name = "pynamical"
    language = "python"

    def _probe(self) -> tuple[bool, str, str]:
        import pynamical

        return True, getattr(pynamical, "__version__", "?"), ""

    def task_bifurcation_diagram(self, quick: bool) -> Callable[[], Any]:
        from pynamical import logistic_map, simulate

        lg = self.cfg["logistic"]
        n_rates = 200 if quick else lg["n_rates"]

        def run() -> None:
            simulate(
                model=logistic_map,
                num_gens=lg["n_gens"],
                rate_min=lg["r_min"],
                rate_max=lg["r_max"],
                num_rates=n_rates,
                num_discard=lg["n_discard"],
                initial_pop=lg["ic"],
            )

        return run
